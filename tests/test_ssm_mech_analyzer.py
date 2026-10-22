"""
Integration smoke tests for MambaAnalyzer.

These tests are skipped unless the HF model is locally cached and
torch is available, so they can run in CI without downloading weights.
They use a tiny synthetic Mamba config to avoid model downloads.
"""

import numpy as np
import pytest
import torch

# Marker: skip if transformers or mamba model weights are not available
HAS_TRANSFORMERS = True
try:
    from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
except ImportError:
    HAS_TRANSFORMERS = False


skip_no_transformers = pytest.mark.skipif(
    not HAS_TRANSFORMERS,
    reason="transformers not installed",
)


def _make_tiny_mamba():
    """Build a 2-layer Mamba model with tiny dimensions for CPU testing."""
    cfg = MambaConfig(
        vocab_size=128,
        hidden_size=64,
        state_size=8,
        intermediate_size=128,
        num_hidden_layers=2,
        time_step_rank=16,
        conv_kernel=4,
    )
    model = MambaForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@skip_no_transformers
class TestMambaAnalyzerUnit:
    """Unit tests against a tiny synthetic model (no HF download needed)."""

    def _get_analyzer(self):
        from ssm_mech.analyzer import MambaAnalyzer, BalancedBasis, SurgeryResult
        from ssm_mech.model_utils import n_layers
        from ssm_mech.tasks import build_ioi
        from transformers import AutoTokenizer

        model = _make_tiny_mamba()
        # MambaAnalyzer.__init__ calls load_model; bypass by constructing directly
        analyzer = object.__new__(MambaAnalyzer)
        analyzer.model_id = "tiny-mamba-test"

        # Use the GPT-2 tokenizer (small, always cached)
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            pytest.skip("gpt2 tokenizer not cached")

        analyzer.model     = model
        analyzer.tokenizer = tokenizer
        analyzer.device    = torch.device("cpu")
        analyzer._n_layers = n_layers(model)
        return analyzer

    def test_fit_returns_balanced_basis(self):
        from ssm_mech.analyzer import BalancedBasis
        from ssm_mech.tasks import build_ioi

        a       = self._get_analyzer()
        prompts = [p.text for p in build_ioi(n=10)]
        basis   = a.fit(layer=0, prompts=prompts)
        assert isinstance(basis, BalancedBasis)
        assert basis.layer == 0
        assert basis.V.ndim == 3

    def test_etmfd_returns_float(self):
        from ssm_mech.tasks import build_ioi

        a       = self._get_analyzer()
        prompts = [p.text for p in build_ioi(n=5)]
        val     = a.etmfd(layer=0, prompts=prompts)
        assert isinstance(val, float)
        assert val >= 0.0

    def test_timescales_shape(self):
        from ssm_mech.tasks import build_ioi

        a       = self._get_analyzer()
        prompts = [p.text for p in build_ioi(n=5)]
        basis   = a.fit(layer=0, prompts=prompts)
        taus    = a.timescales(layer=0, basis=basis)
        d_inner, d_state, _ = basis.V.shape
        assert taus.shape == (d_inner, d_state)

    def test_basis_save_load_roundtrip(self, tmp_path):
        from ssm_mech.analyzer import BalancedBasis
        from ssm_mech.tasks import build_ioi

        a       = self._get_analyzer()
        prompts = [p.text for p in build_ioi(n=5)]
        basis   = a.fit(layer=0, prompts=prompts)
        path    = tmp_path / "basis.npz"
        a.save_basis(basis, path)
        basis2 = a.load_basis(path)
        np.testing.assert_array_equal(basis.V, basis2.V)
        assert basis.n_tokens == basis2.n_tokens

    def test_surgery_returns_result(self):
        from ssm_mech.analyzer import SurgeryResult
        from ssm_mech.tasks import build_ioi

        a          = self._get_analyzer()
        ioi        = build_ioi(n=5)
        basis      = a.fit(layer=0, prompts=[p.text for p in ioi])
        result     = a.surgery(layer=0, basis=basis, k=1, eval_prompts=ioi, task="ioi")
        assert isinstance(result, SurgeryResult)
        assert isinstance(result.delta, float)

    def test_surgery_invalid_task_raises(self):
        from ssm_mech.tasks import build_ioi

        a     = self._get_analyzer()
        ioi   = build_ioi(n=3)
        basis = a.fit(layer=0, prompts=[p.text for p in ioi])
        with pytest.raises(ValueError, match="task must be"):
            a.surgery(layer=0, basis=basis, k=1, eval_prompts=ioi, task="fake_task")

    def test_layer_sweep_collects_results(self):
        from ssm_mech.tasks import build_ioi

        a     = self._get_analyzer()
        ioi   = build_ioi(n=3)
        texts = [p.text for p in ioi]
        res   = a.layer_sweep(basis_prompts=texts, eval_prompts=ioi, task="ioi", k=1)
        # Should have one result per layer (model has 2 layers)
        assert len(res) == 2
