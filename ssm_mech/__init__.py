"""
ssm_mech -- Causal mode analysis for SSM language models.

Provides basis-invariant extraction of balanced modes via Gramian decomposition,
causal surgery (rank-k state projection), ETMFD diagnostics, and mode timescale
analysis for HuggingFace Mamba / FalconH1 / Nemotron-H models.

Usage:
    from ssm_mech import MambaAnalyzer

    a = MambaAnalyzer("state-spaces/mamba-370m-hf")
    basis = a.fit(layer=39, prompts=ioi_prompts)   # IOI-specific Gramian
    delta = a.surgery(layer=39, basis=basis, k=1, eval_prompts=eval_set)
"""

from .analyzer import MambaAnalyzer

__version__ = "0.1.0"
__all__ = ["MambaAnalyzer"]
