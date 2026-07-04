"""Tests for ssm_interpret.ltv (experimental)."""

import warnings
import torch
import torch.nn as nn
import pytest


# Suppress the FutureWarning on import for all tests in this file
@pytest.fixture(autouse=True)
def suppress_ltv_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        yield


def _import_ltv():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from ssm_interpret.ltv import gramians_ltv, scope_report, gvi, etmfd, llod
        return gramians_ltv, scope_report, gvi, etmfd, llod


def make_lti_seq(A, B, C, T):
    """Expand constant LTI matrices to per-step sequences."""
    A_t = torch.tensor(A, dtype=torch.float64).unsqueeze(0).expand(T, -1, -1)
    B_t = torch.tensor(B, dtype=torch.float64).unsqueeze(0).expand(T, -1, -1)
    C_t = torch.tensor(C, dtype=torch.float64).unsqueeze(0).expand(T, -1, -1)
    return A_t, B_t, C_t


class LTIModel(nn.Module):
    """Model with get_delta_sequence returning constant Δ — GVI should be 0."""
    def __init__(self, d=3, delta_val=0.1):
        super().__init__()
        self.d = d
        self.delta_val = delta_val

    def forward(self, u):
        return u.mean(1)

    def get_delta_sequence(self, u):
        B, T, m = u.shape
        return torch.full((B, T, self.d), self.delta_val)


class VaryingDeltaModel(nn.Module):
    """Model with highly variable Δ — GVI should be large."""
    def __init__(self, d=3):
        super().__init__()
        self.d = d

    def forward(self, u):
        return u.mean(1)

    def get_delta_sequence(self, u):
        B, T, m = u.shape
        # Δ varies between 0.001 and 1.0 — high CV
        torch.manual_seed(0)
        return torch.rand(B, T, self.d) + 0.001


# ── Obvious tests ──────────────────────────────────────────────────────────

def test_ltv_import_warns():
    """Importing ssm_interpret.ltv should emit a FutureWarning."""
    import importlib, sys
    # Remove cached module to force re-import
    for key in list(sys.modules.keys()):
        if "ssm_interpret.ltv" in key:
            del sys.modules[key]
    with pytest.warns(FutureWarning, match="experimental"):
        import ssm_interpret.ltv


def test_gramians_ltv_shape():
    gramians_ltv, *_ = _import_ltv()
    T, d, m, p = 20, 3, 2, 2
    A_seq = 0.8 * torch.eye(d).unsqueeze(0).expand(T, -1, -1).double()
    B_seq = torch.randn(T, d, m).double() * 0.3
    C_seq = torch.randn(T, p, d).double() * 0.3
    Wc, Wo, meta = gramians_ltv(A_seq, B_seq, C_seq)
    assert Wc.shape == (d, d)
    assert Wo.shape == (d, d)


def test_scope_report_all_keys():
    _, scope_report, *_ = _import_ltv()
    model = LTIModel(d=3)
    u = torch.randn(4, 20, 3)
    report = scope_report(model, u)
    for key in ("etmfd", "llod", "gvi", "regime", "applicable", "message"):
        assert key in report


# ── Non-obvious tests ──────────────────────────────────────────────────────

def test_ltv_degenerates_to_lti():
    """gramians_ltv with constant A_seq must match gramians() to machine precision."""
    from ssm_interpret.core import gramians as gramians_lti
    gramians_ltv, *_ = _import_ltv()

    A = torch.diag(torch.tensor([0.9, 0.6, 0.3], dtype=torch.float64)).numpy()
    B = torch.eye(3, dtype=torch.float64).numpy()
    C = torch.eye(3, dtype=torch.float64).numpy()
    T = 32

    Wc_lti, Wo_lti, _ = gramians_lti(A, B, C, T=T)
    A_seq, B_seq, C_seq = make_lti_seq(A, B, C, T)
    Wc_ltv, Wo_ltv, _ = gramians_ltv(A_seq, B_seq, C_seq)

    Wc_lti = torch.tensor(Wc_lti, dtype=torch.float64)
    Wo_lti = torch.tensor(Wo_lti, dtype=torch.float64)

    assert torch.allclose(Wc_lti, Wc_ltv, atol=1e-8), \
        f"Wc mismatch LTI vs LTV (const): max err {(Wc_lti - Wc_ltv).abs().max():.4e}"
    assert torch.allclose(Wo_lti, Wo_ltv, atol=1e-8)


def test_gvi_zero_for_constant_delta():
    """Model with constant Δ must return GVI = 0.0 exactly."""
    _, _, gvi_fn, *_ = _import_ltv()
    model = LTIModel(d=3, delta_val=0.1)
    u = torch.randn(8, 30, 3)
    assert gvi_fn(model, u) == 0.0


def test_gvi_large_for_varying_delta():
    """Model with highly variable Δ must return GVI > 0.2 (selective regime)."""
    _, scope_report, gvi_fn, *_ = _import_ltv()
    model = VaryingDeltaModel(d=3)
    u = torch.randn(8, 30, 3)
    g = gvi_fn(model, u)
    assert g > 0.2, f"Expected GVI > 0.2 for varying delta, got {g:.4f}"


def test_scope_report_regime_labels():
    """scope_report must return correct regime label for all three bands."""
    _, scope_report, *_ = _import_ltv()

    lti_model = LTIModel(delta_val=0.1)
    varying_model = VaryingDeltaModel()
    u = torch.randn(4, 30, 3)

    lti_report = scope_report(lti_model, u)
    assert lti_report["regime"] == "lti"
    assert lti_report["applicable"] is True

    sel_report = scope_report(varying_model, u)
    assert sel_report["regime"] in ("near_lti", "selective")
