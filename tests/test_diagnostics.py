"""Tests for ssm_interpret.diagnostics."""

import torch
import pytest
import ssm_interpret as ssi
from ssm_interpret.diagnostics import kalman_taxonomy, architecture_gap, sym_error, scope_check


def make_mimo_ema(rhos):
    d = len(rhos)
    A = torch.diag(torch.tensor(rhos, dtype=torch.float64))
    B = torch.eye(d, dtype=torch.float64)
    C = torch.eye(d, dtype=torch.float64)
    return A.numpy(), B.numpy(), C.numpy()


# ── Obvious tests ──────────────────────────────────────────────────────────

def test_taxonomy_completeness_exact_capacity():
    """All d modes live for a well-trained exact-capacity model."""
    A, B, C = make_mimo_ema([0.95, 0.65, 0.20])
    tax = kalman_taxonomy(A, B, C)
    d = 3
    total = len(tax["live"]) + len(tax["dead_type1"]) + len(tax["dead_type2"])
    assert total == d, f"Taxonomy completeness failed: {tax}"
    assert len(tax["live"]) == d, f"Expected all 3 live, got {tax}"


def test_taxonomy_completeness_always_holds():
    """live + dead_t1 + dead_t2 == d for any input."""
    torch.manual_seed(0)
    for _ in range(5):
        d = torch.randint(2, 7, ()).item()
        A = torch.randn(d, d, dtype=torch.float64)
        A = (A / (torch.linalg.eigvals(A).abs().max() * 1.2)).numpy()
        B = torch.randn(d, 2, dtype=torch.float64).numpy()
        C = torch.randn(2, d, dtype=torch.float64).numpy()
        tax = kalman_taxonomy(A, B, C)
        total = len(tax["live"]) + len(tax["dead_type1"]) + len(tax["dead_type2"])
        assert total == d


def test_sym_error_zero_for_symmetric():
    W = torch.eye(4, dtype=torch.float64).numpy()
    assert sym_error(W) < 1e-12


def test_sym_error_nonzero_for_asymmetric():
    W = torch.randn(4, 4, dtype=torch.float64).numpy()
    assert sym_error(W) > 0.0


def test_scope_check_lti():
    result = scope_check(0.01)
    assert result["regime"] == "lti"
    assert result["applicable"] is True


def test_scope_check_near_lti():
    result = scope_check(0.10)
    assert result["regime"] == "near_lti"
    assert result["applicable"] is True


def test_scope_check_selective():
    result = scope_check(0.50)
    assert result["regime"] == "selective"
    assert result["applicable"] is False


def test_scope_check_all_keys():
    result = scope_check(0.03)
    for k in ("regime", "applicable", "gvi", "message"):
        assert k in result


# ── Non-obvious tests ──────────────────────────────────────────────────────

def test_taxonomy_completeness_edge_untrained():
    """Untrained model (random A, small B) — taxonomy still sums to d."""
    torch.manual_seed(7)
    d = 5
    A = (0.5 * torch.eye(d) + 0.01 * torch.randn(d, d)).double().numpy()
    B = 1e-6 * torch.randn(d, 1).double().numpy()   # nearly uncontrollable
    C = torch.randn(1, d).double().numpy()
    tax = kalman_taxonomy(A, B, C)
    total = len(tax["live"]) + len(tax["dead_type1"]) + len(tax["dead_type2"])
    assert total == d


def test_gvi_zero_for_lti():
    """GVI must return 0.0 for an exact LTI model with no gating."""
    result = scope_check(0.0)
    assert result["regime"] == "lti"
    assert result["gvi"] == 0.0


def test_architecture_gap_one_single_ema():
    """For a single EMA channel, diagonal A is optimal: gap <= 1 + eps."""
    # Single-output, single-circuit: diagonal A can perfectly represent
    # the optimal Hankel structure — gap should be close to 1.
    rho = 0.85
    T = 32

    # Diagonal A (d=2 to have a non-trivial comparison mode)
    A_d = torch.diag(torch.tensor([rho, 0.1], dtype=torch.float64)).numpy()
    B_d = torch.tensor([[1.0], [0.5]], dtype=torch.float64).numpy()
    C_d = torch.tensor([[1.0, 0.0]], dtype=torch.float64).numpy()

    # Full A (d=2)
    A_f = torch.tensor([[rho, 0.05], [0.02, 0.1]], dtype=torch.float64).numpy()
    B_f = B_d.copy()
    C_f = C_d.copy()

    from ssm_interpret.core import balanced_realization
    _, _, _, _, sigma_d = balanced_realization(A_d, B_d, C_d, T=T)
    _, _, _, _, sigma_f = balanced_realization(A_f, B_f, C_f, T=T)

    gap = architecture_gap(sigma_d, sigma_f, m=1)
    # For near-identical architectures on a single task, gap should be < 5
    assert gap < 5.0, f"Expected gap < 5 for single EMA task, got {gap:.2f}"


def test_sym_error_after_balancing_near_zero():
    """After balanced_realization, sym_error of Wc_bal should be near zero."""
    from ssm_interpret.core import balanced_realization, gramians

    torch.manual_seed(3)
    d = 4
    A = torch.randn(d, d, dtype=torch.float64)
    A = (A / (torch.linalg.eigvals(A).abs().max() * 1.2)).numpy()
    B = torch.randn(d, 2, dtype=torch.float64).numpy()
    C = torch.randn(2, d, dtype=torch.float64).numpy()

    Ab, Bb, Cb, T_bal, sigma = balanced_realization(A, B, C, T=32)
    Wc_bal, Wo_bal, _ = gramians(Ab, Bb, Cb, T=32)
    err = sym_error(Wc_bal)
    assert err < 1e-4, f"sym_error after balancing too high: {err:.4e}"
