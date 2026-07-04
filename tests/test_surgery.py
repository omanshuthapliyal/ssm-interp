"""Tests for ssm_interpret.surgery."""

import torch
import pytest
import ssm_interpret as ssi
from ssm_interpret.core import balanced_realization, impulse_response


def _run_ssm(A, B, C, u):
    """Forward pass of a discrete-time LTI SSM. u: (T, m) → y: (T, p)."""
    A, B, C = [torch.tensor(x, dtype=torch.float64) for x in (A, B, C)]
    u = torch.tensor(u, dtype=torch.float64)
    T, m = u.shape
    d = A.shape[0]
    x = torch.zeros(d, dtype=torch.float64)
    ys = []
    for t in range(T):
        x = A @ x + B @ u[t]
        ys.append(C @ x)
    return torch.stack(ys)   # (T, p)


def make_mimo_ema(rhos, seed=0):
    """Diagonal MIMO SSM — one circuit per output channel."""
    d = len(rhos)
    A = torch.diag(torch.tensor(rhos, dtype=torch.float64))
    B = torch.eye(d, dtype=torch.float64)
    C = torch.eye(d, dtype=torch.float64)
    return A, B, C


# ── Obvious tests ──────────────────────────────────────────────────────────

def test_ablate_dynamics_changes_A():
    rhos = [0.9, 0.6, 0.3]
    A, B, C = make_mimo_ema(rhos)
    _, _, _, T_bal, _ = balanced_realization(A, B, C, T=32)
    A2, B2, C2 = ssi.ablate(A, B, C, T_bal, k=0, mode="dynamics")
    A2 = torch.tensor(A2, dtype=torch.float64)
    assert not torch.allclose(torch.tensor(A, dtype=torch.float64), A2)


def test_ablate_output_changes_C():
    rhos = [0.9, 0.6, 0.3]
    A, B, C = make_mimo_ema(rhos)
    _, _, _, T_bal, _ = balanced_realization(A, B, C, T=32)
    A2, B2, C2 = ssi.ablate(A, B, C, T_bal, k=0, mode="output")
    assert torch.allclose(torch.tensor(A, dtype=torch.float64),
                          torch.tensor(A2, dtype=torch.float64))
    assert not torch.allclose(torch.tensor(C, dtype=torch.float64),
                               torch.tensor(C2, dtype=torch.float64))


def test_ablate_invalid_mode():
    A, B, C = make_mimo_ema([0.9, 0.5])
    _, _, _, T_bal, _ = balanced_realization(A, B, C, T=20)
    with pytest.raises(ValueError, match="mode"):
        ssi.ablate(A, B, C, T_bal, k=0, mode="wrong")


def test_mimo_selectivity_exact():
    """Diagonal MIMO: ablating mode k → 100% target, ~0% others (exact by algebra)."""
    torch.manual_seed(0)
    rhos = [0.95, 0.65, 0.20]
    A, B, C = make_mimo_ema(rhos)
    Ab, Bb, Cb, T_bal, sigma = balanced_realization(A, B, C, T=20)

    T_seq = 20
    u = torch.randn(T_seq, 3, dtype=torch.float64)
    y_base = _run_ssm(A, B, C, u)
    y_null = torch.zeros_like(y_base)

    for k in range(3):
        A2, B2, C2 = ssi.ablate(A, B, C, T_bal, k=k, mode="dynamics")
        y_surg = _run_ssm(A2, B2, C2, u)
        sel = ssi.selectivity(
            y_surg.unsqueeze(0), y_base.unsqueeze(0), y_null.unsqueeze(0),
            target_channel=k,
        )
        assert sel > 10.0, f"Mode {k}: expected sel >> 1, got {sel:.2f}"


def test_faithfulness_ratio():
    delta_active   = 3.5
    delta_inactive = 0.28
    ratio = ssi.faithfulness_ratio(delta_active, delta_inactive)
    assert abs(ratio - 3.5 / 0.28) < 1e-8


def test_faithfulness_zero_inactive():
    assert ssi.faithfulness_ratio(1.0, 0.0) == float("inf")


# ── Non-obvious tests ──────────────────────────────────────────────────────

def test_dynamics_ne_output_for_live_mode():
    """Dynamics surgery and output surgery give DIFFERENT post-surgery outputs
    for a live mode — they are not equivalent in general."""
    rhos = [0.9, 0.6, 0.3]
    A, B, C = make_mimo_ema(rhos)
    Ab, Bb, Cb, T_bal, sigma = balanced_realization(A, B, C, T=32)

    torch.manual_seed(1)
    u = torch.randn(30, 3, dtype=torch.float64)

    A_d, B_d, C_d = ssi.ablate(A, B, C, T_bal, k=0, mode="dynamics")
    A_o, B_o, C_o = ssi.ablate(A, B, C, T_bal, k=0, mode="output")

    y_dynamics = _run_ssm(A_d, B_d, C_d, u)
    y_output   = _run_ssm(A_o, B_o, C_o, u)

    diff = (y_dynamics - y_output).norm()
    assert diff > 1e-6, \
        f"Dynamics and output surgery gave identical results (diff={diff:.2e}). " \
        "They should differ for live modes."


def test_surgery_truncation_noncommutativity():
    """ablate(dynamics, k=0) then truncate(d-1) != truncate(k=d-2) directly.

    After dynamics surgery the system is still d-dimensional (one pole zeroed);
    truncation gives a (d-1)-dimensional system. Direct truncation gives a
    different (d-1)-dimensional system — their impulse responses differ.
    """
    rhos = [0.9, 0.6, 0.3]
    A, B, C = make_mimo_ema(rhos)
    Ab, Bb, Cb, T_bal, sigma = balanced_realization(A, B, C, T=32)
    Ab, Bb, Cb, T_bal, sigma = [torch.tensor(x, dtype=torch.float64)
                                  for x in (Ab, Bb, Cb, T_bal, sigma)]

    # Path 1: ablate k=0 then truncate last mode
    A_s, B_s, C_s = ssi.ablate(A, B, C, T_bal, k=0, mode="dynamics")
    Ab_s, Bb_s, Cb_s, _, _ = balanced_realization(A_s, B_s, C_s, T=32)
    Ab_s_k, Bb_s_k, Cb_s_k = ssi.truncate(Ab_s, Bb_s, Cb_s, k=2)
    H_path1 = torch.tensor(impulse_response(Ab_s_k, Bb_s_k, Cb_s_k, T=20))

    # Path 2: direct truncation to k=2
    Ab_k, Bb_k, Cb_k = ssi.truncate(Ab, Bb, Cb, k=2)
    H_path2 = torch.tensor(impulse_response(Ab_k, Bb_k, Cb_k, T=20))

    diff = (H_path1 - H_path2).norm()
    assert diff > 1e-4, \
        f"Surgery+truncation should differ from direct truncation, diff={diff:.4e}"


def test_anti_selectivity_entangled_timescales():
    """For a SISO mixture with very close rhos, balanced modes mix the circuits
    and surgery on mode 0 is not isolated to one 'channel' — selectivity degrades.

    We use a 2-output SISO-style system where both outputs are weighted sums
    of the same two EMA modes, forcing the balanced modes to straddle both channels.
    """
    # Two EMA modes with nearly identical timescales
    rhos = [0.91, 0.89]
    A = torch.diag(torch.tensor(rhos, dtype=torch.float64))
    # Both inputs feed both states (not identity B)
    B = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
    # Both states contribute to both outputs (not identity C)
    C = torch.tensor([[1.0, 0.8], [0.8, 1.0]], dtype=torch.float64)
    Ab, Bb, Cb, T_bal, sigma = balanced_realization(A, B, C, T=64)

    torch.manual_seed(5)
    u = torch.randn(40, 2, dtype=torch.float64)
    y_base = _run_ssm(A, B, C, u)
    y_null = torch.zeros_like(y_base)

    A2, B2, C2 = ssi.ablate(A, B, C, T_bal, k=0, mode="dynamics")
    y_surg = _run_ssm(A2, B2, C2, u)

    sel = ssi.selectivity(
        y_surg.unsqueeze(0), y_base.unsqueeze(0), y_null.unsqueeze(0),
        target_channel=0,
    )
    # With mixed B/C and entangled timescales, selectivity should be much lower
    # than the 251x achieved on well-separated MIMO tasks
    assert sel < 50.0, \
        f"Expected degraded selectivity for entangled mixed system, got {sel:.2f}"


def test_random_direction_baseline():
    """Ablating a random (non-balanced) direction gives low selectivity."""
    rhos = [0.95, 0.65, 0.20]
    A, B, C = make_mimo_ema(rhos)
    Ab, Bb, Cb, T_bal, sigma = balanced_realization(A, B, C, T=20)
    T_bal_t = torch.tensor(T_bal, dtype=torch.float64)

    torch.manual_seed(42)
    # Replace T_bal with a random matrix to ablate a non-balanced direction
    T_rand = torch.randn(3, 3, dtype=torch.float64)
    T_rand = T_rand + 3 * torch.eye(3, dtype=torch.float64)

    u = torch.randn(20, 3, dtype=torch.float64)
    y_base = _run_ssm(A, B, C, u)
    y_null = torch.zeros_like(y_base)

    A2, B2, C2 = ssi.ablate(A, B, C, T_rand.numpy(), k=0, mode="dynamics")
    y_surg = _run_ssm(A2, B2, C2, u)

    sel_random = ssi.selectivity(
        y_surg.unsqueeze(0), y_base.unsqueeze(0), y_null.unsqueeze(0),
        target_channel=0,
    )
    # Random direction should be far less selective than balanced
    A_bal, B_bal, C_bal = ssi.ablate(A, B, C, T_bal, k=0, mode="dynamics")
    y_surg_bal = _run_ssm(A_bal, B_bal, C_bal, u)
    sel_balanced = ssi.selectivity(
        y_surg_bal.unsqueeze(0), y_base.unsqueeze(0), y_null.unsqueeze(0),
        target_channel=0,
    )
    assert sel_balanced > sel_random * 2, \
        f"Balanced sel={sel_balanced:.2f} should >> random sel={sel_random:.2f}"
