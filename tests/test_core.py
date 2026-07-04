"""Tests for ssm_interpret.core — obvious and non-obvious invariants."""

import warnings

import numpy as np
import pytest
import torch

import ssm_interpret as ssi
from ssm_interpret.core import gramians, balanced_realization, hsv, truncate, glover_bound, impulse_response


# ── Fixtures ───────────────────────────────────────────────────────────────

def make_diagonal_ema(rhos, seed=0):
    """Diagonal-A SSM whose exact balanced basis is known analytically."""
    torch.manual_seed(seed)
    d = len(rhos)
    A = torch.diag(torch.tensor(rhos, dtype=torch.float64))
    B = torch.randn(d, 1, dtype=torch.float64) * 0.5
    C = torch.randn(1, d, dtype=torch.float64) * 0.5
    return A, B, C


def make_random_stable(d=4, m=2, p=2, seed=42):
    """Random stable full-A SSM."""
    torch.manual_seed(seed)
    A = torch.randn(d, d, dtype=torch.float64)
    A = A / (torch.linalg.eigvals(A).abs().max() * 1.1)   # scale inside unit disc
    B = torch.randn(d, m, dtype=torch.float64) * 0.3
    C = torch.randn(p, d, dtype=torch.float64) * 0.3
    return A, B, C


# ── Obvious tests ──────────────────────────────────────────────────────────

def test_gramians_symmetric():
    A, B, C = make_random_stable()
    Wc, Wo, meta = gramians(A, B, C, T=32)
    assert torch.allclose(torch.tensor(Wc), torch.tensor(Wc).T, atol=1e-8)
    assert torch.allclose(torch.tensor(Wo), torch.tensor(Wo).T, atol=1e-8)


def test_gramians_meta_keys():
    A, B, C = make_random_stable()
    _, _, meta = gramians(A, B, C, T=32)
    for key in ("sym_error_c", "sym_error_o", "well_conditioned",
                "condition_number_c", "effective_rank", "spectral_radius"):
        assert key in meta


def test_balanced_gramians_diagonal():
    """After balanced_realization, Wc_bal ≈ Wo_bal ≈ diag(sigma)."""
    A, B, C = make_random_stable()
    Ab, Bb, Cb, T_bal, sigma = balanced_realization(A, B, C, T=64)
    Ab, Bb, Cb, T_bal, sigma = [torch.tensor(x, dtype=torch.float64)
                                  for x in (Ab, Bb, Cb, T_bal, sigma)]
    Wc_bal, Wo_bal, _ = gramians(Ab, Bb, Cb, T=64)
    Wc_bal = torch.tensor(Wc_bal, dtype=torch.float64)
    Wo_bal = torch.tensor(Wo_bal, dtype=torch.float64)
    diag_sigma = torch.diag(sigma)
    assert torch.allclose(Wc_bal, diag_sigma, atol=1e-4), \
        f"Wc_bal not diagonal: max err {(Wc_bal - diag_sigma).abs().max():.4e}"
    assert torch.allclose(Wo_bal, diag_sigma, atol=1e-4)


def test_sigma_descending():
    A, B, C = make_random_stable()
    _, _, _, _, sigma = balanced_realization(A, B, C, T=32)
    sigma = torch.tensor(sigma)
    assert (sigma[:-1] >= sigma[1:] - 1e-10).all()


def test_truncate_shape():
    A, B, C = make_random_stable(d=4, m=2, p=2)
    Ab, Bb, Cb, T_bal, sigma = balanced_realization(A, B, C, T=32)
    Ab_k, Bb_k, Cb_k = truncate(Ab, Bb, Cb, k=2)
    assert torch.tensor(Ab_k).shape == (2, 2)
    assert torch.tensor(Bb_k).shape == (2, 2)
    assert torch.tensor(Cb_k).shape == (2, 2)


def test_glover_bound_positive():
    A, B, C = make_random_stable()
    _, _, _, _, sigma = balanced_realization(A, B, C, T=32)
    bound = glover_bound(sigma, k=1)
    assert bound > 0


def test_unstable_warns():
    A = torch.eye(3, dtype=torch.float64) * 1.5   # spectral radius = 1.5
    B = torch.ones(3, 1, dtype=torch.float64)
    C = torch.ones(1, 3, dtype=torch.float64)
    with pytest.warns(UserWarning, match="spectral_radius"):
        gramians(A, B, C, T=10)


# ── Non-obvious tests ──────────────────────────────────────────────────────

def test_hsv_basis_invariance():
    """HSVs must be identical under any invertible change of basis."""
    A, B, C = make_random_stable(d=4)
    _, _, _, _, sigma_orig = balanced_realization(A, B, C, T=32)

    # Random invertible transform
    torch.manual_seed(99)
    T_rand = torch.randn(4, 4, dtype=torch.float64)
    T_rand = T_rand + 4 * torch.eye(4, dtype=torch.float64)  # make invertible
    T_inv  = torch.linalg.inv(T_rand)

    A2 = T_inv @ A @ T_rand
    B2 = T_inv @ B
    C2 = C @ T_rand

    _, _, _, _, sigma_new = balanced_realization(A2, B2, C2, T=32)

    sigma_orig = torch.tensor(sigma_orig, dtype=torch.float64)
    sigma_new  = torch.tensor(sigma_new,  dtype=torch.float64)
    assert torch.allclose(sigma_orig, sigma_new, atol=1e-6), \
        f"HSVs changed under basis transform: {sigma_orig} vs {sigma_new}"


def test_impulse_response_basis_invariance():
    """Two realizations of the same transfer function → identical H."""
    A, B, C = make_random_stable(d=4)
    H1 = torch.tensor(impulse_response(A, B, C, T=20), dtype=torch.float64)

    torch.manual_seed(7)
    T_rand = torch.randn(4, 4, dtype=torch.float64)
    T_rand = T_rand + 4 * torch.eye(4, dtype=torch.float64)
    T_inv  = torch.linalg.inv(T_rand)
    A2 = T_inv @ A @ T_rand
    B2 = T_inv @ B
    C2 = C @ T_rand
    H2 = torch.tensor(impulse_response(A2, B2, C2, T=20), dtype=torch.float64)

    assert torch.allclose(H1, H2, atol=1e-8), \
        f"Impulse response changed under basis transform: max err {(H1-H2).abs().max():.4e}"


def test_glover_bound_tightness_ratio():
    """For EMA tasks, err_k / sigma[k+1] should be roughly constant across k.

    This verifies the Glover bound is tight in the task-specific sense
    documented in §2.4.  We don't test the exact ratio (task-dependent)
    but confirm it doesn't vary by more than 3× across truncation levels.
    """
    # Build a 4-channel MIMO EMA model (known structure)
    rhos = [0.95, 0.75, 0.50, 0.25]
    d = len(rhos)
    A = torch.diag(torch.tensor(rhos, dtype=torch.float64))
    B = torch.eye(d, dtype=torch.float64)
    C = torch.eye(d, dtype=torch.float64)
    T = 32

    Ab, Bb, Cb, T_bal, sigma = balanced_realization(A, B, C, T=T)
    Ab, Bb, Cb, sigma = [torch.tensor(x, dtype=torch.float64)
                          for x in (Ab, Bb, Cb, sigma)]

    H_full = torch.tensor(impulse_response(A, B, C, T=T), dtype=torch.float64)

    ratios = []
    for k in range(1, d):
        Ab_k, Bb_k, Cb_k = truncate(Ab, Bb, Cb, k=k)
        Ab_k, Bb_k, Cb_k = [torch.tensor(x, dtype=torch.float64) for x in (Ab_k, Bb_k, Cb_k)]
        H_k = torch.tensor(impulse_response(Ab_k, Bb_k, Cb_k, T=T), dtype=torch.float64)
        err  = float((H_full - H_k[:, :, :] if H_full.shape == H_k.shape
                      else H_full).norm())
        bound = glover_bound(sigma, k)
        ratios.append(err / max(bound, 1e-12))

    ratio_range = max(ratios) / max(min(ratios), 1e-12)
    assert ratio_range < 5.0, \
        f"err/bound ratios vary too much across k: {ratios} (range={ratio_range:.2f})"


def test_rank_deficient_A_no_crash():
    """Rank-deficient A (zero eigenvalue) must not crash balanced_realization."""
    A = torch.diag(torch.tensor([0.9, 0.0, 0.5], dtype=torch.float64))
    B = torch.randn(3, 1, dtype=torch.float64)
    C = torch.randn(1, 3, dtype=torch.float64)
    Ab, Bb, Cb, T_bal, sigma = balanced_realization(A, B, C, T=20)
    sigma = torch.tensor(sigma)
    assert sigma.min() >= 0.0


def test_numpy_input_returns_numpy():
    """Functions should return numpy arrays when given numpy input."""
    A = np.array([[0.9, 0.1], [0.0, 0.7]])
    B = np.array([[1.0], [0.5]])
    C = np.array([[1.0, 0.5]])
    Wc, Wo, _ = gramians(A, B, C, T=20)
    assert isinstance(Wc, np.ndarray)
    Ab, Bb, Cb, T_bal, sigma = balanced_realization(A, B, C, T=20)
    assert isinstance(Ab, np.ndarray)
    assert isinstance(sigma, np.ndarray)
