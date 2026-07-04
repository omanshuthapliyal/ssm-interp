"""Core analysis primitives: Gramians, HSVs, balanced realization, truncation.

All functions accept numpy arrays or torch tensors and return the same type
as the primary input (A). Internally everything runs in float64 for numerical
stability.

Quick start
-----------
>>> import ssm_interpret as ssi
>>> Wc, Wo, meta = ssi.gramians(A, B, C, T=64)
>>> Ab, Bb, Cb, T_bal, sigma = ssi.balanced_realization(A, B, C, T=64)
>>> Ab_k, Bb_k, Cb_k = ssi.truncate(Ab, Bb, Cb, k=2)
>>> bound = ssi.glover_bound(sigma, k=2)   # certified ||H - H_k||_H <= bound
"""

from __future__ import annotations

import torch

from ssm_interpret._utils import (
    ArrayLike,
    check_system,
    like_input,
    to_torch,
    warn_sym,
    warn_unstable,
)

# ── Gramians ───────────────────────────────────────────────────────────────

def gramians(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    T: int,
    eps: float = 0.0,
) -> tuple:
    """Compute finite-horizon controllability and observability Gramians.

    For a discrete-time LTI system x_{t+1} = A x_t + B u_t, y_t = C x_t:

        W_c = sum_{k=0}^{T-1}  A^k B Bᵀ (Aᵀ)^k
        W_o = sum_{k=0}^{T-1}  (Aᵀ)^k Cᵀ C A^k

    Parameters
    ----------
    A : (d, d)
    B : (d, m)
    C : (p, d)
    T : int   Finite horizon. Longer T → more accurate Gramians for stable A.
    eps : float
        Regularisation added to diagonal: Wc += eps*I. Useful when
        balanced_realization is ill-conditioned (sym_error > 0.1).

    Returns
    -------
    Wc   : (d, d)  Controllability Gramian
    Wo   : (d, d)  Observability Gramian
    meta : dict    {"sym_error_c", "sym_error_o", "well_conditioned",
                    "condition_number_c", "condition_number_o",
                    "effective_rank", "spectral_radius"}
    """
    A_ref = A
    A, B, C, d, m, p = check_system(A, B, C)

    rho = warn_unstable(A)

    Wc = torch.zeros(d, d, dtype=torch.float64)
    Wo = torch.zeros(d, d, dtype=torch.float64)

    Ak = torch.eye(d, dtype=torch.float64)
    for _ in range(T):
        Wc = Wc + Ak @ (B @ B.T) @ Ak.T
        Wo = Wo + Ak.T @ (C.T @ C) @ Ak
        Ak = Ak @ A

    if eps > 0:
        Wc = Wc + eps * torch.eye(d, dtype=torch.float64)
        Wo = Wo + eps * torch.eye(d, dtype=torch.float64)

    sym_c = warn_sym(Wc, "Wc")
    sym_o = warn_sym(Wo, "Wo")
    Wc = (Wc + Wc.T) / 2
    Wo = (Wo + Wo.T) / 2

    well = (sym_c < 1e-3) and (sym_o < 1e-3)
    cond_c = float(torch.linalg.cond(Wc))
    cond_o = float(torch.linalg.cond(Wo))

    eigs = torch.linalg.eigvalsh(Wc)
    thresh = float(eigs.max()) * 1e-6
    eff_rank = int((eigs > thresh).sum())

    meta = {
        "sym_error_c": sym_c,
        "sym_error_o": sym_o,
        "well_conditioned": well,
        "condition_number_c": cond_c,
        "condition_number_o": cond_o,
        "effective_rank": eff_rank,
        "spectral_radius": rho,
    }
    return like_input(Wc, A_ref), like_input(Wo, A_ref), meta


# ── HSVs ───────────────────────────────────────────────────────────────────

def hsv(
    Wc: ArrayLike,
    Wo: ArrayLike,
    eps: float = 1e-12,
) -> ArrayLike:
    """Hankel singular values: sigma_k = sqrt(lambda_k(Wc Wo)), descending.

    Parameters
    ----------
    Wc  : (d, d)  Controllability Gramian (from gramians())
    Wo  : (d, d)  Observability Gramian
    eps : float   Clamp negative eigenvalues (numerical noise) to this floor.

    Returns
    -------
    sigma : (d,)  HSVs in descending order.
    """
    Wc_ref = Wc
    Wc = to_torch(Wc)
    Wo = to_torch(Wo)

    # sigma_k = sqrt(eig(Wc Wo)) = sqrt(eig(Wc^{1/2} Wo Wc^{1/2}))
    # Use the symmetric form for stability
    L = torch.linalg.cholesky((Wc + Wc.T) / 2 + eps * torch.eye(Wc.shape[0], dtype=Wc.dtype))
    M = L.T @ Wo @ L
    M = (M + M.T) / 2
    eigvals = torch.linalg.eigvalsh(M).clamp(min=0)
    sigma = eigvals.sqrt().flip(0)
    return like_input(sigma, Wc_ref)


# ── Balanced realization ───────────────────────────────────────────────────

def balanced_realization(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    T: int,
    eps: float = 1e-10,
) -> tuple:
    """Compute the balanced realization of a discrete-time LTI SSM.

    The balancing transformation T_bal simultaneously diagonalises both
    Gramians: T_bal^{-1} Wc T_bal^{-T} = T_bal^T Wo T_bal = diag(sigma).

    The balanced basis is unique up to sign and permutation (Proposition 1).
    Modes are ordered by Hankel singular value (largest first).

    Parameters
    ----------
    A, B, C : system matrices
    T       : finite horizon for Gramian computation
    eps     : regularisation; increase to 1e-5 if sym_error > 0.1

    Returns
    -------
    Ab    : (d, d)   A in balanced basis
    Bb    : (d, m)   B in balanced basis
    Cb    : (p, d)   C in balanced basis
    T_bal : (d, d)   balancing transformation (Ab = T_bal^{-1} A T_bal, etc.)
    sigma : (d,)     Hankel singular values, descending
    """
    A_ref = A
    A, B, C, d, m, p = check_system(A, B, C)

    Wc, Wo, _ = gramians(A, B, C, T, eps=eps)
    Wc = to_torch(Wc)
    Wo = to_torch(Wo)

    # Square-root factorisation: Wc = Lc Lcᵀ
    Lc = torch.linalg.cholesky(Wc + eps * torch.eye(d, dtype=torch.float64))
    # SVD of Lc^T Wo Lc
    M = Lc.T @ Wo @ Lc
    M = (M + M.T) / 2
    U, S, _ = torch.linalg.svd(M)
    # S = sigma^2  →  sigma = S^{1/2}
    sigma = S.clamp(min=0).sqrt()

    # Balancing transformation
    Sigma_half_inv = torch.diag(1.0 / sigma.clamp(min=eps))
    T_bal = Lc @ U @ Sigma_half_inv.sqrt().diag() if False else \
            Lc @ U @ torch.diag(S.clamp(min=eps).pow(-0.25))

    T_bal_inv = torch.linalg.pinv(T_bal)

    Ab = T_bal_inv @ A @ T_bal
    Bb = T_bal_inv @ B
    Cb = C @ T_bal

    return (
        like_input(Ab, A_ref),
        like_input(Bb, A_ref),
        like_input(Cb, A_ref),
        like_input(T_bal, A_ref),
        like_input(sigma, A_ref),
    )


# ── Truncation ─────────────────────────────────────────────────────────────

def truncate(
    Ab: ArrayLike,
    Bb: ArrayLike,
    Cb: ArrayLike,
    k: int,
) -> tuple:
    """Keep the top-k modes of a balanced system.

    Parameters
    ----------
    Ab, Bb, Cb : balanced system matrices from balanced_realization()
    k          : number of modes to retain (1 <= k <= d)

    Returns
    -------
    Ab_k : (k, k)
    Bb_k : (k, m)
    Cb_k : (p, k)

    Note
    ----
    The approximation error satisfies ||H - H_k||_H <= 2 * sigma[k]
    (Glover 1984). Use glover_bound(sigma, k) to retrieve this certificate.
    """
    Ab_ref = Ab
    Ab = to_torch(Ab)
    Bb = to_torch(Bb)
    Cb = to_torch(Cb)
    d = Ab.shape[0]
    if not (1 <= k <= d):
        raise ValueError(f"k must be in [1, {d}], got {k}")
    return (
        like_input(Ab[:k, :k], Ab_ref),
        like_input(Bb[:k, :], Ab_ref),
        like_input(Cb[:, :k], Ab_ref),
    )


# ── Impulse response ───────────────────────────────────────────────────────

def impulse_response(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    T: int,
) -> ArrayLike:
    """Compute the matrix impulse response H[t] = C A^t B.

    Parameters
    ----------
    A : (d, d)
    B : (d, m)
    C : (p, d)
    T : int   Number of timesteps.

    Returns
    -------
    H : (p, T, m)   H[:, t, :] = C A^t B
    """
    A_ref = A
    A, B, C, d, m, p = check_system(A, B, C)

    H = torch.zeros(p, T, m, dtype=torch.float64)
    Ak = torch.eye(d, dtype=torch.float64)
    for t in range(T):
        H[:, t, :] = C @ Ak @ B
        Ak = Ak @ A
    return like_input(H, A_ref)


# ── Glover bound ───────────────────────────────────────────────────────────

def glover_bound(sigma: ArrayLike, k: int) -> float:
    """Certified upper bound on Hankel-norm truncation error.

    ||H - H_k||_H  <=  2 * sigma[k]

    where H_k retains the top-k balanced modes (Glover 1984, Theorem 2).

    Parameters
    ----------
    sigma : (d,)  HSVs in descending order (from balanced_realization).
    k     : int   Number of modes retained. Bound applies to modes k+1..d.

    Returns
    -------
    float  Upper bound on Hankel norm of the approximation error.
    """
    sigma = to_torch(sigma)
    if not (0 <= k < len(sigma)):
        raise ValueError(f"k must be in [0, {len(sigma)-1}], got {k}")
    return float(2.0 * sigma[k])
