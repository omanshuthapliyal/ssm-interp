"""Diagnostics: Kalman taxonomy, architecture gap, scope checks.

These functions answer two questions:
  1. What is the structure of the learned circuits?  (kalman_taxonomy, architecture_gap)
  2. Is this framework applicable to this model?      (scope_check, sym_error)
"""

from __future__ import annotations

import torch

from ssm_interpret._utils import ArrayLike, check_system, like_input, to_torch


# ── Kalman subspace taxonomy ───────────────────────────────────────────────

def kalman_taxonomy(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    eps: float = 1e-6,
) -> dict:
    """Classify each state dimension as live, dead-type-1, or dead-type-2.

    Taxonomy (from the balanced circuit perspective)
    ------------------------------------------------
    Live         : sigma_k >> 0.  Both controllable and observable.
                   Ablating these modes degrades model output.

    Dead Type-1  : B-norm ≈ 0.  Strictly uncontrollable — input never
                   reaches this state dimension.  Ablating has no effect
                   because x_k ≡ 0 always.

    Dead Type-2  : B-norm > 0 but sigma_k ≈ 0.  Dynamically trivial —
                   input reaches the state but A does not sustain it long
                   enough to be observable.  Ablating has no effect because
                   C_bal[:,k] ≈ 0.  Gradient-based training produces these
                   when surplus dimensions receive B-coupling but lose C-
                   coupling due to zero gradient from output.

    Invariant
    ---------
    len(live) + len(dead_t1) + len(dead_t2) == d  (always)

    Parameters
    ----------
    A, B, C : system matrices (original basis)
    eps     : relative threshold for "near-zero" B-norm and sigma

    Returns
    -------
    dict with keys "live", "dead_type1", "dead_type2" — each a list of
    mode indices (0-indexed, ordered by decreasing Hankel energy).
    """
    from ssm_interpret.core import balanced_realization

    A, B, C, d, m, p = check_system(A, B, C)

    # Use a modest horizon; taxonomy is stable across T for converged models
    T_tax = min(64, 20 * d)
    Ab, Bb, Cb, T_bal, sigma = balanced_realization(A, B, C, T=T_tax)
    Ab = to_torch(Ab)
    Bb = to_torch(Bb)
    sigma = to_torch(sigma)

    sigma_max = float(sigma[0]) if sigma.numel() > 0 else 1.0
    threshold = eps * max(sigma_max, 1e-12)

    b_norms = torch.norm(to_torch(Bb), dim=1)   # (d,) — per-mode input coupling

    live, dead_t1, dead_t2 = [], [], []
    for k in range(d):
        if float(sigma[k]) > threshold:
            live.append(k)
        elif float(b_norms[k]) < threshold:
            dead_t1.append(k)
        else:
            dead_t2.append(k)

    return {"live": live, "dead_type1": dead_t1, "dead_type2": dead_t2}


# ── Architecture gap ───────────────────────────────────────────────────────

def architecture_gap(
    sigma_diag: ArrayLike,
    sigma_full: ArrayLike,
    m: int,
) -> float:
    """Leakage gap between diagonal-A and full-A architectures.

    Measures how much more a full-A SSM can suppress its (m+1)-th Hankel
    mode compared to a diagonal-A SSM on the same task:

        gap = sigma_diag[m] / sigma_full[m]

    gap >> 1  →  diagonal A leaks energy into its (m+1)-th mode that
                 full A can suppress via off-diagonal coupling.
    gap ≈  1  →  both architectures are equivalent for this task
                 (occurs when the dominant Hankel mode is a running sum,
                 i.e. rho → 1; see §2.5).

    Parameters
    ----------
    sigma_diag : (d,) HSVs of a diagonal-A model, descending
    sigma_full : (d,) HSVs of a full-A model on the same task, descending
    m          : int  number of target circuits; gap computed at index m

    Returns
    -------
    float  gap ratio (> 0); returns inf if sigma_full[m] < 1e-12.
    """
    sigma_diag = to_torch(sigma_diag)
    sigma_full = to_torch(sigma_full)
    d = min(len(sigma_diag), len(sigma_full))
    if not (0 <= m < d):
        raise ValueError(f"m must be in [0, {d-1}], got {m}")
    denom = float(sigma_full[m])
    if denom < 1e-12:
        return float("inf")
    return float(sigma_diag[m]) / denom


# ── Symmetry error ─────────────────────────────────────────────────────────

def sym_error(W: ArrayLike) -> float:
    """Normalised asymmetry of a Gramian: ||W - Wᵀ|| / ||W||.

    Values below 1e-3 indicate a well-conditioned balanced realization.
    Values above 0.1 indicate the model has not converged or requires
    regularisation (gramians(..., eps=1e-5)).

    Parameters
    ----------
    W : (d, d)  a Gramian matrix (Wc or Wo)

    Returns
    -------
    float  in [0, ∞)
    """
    W = to_torch(W)
    norm = float(W.norm())
    if norm < 1e-30:
        return 0.0
    return float((W - W.T).norm()) / norm


# ── LTI scope check ────────────────────────────────────────────────────────

def scope_check(gvi_value: float) -> dict:
    """Classify a model's regime from its Gate Variation Index (GVI = CV(Δ)).

    The GVI is computed by ssm_interpret.ltv.gvi() for gated / selective SSMs.
    For a standard linear SSM with fixed parameters, GVI = 0 exactly.

    Regime thresholds (empirically calibrated, §5):
        lti       : GVI < 0.05   — near-LTI; balanced realization fully applies
        near_lti  : 0.05–0.20   — borderline; apply with caution / use LTV extension
        selective : GVI > 0.20   — genuine selectivity; LTI framework requires extension

    Parameters
    ----------
    gvi_value : float  Gate Variation Index (coefficient of variation of Δ)

    Returns
    -------
    dict with keys:
        "regime"     : "lti" | "near_lti" | "selective"
        "applicable" : bool  — whether the LTI framework is directly applicable
        "gvi"        : float — the input value, echoed for convenience
        "message"    : str   — human-readable verdict
    """
    if gvi_value < 0.05:
        regime, applicable = "lti", True
        msg = (
            f"GVI={gvi_value:.4f} < 0.05: near-LTI regime. "
            "Balanced realization and subspace surgery apply directly."
        )
    elif gvi_value < 0.20:
        regime, applicable = "near_lti", True
        msg = (
            f"GVI={gvi_value:.4f} in [0.05, 0.20]: borderline regime. "
            "Apply with caution; use ssm_interpret.ltv for data-averaged Gramians."
        )
    else:
        regime, applicable = "selective", False
        msg = (
            f"GVI={gvi_value:.4f} > 0.20: selective (genuinely gated) regime. "
            "The LTI framework requires extension via ssm_interpret.ltv."
        )
    return {"regime": regime, "applicable": applicable, "gvi": gvi_value, "message": msg}
