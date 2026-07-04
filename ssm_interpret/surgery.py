"""Subspace surgery: causal circuit ablation and selectivity measurement.

Design note — two ablation modes
---------------------------------
mode="dynamics"  (default, recommended)
    Modifies A: A' = A - P_k A P_k,  where P_k = T_bal e_k eₖᵀ T_bal⁻¹

    This projects mode k out of the state-transition dynamics.  After surgery
    the mode no longer evolves: energy injected into the k-direction by B is
    not sustained across timesteps.  The ablation is permanent and causal —
    it acts identically at every timestep, from t=0 to t=T.

    Control theory analogy: removing a pole from the transfer function.
    Approximation guarantee: ||H - H^{(-k)}||_H <= 2 * sigma[k+1..] (Glover).

mode="output"
    Modifies C: C' = C (I - P_k)

    This silences the read-out of mode k without changing the dynamics.
    Mode k still evolves inside the state (energy accumulates) but contributes
    nothing to the output.  Cheaper (O(d·p) vs O(d²)) and equivalent to
    dynamics surgery *only* when mode k is perfectly observable and T_bal is
    exact.  For overcomplete models or ill-conditioned T_bal, prefer
    mode="dynamics".

    Control theory analogy: muting one output channel of a MIMO system.

References
----------
Glover, K. (1984). All optimal Hankel-norm approximations of linear
multivariable systems and their L∞-error bounds. Int. J. Control, 39(6).

Moore, B. (1981). Principal component analysis in linear systems.
IEEE Trans. Autom. Control, 26(1).
"""

from __future__ import annotations

import torch

from ssm_interpret._utils import ArrayLike, check_system, check_T_bal, like_input, to_torch

# ── Ablation ───────────────────────────────────────────────────────────────

def ablate(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    T_bal: ArrayLike,
    k: int,
    mode: str = "dynamics",
) -> tuple:
    """Remove circuit k from a balanced SSM via subspace surgery.

    Parameters
    ----------
    A, B, C : system matrices (original parameter basis, NOT balanced)
    T_bal   : (d, d) balancing transformation from balanced_realization()
    k       : int   mode index to ablate (0 = highest Hankel energy)
    mode    : "dynamics" | "output"
              "dynamics" — modifies A (correct, permanent, O(d²))
              "output"   — modifies C (faster approximation, O(d·p))

    Returns
    -------
    A_new : (d, d)   modified A (identical to A if mode="output")
    B_new : (d, m)   always identical to B
    C_new : (p, d)   modified C if mode="output", else identical to C

    Example
    -------
    >>> Ab, Bb, Cb, T_bal, sigma = ssi.balanced_realization(A, B, C, T=64)
    >>> A2, B2, C2 = ssi.ablate(A, B, C, T_bal, k=0, mode="dynamics")
    >>> y_surg = run_model(A2, B2, C2, u_test)
    """
    if mode not in ("dynamics", "output"):
        raise ValueError(f"mode must be 'dynamics' or 'output', got '{mode}'")

    A_ref = A
    A, B, C, d, m, p = check_system(A, B, C)
    T_bal = check_T_bal(T_bal, d)

    if not (0 <= k < d):
        raise ValueError(f"k must be in [0, {d-1}], got {k}")

    T_bal_inv = torch.linalg.pinv(T_bal)

    # Rank-1 projector onto mode k in the original basis
    # P_k = T_bal @ e_k eₖᵀ @ T_bal⁻¹
    e_k = torch.zeros(d, 1, dtype=torch.float64)
    e_k[k] = 1.0
    P_k = T_bal @ (e_k @ e_k.T) @ T_bal_inv

    if mode == "dynamics":
        A_new = A - P_k @ A @ P_k
        return like_input(A_new, A_ref), like_input(B, A_ref), like_input(C, A_ref)

    # mode == "output"
    C_new = C @ (torch.eye(d, dtype=torch.float64) - P_k)
    return like_input(A, A_ref), like_input(B, A_ref), like_input(C_new, A_ref)


# ── Selectivity ────────────────────────────────────────────────────────────

def selectivity(
    y_surg: ArrayLike,
    y_base: ArrayLike,
    y_null: ArrayLike,
    target_channel: int,
) -> float:
    """Compute surgery selectivity for a single ablated mode.

    Selectivity = normalised damage to target channel
                  / mean normalised damage to all other channels

        Delta[j] = (MSE_surg[j] - MSE_base[j]) / MSE_null[j]
        sel_k    = Delta[target] / mean(Delta[j != target])

    sel_k >> 1  →  surgery is channel-selective (good)
    sel_k ~  1  →  surgery is unselective (modes are entangled)
    sel_k <  1  →  anti-selective (Hankel modes straddle multiple channels)

    Parameters
    ----------
    y_surg         : (B, T, p) or (B, p)  post-surgery model output
    y_base         : (B, T, p) or (B, p)  baseline model output
    y_null         : (B, T, p) or (B, p)  null-model output (e.g. zero predictor)
    target_channel : int   index of the channel expected to be most degraded

    Returns
    -------
    float  selectivity ratio; inf if all non-target deltas are zero.
    """
    y_surg = to_torch(y_surg).double()
    y_base = to_torch(y_base).double()
    y_null = to_torch(y_null).double()

    def mse_per_channel(pred, target):
        """Mean squared error per output channel → shape (p,)."""
        diff2 = (pred - target) ** 2          # (..., p)
        # Average over all leading dims (batch, time) leaving only channel dim
        while diff2.ndim > 1:
            diff2 = diff2.mean(0)
        return diff2                           # (p,)

    mse_s = mse_per_channel(y_surg, y_base)   # damage from surgery
    mse_n = mse_per_channel(y_null, y_base)   # null-model damage (normaliser)

    delta = mse_s / mse_n.clamp(min=1e-12)
    p = delta.shape[0]

    if not (0 <= target_channel < p):
        raise ValueError(f"target_channel {target_channel} out of range [0, {p-1}]")

    delta_target = float(delta[target_channel])
    others = [float(delta[j]) for j in range(p) if j != target_channel]
    mean_other = sum(others) / len(others) if others else 0.0

    if mean_other < 1e-12:
        return float("inf")
    return delta_target / mean_other


# ── Faithfulness ───────────────────────────────────────────────────────────

def faithfulness_ratio(
    delta_active: float,
    delta_inactive: float,
) -> float:
    """Input-conditional causal validity score.

    Measures whether surgery damages performance *only when the circuit is
    causally active* (i.e. when the relevant input feature is present).

    faithfulness = Delta(active sequences) / Delta(inactive sequences)

    A ratio >> 1 confirms that surgery is not just channel-selective but
    input-conditionally causal — it is the SSM analogue of causal scrubbing
    (Chan et al. 2022).

    Parameters
    ----------
    delta_active   : normalised damage on sequences where the circuit is active
    delta_inactive : normalised damage on sequences where the circuit is idle

    Returns
    -------
    float  ratio; inf if delta_inactive == 0.
    """
    if delta_inactive < 1e-12:
        return float("inf")
    return delta_active / delta_inactive
