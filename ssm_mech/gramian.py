"""
ssm_mech.gramian -- Online Gramian estimation for HuggingFace Mamba models.

Gramians are estimated from actual model forward passes using forward hooks.
This is distribution-dependent: the resulting balanced basis reflects the
estimation corpus, not the model weights alone. Use task-specific prompts
(e.g. IOI sentences) when you want a task-specific balanced basis.

Key classes / functions
-----------------------
OnlineStats       -- Welford accumulator for discrete-A statistics
merge_stats       -- Combine two OnlineStats (parallel / multi-GPU merge)
gramian_from_stats -- Compute per-channel Wc from an OnlineStats object
balanced_eigvec   -- Compute balanced mode eigenvectors from Wc
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.linalg import solve_discrete_lyapunov


# ---------------------------------------------------------------------------
# Online accumulator
# ---------------------------------------------------------------------------

class OnlineStats:
    """
    Welford-style accumulator for the discrete-A statistics needed to
    estimate the controllability Gramian W_c of one SSM layer.

    Tracks, per token seen:
      - mean and M2 of the per-channel discrete eigenvalues A_disc (d_inner, d_state)
      - sum of the input projection B (d_state,) and output projection C (d_state,)

    All arithmetic is done in float64 for numerical stability.

    Parameters
    ----------
    d_inner : int
        Number of SSM channels (model dimension / expand factor).
    d_state : int
        Number of recurrent state dimensions per channel.
    """

    def __init__(self, d_inner: int, d_state: int) -> None:
        self.d_inner = d_inner
        self.d_state = d_state
        self.n: int = 0
        self.mean  = np.zeros((d_inner, d_state), dtype=np.float64)
        self.M2    = np.zeros((d_inner, d_state), dtype=np.float64)
        self.B_sum = np.zeros(d_state, dtype=np.float64)
        self.C_sum = np.zeros(d_state, dtype=np.float64)
        self.BC_n: int = 0

    # ------------------------------------------------------------------
    # Batch update (Chan's parallel Welford)
    # ------------------------------------------------------------------

    def update(
        self,
        A_disc_batch: np.ndarray,
        B_batch: Optional[np.ndarray] = None,
        C_batch: Optional[np.ndarray] = None,
    ) -> None:
        """
        Incorporate a batch of per-token statistics.

        Parameters
        ----------
        A_disc_batch : (N, d_inner, d_state) float array
            Discrete eigenvalues A_t = exp(dt_t * A_cont) for each token.
        B_batch : (N, d_state) float array, optional
            Input projections for each token.
        C_batch : (N, d_state) float array, optional
            Output projections for each token.
        """
        N = A_disc_batch.shape[0]
        if N == 0:
            return
        batch      = A_disc_batch.astype(np.float64)
        batch_mean = batch.mean(axis=0)
        batch_M2   = ((batch - batch_mean[None]) ** 2).sum(axis=0)
        delta      = batch_mean - self.mean
        combined_n = self.n + N
        new_mean   = self.mean + delta * (N / combined_n)
        new_M2     = self.M2 + batch_M2 + delta ** 2 * (self.n * N / combined_n)
        self.mean, self.M2, self.n = new_mean, new_M2, combined_n
        if B_batch is not None:
            self.B_sum += B_batch.astype(np.float64).sum(axis=0)
            self.BC_n  += int(B_batch.shape[0])
        if C_batch is not None:
            self.C_sum += C_batch.astype(np.float64).sum(axis=0)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def A_avg(self) -> np.ndarray:
        """Mean discrete eigenvalues, shape (d_inner, d_state)."""
        return self.mean.copy()

    @property
    def B_avg(self) -> np.ndarray:
        """Mean input projection, shape (d_state,)."""
        n = self.BC_n if self.BC_n > 0 else 1
        return self.B_sum / n

    @property
    def C_avg(self) -> np.ndarray:
        """Mean output projection, shape (d_state,)."""
        n = self.BC_n if self.BC_n > 0 else 1
        return self.C_sum / n

    @property
    def etmfd(self) -> float:
        """
        Effective Timescale Modulation Factor Deviation.

        Measures how much the discrete-A matrix varies across tokens relative
        to its mean:

            ETMFD = sqrt(sum_ij Var[A_disc_ij]) / ||A_avg||_F

        Zero for a truly LTI system (constant A_disc).
        Values above ~0.18 indicate genuine input-dependent timescale modulation.
        """
        if self.n < 2:
            return float("nan")
        var         = self.M2 / (self.n - 1)
        avg_norm_sq = float((self.mean ** 2).sum())
        if avg_norm_sq < 1e-20:
            return float("nan")
        return float(np.sqrt(var.sum()) / np.sqrt(avg_norm_sq))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to a compressed NumPy archive (.npz)."""
        np.savez_compressed(
            path,
            d_inner=self.d_inner, d_state=self.d_state,
            n=self.n, mean=self.mean, M2=self.M2,
            B_sum=self.B_sum, C_sum=self.C_sum, BC_n=self.BC_n,
        )

    @classmethod
    def load(cls, path: str | Path) -> "OnlineStats":
        """Load from a .npz file written by :meth:`save`."""
        d   = np.load(path, allow_pickle=False)
        obj = cls(int(d["d_inner"]), int(d["d_state"]))
        obj.n      = int(d["n"])
        obj.mean   = d["mean"].astype(np.float64)
        obj.M2     = d["M2"].astype(np.float64)
        obj.B_sum  = d["B_sum"].astype(np.float64)
        obj.C_sum  = d["C_sum"].astype(np.float64)
        obj.BC_n   = int(d["BC_n"])
        return obj


# ---------------------------------------------------------------------------
# Parallel merge
# ---------------------------------------------------------------------------

def merge_stats(s1: OnlineStats, s2: OnlineStats) -> OnlineStats:
    """
    Combine two OnlineStats objects via Chan's parallel formula.

    Exact (no approximation). Useful for merging shards from multi-GPU
    or multi-worker Gramian estimation runs.
    """
    if s1.d_inner != s2.d_inner or s1.d_state != s2.d_state:
        raise ValueError("OnlineStats shapes must match to merge.")
    if s1.n == 0:
        return s2
    if s2.n == 0:
        return s1
    out        = OnlineStats(s1.d_inner, s1.d_state)
    n1, n2     = s1.n, s2.n
    combined_n = n1 + n2
    delta      = s2.mean - s1.mean
    out.n      = combined_n
    out.mean   = (n1 * s1.mean + n2 * s2.mean) / combined_n
    out.M2     = s1.M2 + s2.M2 + delta ** 2 * (n1 * n2 / combined_n)
    out.B_sum  = s1.B_sum + s2.B_sum
    out.C_sum  = s1.C_sum + s2.C_sum
    out.BC_n   = s1.BC_n + s2.BC_n
    return out


# ---------------------------------------------------------------------------
# Gramian and eigendecomposition
# ---------------------------------------------------------------------------

def gramian_from_stats(
    a_diag: np.ndarray,
    b_vec:  np.ndarray,
    c_vec:  np.ndarray,
    clip:   float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Closed-form controllability and observability Gramians for a single SSM
    channel with diagonal discrete-A.

        W_c[i,j] = b[i] * b[j] / (1 - a[i] * a[j])
        W_o[i,j] = c[i] * c[j] / (1 - a[i] * a[j])

    Parameters
    ----------
    a_diag : (d_state,) array
        Mean discrete eigenvalues for one channel. Must be in (0, 1).
    b_vec : (d_state,) array
        Mean input projection for this channel.
    c_vec : (d_state,) array
        Mean output projection for this channel.
    clip : float
        Minimum absolute denominator value to avoid division by zero.

    Returns
    -------
    Wc, Wo : (d_state, d_state) arrays
    """
    denom = 1.0 - np.outer(a_diag, a_diag)
    denom = np.where(np.abs(denom) < clip, np.sign(denom + 1e-30) * clip, denom)
    Wc = np.outer(b_vec, b_vec) / denom
    Wo = np.outer(c_vec, c_vec) / denom
    return Wc, Wo


def balanced_eigvec(
    stats: OnlineStats,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Compute the balanced mode eigenvectors for all channels from an OnlineStats.

    For each channel, solves for the eigenvectors of the controllability Gramian
    W_c estimated from :func:`gramian_from_stats`. Modes are sorted descending
    by eigenvalue (mode-0 = most controllable / observable = dominant).

    Channels where A_avg has eigenvalues outside (0, 1) (numerically unstable)
    fall back to a random orthonormal basis.

    Parameters
    ----------
    stats : OnlineStats
        Fitted statistics for one layer.
    rng : np.random.Generator, optional
        RNG used for fallback random bases. Defaults to np.random.default_rng(0).

    Returns
    -------
    V : (d_inner, d_state, d_state) float32 array
        V[ch, :, k] is the k-th balanced mode eigenvector for channel ch.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    d_inner, d_state = stats.d_inner, stats.d_state
    V = np.zeros((d_inner, d_state, d_state), dtype=np.float32)
    for ch in range(d_inner):
        a_diag = stats.A_avg[ch, :]
        if a_diag.max() >= 1.0 or a_diag.min() < 0.0:
            V[ch] = np.linalg.qr(
                rng.standard_normal((d_state, d_state))
            )[0].astype(np.float32)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Wc, _ = gramian_from_stats(a_diag, stats.B_avg, stats.C_avg)
        try:
            evals, evecs = np.linalg.eigh(Wc)
            V[ch] = evecs[:, np.argsort(evals)[::-1]].astype(np.float32)
        except np.linalg.LinAlgError:
            V[ch] = np.linalg.qr(
                rng.standard_normal((d_state, d_state))
            )[0].astype(np.float32)
    return V
