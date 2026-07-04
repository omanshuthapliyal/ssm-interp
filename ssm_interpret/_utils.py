"""Internal utilities: tensor coercion, shape validation, numerical warnings."""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


# ── Type coercion ──────────────────────────────────────────────────────────

def to_torch(x: ArrayLike, dtype=torch.float64) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.tensor(np.asarray(x), dtype=dtype)


def to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def like_input(result: torch.Tensor, reference: ArrayLike) -> ArrayLike:
    """Return result in the same container type (numpy/torch) as reference."""
    if isinstance(reference, np.ndarray):
        return to_numpy(result)
    return result


# ── Shape validation ───────────────────────────────────────────────────────

def check_system(A, B, C):
    """Validate (A, B, C) shapes. Returns (d, m, p) and torch tensors."""
    A = to_torch(A)
    B = to_torch(B)
    C = to_torch(C)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square (d, d), got {tuple(A.shape)}")
    d = A.shape[0]
    if B.ndim != 2 or B.shape[0] != d:
        raise ValueError(f"B must be (d, m), got {tuple(B.shape)}")
    if C.ndim != 2 or C.shape[1] != d:
        raise ValueError(f"C must be (p, d), got {tuple(C.shape)}")
    return A, B, C, d, B.shape[1], C.shape[0]


def check_T_bal(T_bal, d: int) -> torch.Tensor:
    T_bal = to_torch(T_bal)
    if T_bal.shape != (d, d):
        raise ValueError(f"T_bal must be ({d}, {d}), got {tuple(T_bal.shape)}")
    return T_bal


# ── Numerical warnings ─────────────────────────────────────────────────────

def warn_unstable(A: torch.Tensor) -> float:
    """Warn if spectral radius >= 1. Returns spectral radius."""
    eigvals = torch.linalg.eigvals(A)
    rho = float(eigvals.abs().max())
    if rho >= 1.0:
        warnings.warn(
            f"spectral_radius(A) = {rho:.6f} >= 1.0. "
            "The system is unstable; Gramians may diverge. "
            "Results are returned anyway — inspect with care.",
            UserWarning,
            stacklevel=3,
        )
    return rho


def warn_sym(W: torch.Tensor, name: str = "W", threshold: float = 1e-3) -> float:
    """Warn if Gramian is asymmetric. Returns sym_error scalar."""
    norm_W = float(W.norm())
    if norm_W < 1e-30:
        return 0.0
    err = float((W - W.T).norm()) / norm_W
    if err > 0.1:
        warnings.warn(
            f"sym_error({name}) = {err:.4e} > 0.1. "
            "Gramian is highly asymmetric — model may not have converged. "
            "Consider regularising with the eps parameter.",
            UserWarning,
            stacklevel=3,
        )
    elif err > threshold:
        warnings.warn(
            f"sym_error({name}) = {err:.4e} > {threshold:.0e}. "
            "Balanced realization may be ill-conditioned.",
            UserWarning,
            stacklevel=3,
        )
    return err
