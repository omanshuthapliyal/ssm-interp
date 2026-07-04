"""LTV Gramians: data-averaged controllability / observability for gated SSMs.

For a linear time-varying (LTV) system with per-step matrices:
    x_{t+1} = A_t x_t + B_t u_t
    y_t     = C_t x_t

the data-averaged Gramians are computed by averaging the empirical Gramian
contributions over a batch of input sequences.  When A_t = A for all t
(exact LTI), these recover the standard finite-horizon Gramians exactly.

This is the approach used in §5 to handle Mamba's input-dependent Δ:
A_t = exp(Δ_t ⊙ A_log) where Δ_t depends on u_t.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ssm_interpret._utils import ArrayLike, to_torch, warn_sym


def gramians_ltv(
    A_seq: ArrayLike,
    B_seq: ArrayLike,
    C_seq: ArrayLike,
    eps: float = 0.0,
) -> tuple:
    """Compute data-averaged LTV Gramians from explicit per-step matrices.

    Parameters
    ----------
    A_seq : (T, d, d)   per-step state-transition matrices
    B_seq : (T, d, m)   per-step input matrices
    C_seq : (T, p, d)   per-step output matrices
    eps   : float       diagonal regularisation

    Returns
    -------
    Wc   : (d, d)  data-averaged controllability Gramian
    Wo   : (d, d)  data-averaged observability Gramian
    meta : dict    {"sym_error_c", "sym_error_o", "well_conditioned"}
    """
    A_seq = to_torch(A_seq)
    B_seq = to_torch(B_seq)
    C_seq = to_torch(C_seq)

    T, d, _ = A_seq.shape
    Wc = torch.zeros(d, d, dtype=torch.float64)
    Wo = torch.zeros(d, d, dtype=torch.float64)

    # Forward pass: cumulative state-transition product  Phi_{t,0}
    Phi = torch.eye(d, dtype=torch.float64)
    for t in range(T):
        At = A_seq[t].double()
        Bt = B_seq[t].double()
        Ct = C_seq[t].double()
        Wc = Wc + Phi @ (Bt @ Bt.T) @ Phi.T
        Wo = Wo + Phi.T @ (Ct.T @ Ct) @ Phi
        Phi = At @ Phi

    if eps > 0:
        eye = torch.eye(d, dtype=torch.float64)
        Wc = Wc + eps * eye
        Wo = Wo + eps * eye

    sym_c = warn_sym(Wc, "Wc_ltv")
    sym_o = warn_sym(Wo, "Wo_ltv")
    Wc = (Wc + Wc.T) / 2
    Wo = (Wo + Wo.T) / 2

    meta = {
        "sym_error_c": sym_c,
        "sym_error_o": sym_o,
        "well_conditioned": (sym_c < 1e-3) and (sym_o < 1e-3),
    }
    return Wc, Wo, meta


def gramians_from_model(
    model: nn.Module,
    u_batch: ArrayLike,
    eps: float = 1e-6,
) -> tuple:
    """Compute data-averaged LTV Gramians by hooking into a gated SSM.

    The model must expose per-step (A_t, B_t, C_t) matrices.  Implement:

        model.get_abc_sequence(u)  -> (A_seq, B_seq, C_seq)
            A_seq : (T, d, d)
            B_seq : (T, d, m)
            C_seq : (T, p, d)

    If not implemented, falls back to treating the model as LTI by calling
    model.get_abc() -> (A, B, C) and repeating over T steps.

    Parameters
    ----------
    model   : nn.Module with get_abc_sequence or get_abc
    u_batch : (B, T, m)  batch of input sequences; Gramians averaged over B
    eps     : float      regularisation

    Returns
    -------
    Wc, Wo, meta  (averaged over batch dimension B)
    """
    u_batch = to_torch(u_batch, dtype=torch.float32)
    B_size, T, m = u_batch.shape

    Wc_acc = None
    Wo_acc = None

    for b in range(B_size):
        u = u_batch[b:b+1]   # (1, T, m)

        if hasattr(model, "get_abc_sequence"):
            with torch.no_grad():
                A_seq, B_seq, C_seq = model.get_abc_sequence(u)
        elif hasattr(model, "get_abc"):
            with torch.no_grad():
                A, B, C = model.get_abc()
            A_seq = A.unsqueeze(0).expand(T, -1, -1)
            B_seq = B.unsqueeze(0).expand(T, -1, -1)
            C_seq = C.unsqueeze(0).expand(T, -1, -1)
        else:
            raise AttributeError(
                "model must implement get_abc_sequence(u) or get_abc(). "
                "See ssm_interpret.ltv.gramians module docstring."
            )

        Wc_b, Wo_b, _ = gramians_ltv(A_seq, B_seq, C_seq, eps=eps)
        Wc_acc = Wc_b if Wc_acc is None else Wc_acc + Wc_b
        Wo_acc = Wo_b if Wo_acc is None else Wo_acc + Wo_b

    Wc = Wc_acc / B_size
    Wo = Wo_acc / B_size

    sym_c = warn_sym(Wc, "Wc_model")
    sym_o = warn_sym(Wo, "Wo_model")
    Wc = (Wc + Wc.T) / 2
    Wo = (Wo + Wo.T) / 2

    meta = {
        "sym_error_c": sym_c,
        "sym_error_o": sym_o,
        "well_conditioned": (sym_c < 1e-3) and (sym_o < 1e-3),
        "n_sequences": B_size,
    }
    return Wc, Wo, meta
