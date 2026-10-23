"""
ssm_mech.surgery -- Causal mode surgery for HuggingFace Mamba models.

Surgery projects the SSM hidden state onto a rank-k subspace defined by the
top-k balanced mode eigenvectors, then reads out normally. This ablates
information that cannot be expressed in those k modes.

The patched forward function replaces the model's default fast-path scan
with a Python loop that applies the projection at each timestep. Surgery
is applied to one layer at a time; all other layers are unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Python scan with optional rank-k projection
# ---------------------------------------------------------------------------

def _python_scan(
    u: torch.Tensor,
    dt: torch.Tensor,
    A_cont: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    V_k: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sequential SSM scan with optional rank-k balanced mode projection.

    Parameters
    ----------
    u : (batch, seq, d_inner)
    dt : (batch, seq, d_inner)
    A_cont : (d_inner, d_state)  -- continuous-time log eigenvalues (negative)
    B : (batch, seq, d_state)
    C : (batch, seq, d_state)
    D : (d_inner,)
    V_k : (d_inner, d_state, k), optional
        Top-k balanced eigenvectors per channel. When provided, the hidden
        state is projected onto this subspace at each step.

    Returns
    -------
    y : (batch, seq, d_inner)
    """
    batch, seqlen, d_inner = u.shape
    d_state = A_cont.shape[1]
    dtype   = u.dtype

    # Always compute the exponential in float32 for numerical stability on all
    # backends (MPS float16 is incomplete; bfloat16 has limited exp support).
    A_disc = torch.exp(
        dt.float().unsqueeze(-1) * A_cont.float().unsqueeze(0).unsqueeze(0)
    ).to(dtype)  # (batch, seq, d_inner, d_state)

    x = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=dtype)
    outputs = []

    for t in range(seqlen):
        x = (
            A_disc[:, t] * x
            + (dt[:, t] * u[:, t]).unsqueeze(-1) * B[:, t].unsqueeze(1)
        )
        if V_k is not None:
            # rank-k projection: x <- V_k @ V_k^T @ x  (per channel)
            coeffs = (x.unsqueeze(-1) * V_k.unsqueeze(0)).sum(dim=2)
            x = (coeffs.unsqueeze(2) * V_k.unsqueeze(0)).sum(dim=-1)
        y_t = (x * C[:, t].unsqueeze(1)).sum(-1) + D.to(dtype) * u[:, t]
        outputs.append(y_t)

    return torch.stack(outputs, dim=1)


# ---------------------------------------------------------------------------
# Patched forward for a single MambaMixer layer
# ---------------------------------------------------------------------------

def _make_surgery_forward(
    mixer: torch.nn.Module,
    A_cont: torch.Tensor,
    V_k_ref: list,          # list of length 1; None = no surgery
):
    """
    Return a replacement forward function for one MambaMixer.

    ``V_k_ref`` is a mutable single-element list so the caller can swap the
    surgery basis between calls without rebuilding the closure.

    Parameters
    ----------
    mixer : MambaMixer module
    A_cont : (d_inner, d_state) tensor  -- continuous A on the target device
    V_k_ref : [tensor or None]
        V_k_ref[0] is (d_inner, d_state, k) when surgery is active, else None.
    """
    dt_rank = mixer.time_step_rank
    d_state = mixer.ssm_state_size

    def forward(hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        batch, seqlen, _ = hidden_states.shape
        dtype   = hidden_states.dtype

        xz      = mixer.in_proj(hidden_states)
        d_inner = xz.shape[-1] // 2
        xz_t    = xz.transpose(1, 2)
        x_t, z_t = xz_t[:, :d_inner], xz_t[:, d_inner:]

        x_conv = mixer.act(mixer.conv1d(x_t)[..., :seqlen])
        x_flat = x_conv.transpose(1, 2).contiguous().reshape(batch * seqlen, d_inner)
        x_dbl  = mixer.x_proj(x_flat)

        dt_raw = x_dbl[:, :dt_rank]
        B_flat = x_dbl[:, dt_rank: dt_rank + d_state]
        C_flat = x_dbl[:, dt_rank + d_state:]

        dt = F.softplus(
            F.linear(dt_raw, mixer.dt_proj.weight, mixer.dt_proj.bias)
        ).view(batch, seqlen, d_inner).to(dtype)

        B = B_flat.view(batch, seqlen, d_state).to(dtype)
        C = C_flat.view(batch, seqlen, d_state).to(dtype)
        u = x_conv.transpose(1, 2)

        y = _python_scan(u, dt, A_cont.to(dtype), B, C, mixer.D, V_k_ref[0])
        y = y * F.silu(z_t.transpose(1, 2))
        return mixer.out_proj(y.to(dtype))

    return forward


# ---------------------------------------------------------------------------
# Context manager for clean surgery application
# ---------------------------------------------------------------------------

@contextmanager
def surgery_context(
    model: torch.nn.Module,
    layer_idx: int,
    V_k: torch.Tensor,
):
    """
    Context manager that patches one layer with rank-k surgery and restores
    the original forward on exit.

    Parameters
    ----------
    model : MambaForCausalLM
    layer_idx : int
    V_k : (d_inner, d_state, k) tensor on the model's device
        Top-k balanced eigenvectors for the target layer.

    Example
    -------
    >>> with surgery_context(model, layer=20, V_k=basis.V_top(k=1, device=device)):
    ...     logits = model(input_ids=ids).logits
    """
    from .model_utils import find_mixer

    mixer = find_mixer(model, layer_idx)
    if mixer is None:
        raise ValueError(f"Layer {layer_idx} not found in model.")

    A_cont = -torch.exp(mixer.A_log.float()).to(V_k.device)
    V_k_ref = [V_k]
    original_forward = mixer.forward
    mixer.forward = _make_surgery_forward(mixer, A_cont, V_k_ref)
    try:
        yield
    finally:
        mixer.forward = original_forward
