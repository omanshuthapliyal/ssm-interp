"""LTI scope diagnostics: ETMFD, LLOD, GVI.

Three scalar metrics that quantify how far a trained model departs from
linear time-invariant dynamics.  Together they determine whether the LTI
balanced-realization framework applies to a given model on a given task.

Metric definitions (§5)
------------------------
ETMFD  Effective Time-Mean Forgetting Decay
       Mean absolute deviation of per-step A_t from the time-mean A_bar.
       Near zero for LTI; large when gating varies heavily across time.

LLOD   Log Linearity-of-Output Deviation
       Measures output nonlinearity by comparing the actual output to
       the output predicted by a linear approximation around the mean input.
       Near zero for LTI; large when gating creates strong nonlinearity.

GVI    Gate Variation Index  (= CV(Δ) for Mamba-style models)
       Coefficient of variation of the per-step step-size parameter Δ.
       GVI = 0 exactly for an LTI model (fixed Δ).
       GVI is the primary scope indicator; ETMFD and LLOD are auxiliary.

Thresholds (scope_report)
--------------------------
GVI < 0.05           → "lti"      : framework applies directly
0.05 <= GVI < 0.20   → "near_lti" : apply with caution; use LTV Gramians
GVI >= 0.20          → "selective": LTI framework requires extension
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ssm_interpret._utils import to_torch


def etmfd(model: nn.Module, u_batch: torch.Tensor) -> float:
    """Effective Time-Mean Forgetting Decay.

    Requires model to implement get_abc_sequence(u) -> (A_seq, B_seq, C_seq).
    Falls back to 0.0 (exact LTI) if model only implements get_abc().

    Parameters
    ----------
    model   : nn.Module
    u_batch : (B, T, m)

    Returns
    -------
    float  mean ||A_t - A_bar||_F over batch and time; 0.0 for exact LTI.
    """
    u_batch = to_torch(u_batch, dtype=torch.float32)

    if not hasattr(model, "get_abc_sequence"):
        return 0.0

    deviations = []
    for b in range(u_batch.shape[0]):
        with torch.no_grad():
            A_seq, _, _ = model.get_abc_sequence(u_batch[b:b+1])
        A_seq = A_seq.double()              # (T, d, d)
        A_bar = A_seq.mean(0, keepdim=True) # (1, d, d)
        dev = (A_seq - A_bar).norm(dim=(1, 2)).mean()
        deviations.append(float(dev))

    return sum(deviations) / len(deviations)


def llod(model: nn.Module, u_batch: torch.Tensor) -> float:
    """Log Linearity-of-Output Deviation.

    Compares true output to a linear prediction from the mean input.
    Requires model.forward(u) -> output.

    Parameters
    ----------
    model   : nn.Module
    u_batch : (B, T, m)

    Returns
    -------
    float  mean relative deviation; 0.0 for exact LTI.
    """
    u_batch = to_torch(u_batch, dtype=torch.float32)

    with torch.no_grad():
        y_true = model(u_batch)
        u_mean = u_batch.mean(0, keepdim=True).expand_as(u_batch)
        y_linear = model(u_mean)

    dev = (y_true - y_linear).norm()
    ref = y_true.norm().clamp(min=1e-12)
    return float(dev / ref)


def gvi(model: nn.Module, u_batch: torch.Tensor) -> float:
    """Gate Variation Index = CV(Δ) for Mamba-style selective SSMs.

    Requires model to implement get_delta_sequence(u) -> delta_seq
    where delta_seq has shape (B, T, d) or (T, d).

    For a standard LTI model (no gating), returns 0.0 exactly.

    Parameters
    ----------
    model   : nn.Module
    u_batch : (B, T, m)

    Returns
    -------
    float  coefficient of variation of Δ; 0.0 for exact LTI.
    """
    u_batch = to_torch(u_batch, dtype=torch.float32)

    if not hasattr(model, "get_delta_sequence"):
        return 0.0

    with torch.no_grad():
        delta = model.get_delta_sequence(u_batch)  # (B, T, d) or (T, d)
    delta = to_torch(delta).double().flatten()
    mean = delta.mean().clamp(min=1e-12)
    std  = delta.std()
    return float(std / mean)


def scope_report(model: nn.Module, u_batch: torch.Tensor) -> dict:
    """Run all three scope metrics and return a unified verdict.

    Parameters
    ----------
    model   : nn.Module
    u_batch : (B, T, m)

    Returns
    -------
    dict with keys:
        "etmfd"      : float
        "llod"       : float
        "gvi"        : float
        "regime"     : "lti" | "near_lti" | "selective"
        "applicable" : bool
        "message"    : str
    """
    from ssm_interpret.diagnostics import scope_check

    u_batch = to_torch(u_batch, dtype=torch.float32)

    etmfd_val = etmfd(model, u_batch)
    llod_val  = llod(model, u_batch)
    gvi_val   = gvi(model, u_batch)

    verdict = scope_check(gvi_val)

    return {
        "etmfd": etmfd_val,
        "llod": llod_val,
        "gvi": gvi_val,
        "regime": verdict["regime"],
        "applicable": verdict["applicable"],
        "message": verdict["message"],
    }
