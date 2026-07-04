"""Sparse Autoencoder (SAE) validation: state-trajectory alignment with balanced modes.

The SAE here is a *validation tool*, not a discovery tool.  For transformers,
SAEs are used to find features (no canonical basis exists).  For LTI SSMs,
theory provides the balanced basis; the SAE independently verifies that these
directions correspond to structure the network actually learned — not a
post-hoc mathematical artefact.

Key result (§3.7): an SAE trained only on state trajectories {x_t}, with no
access to A, B, C or the balancing transformation, recovers balanced mode
directions at cosine similarity >= 0.90 for well-separated EMA tasks.

Integration contract
--------------------
To use collect_states() with your model, implement:

    class MySSM(nn.Module):
        def forward_states(self, u):
            # u     : (B, T, m)
            # return: states (B, T+1, d),  output (B, T, p)
            ...

This is the documented integration contract for ssm-interpret.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn

from ssm_interpret._utils import ArrayLike, to_torch


# ── Sparse Autoencoder ─────────────────────────────────────────────────────

class SparseAutoEncoder(nn.Module):
    """One-layer sparse autoencoder: encode → ReLU → decode.

    Parameters
    ----------
    d_input  : int   dimension of input states (= SSM state dim d)
    d_hidden : int   number of SAE features (typically 2*d or 4*d)
    """

    def __init__(self, d_input: int, d_hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(d_input, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_input, bias=False)
        # Unit-norm decoder columns (tied-norm convention)
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_hat, features) where features has ReLU sparsity."""
        features = torch.relu(self.encoder(x))
        x_hat = self.decoder(features)
        return x_hat, features

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))


# ── Training ───────────────────────────────────────────────────────────────

def train_sae(
    X: ArrayLike,
    d_hidden: int,
    l1_weight: float = 1e-3,
    epochs: int = 3000,
    lr: float = 1e-3,
    seed: int = 0,
    verbose: bool = False,
) -> SparseAutoEncoder:
    """Train a sparse autoencoder on state trajectories.

    Parameters
    ----------
    X        : (N, d)  state trajectory matrix; rows are individual states x_t.
               Build from collect_states(model, u_batch).
    d_hidden : int     number of SAE features
    l1_weight: float   sparsity penalty coefficient
    epochs   : int     training epochs
    lr       : float   Adam learning rate
    seed     : int     for reproducibility
    verbose  : bool    print loss every 500 epochs

    Returns
    -------
    SparseAutoEncoder  trained model (eval mode)
    """
    torch.manual_seed(seed)
    X = to_torch(X, dtype=torch.float32)
    d_input = X.shape[1]

    sae = SparseAutoEncoder(d_input, d_hidden)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    sae.train()
    for epoch in range(1, epochs + 1):
        x_hat, features = sae(X)
        recon_loss = ((X - x_hat) ** 2).mean()
        l1_loss = features.abs().mean()
        loss = recon_loss + l1_weight * l1_loss

        opt.zero_grad()
        loss.backward()
        # Re-normalise decoder columns after each step (tied-norm)
        opt.step()
        with torch.no_grad():
            sae.decoder.weight.data = nn.functional.normalize(
                sae.decoder.weight.data, dim=0
            )

        if verbose and epoch % 500 == 0:
            print(f"  epoch {epoch:4d}  recon={recon_loss.item():.4e}  "
                  f"l1={l1_loss.item():.4e}")

    sae.eval()
    return sae


# ── Alignment scoring ──────────────────────────────────────────────────────

def alignment_score(
    sae: SparseAutoEncoder,
    T_bal: ArrayLike,
    n_modes: int,
) -> ArrayLike:
    """Maximum cosine similarity between each balanced mode and SAE features.

    For each balanced mode direction (column k of T_bal), find the SAE
    decoder feature with the highest absolute cosine similarity.

    Parameters
    ----------
    sae     : trained SparseAutoEncoder
    T_bal   : (d, d)  balancing transformation from balanced_realization()
    n_modes : int     number of modes to score (1 <= n_modes <= d)

    Returns
    -------
    cosine_sims : (n_modes,)  max |cos| between mode k and any SAE feature.
                  Values near 1.0 indicate the SAE has recovered the mode.
    """
    T_bal = to_torch(T_bal, dtype=torch.float32)
    d = T_bal.shape[0]

    # Decoder matrix: columns are feature directions, shape (d, d_hidden)
    W_dec = sae.decoder.weight.detach()   # (d, d_hidden)
    W_dec = nn.functional.normalize(W_dec, dim=0)

    scores = []
    for k in range(n_modes):
        mode_dir = T_bal[:, k]
        mode_dir = mode_dir / mode_dir.norm().clamp(min=1e-12)
        cos_sims = (W_dec.T @ mode_dir).abs()   # (d_hidden,)
        scores.append(float(cos_sims.max()))

    return torch.tensor(scores)


def dead_fraction(
    sae: SparseAutoEncoder,
    X: ArrayLike,
    tol: float = 0.05,
) -> float:
    """Fraction of SAE features that are never active on dataset X.

    A feature is considered dead if its mean activation across X is below
    tol * max_activation.

    Parameters
    ----------
    sae : trained SparseAutoEncoder
    X   : (N, d) state matrix (same distribution used for training)
    tol : float  relative threshold for "dead"

    Returns
    -------
    float in [0, 1]
    """
    X = to_torch(X, dtype=torch.float32)
    with torch.no_grad():
        features = sae.encode(X)   # (N, d_hidden)
    mean_acts = features.mean(0)   # (d_hidden,)
    threshold = tol * float(mean_acts.max().clamp(min=1e-12))
    n_dead = int((mean_acts < threshold).sum())
    return n_dead / features.shape[1]


# ── State collection ───────────────────────────────────────────────────────

def collect_states(
    model: nn.Module,
    u_batch: ArrayLike,
    method: str = "auto",
    max_seqs: Optional[int] = None,
    state_module: Optional[str] = None,
) -> torch.Tensor:
    """Collect state trajectories from a trained SSM for SAE training.

    Parameters
    ----------
    model        : nn.Module.  Must implement forward_states(u) -> (states, output)
                   if method="explicit" or method="auto".  See integration contract
                   at the top of this module.
    u_batch      : (B, T, m)  input sequences
    method       : "auto" | "explicit" | "hook"
                   "explicit" — calls model.forward_states(u)
                   "hook"     — registers a forward hook on state_module
                   "auto"     — tries explicit, falls back to hook with warning
    max_seqs     : int | None  cap on returned rows (N = min(B*T, max_seqs)).
                   Use to avoid OOM for large models / long sequences.
    state_module : str | None  submodule name for hook-based collection.
                   Required when method="hook".

    Returns
    -------
    X : (N, d)  state matrix where N = min(B*T, max_seqs).
                Rows are individual state vectors x_t, shuffled.
    """
    u_batch = to_torch(u_batch, dtype=torch.float32)
    B, T, m = u_batch.shape

    # ── Attempt explicit method ────────────────────────────────────────────
    if method in ("explicit", "auto"):
        if hasattr(model, "forward_states"):
            with torch.no_grad():
                states, _ = model.forward_states(u_batch)   # (B, T+1, d)
            # Drop initial zero state x_0
            states = states[:, 1:, :]                        # (B, T, d)
            X = states.reshape(B * T, -1)                    # (B*T, d)
            if max_seqs is not None and X.shape[0] > max_seqs:
                idx = torch.randperm(X.shape[0])[:max_seqs]
                X = X[idx]
            return X.detach()
        elif method == "explicit":
            raise AttributeError(
                "model does not implement forward_states(u). "
                "Add the method or use method='hook'."
            )
        else:
            warnings.warn(
                "model does not implement forward_states(). "
                "Falling back to forward hook — set method='hook' to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )
            method = "hook"

    # ── Hook-based fallback ────────────────────────────────────────────────
    if method == "hook":
        if state_module is None:
            raise ValueError(
                "state_module must be specified when method='hook'. "
                "Pass the name of the submodule whose output is the SSM state."
            )
        captured: list[torch.Tensor] = []

        def _hook(module, input, output):
            if isinstance(output, tuple):
                captured.append(output[0].detach())
            else:
                captured.append(output.detach())

        target = dict(model.named_modules()).get(state_module)
        if target is None:
            raise ValueError(
                f"Submodule '{state_module}' not found in model. "
                f"Available: {list(dict(model.named_modules()).keys())}"
            )
        handle = target.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                model(u_batch)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("Hook did not capture any output.")

        X = torch.cat([t.reshape(-1, t.shape[-1]) for t in captured], dim=0)
        if max_seqs is not None and X.shape[0] > max_seqs:
            idx = torch.randperm(X.shape[0])[:max_seqs]
            X = X[idx]
        return X.detach()

    raise ValueError(f"method must be 'auto', 'explicit', or 'hook', got '{method}'")
