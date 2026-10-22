"""
ssm_mech.analyzer -- High-level MambaAnalyzer API.

MambaAnalyzer wraps model loading, Gramian estimation, surgery, and
diagnostics into a single object. It is the recommended entry point.

Example
-------
>>> from ssm_mech import MambaAnalyzer
>>> from ssm_mech.tasks import build_ioi, eval_ioi
>>>
>>> a = MambaAnalyzer("state-spaces/mamba-370m-hf")
>>>
>>> # Fit a task-specific Gramian at layer 39
>>> ioi_prompts = build_ioi(n=200)
>>> basis = a.fit(layer=39, prompts=[p.text for p in ioi_prompts])
>>>
>>> # Run surgery and measure the effect on IOI logit difference
>>> eval_prompts = build_ioi(n=80, seed=99)
>>> result = a.surgery(layer=39, basis=basis, k=1, eval_prompts=eval_prompts, task="ioi")
>>> print(f"delta = {result.delta:+.3f} nats")
>>>
>>> # ETMFD diagnostic (how input-dependent is the layer's timescale?)
>>> print(f"ETMFD = {a.etmfd(layer=39, prompts=[p.text for p in ioi_prompts]):.4f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from .gramian import OnlineStats, balanced_eigvec, merge_stats
from .model_utils import find_mixer, load_model, n_layers
from .surgery import surgery_context
from .tasks import (
    IOIPrompt, InductionPrompt, SVAPrompt,
    eval_ioi, eval_induction, eval_sva,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BalancedBasis:
    """
    Balanced mode eigenvectors for one layer, estimated from a given prompt set.

    Attributes
    ----------
    layer : int
    model_id : str
    V : (d_inner, d_state, d_state) float32 array
        V[ch, :, k] is the k-th balanced eigenvector for channel ch,
        sorted descending by eigenvalue.
    A_avg : (d_inner, d_state) float32 array
        Mean discrete eigenvalues used to compute W_c.
    n_tokens : int
        Number of tokens used to estimate this basis.
    task_specific : bool
        True if fitted from task-specific prompts (recommended for selectivity).
    etmfd : float
        ETMFD of the estimation distribution.
    """
    layer:         int
    model_id:      str
    V:             np.ndarray
    A_avg:         np.ndarray
    n_tokens:      int
    task_specific: bool
    etmfd:         float

    def V_top(self, k: int, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        """
        Return the top-k eigenvectors as a (d_inner, d_state, k) tensor.

        Parameters
        ----------
        k : int
            Number of modes to retain. k=1 is typically sufficient.
        device : str or torch.device
        """
        if k > self.V.shape[-1]:
            raise ValueError(f"k={k} exceeds d_state={self.V.shape[-1]}")
        return torch.tensor(self.V[:, :, :k], device=device)

    def save(self, path: str | Path) -> None:
        """Save basis to a .npz file."""
        np.savez_compressed(
            path,
            layer=self.layer, model_id=self.model_id,
            V=self.V, A_avg=self.A_avg,
            n_tokens=self.n_tokens,
            task_specific=self.task_specific,
            etmfd=self.etmfd,
        )

    @classmethod
    def load(cls, path: str | Path) -> "BalancedBasis":
        """Load a basis saved with :meth:`save`."""
        d = np.load(path, allow_pickle=False)
        return cls(
            layer=int(d["layer"]),
            model_id=str(d["model_id"]),
            V=d["V"].astype(np.float32),
            A_avg=d["A_avg"].astype(np.float32),
            n_tokens=int(d["n_tokens"]),
            task_specific=bool(d["task_specific"]),
            etmfd=float(d["etmfd"]),
        )


@dataclass
class SurgeryResult:
    """
    Result of a rank-k surgery experiment at one layer.

    Attributes
    ----------
    layer : int
    k : int
        Rank of the projection.
    task : str
        Task name ("ioi", "induction", "sva").
    baseline : float
        Mean logit difference before surgery.
    surgery_val : float
        Mean logit difference after surgery.
    delta : float
        baseline - surgery_val. Positive = surgery disrupts the task.
    n_prompts : int
    """
    layer:       int
    k:           int
    task:        str
    baseline:    float
    surgery_val: float
    delta:       float
    n_prompts:   int

    def __str__(self) -> str:
        return (
            f"Layer {self.layer} | task={self.task} | k={self.k} | "
            f"baseline={self.baseline:+.3f} | surgery={self.surgery_val:+.3f} | "
            f"delta={self.delta:+.3f} nats"
        )


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class MambaAnalyzer:
    """
    High-level interface for Gramian estimation and causal mode surgery
    on HuggingFace Mamba models.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier, e.g. "state-spaces/mamba-130m-hf".
    dtype : torch.dtype
        Model weight dtype. float32 is recommended; float16 reduces memory.
    token : str, optional
        HuggingFace access token for gated models.
    """

    def __init__(
        self,
        model_id: str,
        dtype: torch.dtype = torch.float32,
        token: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        log.info("Loading %s ...", model_id)
        self.model, self.tokenizer, self.device = load_model(
            model_id, dtype=dtype, token=token
        )
        self._n_layers = n_layers(self.model)
        log.info("Loaded: %d layers, device=%s", self._n_layers, self.device)

    # ------------------------------------------------------------------
    # Gramian estimation
    # ------------------------------------------------------------------

    def fit(
        self,
        layer: int,
        prompts: List[str],
        batch_size: int = 1,
    ) -> BalancedBasis:
        """
        Estimate the balanced basis at a given layer from a list of prompts.

        The Gramian is estimated online from actual forward passes: the
        estimation distribution matters. For task-specific analysis, pass
        prompts that are representative of the task (e.g. IOI sentences).
        For a general corpus basis, pass generic text.

        Parameters
        ----------
        layer : int
            Layer index (0-based).
        prompts : list of str
            Text prompts used to estimate the Gramian. More prompts = more
            accurate basis. 200--500 is typically sufficient.
        batch_size : int
            Number of prompts per forward pass. Currently only 1 is supported
            for correctness with variable-length sequences.

        Returns
        -------
        BalancedBasis
        """
        if not prompts:
            raise ValueError(
                "prompts cannot be empty. Pass a list of strings for corpus estimation, "
                "or use tasks.build_ioi() for an IOI-specific Gramian."
            )
        mixer = find_mixer(self.model, layer)
        if mixer is None:
            raise ValueError(f"Layer {layer} not found (model has {self._n_layers} layers).")

        d_inner = mixer.in_proj.out_features // 2
        d_state = mixer.ssm_state_size
        stats   = OnlineStats(d_inner, d_state)

        self._collect_gramian_stats(mixer, prompts, stats)

        V    = balanced_eigvec(stats)
        basis = BalancedBasis(
            layer=layer,
            model_id=self.model_id,
            V=V,
            A_avg=stats.A_avg.astype(np.float32),
            n_tokens=stats.n,
            task_specific=True,
            etmfd=stats.etmfd,
        )
        log.info(
            "Layer %d basis fitted from %d tokens (ETMFD=%.4f)",
            layer, stats.n, stats.etmfd,
        )
        return basis

    # ------------------------------------------------------------------
    # Surgery
    # ------------------------------------------------------------------

    def surgery(
        self,
        layer: int,
        basis: BalancedBasis,
        k: int,
        eval_prompts: Union[List[IOIPrompt], List[InductionPrompt], List[SVAPrompt]],
        task: str = "ioi",
    ) -> SurgeryResult:
        """
        Run rank-k causal mode surgery at one layer and measure the effect.

        Surgery replaces the SSM's hidden state with its projection onto the
        top-k balanced eigenvectors at each timestep, then evaluates the
        task metric before and after.

        Parameters
        ----------
        layer : int
        basis : BalancedBasis
            Returned by :meth:`fit`. Should be estimated from a distribution
            close to the evaluation task for best results.
        k : int
            Rank of the projection. k=1 targets the single most influential mode.
        eval_prompts : list
            Prompts to evaluate. Use :func:`tasks.build_ioi` etc.
        task : {"ioi", "induction", "sva"}

        Returns
        -------
        SurgeryResult
        """
        task = task.lower()
        if task not in ("ioi", "induction", "sva"):
            raise ValueError(f"task must be 'ioi', 'induction', or 'sva', got {task!r}")

        eval_fn = {"ioi": eval_ioi, "induction": eval_induction, "sva": eval_sva}[task]

        baseline    = eval_fn(self.model, self.tokenizer, eval_prompts, self.device)
        V_k         = basis.V_top(k=k, device=self.device)

        with surgery_context(self.model, layer_idx=layer, V_k=V_k):
            surgery_val = eval_fn(self.model, self.tokenizer, eval_prompts, self.device)

        result = SurgeryResult(
            layer=layer, k=k, task=task,
            baseline=baseline, surgery_val=surgery_val,
            delta=baseline - surgery_val,
            n_prompts=len(eval_prompts),
        )
        log.info(str(result))
        return result

    # ------------------------------------------------------------------
    # ETMFD diagnostic
    # ------------------------------------------------------------------

    def etmfd(self, layer: int, prompts: List[str]) -> float:
        """
        Compute the ETMFD (Effective Timescale Modulation Factor Deviation)
        for a given layer on the given prompts.

        Values near zero indicate near-LTI dynamics (constant discrete A).
        Values above ~0.18 indicate genuine input-dependent timescale modulation.

        Parameters
        ----------
        layer : int
        prompts : list of str

        Returns
        -------
        float
        """
        mixer = find_mixer(self.model, layer)
        if mixer is None:
            raise ValueError(f"Layer {layer} not found.")
        d_inner = mixer.in_proj.out_features // 2
        d_state = mixer.ssm_state_size
        stats   = OnlineStats(d_inner, d_state)
        self._collect_gramian_stats(mixer, prompts, stats)
        return stats.etmfd

    # ------------------------------------------------------------------
    # Mode timescales
    # ------------------------------------------------------------------

    def timescales(self, layer: int, basis: BalancedBasis) -> np.ndarray:
        """
        Compute the effective timescale (in tokens) for each mode of each channel.

        For a mode with eigenvector v and discrete eigenvalues a_j:
            tau_mode = sum_j v[j]^2 / (1 - a_j)

        Parameters
        ----------
        layer : int
        basis : BalancedBasis
            Must have been fitted at the given layer.

        Returns
        -------
        timescales : (d_inner, d_state) float array
            timescales[ch, k] = effective timescale of mode k for channel ch.
        """
        if basis.layer != layer:
            raise ValueError(
                f"basis.layer={basis.layer} does not match requested layer={layer}"
            )
        d_inner, d_state, _ = basis.V.shape
        taus = np.full((d_inner, d_state), np.nan, dtype=np.float64)
        for ch in range(d_inner):
            a = basis.A_avg[ch, :].astype(np.float64)
            if a.max() >= 1.0 or a.min() < 0.0:
                continue
            tau_j = 1.0 / (1.0 - a + 1e-12)
            for mode in range(d_state):
                v = basis.V[ch, :, mode].astype(np.float64)
                taus[ch, mode] = float(np.dot(v ** 2, tau_j))
        return taus

    # ------------------------------------------------------------------
    # Layer sweep
    # ------------------------------------------------------------------

    def layer_sweep(
        self,
        basis_prompts: List[str],
        eval_prompts: Union[List[IOIPrompt], List[InductionPrompt], List[SVAPrompt]],
        task: str = "ioi",
        k: int = 1,
        layers: Optional[List[int]] = None,
    ) -> List[SurgeryResult]:
        """
        Run surgery at every layer (or a given subset) and collect results.

        Parameters
        ----------
        basis_prompts : list of str
            Prompts used to fit the Gramian at each layer.
        eval_prompts : list
            Prompts used to evaluate task performance after surgery.
        task : {"ioi", "induction", "sva"}
        k : int
        layers : list of int, optional
            Layers to sweep. Defaults to all layers.

        Returns
        -------
        List[SurgeryResult], one per layer.
        """
        if layers is None:
            layers = list(range(self._n_layers))
        results = []
        for layer in layers:
            log.info("Sweeping layer %d / %d ...", layer, self._n_layers - 1)
            try:
                basis  = self.fit(layer=layer, prompts=basis_prompts)
                result = self.surgery(
                    layer=layer, basis=basis, k=k,
                    eval_prompts=eval_prompts, task=task,
                )
                results.append(result)
            except Exception as exc:
                log.warning("Layer %d skipped: %s", layer, exc)
        return results

    # ------------------------------------------------------------------
    # Basis I/O convenience
    # ------------------------------------------------------------------

    def save_basis(self, basis: BalancedBasis, path: str | Path) -> None:
        """Save a BalancedBasis to disk."""
        basis.save(path)
        log.info("Basis saved to %s", path)

    def load_basis(self, path: str | Path) -> BalancedBasis:
        """Load a BalancedBasis from disk."""
        return BalancedBasis.load(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_gramian_stats(
        self,
        mixer: torch.nn.Module,
        prompts: List[str],
        stats: OnlineStats,
    ) -> None:
        """Accumulate Gramian statistics from forward passes via a hook."""
        import torch.nn.functional as F

        dt_rank = mixer.time_step_rank
        d_state = mixer.ssm_state_size

        def hook(module, inp, output):
            hidden = inp[0]
            b, seqlen, _ = hidden.shape
            xz    = module.in_proj(hidden)
            d_in  = xz.shape[-1] // 2
            xz_t  = xz.transpose(1, 2)
            x_t   = xz_t[:, :d_in]
            x_conv = module.act(module.conv1d(x_t)[..., :seqlen])
            x_flat = x_conv.transpose(1, 2).contiguous().reshape(b * seqlen, d_in)
            x_dbl  = module.x_proj(x_flat)
            dt_raw = x_dbl[:, :dt_rank]
            B_flat = x_dbl[:, dt_rank: dt_rank + d_state]
            C_flat = x_dbl[:, dt_rank + d_state:]
            dt_val = F.softplus(
                F.linear(dt_raw, module.dt_proj.weight, module.dt_proj.bias)
            ).float().cpu()
            B_vals = B_flat.float().cpu()
            C_vals = C_flat.float().cpu()
            A_log  = module.A_log.float().cpu()
            dt_exp = dt_val.unsqueeze(-1)
            A_disc = torch.exp(-dt_exp * torch.exp(A_log).unsqueeze(0)).numpy()
            stats.update(A_disc, B_vals.numpy(), C_vals.numpy())

        handle = mixer.register_forward_hook(hook)
        try:
            for text in prompts:
                ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
                self.model(input_ids=ids)
        finally:
            handle.remove()
