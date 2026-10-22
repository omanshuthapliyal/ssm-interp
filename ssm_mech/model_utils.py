"""
ssm_mech.model_utils -- HuggingFace Mamba model loading and layer access.

Supports MambaForCausalLM from the `transformers` library. Automatically
selects CUDA, MPS, or CPU based on availability. Models are loaded in eval
mode with gradients disabled.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch


def load_model(
    model_id: str,
    dtype: torch.dtype = torch.float32,
    token: Optional[str] = None,
) -> Tuple[torch.nn.Module, object, torch.device]:
    """
    Load a HuggingFace MambaForCausalLM model and its tokenizer.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. "state-spaces/mamba-130m-hf".
    dtype : torch.dtype
        Weight dtype. float32 is recommended for numerical accuracy.
        float16 or bfloat16 reduce memory use on large models.
    token : str, optional
        HuggingFace access token for gated models.

    Returns
    -------
    model : MambaForCausalLM
        Model in eval mode, gradients disabled.
    tokenizer : AutoTokenizer
    device : torch.device
    """
    from transformers import AutoTokenizer, MambaForCausalLM

    device = _select_device()

    if device.type == "mps" and dtype in (torch.float16, torch.bfloat16):
        warnings.warn(
            f"dtype={dtype} is not fully supported on MPS. "
            "Falling back to torch.float32.",
            UserWarning,
            stacklevel=2,
        )
        dtype = torch.float32

    kwargs: dict = {"torch_dtype": dtype}
    if token:
        kwargs["token"] = token

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        model = MambaForCausalLM.from_pretrained(model_id, **kwargs)

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer, device


def find_mixer(model: torch.nn.Module, layer_idx: int) -> Optional[torch.nn.Module]:
    """
    Return the MambaMixer submodule at the given layer index.

    Works with the standard HuggingFace MambaForCausalLM layout:
    ``model.backbone.layers[layer_idx].mixer``

    Returns None if the layer or mixer attribute does not exist.
    """
    try:
        return model.backbone.layers[layer_idx].mixer
    except (AttributeError, IndexError):
        return None


def n_layers(model: torch.nn.Module) -> int:
    """Return the total number of layers in a HuggingFace Mamba model."""
    try:
        return len(model.backbone.layers)
    except AttributeError:
        return 0


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
