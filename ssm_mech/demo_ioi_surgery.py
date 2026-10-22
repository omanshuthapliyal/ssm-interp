"""
ssm_mech demo: IOI causal mode surgery on Mamba-130M.

This script shows the full ssm_mech workflow end-to-end:
  1. Load a Mamba model from HuggingFace.
  2. Build IOI prompts and fit a task-specific balanced basis at one layer.
  3. Check the ETMFD diagnostic.
  4. Run rank-1 causal mode surgery and print the logit-difference drop.
  5. Save the basis to disk for reuse.

Run:
    python ssm_mech/demo_ioi_surgery.py

Requirements: GPU strongly recommended for 130M. CPU will work but is slow.
"""

import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------

MODEL_ID     = "state-spaces/mamba-130m-hf"
TARGET_LAYER = 20          # L20 is the dominant IOI layer for 130M
N_FIT        = 200         # prompts for Gramian estimation
N_EVAL       = 80          # prompts for surgery evaluation
K            = 1           # rank of the projection

# If you have a gated model, set HF_TOKEN in your environment:
HF_TOKEN = os.environ.get("HF_TOKEN")

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

from ssm_mech import MambaAnalyzer
from ssm_mech.tasks import build_ioi

# ---------------------------------------------------------------------------
# 2. Load model
# ---------------------------------------------------------------------------

print(f"\nLoading {MODEL_ID} ...")
analyzer = MambaAnalyzer(MODEL_ID, token=HF_TOKEN)
print(f"Model loaded: {analyzer._n_layers} layers, device={analyzer.device}")

# ---------------------------------------------------------------------------
# 3. Build prompts
# ---------------------------------------------------------------------------

fit_prompts  = build_ioi(n=N_FIT, seed=42)
eval_prompts = build_ioi(n=N_EVAL, seed=99)   # held-out seed

print(f"\nExample prompt: '{fit_prompts[0].text}' (IO={fit_prompts[0].io_name})")

# ---------------------------------------------------------------------------
# 4. Fit task-specific Gramian
# ---------------------------------------------------------------------------

print(f"\nFitting balanced basis at layer {TARGET_LAYER} ({N_FIT} prompts) ...")
basis = analyzer.fit(
    layer=TARGET_LAYER,
    prompts=[p.text for p in fit_prompts],
)
print(f"  Tokens seen : {basis.n_tokens}")
print(f"  ETMFD       : {basis.etmfd:.4f}  (>0.18 = meaningful LTV deviation)")

# ---------------------------------------------------------------------------
# 5. ETMFD diagnostic
# ---------------------------------------------------------------------------

etmfd = analyzer.etmfd(
    layer=TARGET_LAYER,
    prompts=[p.text for p in fit_prompts],
)
print(f"  ETMFD (standalone call): {etmfd:.4f}")

# ---------------------------------------------------------------------------
# 6. Surgery
# ---------------------------------------------------------------------------

print(f"\nRunning rank-{K} surgery at layer {TARGET_LAYER} ({N_EVAL} eval prompts) ...")
result = analyzer.surgery(
    layer=TARGET_LAYER,
    basis=basis,
    k=K,
    eval_prompts=eval_prompts,
    task="ioi",
)
print(f"\nResult:")
print(f"  Baseline IOI logit diff : {result.baseline:+.3f} nats")
print(f"  Post-surgery            : {result.surgery_val:+.3f} nats")
print(f"  Delta (disruption)      : {result.delta:+.3f} nats")

if result.delta > 1.0:
    print("\n  Layer is causally important for IOI (delta > 1.0 nat).")
else:
    print("\n  Layer has limited causal role for IOI at this rank.")

# ---------------------------------------------------------------------------
# 7. Timescales
# ---------------------------------------------------------------------------

taus = analyzer.timescales(layer=TARGET_LAYER, basis=basis)
import numpy as np
mode0_taus = taus[:, 0]
print(f"\nMode-0 timescale across channels:")
print(f"  Median tau = {np.nanmedian(mode0_taus):.1f} tokens")
print(f"  Max tau    = {np.nanmax(mode0_taus):.1f} tokens")

# ---------------------------------------------------------------------------
# 8. Save basis
# ---------------------------------------------------------------------------

basis_path = f"basis_l{TARGET_LAYER}_k{K}.npz"
analyzer.save_basis(basis, basis_path)
print(f"\nBasis saved to '{basis_path}'. Load with:")
print(f"  basis = analyzer.load_basis('{basis_path}')")
