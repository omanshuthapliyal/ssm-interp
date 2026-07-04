# ssm-interpret

**Control-theoretic mechanistic interpretability for linear state space models.**

`ssm-interpret` provides tools to extract, surgically ablate, and validate circuits in discrete-time linear SSMs (S4, Mamba in near-LTI regime, RWKV, and custom architectures).

---

## Installation

```bash
pip install ssm-interpret
```

Or from source:

```bash
git clone https://github.com/TODO/ssm-interpret
cd ssm-interpret
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, NumPy ≥ 1.24, SciPy ≥ 1.10.

---

## Quick Start

```python
import ssm_interpret as ssi

# 1. Extract the circuit basis from a trained SSM
Ab, Bb, Cb, T_bal, sigma = ssi.balanced_realization(A, B, C, T=64)
print(sigma)  # Hankel singular values — circuit energy ranking

# 2. Check what the library guarantees for this approximation
print(ssi.glover_bound(sigma, k=2))  # certified ||H - H_2||_H <= bound

# 3. Surgically remove circuit k
A2, B2, C2 = ssi.ablate(A, B, C, T_bal, k=0, mode="dynamics")

# 4. Measure selectivity: how much did we damage the target vs other channels?
sel = ssi.selectivity(y_surg, y_base, y_null, target_channel=0)
print(f"Selectivity: {sel:.1f}×")  # paper reports 251× vs 1.6× for patching

# 5. Classify state dimensions
print(ssi.kalman_taxonomy(A, B, C))
# {'live': [0, 1, 2], 'dead_type1': [], 'dead_type2': [3, 4, 5]}

# 6. Check if this framework applies to your gated model
from ssm_interpret.ltv import scope_report
report = scope_report(mamba_layer, u_batch)
print(report["regime"])   # "lti" | "near_lti" | "selective"
```

---

## Core Concepts

### Circuits as Balanced Modes

An SSM circuit is a direction in state space characterised by its Hankel singular value σ_k — the geometric mean of how controllable and observable that direction is. The *balanced realization* simultaneously diagonalises both Gramians, giving the unique canonical circuit basis (invariant to parameterisation, random seeds, and architecture).

### Subspace Surgery

Activation patching (zeroing x_t at one timestep) does not ablate an SSM circuit because x_{t+1} = A x_t + B u_t immediately regenerates it. Surgery modifies the dynamics matrix directly:

```
mode="dynamics"  →  A' = A − P_k A P_k      (remove the pole; permanent)
mode="output"    →  C' = C(I − P_k)          (silence the readout; faster)
```

See `surgery.py` docstring for the control-theory derivation and when each mode is appropriate.

### Glover Bound

Balanced truncation carries a certified error guarantee:

```
||H − H_k||_H  ≤  2 σ_{k+1}
```

Use `glover_bound(sigma, k)` to retrieve the certificate for any truncation level.

---

## Module Overview

| Module | Contents |
|---|---|
| `ssm_interpret.core` | `gramians`, `hsv`, `balanced_realization`, `truncate`, `impulse_response`, `glover_bound` |
| `ssm_interpret.surgery` | `ablate`, `selectivity`, `faithfulness_ratio` |
| `ssm_interpret.diagnostics` | `kalman_taxonomy`, `architecture_gap`, `sym_error`, `scope_check` |
| `ssm_interpret.sae` | `train_sae`, `alignment_score`, `dead_fraction`, `collect_states` |
| `ssm_interpret.ltv` *(experimental)* | `gramians_ltv`, `gramians_from_model`, `etmfd`, `llod`, `gvi`, `scope_report` |

---

## Integrating Your Own Model

To use `collect_states()` for SAE training, implement one method:

```python
class MySSM(nn.Module):
    def forward_states(self, u):
        # u      : (B, T, m)
        # return : states (B, T+1, d),  output (B, T, p)
        ...
```

For `ssm_interpret.ltv` diagnostics on a gated model:

```python
class MyMamba(nn.Module):
    def get_delta_sequence(self, u):
        # u      : (B, T, m)
        # return : delta (B, T, d)   per-step step-size parameter
        ...
    def get_abc_sequence(self, u):
        # return : A_seq (T,d,d), B_seq (T,d,m), C_seq (T,p,d)
        ...
```

---

## Demos

Six self-contained notebooks in `notebooks/`:

| Notebook | Paper section | Key result |
|---|---|---|
| `demo_hsv_universality.ipynb` | §2.2 | 5 seeds → same spectrum within 0.5% |
| `demo_surgery_selectivity.ipynb` | §3.3 | 251× surgery vs 1.6× patching |
| `demo_glover_bound.ipynb` | §2.4 | err/σ_{k+1} constant across k |
| `demo_sae_alignment.ipynb` | §3.7 | SAE cosine sim ≥ 0.90 without theory access |
| `demo_scope_diagnostics.ipynb` | §5 | ETMFD/LLOD/GVI: 14 orders of magnitude |
| `demo_nlp_shakespeare.ipynb` | §3, §6 | HSV spectrum, surgery bars, faithfulness |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Citation

<!-- ```bibtex
@article{ssm_circuits_2026,
  title   = {The Geometry of SSM Circuits: Control-Theoretic Extraction,
             Causal Surgery, and Scope},
  author  = {Anonymous},
  year    = {2026},
  note    = {Workshop on Actionable Interpretability, COLM 2026}
}
``` -->

---

## Acknowledgements

Development of this library was assisted by [Claude](https://claude.ai) (Anthropic),
used as a coding tool during research and implementation.
All scientific claims, experimental design, and mathematical content
are the work of the author.

## License

Apache 2.0 — see `LICENSE`.
