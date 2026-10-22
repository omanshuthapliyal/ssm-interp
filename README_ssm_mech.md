# ssm_mech

**Causal mode analysis for SSM language models.**

`ssm_mech` provides tools to identify which layers and internal modes of a State Space Model (SSM) are causally responsible for specific linguistic behaviors, using Gramian decomposition and state surgery. It works directly with HuggingFace Mamba models and requires no modification to model weights.

---

## What it does

SSMs like Mamba maintain a recurrent hidden state that compresses information across tokens. The hidden state lives in a high-dimensional space, but not all dimensions carry the same information. `ssm_mech` helps you find the most causally significant directions in that space:

1. **Gramian estimation** -- run forward passes on a set of prompts and compute a balanced controllability Gramian for any layer. The Gramian identifies the state directions that most efficiently encode input information.

2. **Causal mode surgery** -- project the hidden state onto a rank-k subspace (the top-k "causal modes") at inference time, then measure how much a task score drops. A large drop means that mode is load-bearing for that task.

3. **ETMFD diagnostic** -- quantify how much a layer's recurrent dynamics vary with input. Values near 0 indicate approximately linear time-invariant dynamics; values above ~0.18 indicate genuine input-dependent timescale modulation.

4. **Mode timescales** -- compute the effective memory horizon (in tokens) for each recurrent mode.

---

## Installation

```bash
pip install torch transformers scipy numpy
pip install -e .   # from the repo root
```

GPU is strongly recommended for models larger than 130M parameters. CPU works for small models and for testing.

---

## Quick start

```python
from ssm_mech import MambaAnalyzer
from ssm_mech.tasks import build_ioi

# Load model
analyzer = MambaAnalyzer("state-spaces/mamba-130m-hf")

# Build prompts -- use task-specific prompts for a task-specific Gramian
fit_prompts  = build_ioi(n=200)
eval_prompts = build_ioi(n=80, seed=99)

# Fit the balanced basis at layer 20
basis = analyzer.fit(
    layer=20,
    prompts=[p.text for p in fit_prompts],
)
print(f"ETMFD: {basis.etmfd:.4f}")

# Run rank-1 surgery and measure the logit-difference drop
result = analyzer.surgery(
    layer=20,
    basis=basis,
    k=1,
    eval_prompts=eval_prompts,
    task="ioi",
)
print(f"Baseline: {result.baseline:+.3f}  After surgery: {result.surgery_val:+.3f}")
print(f"Disruption (delta): {result.delta:+.3f} nats")
```

---

## Layer sweep

To find which layers are causally important for a given task across the whole model:

```python
results = analyzer.layer_sweep(
    basis_prompts=[p.text for p in fit_prompts],
    eval_prompts=eval_prompts,
    task="ioi",
    k=1,
)
for r in sorted(results, key=lambda x: -x.delta):
    print(r)
```

---

## Built-in tasks

Three tasks are provided out of the box.

**IOI (Indirect Object Identification)** -- given a sentence like "When Mary and John went to the store, John gave a book to", predict that the next token is "Mary". Metric: `logit(IO_name) - logit(subject_name)`.

```python
from ssm_mech.tasks import build_ioi, eval_ioi

prompts = build_ioi(n=100)
score   = eval_ioi(analyzer.model, analyzer.tokenizer, prompts, analyzer.device)
```

**Induction** -- given a k-shot pattern `a b a b ... a`, predict `b`. Metric: `logit(b) - logit(a)`.

```python
from ssm_mech.tasks import build_induction

prompts = build_induction(n=100, k_shot=2)
```

**SVA (Subject-Verb Agreement)** -- predict whether a singular or plural verb is correct given a noun phrase with an attractor. Metric: `logit(correct_verb) - logit(incorrect_verb)`.

```python
from ssm_mech.tasks import build_sva

prompts = build_sva(n=100)
```

---

## Saving and loading bases

Reuse a fitted basis without re-running forward passes:

```python
# Save
analyzer.save_basis(basis, "mamba130m_l20_ioi.npz")

# Load later
basis = analyzer.load_basis("mamba130m_l20_ioi.npz")
result = analyzer.surgery(layer=20, basis=basis, k=1, eval_prompts=eval_prompts, task="ioi")
```

---

## ETMFD diagnostic

The ETMFD (Effective Timescale Modulation Factor Deviation) measures how much the layer's discrete recurrent matrix varies with input, relative to its mean value:

```
ETMFD = sqrt(sum_ij Var[A_disc_ij]) / ||A_avg||_F
```

A value near 0 means the layer behaves like a fixed linear filter regardless of what it reads. Values above approximately 0.18 indicate that the layer actively adjusts its temporal integration based on content. The boundary at 0.18 is empirically calibrated against near-LTI layers.

```python
val = analyzer.etmfd(layer=20, prompts=[p.text for p in fit_prompts])
print(f"Layer 20 ETMFD: {val:.4f}")
```

---

## Merging statistics across GPUs or workers

If you collect Gramian statistics on multiple machines or data shards, merge them exactly using Chan's parallel formula:

```python
from ssm_mech.gramian import OnlineStats, merge_stats

s1 = OnlineStats.load("shard1.npz")
s2 = OnlineStats.load("shard2.npz")
combined = merge_stats(s1, s2)
```

---

## Mode timescales

After fitting a basis, inspect the memory horizon of each mode:

```python
import numpy as np

taus = analyzer.timescales(layer=20, basis=basis)
# taus[ch, k] = effective timescale of mode k for channel ch (in tokens)

median_tau = np.nanmedian(taus[:, 0])   # mode-0 across channels
print(f"Dominant mode median timescale: {median_tau:.1f} tokens")
```

---

## The distribution-mismatch effect

The Gramian is estimated from a corpus of prompts, not from the model weights alone. This means the balanced basis depends on what distribution you use to estimate it. For task-specific interpretability:

- Use task prompts (`build_ioi`, `build_induction`) to fit the basis when you want to identify task-specific causal modes.
- A general corpus basis is a valid baseline but may underestimate mode importance for specific tasks, because the most causally important modes for task X are aligned with task X's input distribution.

This is expected behavior, not a flaw. The ETMFD diagnostic helps quantify how much the distribution matters: high ETMFD layers are more sensitive to the choice of estimation corpus.

---

## Supported models

`ssm_mech` supports all HuggingFace `MambaForCausalLM` models from the `transformers` library:

| Model | HF ID |
|-------|-------|
| Mamba-130M | `state-spaces/mamba-130m-hf` |
| Mamba-370M | `state-spaces/mamba-370m-hf` |
| Mamba-790M | `state-spaces/mamba-790m-hf` |
| Mamba-1.4B | `state-spaces/mamba-1.4b-hf` |
| Mamba-2.8B | `state-spaces/mamba-2.8b-hf` |

For FalconH1, Nemotron-H, and other hybrid architectures, the Mamba mixer submodule must be accessible via `model.backbone.layers[i].mixer`. Contributions for additional model families are welcome.

---

## Running the demo

```bash
python ssm_mech/demo_ioi_surgery.py
```

This runs the full IOI surgery workflow on Mamba-130M (downloads ~250 MB on first run) and prints the layer-20 disruption delta.

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

The Gramian and task tests run without a GPU. The analyzer integration tests (`test_analyzer.py`) require `transformers` and will build a tiny synthetic model for CPU-only execution.

---

## License

Apache 2.0
