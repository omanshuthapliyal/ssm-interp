# The Geometry of SSM Causal Modes

Code and results for the paper **"The Geometry of SSM Causal Modes: Control-Theoretic Extraction, Causal Surgery, and Scope"** (under review).

Balanced realization is a basis-invariant decomposition of SSM hidden states into causal modes ranked by their joint controllability and observability. We show that mode surgery -- projecting the SSM state onto a rank-k subspace of the dominant balanced mode -- disrupts specific behavioral tasks (IOI, induction, SVA, copy) with high selectivity, and that the disruption pattern is consistent across scales (130M to 2.8B parameters) and architectures (Mamba, FalconH1, Nemotron-H, RWKV).

---

## Quick start

```bash
pip install -r requirements.txt
```

Python 3.11+ required. NLP synthetic experiments run on CPU. Mamba behavioral experiments require a CUDA GPU.

For the Mamba experiments, install additional dependencies:

```bash
pip install transformers datasets huggingface_hub
# mamba_ssm requires CUDA -- see https://github.com/state-spaces/mamba
```

---

## Key results

| Scale | IOI dominant layer | delta_IOI (nats) | Tier |
|---|---|---|---|
| 130M | L20 | +4.822 | CAUSAL-MODE |
| 370M | L39 | +3.654 (corpus) / +4.344 (IOI-specific) | CAUSAL-MODE |
| 790M | L23 | +2.643 | CAUSAL-MODE |
| 2.8B | L44 | +2.658 | CAUSAL-MODE |

**PCA vs balanced realization (Table 24):** at 130M L20, corpus-balanced outperforms PCA (1.074x). At 370M L39, corpus-balanced reverses; computing an IOI-specific Gramian restores the balanced > PCA hierarchy (+4.344 vs +4.259 nats), confirming the reversal is a corpus-distribution estimation artifact.

**Induction k-shot:** 370M monotone through k=4; 790M shows smooth sign reversal at k>=3 (causal mode migration confirmed); 2.8B shows abrupt reversal at k=4 only (reclassified as noise).

---

## Reproducing paper results (NLP synthetic experiments, CPU)

Each script below reproduces one result. Run from the repo root.

```bash
# Table 1 -- Surgery selectivity (251x, 28x, 0.28x) and negative control
python code/experiment_negative_control.py

# Figure 3 -- Surgery selectivity bar chart
python code/experiment_nlp_surgery.py

# Figure p2 -- Basis comparison (balanced vs PCA vs Schur)
python code/experiment_p2_basis_comparison.py

# Figure 10 -- Faithfulness ratio (12.3x, CI [9.9x, 16.5x])
python code/experiment_faithfulness.py

# Figure 4 -- Architecture gap (Full A vs Diagonal A)
python code/experiment2_q2c_arch_gap_robust.py

# Figure 11 -- Overcomplete SSM (d=6): live/dead gap (79x)
python code/experiment_overcomplete.py

# Figure 6 -- SAE alignment with balanced modes (mean |cos| = 0.974)
python code/experiment_p4_sae_d6.py
```

---

## Reproducing Mamba behavioral experiments (GPU required)

Pre-computed outputs are in `code/phase0/outputs/`. To rerun:

```bash
cd code/phase0

# 130M IOI layer sweep
python exp_layer_sweep_130m.py

# 130M vs PCA baseline (Table 24 -- 130M L20)
python exp_pca_baseline.py

# 370M IOI-specific Gramian (Table 24 -- distribution-mismatch test)
python exp_ioi_gramian_370m_l39.py

# Induction k-shot at 370M
python exp_induction_370m_k3.py

# Mode timescale analysis (L0 ~22 tokens, L20 ~57K tokens)
python exp_mode_timescale.py

# ETMFD diagnostic at 370M L39
python exp_etmfd_370m_l39.py
```

Scale sweep results (370M, 790M, 2.8B) are in `code/phase0/outputs/` and `code/phase0/n7_str/`.

---

## Repository structure

```
ssm_mech/              Library (pip-installable; in development)
  analyzer.py          MambaAnalyzer -- fit, surgery, etmfd, timescales
  gramian.py           OnlineStats, gramian_diagonal_A, merge_stats
  model_utils.py       load_hf_model, find_mixer_hf
  surgery.py           make_surgery_forward, eval_ioi, eval_induction
  tasks.py             IOI / induction / SVA prompt builders

code/                  NLP synthetic model experiments (CPU, no GPU)
  experiment_*.py      One script per paper claim
  ssm_utils.py         Shared utilities (ETMFD, LLOD, GVI diagnostics)

code/phase0/           Mamba behavioral experiments (GPU required)
  exp_*.py             One script per experiment
  utils.py             OnlineStats, Gramian helpers, logging
  hf_mamba_utils.py    HuggingFace Mamba model loading + surgery hooks
  outputs/             Pre-computed NPZ, CSV, JSON results

docs/
  manuscript.html      Main paper (full HTML)
  index.html           Blog post / explainer
```

---

## Citation

```bibtex
@article{thapliyal2026ssm,
  title   = {The Geometry of {SSM} Causal Modes: Control-Theoretic Extraction, Causal Surgery, and Scope},
  author  = {Thapliyal, Omanshu},
  year    = {2026},
  note    = {Under review}
}
```
