# PCMCI and its conditional independence tests

PCMCI (Runge et al., 2019) estimates a **static** lagged causal graph by conditional
independence testing, under the assumption of **causal sufficiency** — no unobserved
confounders. Two phases: a greedy PC skeleton step that prunes candidate parents, then the
MCI step that tests each surviving link conditional on both parents' parent sets.

Three notebooks, one per conditional independence test. They differ **only** in the CI test;
data, preprocessing and feature sets are identical.

| Notebook | CI test |
|----------|---------|
| `notebooks/pcmci_parcorr.ipynb` | ParCorr — linear |
| `notebooks/pcmci_gpdc.ipynb` | GPDCtorch — nonlinear additive |
| `notebooks/pcmci_cmi.ipynb` | FAISS-CMI — model-free |

Data, variable groups, feature sets and the preprocessing pipeline are documented in
[`data-and-preprocessing.md`](data-and-preprocessing.md). Note in particular the ⚠ warning
there about the `climate` feature set being reused across crops in these notebooks.

---

## ParCorr — Linear

Partial correlation via OLS residuals, with an analytic t-test null. Fast and interpretable;
misses any nonlinear relationship.

| Parameter | Value | Reason |
|-----------|-------|--------|
| `tau_min` | 1 | No contemporaneous links |
| `tau_max` | 6 (monthly) / 16 (weekly) | From RV auto-bivCI in the EDA notebook |
| `pc_alpha` | 0.2 | Liberal skeleton step preserves power for MCI |
| `alpha_level` | 0.05 | Final significance threshold |
| `fdr_method` | `fdr_bh` | Benjamini-Hochberg FDR control |
| `significance` | `analytic` | Valid after the probit transform; no permutation cost |

---

## GPDCtorch — Nonlinear Additive

Gaussian-process regression on X and Z residuals, then Distance Correlation on the residual
pair. Detects curved relationships ParCorr misses. Slow on CPU (~0.5–2 s per GP fit);
tractable with `max_combinations=1`.

| Parameter | Value | Reason |
|-----------|-------|--------|
| `tau_max` | 3 | GP conditioning sets grow with tau; wider sets overfit at T=192 |
| `pc_alpha` | 0.1 | GPDC has lower power for linear links; a tighter skeleton reduces spurious GP fits |
| `max_combinations` | 1 | Test only the top-scoring conditioning set per link — the primary runtime control |
| `significance` | `analytic` | Pre-computes the DC null distribution once per T |
| `sig_samples` | 200 | Samples for the analytic null cache |
| `alpha_level` | 0.05 | Final FDR-BH threshold |

---

## FAISS-CMI — Model-Free

Kraskov et al. (2004) kNN estimator of conditional mutual information, using FAISS
`IndexFlatL2` for joint-space nearest-neighbour search. Detects any statistical dependency —
nonlinear, non-additive, non-monotone. Requires shuffle testing; no analytic null exists.

```
I(X;Y|Z) = ψ(k) + ⟨ψ(n_z+1)⟩ − ⟨ψ(n_xz+1) + ψ(n_yz+1)⟩
```

The `FAISSCMI` class is implemented **inline** in `pcmci_cmi.ipynb`, not imported.

| Parameter | Value | Reason |
|-----------|-------|--------|
| `tau_max` | 3 | kNN estimates degrade with wide conditioning sets at T=192 |
| `pc_alpha` | 0.05 | Tighter than ParCorr — shuffle tests are expensive |
| `k` | 5 | Standard Kraskov choice; balances bias/variance at T≈192 |
| `sig_samples` | 200 | Shuffle permutations per test — minimum for stable p-values |
| `significance` | `shuffle_test` | Only valid option for CMI |
| `max_combinations` | 1 | Limits kNN fits per MCI link |
| FAISS backend | CPU (Apple Silicon) / CUDA (Linux+NVIDIA) | MPS unsupported by FAISS; falls back to CPU automatically |

---

## Method comparison

| | ParCorr | GPDCtorch | FAISS-CMI |
|-|---------|-----------|-----------|
| Detects | Linear conditional deps | Nonlinear additive | Any dependency |
| Speed | Fast (analytic) | Slow (~1–2 s/fit on CPU) | Medium (kNN + shuffle) |
| `tau_max` used | 6 / 16 | 3 | 3 |
| Significance | Analytic t-test | Analytic DC null | Shuffle test only |
| Key constraint | Misses nonlinear | `max_combinations=1` needed | Shuffle cost; T must be ≫ k |

For the confounder-aware counterparts see [`lpcmci.md`](lpcmci.md); for a neural,
regime-switching alternative see [`sdci.md`](sdci.md).
