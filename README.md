# Gaia

Causal inference on U.S. crop return volatility (corn, soybean, wheat). Tests causal dependencies between realized volatility and macroeconomic indicators, news sentiment, and climate variables using multivariate time-series methods — PCMCI with three conditional independence tests: ParCorr, GPDCtorch, and FAISS-CMI.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/eda_preprocessing.ipynb` | EDA and full preprocessing pipeline (impute → detrend → anomalize → Gaussianise). Distribution analysis, seasonality checks, `tau_max` selection via bivCI, scatter/density plots for CI test selection, sanity-check PCMCI + ParCorr run. |
| `notebooks/pcmci_parcorr.ipynb` | PCMCI + ParCorr on all three crops at monthly frequency across three feature sets: `climate`, `climate_rv`, `macro_news_climate_rv`. Cross-crop comparison of significant causal links into realized volatility. |
| `notebooks/pcmci_gpdc.ipynb` | PCMCI + GPDCtorch (GP regression + Distance Correlation) on all three crops. Detects nonlinear additive dependencies that ParCorr misses. Includes interpretation of notable links (SOI confounder, crop-specific ENSO asymmetry). |
| `notebooks/pcmci_cmi.ipynb` | PCMCI + FAISS-CMI (Kraskov kNN conditional mutual information). Model-free: detects any statistical dependency. FAISSCMI class is implemented inline — no import from external source. |
| `notebooks/lpcmci_parcorr.ipynb` | LPCMCI + ParCorr. Drops PCMCI's causal-sufficiency assumption: learns a DPAG that distinguishes genuine directed causes (`-->`) from links that are merely confounded by an unobserved common driver (`<->`). Compares its verdicts against `pcmci_parcorr.ipynb`'s significant links into `weekly_rv`. |
| `notebooks/lpcmci_gpdc.ipynb` | LPCMCI + GPDCtorch. Same confounder-awareness, paired with the nonlinear-additive CI test. Runs at a shorter `tau_max` and tighter search bounds than the ParCorr version — GP fits make LPCMCI's non-ancestral phase far more expensive. Compares against `pcmci_gpdc.ipynb`. |
| `notebooks/lpcmci_cmi.ipynb` | LPCMCI + FAISS-CMI. Model-free CI test (FAISSCMI implemented inline, as in `pcmci_cmi.ipynb`) combined with confounder-awareness. Uses a reduced `sig_samples` for LPCMCI's own runs (the shuffle test multiplies cost across LPCMCI's many CI calls); the `pcmci_cmi.ipynb`-comparison baseline keeps the original `sig_samples=200`. |

---

## Data

### Datasets

| Path | Frequency | Rows | Date Range |
|------|-----------|------|------------|
| `data/monthly/{corn,soybean,wheat}.csv` | Monthly | 192 | 2009-04 → 2025-03 |
| `data/weekly/{corn,soybean,wheat}.csv` | Weekly | 832 | 2009-04 → 2025-03 |

All three crops share the same 22 columns. Managed via DVC: `dvc pull`.

### Variables

| Group | Variables | N |
|-------|-----------|---|
| `rv` | `weekly_rv` | 1 |
| `climate` | `prcp`, `awnd`, `tmin`, `tmax`, `co2`, `pdsi`, `EVAP`, `soil_moisture_mean_m3m3`, `cloud_cover_mean_pct`, `humidity_mean_pct`, `ssta_elino`, `ssta_lanina`, `SOI_index`, `NAO_index` | 14 |
| `macro` | `DJIA_Index`, `WTI_Index`, `Broad_Dollar_index`, `Stock_Uncertainty` | 4 |
| `news` | `frbsf_sentiment`, `epu_index`, `Text_Climate_Anomaly` | 3 |

### Feature Sets

| Feature Set | Groups | N vars |
|-------------|--------|--------|
| `climate` | climate | 14 |
| `climate_rv` | climate + rv | 15 |
| `macro_news_climate_rv` | macro + news + climate + rv | 22 |

---

## Preprocessing Pipeline

All PCMCI runs use the same 4-step pipeline (established in `eda_preprocessing.ipynb`):

| Step | Operation | Why |
|------|-----------|-----|
| **Impute** | Linear interpolation for `pdsi`, `Broad_Dollar_index`; forward-fill for `DJIA_Index` | Eliminates NaNs before array operations |
| **Detrend** | `pp.smooth(smooth_width=180mo, kernel='gaussian', residuals=True)` | Removes secular drift (CO₂ rise, DJIA trend, dollar cycle) that would inflate long-range spurious links |
| **Anomalize** | Subtract phase mean, divide by phase std (cycle_length=12 monthly / 52 weekly) | Removes annual seasonality so CI tests see residuals, not seasonal patterns |
| **Gaussianise** | `pp.trafo2normal` (rank-based probit transform) | Marginal Gaussianity validates ParCorr's analytic t-test; improves GP covariance conditioning for GPDCtorch; does not alter rank-order dependence |

---

## Conditional Independence Tests

### ParCorr — Linear

Partial correlation via OLS residuals. Analytic t-test null. Fast and interpretable; misses any nonlinear relationship.

| Parameter | Value | Reason |
|-----------|-------|--------|
| `tau_min` | 1 | No contemporaneous links |
| `tau_max` | 6 (monthly) / 16 (weekly) | From RV auto-bivCI in EDA notebook |
| `pc_alpha` | 0.2 | Liberal skeleton step preserves power for MCI |
| `alpha_level` | 0.05 | Final significance threshold |
| `fdr_method` | `fdr_bh` | Benjamini-Hochberg FDR control |
| `significance` | `analytic` | Valid after `trafo2normal`; no permutation cost |

### GPDCtorch — Nonlinear Additive

GP regression on X and Z residuals, then Distance Correlation on the residual pair. Detects curved relationships ParCorr misses. Slow on CPU (~0.5–2 s per GP fit); tractable with `max_combinations=1`.

| Parameter | Value | Reason |
|-----------|-------|--------|
| `tau_max` | 3 | GP conditioning sets grow with tau; wider sets overfit at T=192 |
| `pc_alpha` | 0.1 | GPDC has lower power for linear links; tighter skeleton reduces spurious GP fits |
| `max_combinations` | 1 | Test only the top-scoring conditioning set per link — primary runtime control |
| `significance` | `analytic` | Pre-computes DC null distribution once per T |
| `sig_samples` | 200 | Samples for the analytic null cache |
| `alpha_level` | 0.05 | Final FDR-BH threshold |

### FAISS-CMI — Model-Free

Kraskov et al. (2004) kNN estimator of conditional mutual information using FAISS IndexFlatL2 for joint-space nearest-neighbour search. Detects any statistical dependency (nonlinear, non-additive, non-monotone). Requires shuffle testing — no analytic null available.

```
I(X;Y|Z) = ψ(k) + ⟨ψ(n_z+1)⟩ − ⟨ψ(n_xz+1) + ψ(n_yz+1)⟩
```

The `FAISSCMI` class is implemented inline in `pcmci_cmi.ipynb` (not imported from source).

| Parameter | Value | Reason |
|-----------|-------|--------|
| `tau_max` | 3 | kNN estimates degrade with wide conditioning sets at T=192 |
| `pc_alpha` | 0.05 | Tighter than ParCorr — shuffle tests are expensive; avoid unnecessary fits |
| `k` | 5 | Standard Kraskov choice; balances bias/variance at T~192 |
| `sig_samples` | 200 | Shuffle permutations per test — min for stable p-values |
| `significance` | `shuffle_test` | Only valid option for CMI; no analytic null |
| `max_combinations` | 1 | Limits kNN fits per MCI link |
| FAISS backend | CPU (Apple Silicon) / CUDA (Linux+NVIDIA) | MPS not supported by FAISS; falls back to CPU automatically |

### LPCMCI — Latent-Confounder-Aware

Gerhardus & Runge (2020). Drops PCMCI's causal-sufficiency assumption (no unobserved confounders).
Learns a time series DPAG with edge types instead of a DAG: `-->` (candidate cause, not
reverse-causal), `<->` (confounded — not causal), and `o->`/`o-o` (orientation undetermined).
Implemented as three sibling notebooks — `lpcmci_parcorr.ipynb`, `lpcmci_gpdc.ipynb`,
`lpcmci_cmi.ipynb` — one per CI test, mirroring `pcmci_parcorr.ipynb` / `pcmci_gpdc.ipynb` /
`pcmci_cmi.ipynb` so only the algorithm (PCMCI vs. LPCMCI) changes within each pair.

LPCMCI's non-ancestral search phase issues far more CI tests than plain PCMCI's single greedy PC
step, so every pairing needs tighter bounds than its PCMCI counterpart — and the tighter the
underlying CI test's per-call cost (ParCorr's analytic t-test < FAISSCMI's shuffle test < GPDCtorch's
GP fit), the more aggressive the bounds have to get:

| Parameter | ParCorr | GPDCtorch | FAISS-CMI |
|-----------|---------|-----------|-----------|
| `tau_max` (LPCMCI's own runs) | 3 | 1 | 1 |
| `pc_alpha` | 0.01 | 0.01 | 0.01 |
| `n_preliminary_iterations` | 1 | 0 | 0 |
| `max_p_global` / `max_p_non_ancestral` / `max_q_global` | 2 / 1 / 2 | 1 / 0 / 1 | 1 / 0 / 1 |
| CI-test-specific | `significance='analytic'` | `sig_samples=200` (analytic null cache) | `k=0.1`, `sig_samples=50` (vs. 200 in `pcmci_cmi.ipynb`) |
| Measured: N=22, tightest bounds | 25 s | 120 s | 37 s |

Each notebook documents its own empirical timings (including configs that didn't converge in 3-5+
minutes and were killed) in a "Runtime Tuning" section — see the tutorial's S2.5 discussion of
`max_p_non_ancestral` as the intended way to trade the algorithm's asymptotic completeness for
tractability.

The `FAISSCMI`-style pattern of implementing something inline continues in spirit: LPCMCI's own
`get_corrected_pvalues` is broken in tigramite 5.2.10.1 (`AttributeError: 'LPCMCI' object has no
attribute '_check_tau_limits'`), so all three `lpcmci_*.ipynb` notebooks reimplement the
Benjamini-Hochberg correction directly on `p_matrix`.

Each notebook also reruns its plain-PCMCI counterpart on the `climate_rv` feature set and checks,
for every parent PCMCI found for `weekly_rv`, whether LPCMCI confirms it as causal or downgrades it
to confounded (`<->`) — the question causal-sufficiency-assuming methods cannot ask of themselves.

---

## Method Comparison

| | ParCorr | GPDCtorch | FAISS-CMI | LPCMCI (x3 CI tests) |
|-|---------|-----------|-----------|--------|
| Detects | Linear conditional deps | Nonlinear additive | Any dependency | Same as paired CI test, confounder-aware |
| Speed | Fast (analytic) | Slow (~1–2 s/fit on CPU) | Medium (kNN + shuffle) | Fast-to-slow depending on paired CI test (see LPCMCI table above) |
| `tau_max` used | 6 / 16 | 3 | 3 | 3 (ParCorr) / 1 (GPDCtorch, FAISS-CMI) |
| Significance | Analytic t-test | Analytic DC null | Shuffle test only | Same as paired CI test |
| Links GPDCtorch finds, ParCorr misses | — | Nonlinear additive | — | — |
| Key constraint | Misses nonlinear | `max_combinations=1` needed | Shuffle cost; T must be >> k | Assumes causal sufficiency is **false**; `max_p_non_ancestral` needed for tractability, more so as the paired CI test gets more expensive |

---

## Environment Setup

```bash
# CPU / development (default)
conda env create -f environment.yml   # creates env 'gaia'
conda activate gaia

# CUDA GPU (Linux + NVIDIA)
conda env create -f environment.cuda.yml   # creates env 'gaia-cuda'
```

Key dependencies: `tigramite`, `faiss-cpu`, `torch`, `gpytorch`, `dcor`, `statsmodels`.
