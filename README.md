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
| `notebooks/causal_check_soi_rv_lanina.ipynb` | Diagnostic notebook for corn only. Isolates the causal triangle between `SOI_index`, `weekly_rv`, and `ssta_lanina`. Runs PCMCI with and without the ENSO confounder to verify the `weekly_rv → SOI` link is spurious. |

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

---

## Method Comparison

| | ParCorr | GPDCtorch | FAISS-CMI |
|-|---------|-----------|-----------|
| Detects | Linear conditional deps | Nonlinear additive | Any dependency |
| Speed | Fast (analytic) | Slow (~1–2 s/fit on CPU) | Medium (kNN + shuffle) |
| `tau_max` used | 6 / 16 | 3 | 3 |
| Significance | Analytic t-test | Analytic DC null | Shuffle test only |
| Links GPDCtorch finds, ParCorr misses | — | Nonlinear additive | — |
| Key constraint | Misses nonlinear | `max_combinations=1` needed | Shuffle cost; T must be >> k |

---

## Notable Results

### SOI_index — Confounder Artifact

The `weekly_rv(-1) → SOI_index` link (val ≈ 0.21) that appears in ParCorr and GPDCtorch runs is **spurious**. The true structure is a fork:

```
ssta_lanina(t-k) ──→ SOI_index(t)     [atmospheric lag 1-3 months]
ssta_lanina(t-k) ──→ weekly_rv(t)     [market ENSO-forecast channel]
```

Without `ssta_lanina` in the conditioning set, PCMCI routes the shared SST signal as `rv(t-1) → SOI(t)`. `causal_check_soi_rv_lanina.ipynb` runs this experiment explicitly: the link vanishes in Run B once `ssta_lanina` is added.

### Crop-Specific ENSO Causality

After adding `ssta_lanina`, ENSO causal links appear for corn but not wheat:

- **Corn** — Midwest Corn Belt; July–August pollination window aligns with La Niña drought peak; consistent, repeatable signal across years.
- **Wheat** — Mixed regional effects (La Niña helps Northern Plains, hurts Southern Plains); March–May critical window coincides with La Niña's decaying phase; global supply competition (Russia, Ukraine, Australia) overwhelms the US ENSO signal.

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
