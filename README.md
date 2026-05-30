# Gaia

Applies causal inference to U.S. crop return volatility (corn, soybean, wheat), testing causal dependencies between realized volatility and macroeconomic indicators, news sentiment, and climate variables using multivariate time-series methods (PCMCI, CMI, Gaussian Processes, partial correlation).

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/eda_preprocessing.ipynb` | Exploratory data analysis and full preprocessing pipeline (detrend → anomalize → Gaussianise). Includes distribution analysis, seasonality checks, tau_max selection via bivCI, scatter/density plots for CI test selection, and a sanity-check PCMCI + ParCorr run. |
| `notebooks/pcmci_parcorr.ipynb` | Runs PCMCI + ParCorr on all three crops (corn, soybean, wheat) at monthly frequency across three feature sets: `climate`, `climate_rv`, and `macro_news_climate_rv`. Includes cross-crop comparison of significant causal links into realized volatility. |

## Feature Sets

| Feature Set | Groups | N vars |
|-------------|--------|--------|
| `climate` | climate | 10 |
| `climate_rv` | climate + rv | 11 |
| `macro_news_climate_rv` | macro + news + climate + rv | 18 |

## Preprocessing Pipeline

All PCMCI runs use the same pipeline (matching `eda_preprocessing.ipynb`):

1. **Impute** — linear interpolation for `pdsi` / `Broad_Dollar_index`; forward-fill for `DJIA_Index`
2. **Detrend** — `pp.smooth(smooth_width=15yr, kernel='gaussian', residuals=True)` to remove secular drift in CO₂, DJIA, Broad Dollar
3. **Anomalize** — subtract seasonal phase mean and divide by phase std to remove annual cycles
4. **Gaussianise** — `pp.trafo2normal` (rank-based) so ParCorr's analytic t-test null is valid

## PCMCI + ParCorr Config

| Parameter | Value | Reason |
|-----------|-------|--------|
| `tau_min` | 1 | No contemporaneous links |
| `tau_max` | 6 (monthly) / 16 (weekly) | From RV auto-bivCI analysis in EDA notebook |
| `pc_alpha` | 0.2 | Liberal skeleton pruning preserves MCI step power |
| `alpha_level` | 0.05 | Final significance threshold |
| `fdr_method` | `fdr_bh` | Benjamini-Hochberg FDR correction |
| `max_conds_dim` | `None` | PC step controls conditioning-set size |
| `significance` | `analytic` | Valid after `trafo2normal`; no permutation cost |

## Environment Setup

```bash
conda env create -f environment.yml   # creates env 'gaia'
conda activate gaia
```

## Running PCMCI Scripts

```bash
# Run all configs
python scripts/pcmci.py

# Filter by method or frequency
python scripts/pcmci.py --method parcorr --frequency monthly
```

## Data

- `data/{weekly,monthly}/{corn,soybean,wheat}.csv` — processed datasets
- Managed via DVC: `dvc pull`
