# Data and Preprocessing

Shared reference for every method in this project — [PCMCI](pcmci.md),
[LPCMCI](lpcmci.md) and [SDCI](sdci.md) all consume the same panels through the same
four-step pipeline.

---

## Datasets

| Path | Frequency | Rows | Date Range |
|------|-----------|------|------------|
| `data/monthly/{corn,soybean,wheat}.csv` | Monthly | 192 | 2009-04 → 2025-03 |
| `data/weekly/{corn,soybean,wheat}.csv` | Weekly | 832 | 2009-04 → 2025-03 |

All three crops share the same 22 columns. Managed via DVC: `dvc pull`.

## Variable groups

| Group | Variables | N |
|-------|-----------|---|
| `rv` | `weekly_rv` | 1 |
| `climate` | `prcp`, `awnd`, `tmin`, `tmax`, `co2`, `pdsi`, `EVAP`, `soil_moisture_mean_m3m3`, `cloud_cover_mean_pct`, `humidity_mean_pct`, `ssta_elino`, `ssta_lanina`, `SOI_index`, `NAO_index` | 14 |
| `macro` | `DJIA_Index`, `WTI_Index`, `Broad_Dollar_index`, `Stock_Uncertainty` | 4 |
| `news` | `frbsf_sentiment`, `epu_index`, `Text_Climate_Anomaly` | 3 |

## Feature sets

| Feature Set | Groups | N vars |
|-------------|--------|--------|
| `climate` | climate | 14 |
| `climate_rv` | climate + rv | 15 |
| `macro_climate_rv` | macro + climate + rv | 19 |
| `macro_news_climate_rv` | macro + news + climate + rv | 22 |

---

## Which columns are crop-specific

Only 6 of the 22 columns actually differ between `corn.csv`, `soybean.csv` and `wheat.csv`:

| | Columns |
|---|---|
| **Crop-specific (6)** | `weekly_rv`, `prcp`, `awnd`, `tmin`, `tmax`, `pdsi` |
| **Identical across all three (16)** | `co2`, `frbsf_sentiment`, `epu_index`, `DJIA_Index`, `WTI_Index`, `Broad_Dollar_index`, `Stock_Uncertainty`, `Text_Climate_Anomaly`, `EVAP`, `soil_moisture_mean_m3m3`, `cloud_cover_mean_pct`, `humidity_mean_pct`, `ssta_elino`, `ssta_lanina`, `SOI_index`, `NAO_index` |

The split is physically sensible: local weather-station series (`prcp`, `awnd`, `tmin`,
`tmax`, `pdsi`) come from each crop's own growing region, while gridded, global and
financial series do not.

> **⚠ Known issue.** The `climate` group contains **five crop-specific columns**, so it is
> **not** crop-agnostic. The `pcmci_*` notebooks nevertheless short-circuit it in their run
> loop with `# climate is crop-agnostic — run once, reuse`, reusing corn's fit for soybean
> and wheat — while their own markdown correctly states that these variables are
> region-specific. Any `climate`-only result reported for soybean or wheat in those
> notebooks is actually corn's, and cross-crop comparison of that feature set is therefore
> not meaningful there. `notebooks/sdci.ipynb` fits `climate` per crop and includes a cell
> that verifies the above table directly against the CSVs.

---

## Preprocessing pipeline

Established in `notebooks/eda_preprocessing.ipynb` and applied identically by every method.

| Step | Operation | Why |
|------|-----------|-----|
| **Impute** | Linear interpolation for `pdsi`, `Broad_Dollar_index`; forward-fill for `DJIA_Index` | Eliminates NaNs before array operations |
| **Detrend** | Gaussian kernel smooth, residuals only; `smooth_width` = 15 years (180 months / 780 weeks) | Removes secular drift (CO₂ rise, DJIA trend, dollar cycle) that would otherwise inflate long-range spurious links |
| **Anomalise** | Subtract phase mean, divide by phase std (`cycle_length` = 12 monthly / 52 weekly) | Removes annual seasonality so methods see residuals, not seasonal patterns |
| **Gaussianise** | Rank-based probit transform | Validates ParCorr's analytic t-test; improves GP covariance conditioning for GPDCtorch; puts all columns on a common scale for SDCI's single shared noise variance. Does not alter rank-order dependence |

The PCMCI and LPCMCI notebooks call `tigramite.data_processing.smooth` and
`.trafo2normal` directly. `notebooks/sdci.ipynb` reimplements both from their definitions so
it can run without `tigramite`; the reimplementations were verified against tigramite on
`data/monthly/corn.csv` and agree to floating-point roundoff.

### Detrend definition

tigramite's `smooth_width` is **twice sigma**, so the weight between times $a$ and $b$ is

$$w_{ab} = \exp\!\left[-\frac{(a-b)^2}{\big(2 \cdot \texttt{smooth\_width}/2\big)^2}\right]$$

and the pipeline keeps `data - (w @ data) / w.sum(axis=1)`.

### Gaussianise definition

Map each column to its empirical CDF via `np.interp` against the sorted values on the grid
`linspace(1/T, 1, T)`, clip the endpoints to `thres = 0.001`, then apply $\Phi^{-1}$.
