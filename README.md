# Gaia

Causal inference on U.S. crop return volatility (corn, soybean, wheat).

The project asks whether realized volatility of crop returns has identifiable causal
dependencies on macroeconomic indicators, news sentiment, and climate variables. It applies
several multivariate time-series causal-discovery methods to the same panels — 22 variables
observed monthly (192 points) and weekly (832 points) from 2009-04 to 2025-03 — so that
their verdicts can be compared against one another rather than taken on trust individually.

Three families of method are used, chosen because they fail in different ways:

- **PCMCI** — conditional-independence testing for a static lagged graph, under the
  assumption that there are no unobserved confounders. Run with three different CI tests
  (linear, nonlinear-additive, model-free) to separate "no dependence" from "no *detectable*
  dependence".
- **LPCMCI** — drops the no-confounders assumption, and can therefore mark a link as
  *confounded* rather than causal — the question PCMCI cannot ask of itself.
- **SDCI** — a neural, regime-switching alternative: instead of one graph it learns several,
  each active while a variable occupies a different latent state.

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/data-and-preprocessing.md`](docs/data-and-preprocessing.md) | Datasets, variable groups, feature sets, which columns are crop-specific, and the shared 4-step preprocessing pipeline |
| [`docs/pcmci.md`](docs/pcmci.md) | PCMCI and its three conditional independence tests — ParCorr, GPDCtorch, FAISS-CMI — with parameters and rationale |
| [`docs/lpcmci.md`](docs/lpcmci.md) | LPCMCI, its DPAG edge semantics, runtime tuning, and the cross-check against PCMCI |
| [`docs/sdci.md`](docs/sdci.md) | SDCI model architecture and procedure in full — generative model, ELBO, every module, training loop, and structural limitations |

---

## Notebooks

| Notebook | Method |
|----------|--------|
| `notebooks/eda_preprocessing.ipynb` | EDA and the preprocessing pipeline; distribution and seasonality checks, `tau_max` selection |
| `notebooks/pcmci_parcorr.ipynb` | PCMCI + ParCorr |
| `notebooks/pcmci_gpdc.ipynb` | PCMCI + GPDCtorch |
| `notebooks/pcmci_cmi.ipynb` | PCMCI + FAISS-CMI |
| `notebooks/lpcmci_parcorr.ipynb` | LPCMCI + ParCorr |
| `notebooks/lpcmci_gpdc.ipynb` | LPCMCI + GPDCtorch |
| `notebooks/lpcmci_cmi.ipynb` | LPCMCI + FAISS-CMI |
| `notebooks/sdci.ipynb` | SDCI (monthly) — self-contained; vendors the reference implementation inline and uses no `tigramite` |

Results are written to `benchmark/<frequency>/<crop>/<method>/<feature_set>/`.

---

## Environment Setup

```bash
# CPU / development (default)
conda env create -f environment.yml        # creates env 'gaia'
conda activate gaia

# CUDA GPU (Linux + NVIDIA)
conda env create -f environment.cuda.yml   # creates env 'gaia-cuda'
```

Key dependencies: `tigramite`, `faiss-cpu`, `torch`, `gpytorch`, `dcor`, `statsmodels`.

Data is managed with DVC — run `dvc pull` to fetch `data/`.

> On macOS, `libomp` is linked by both numpy and torch, so every Python process needs
> `KMP_DUPLICATE_LIB_OK=TRUE` set before the interpreter starts or it aborts on an OpenMP
> double-initialisation error.
