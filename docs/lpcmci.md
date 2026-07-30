# LPCMCI — Latent-Confounder-Aware Causal Discovery

Gerhardus & Runge (2020). Drops PCMCI's **causal sufficiency** assumption, i.e. it no longer
assumes there are no unobserved confounders.

Instead of a DAG it learns a time-series **DPAG** with typed edges:

| Edge | Meaning |
|------|---------|
| `-->` | candidate cause — directed, not reverse-causal |
| `<->` | confounded by an unobserved common driver — **not** causal |
| `o->` / `o-o` | orientation undetermined |

The `<->` verdict is the whole point: it is the question a causal-sufficiency-assuming
method cannot ask of itself.

Implemented as three sibling notebooks, one per CI test, mirroring the
[PCMCI](pcmci.md) notebooks so that only the algorithm changes within each pair:

| Notebook | Pairs with |
|----------|------------|
| `notebooks/lpcmci_parcorr.ipynb` | `pcmci_parcorr.ipynb` |
| `notebooks/lpcmci_gpdc.ipynb` | `pcmci_gpdc.ipynb` |
| `notebooks/lpcmci_cmi.ipynb` | `pcmci_cmi.ipynb` |

Data, variable groups, feature sets and preprocessing:
[`data-and-preprocessing.md`](data-and-preprocessing.md).

---

## Runtime tuning

LPCMCI's non-ancestral search phase issues far more CI tests than PCMCI's single greedy PC
step, so every pairing needs tighter bounds than its PCMCI counterpart — and the more
expensive the underlying CI test per call (ParCorr's analytic t-test < FAISS-CMI's shuffle
test < GPDCtorch's GP fit), the more aggressive those bounds must be.

| Parameter | ParCorr | GPDCtorch | FAISS-CMI |
|-----------|---------|-----------|-----------|
| `tau_max` (LPCMCI's own runs) | 3 | 1 | 1 |
| `pc_alpha` | 0.01 | 0.01 | 0.01 |
| `n_preliminary_iterations` | 1 | 0 | 0 |
| `max_p_global` / `max_p_non_ancestral` / `max_q_global` | 2 / 1 / 2 | 1 / 0 / 1 | 1 / 0 / 1 |
| CI-test-specific | `significance='analytic'` | `sig_samples=200` (analytic null cache) | `k=0.1`, `sig_samples=50` (vs. 200 in `pcmci_cmi.ipynb`) |
| Measured: N=22, tightest bounds | 25 s | 120 s | 37 s |

Each notebook documents its own empirical timings — including configurations that did not
converge in 3–5+ minutes and were killed — in a "Runtime Tuning" section. See the tutorial's
S2.5 discussion of `max_p_non_ancestral` as the intended way to trade the algorithm's
asymptotic completeness for tractability.

---

## Known upstream bug

LPCMCI's own `get_corrected_pvalues` is broken in tigramite 5.2.10.1:

```
AttributeError: 'LPCMCI' object has no attribute '_check_tau_limits'
```

All three `lpcmci_*.ipynb` notebooks therefore reimplement the Benjamini-Hochberg correction
directly on `p_matrix` — continuing in spirit the inline-implementation pattern established
by `FAISSCMI` in `pcmci_cmi.ipynb`.

---

## Cross-check against PCMCI

Each notebook reruns its plain-PCMCI counterpart on the `climate_rv` feature set and checks,
for every parent PCMCI found for `weekly_rv`, whether LPCMCI **confirms** it as causal
(`-->`) or **downgrades** it to confounded (`<->`).

---

## Comparison

| | PCMCI (×3 CI tests) | LPCMCI (×3 CI tests) |
|-|---------------------|----------------------|
| Causal sufficiency | assumed **true** | assumed **false** |
| Output | DAG, one edge type | DPAG, four edge types |
| Detects | as per paired CI test | same, plus confounding |
| Speed | fast → slow by CI test | same ordering, but far slower |
| Key constraint | cannot detect confounding | `max_p_non_ancestral` needed for tractability, more so as the paired CI test gets more expensive |

For a neural, regime-switching alternative that also assumes causal sufficiency, see
[`sdci.md`](sdci.md).
