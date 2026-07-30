# SDCI — Causal Discovery from Conditionally Stationary Time Series

Complete description of the model architecture and the procedure used in
`notebooks/sdci.ipynb`. No results here — see the notebook's own §14 for those.

> Balsells-Rodas, Sumba, Narendra, Tu, Schweikert, Kjellström, Li.
> *Causal Discovery from Conditionally Stationary Time Series.* **ICML 2025**.
> Reference implementation: [`charlio23/SDCI`](https://github.com/charlio23/SDCI) @ `774f78e`.
> Upstream files derive from [NRI](https://github.com/ethanfetaya/NRI) (MIT) and
> [AmortizedCausalDiscovery](https://github.com/loeweX/AmortizedCausalDiscovery) (MIT).

---

## 1. Why this method, next to PCMCI

`pcmci_*` and `lpcmci_*` estimate a **static** causal graph by conditional-independence
testing: one lagged adjacency for the whole sample, with p-values and FDR control.

SDCI makes a different assumption — **conditional stationarity**. At each time step every
variable $i$ occupies a latent discrete state $s_t^i \in \{1,\dots,K\}$, and the causal graph
in force depends on the state of the **sending** variable. Instead of one graph you get $K$
graphs plus a posterior over which one applies when.

| | PCMCI / LPCMCI | SDCI |
|---|---|---|
| Graph | static | one per latent state |
| Lags | `tau_min…tau_max` | **one-step (τ=1) only**, as configured here |
| Estimation | CI tests | amortised variational inference (ELBO) |
| Output per link | p-value, FDR-corrected | posterior probability $p(\text{edge})$ — a belief, **not** a test statistic |
| Regime timing | — | $q(s_t^i)$, a posterior over time |
| Latent confounders | assumed absent (PCMCI) / handled (LPCMCI) | assumed absent |

There is no null distribution and no FDR step in SDCI. Nothing it outputs is comparable to
an `alpha_level=0.05`.

---

## 2. Generative model

Let $x_t \in \mathbb{R}^{N \times D}$ be $N$ variables of dimension $D$ at time $t$
($D = 1$ throughout this project). The model is:

**Graph prior.** For each ordered pair $(j \to i)$, $j \neq i$, and each state
$k \in \{1,\dots,K\}$, an edge type $z^{(k)}_{ji} \in \{1,\dots,E\}$ is drawn from a fixed
categorical prior. With $E = 2$ and `skip_first=True`, type 1 is the explicit *"no edge"*
class and type 2 is *"edge present"*:

$$p(z^{(k)}_{ji}) = \mathrm{Cat}\big(\texttt{sparsity},\; 1-\texttt{sparsity}\big)$$

The collection $\mathcal{W}^{(k)} = \{z^{(k)}_{ji}\}$ is the graph for state $k$.

**State prior.** Either uniform (the `region` state encoder) or a learned transition
$p(s_t^i \mid x_{t-1}^i, x_t^i, s_{t-1}^i)$ (the `recurrent` encoder).

**Transition likelihood.** Gaussian with fixed scalar variance $\sigma^2$, mean given by an
interaction network that predicts an **increment**:

$$p(x_{t+1} \mid x_t, s_t, \mathcal{W}) = \mathcal{N}\big(x_t + f_\theta(x_t, s_t, \mathcal{W}),\; \sigma^2 I\big)$$

The residual form matters: any structure already explained by $x_t$ itself (trend,
persistence) needs **no edges at all**, so it must be removed in preprocessing or the
sparsity prior faces no competition.

### Variational posteriors

$$q(\mathcal{W}^{(k)}) \;=\; \mathrm{Cat}\big(\mathrm{softmax}(U_{\cdot,k,\cdot})\big), \qquad q(s_t^i \mid \cdot) \;=\; \mathrm{Cat}\big(\mathrm{softmax}(\text{StateEncoder}(\cdot))\big)$$

$q(\mathcal{W})$ is **not amortised** — $U$ is a free parameter tensor, one categorical per
ordered pair × state. This is the right choice when the dataset is many views of *one*
system rather than many independent systems.

### ELBO

$$\mathcal{L} = \underbrace{\mathbb{E}_q\big[\log p(x_{2:T} \mid x_{1:T-1}, s, \mathcal{W})\big]}_{\text{reconstruction}} - \underbrace{\sum_k \mathrm{KL}\big(q(\mathcal{W}^{(k)}) \,\|\, p(\mathcal{W}^{(k)})\big)}_{\text{sparsity}} - \underbrace{\mathrm{KL}\big(q(s) \,\|\, p(s)\big)}_{\text{state}}$$

Training minimises $-\mathcal{L}$, written in code as `loss = nll + kl`.

---

## 3. Edge indexing — `rel_rec` / `rel_send`

Every module speaks in **edge-indexed** tensors. `create_rel_rec_send(N)` builds two
one-hot selector matrices of shape `[R, N]` where `R = N(N-1)`:

```python
off_diag = np.ones([N, N]) - np.eye(N)
rel_rec  = onehot(np.where(off_diag)[0])   # rows -> RECEIVERS
rel_send = onehot(np.where(off_diag)[1])   # cols -> SENDERS
```

Edge index $e$ enumerates the off-diagonal entries **row-major**. `rel_rec[e]` selects the
receiver (the row = *effect*), `rel_send[e]` the sender (the column = *cause*).

Two operations are used everywhere:

| Operation | Code | Shape |
|---|---|---|
| node → edge | `torch.matmul(rel_rec, x)` / `torch.matmul(rel_send, x)` | `[…, N, F] → […, R, F]` |
| edge → node (sum into receiver) | `msgs.transpose(-2,-1).matmul(rel_rec).transpose(-2,-1)` | `[…, R, F] → […, N, F]` |

**Unpacking the result.** `edge_probs_to_adjacency` inverts this to a `[K, cause, effect]`
array:

```python
recv, send = np.where(np.ones((N, N)) - np.eye(N))
adj[:, send, recv] = edge_prob[:, :, 1].T      # last edge type = "edge present"
```

So `adj[k, j, i]` = posterior probability that `cols[j] → cols[i]` is active **while the
sender `cols[j]` is in state `k`**. Getting this transpose wrong silently reverses every
link, which is why it is isolated in one function.

---

## 4. Architecture

Five modules. All are reproduced verbatim in §1 of the notebook.

### 4.1 `MLP` — the shared building block

`modules_ACD.MLP(n_in, n_hid, n_out, do_prob=0, use_batch_norm=True, final_linear=False)`

Two ELU layers with dropout between, optional third linear layer, optional BatchNorm1d over
the flattened first two axes. Xavier-normal init, biases at `0.1`.

One upstream quirk preserved: the `activation` argument is effectively dead — the `if`
chain assigns `F.elu` whenever `activation != 'relu'`, and `forward` hardcodes `F.elu`
regardless. Documented here so it isn't mistaken for a porting bug.

### 4.2 `EncoderFixed` — the graph posterior $q(\mathcal{W})$

```python
self.U = nn.Parameter(torch.randn(N*(N-1), K, E))
def forward(self):
    return self.U - self.U.logsumexp(-1, keepdim=True)      # log-normalised
```

That is the entire causal-discovery parameterisation: `[R, K, E]` log-probabilities, no
inputs, no network. `forward()` returns **log** probabilities; `my_softmax(·, -1)` recovers
probabilities.

Upstream also ships `MLPEncoder` and `NRIMPMEncoders.py` (`mlp` / `rnn` / `att`) which emit a
*different graph per sample*. Those are for datasets of many independent simulations and are
**not** used here — one 192-month panel is one system.

### 4.3 State encoders — $q(s_t^i)$

**`StateEncoderRegion(n_in, n_hid, K)`** — the default.

```python
mlp1 = MLP(n_in, n_hid, n_hid, use_batch_norm=False, final_linear=False)
out  = nn.Linear(n_hid, K)
forward(x): [B, N, T, n_in] -> [B, N, T, K]      # logits
```

The state is read off **each node's own current value** through a shared MLP. Consequences
worth internalising: the state is a learned *band of the value range* ("this variable is
currently high / low"), it carries no temporal memory, and because it is shared across nodes
while every column is Gaussianised to the same marginal, the *occupancy rate* of each state
is near-identical across nodes by construction. What differs meaningfully between nodes is
the **timing**.

**`StateGRNNEncoderSmall(n_in, n_hid, K, rnn='gru', factor=False)`** — the alternative.

`mlp1` per node → `node2edge` → `mlp2` → (`factor` branch) → reshape to per-edge sequences →
3-layer **bidirectional** GRU → 1-layer forward GRU → `edge2node` → `fc_out`. Signature is
`forward(inputs, rel_rec, rel_send)`.

Richer (history + neighbours) and higher variance. Two practical notes: it is bidirectional,
so $q(s_t)$ sees the whole window, not just the past; and its internal `MLP`s use the default
`use_batch_norm=True`, unlike `StateEncoderRegion`.

Selecting it via `STATE_ENCODER = 'recurrent'` also **activates the state KL term**, which is
identically zero under `region`.

### 4.4 `StateDecoder` — the state prior $p(s_t)$

```python
region:    MLP(n_in,          n_hid, K, use_batch_norm=False, final_linear=True)
recurrent: MLP(2*n_in + K,    n_hid, K, use_batch_norm=False, final_linear=True)
forward(x): softmax(prior(x), -1)      # probabilities, not logits
```

Under `recurrent` its input is `cat([x_{t-1}, x_t, s_{t-1}])`, making it a learned transition
prior. Under `region` it is **instantiated but never used** — upstream sets the state KL to
zero in that branch, so the module receives no gradient. The notebook keeps it constructed
for fidelity with `train_GRN.py`.

### 4.5 `MLPDecoder` — the state-gated transition model

This is where conditional stationarity actually happens.

```
MLPDecoder(n_in_node, num_states, edge_types, msg_hid, msg_out, n_hid,
           hidden_states=False, do_prob=0.0, skip_first=True,
           ball_cond=False, embedding=False, num_atoms=None)
```

Constructed here as `MLPDecoder(1, K, E, H, H, H, True, embedding=False, num_atoms=N)` —
note the 7th positional argument sets `hidden_states=True`, which selects the soft
state-mixing path. (`ball_cond` is springs-simulation-specific and unused.)

**`single_step_forward`**, with `inputs [B, T, N, D]`, `states [B, T, N, K]`,
`rel_type [B, T, R, K, E]`:

1. **Node → edge.**
   `receivers = matmul(rel_rec, inputs)`, `senders = matmul(rel_send, inputs)` → `[B,T,R,D]`
   `senders_states = matmul(rel_send, states)` → `[B,T,R,K]` — the state of each edge's **sender**
   `pre_msg = cat([senders, receivers], -1)` → `[B,T,R,2D]`

2. **Per-edge-type messages.** For each edge type $i$ (starting at `1` because
   `skip_first=True` — type 0 is "no edge" and its MLP is never evaluated):

   ```python
   msg = leaky_relu(msg_fc2[i](dropout(leaky_relu(msg_fc1[i](pre_msg)))))
   ```

3. **State gating — the core mechanism.**

   ```python
   rel_types = rel_type[:, :, :, :, i]                        # [B,T,R,K]
   rel_types = torch.sum(rel_types * senders_states, dim=-1)  # [B,T,R,1]
   msg = msg * rel_types
   all_msgs += msg
   ```

   Each message is scaled by that edge's probability **under the state its sender currently
   occupies**. An edge on in state 0 and off in state 1 transmits only while its sender sits
   in state 0. Because `senders_states` is a straight-through one-hot, this is a hard
   selection in the forward pass with a soft gradient.

4. **Edge → node.** Sum incoming messages per receiver via `rel_rec`.

5. **Output.** `aug = cat([sender_features, agg_msgs], -1)` through
   `out_fc1 → out_fc2 → out_fc3` (leaky-ReLU, dropout), then

   ```python
   next_pos = single_timestep_inputs + pred        # predicts the INCREMENT
   ```

**`forward(inputs, state, rel_type, rel_rec, rel_send, pred_steps)`** wraps this with teacher
forcing. It re-anchors on ground truth every `pred_steps` steps by starting from
`inputs[:, 0::pred_steps]`, rolls `pred_steps` predictions, then reassembles the timeline
with `output[:, i::pred_steps] = preds[i]` and truncates to `T-1`. With `pred_steps=1` this
degenerates to a pure one-step model — every prediction is anchored on observed data.

Upstream contains a fallback for when `states.size(1) != last_pred.size(1)` (it retries with
`step-1::pred_steps`), which triggers when `pred_steps` does not divide `T`. Choosing
`pred_steps=1` avoids that path entirely.

### 4.6 Discrete sampling

Edges and states use **different** samplers, deliberately:

| | Sampler | Temperature | Forward value | Gradient |
|---|---|---|---|---|
| Edges | `gumbel_softmax(logits, tau=EDGE_TAU)` | fixed `0.5` | **soft** simplex point | through the relaxation |
| States | `simple(logits, tau=temperature_state)` | annealed `5.0 → 0.5` | **hard** one-hot | through `softmax(logits)` |

`simple` is upstream's reduced-variance straight-through variant: it takes the *argmax of a
Gumbel sample* for the forward value but routes gradients through the plain softmax rather
than the perturbed one.

```python
def simple(logits, tau=1, num_samples=1):
    y = torch.softmax(logits, dim=-1)
    _, k = gumbel_softmax_sample(logits, tau=tau).data.max(-1)
    y_hard = zeros.scatter_(-1, k, 1.0)
    return (y_hard - y.data) + y
```

Note `sample_gumbel` allocates on CPU and moves to device only `if logits.is_cuda`.

---

## 5. Loss terms and their scaling

```python
nll = nll_gaussian(out, target, nll_variance)
kl  = state_kl                                       # 0 under 'region'
for k in range(K):
    kl += kl_mult * kl_categorical(edge_prob[:, :, k, :], log_prior, num_atoms=1)
loss = nll + kl
```

The two normalisations are **not** commensurate, and this drives the most important
configuration decision in the whole notebook:

| Term | Definition | Divides by |
|---|---|---|
| `nll_gaussian(preds, target, var)` | `((preds-target)**2 / (2*var)).sum()` | `target.size(0) * target.size(1)` = **B · N** |
| `kl_categorical(preds, log_prior, num_atoms=1)` | `(preds*(log preds − log_prior)).sum()` | `num_atoms * preds.size(0)` = **B** |

So the NLL is a **per-node** quantity summed over time, while the edge KL is summed over all
$N(N-1)$ pairs and divided only by the batch. The KL is therefore effectively $\mathcal{O}(N)$
up-weighted relative to a per-node ELBO — and the ratio between the two, governed by
`nll_variance` and `kl_mult`, *is* the effective sparsity pressure. §7 of the notebook selects
`nll_variance` on exactly this basis rather than inheriting the upstream value.

---

## 6. Adapting one panel to SDCI's data format

Upstream loaders yield `[B, N, T, D]` — a batch of **independent simulations**. This project
has **one** system observed for 192 months, so windows are slid over the series:

```
[T=192, N] --window(seq_len=24, stride=1)--> [W=169, N, 24, 1]
```

`num_dims = 1` (each node carries a scalar). `EncoderFixed` learns a single graph set across
all windows, which is correct — the windows are views of one system.

**Validation split is chronological, never shuffled.** Overlapping windows leak: windows
starting at $t$ and $t+1$ share 23 of 24 months, so a random split puts near-duplicates on
both sides. `time_split` instead takes the last `VAL_FRAC` of the *timeline* as validation
and **drops every training window that would overlap the validation period**:

```python
first_val_start = starts[-n_val]
train = win[starts + seq_len <= first_val_start]
val   = win[starts >= first_val_start]
```

---

## 7. Preprocessing

Identical to the pipeline in [`data-and-preprocessing.md`](data-and-preprocessing.md) —
impute → 15-year Gaussian detrend → phase-wise anomalise → Gaussianise — but implemented
**without `tigramite`**: `gaussian_detrend` and `trafo2normal` are reimplemented from their
definitions in §2 of the notebook and verified against tigramite to floating-point roundoff.

Why each step matters *specifically for SDCI*:

| Step | Reason |
|---|---|
| Impute | the Gaussian NLL and both state encoders have no NaN path |
| **Detrend** | the decoder's residual form `x_{t+1} = x_t + f(·)` would otherwise absorb secular drift (CO₂, DJIA) with **no edges**, so the sparsity prior wins uncontested |
| Anomalise | otherwise the 12-month cycle is the easiest thing to predict, and shared seasonality appears as dense spurious edges |
| Gaussianise | `nll_variance` is a **single scalar shared by all N nodes**; columns must be on a common scale or heavy-tailed ones dominate the loss |

---

## 8. Training procedure

Per epoch, over shuffled minibatches of windows `var [B, N, T, 1]`, with
`target = var[:, :, 1:, :]`:

1. **State posterior**
   `state_logits = state_encoder(var)` (or `(var, rel_rec, rel_send)` for `recurrent`)
   `state_samp = simple(state_logits, tau=temperature_state)` — straight-through one-hot
   `state_prob = my_softmax(state_logits, -1)`
2. **State KL** — `0` under `region`; under `recurrent`,
   `kl_categorical(state_prob[:, :, 1:], log(state_decoder(cat([x_{t-1}, x_t, s_{t-1}])) + 1e-16), num_atoms=1)`
3. **Edge posterior**
   `edge_logits = encoder().unsqueeze(0).repeat(B, 1, 1, 1)`
   `edges = gumbel_softmax(edge_logits, tau=EDGE_TAU)`; `edge_prob = my_softmax(edge_logits, -1)`
4. **Transition** — `out = decoder(var, state_samp, edges, rel_rec, rel_send, pred_steps)`
5. **Loss** — `nll_gaussian(out, target, nll_variance)` plus the per-state edge KL against
   `log([sparsity, 1-sparsity])`
6. **Backward** — `clip_grad_norm_(·, 5.0)` on `state_encoder`, `encoder`, `decoder`.
   Upstream does **not** clip `state_decoder`; the notebook reproduces that faithfully.
7. **`optimizer.step()`**

Then per epoch: `scheduler.step()` (StepLR, `gamma=0.5`), temperature anneal, and a
validation pass that uses **hard** edge samples (`gumbel_softmax(..., hard=True)`), matching
`train_GRN.py::calculate_metrics`.

**Optimiser.** Adam with two learning rates — `lr_encoder` for `EncoderFixed`, `lr_decoder`
for everything else. The graph parameters get the larger rate (`5e-3` vs `5e-4`) because `U`
is a bare parameter tensor with no upstream network to amplify its gradient.

**Readout.** After training, on the **full series** as a single window `[1, N, T, 1]`:

```python
edge_posterior  = my_softmax(encoder(), -1)          # [R, K, E]
adj             = edge_probs_to_adjacency(...)       # [K, cause, effect]
state_prob_full = my_softmax(state_logits_of(full), -1)[0]   # [N, T, K]
```

### Multi-restart ensembling

192 months against up to 462 ordered pairs means a single fit is not reproducible — different
seeds land on different local optima of a flat ELBO. Every reported number is therefore the
**mean `p(edge)` over `N_RESTARTS` seeds**, accompanied by the **cross-seed standard
deviation** and the count of seeds that individually cross threshold.

`p_std` is the column to read first. `p_mean = 0.8, p_std = 0.35` is one seed shouting;
`p_mean = 0.6, p_std = 0.05` is three seeds agreeing.

---

## 9. Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `NUM_STATES` (K) | 2 | simplest non-trivial regime split; §7b ablates 1 / 2 / 3, where `K=1` is the stationary-graph baseline |
| `NUM_EDGE_TYPES` (E) | 2 | with `skip_first=True`, type 1 = "no edge", type 2 = "edge" |
| `HIDDEN_DIM` | 64 | matches upstream's GRN scale (`--hidden-dim 32`–`64`) |
| `SEQ_LEN` | 24 | two years of monthly context per window |
| `STRIDE` | 1 | maximally overlapping windows — the split, not the stride, prevents leakage |
| `VAL_FRAC` | 0.2 | chronological hold-out |
| `STATE_ENCODER` | `region` | upstream's recommended baseline; `recurrent` is the richer alternative |
| `EPOCHS` / `BATCH_SIZE` | 120 / 32 | |
| `LR_ENCODER` / `LR_DECODER` | 5e-3 / 5e-4 | from `scripts/run_GRN_region_embedding.sh` |
| `STEP_SIZE` / `GAMMA` | 60 / 0.5 | StepLR |
| `SPARSITY` | 0.90 | prior belief that 90% of ordered pairs are not linked |
| `KL_MULT` | 1.0 | standard ELBO weighting |
| `EDGE_TAU` | 0.5 | fixed Gumbel temperature for edges |
| temperature | 5.0 → 0.5, `γ=0.75` | state Gumbel annealing |
| `N_RESTARTS` / `SEEDS` | 3 / `[0,1,2]` | |
| `EDGE_THRESHOLD` | 0.5 | **display convention only** |
| `DEVICE` | `cpu` | see §11 |

### Two deliberate deviations from upstream

**`NLL_VARIANCE = 5e-3`** (upstream: `5e-5`). Upstream's value is calibrated for
gene-expression and spring trajectories where one-step error is tiny. On Gaussianised
monthly anomalies it inflates the NLL by roughly four orders of magnitude relative to the
sparsity KL (see §5), the prior becomes numerically irrelevant, and the graph saturates.
Selected in §7 by held-out MSE **subject to the graph being non-degenerate** — neither
saturated nor empty. Because the density constraint does part of the work, absolute
`p(edge)` values are calibrated only up to this choice; **rank links rather than thresholding
them**.

**`PRED_STEPS = 1`** (upstream: `10`). A 10-step rollout regularises deterministic physics;
on monthly financial anomalies it mostly asks the model to predict noise. Selected in §7 on
held-out MSE. The consequence is structural and must be carried into every interpretation:
**the learned graph is a τ=1 (one-month) structure**, with no way to express the lag-2/lag-3
links `pcmci_*` reports at `tau_max=3`.

**Temperature annealing** is also rescaled: upstream hardcodes `if epoch > 100 and
epoch % 10 == 0`, tuned for 500–1000-epoch runs. The notebook anneals after
`ANNEAL_AFTER × epochs` so the schedule shape survives a 120-epoch run.

---

## 10. Outputs

Written to `benchmark/<frequency>/<crop>/sdci/<feature_set>/`, mirroring the PCMCI layout:

| File | Contents |
|---|---|
| `summary.json` | full configuration, variable list, held-out MSE, link count, seeds |
| `links.csv` | every link above threshold with `p_edge`, `p_std`, `n_seeds>thr`, group labels |
| `adjacency.npy` | `[K, N, N]` seed-averaged posterior, indexed `[state, cause, effect]` |
| `adjacency_std.npy` | cross-seed std, same shape |
| `state_prob.npy` | `[N, T, K]` state posterior over the full series |
| `history.csv` | per-epoch `nll` / `kl` / `mse` / `val_mse` / `temperature`, all seeds |

Plus `benchmark/<frequency>/sdci_links_all.csv` concatenating every crop × feature set.

---

## 11. Structural limitations

These bound any conclusion drawn from this method **regardless of what the numbers say**.

1. **τ=1 only.** With `pred_steps=1` the decoder models a one-month transition. An effect
   acting at lag 2 or 3 has no representation and will simply be absent. **Absence of a link
   is weak evidence.**
2. **Sample size.** 192 months, ~112 usable training windows, up to 462 ordered pairs, and a
   decoder with tens of thousands of parameters. The sparsity prior and the seed ensemble are
   load-bearing, not decoration.
3. **No null, no convergence test.** Unlike a CI test there is no closed form. The only
   evidence a fit is usable is the held-out MSE curve flattening — check it before reading any
   graph, and watch for it turning back **up**, which indicates overfitting past the optimum.
4. **`p(edge)` is prior-dependent.** Its scale moves with `sparsity` and `nll_variance`.
   It is a belief under a chosen prior, not a p-value.
5. **Shallow state semantics under `region`.** The state is a function of the node's own
   current scalar, so "regime" means "this variable is currently high/low", not a global
   market or climate regime. A variable whose value never crosses the learned boundary is
   **frozen in one state**, and since gating is on the *sender's* state, that variable's
   outgoing edges are then governed by a single graph regardless of `K`.
6. **Directionality is predictive, not mechanistic.** In a one-step model, direction is
   decided by whichever series better predicts the other's increment. Persistence
   asymmetries alone can orient an edge.
7. **Causal sufficiency is assumed.** Like PCMCI and unlike LPCMCI, SDCI has no representation
   for an unobserved common driver.
8. **CPU only, by choice.** Upstream `MLPDecoder.single_step_forward` allocates `all_msgs` on
   the CPU and moves it only `if inputs.is_cuda`, so an MPS tensor raises a device mismatch.
   Rather than edit the vendored file, the notebook runs on CPU — ample at 22 nodes × 192
   months. CUDA works unmodified.

---

## 12. Self-containment and provenance

`notebooks/sdci.ipynb` is **fully self-contained**: §1 reproduces every needed SDCI module
verbatim from `charlio23/SDCI` @ `774f78e`, imports stripped. There is no `third_party/`
directory, no `pip install`, and `torch-scatter` — pinned upstream — is **not** required,
because no file under `model/` imports it. Upstream pins `torch==1.13.1` / `numpy==1.26`;
the code runs unmodified on torch 2.x / numpy 2.x.

`tigramite` is not imported anywhere in the notebook.

**Licence.** Upstream ships no `LICENSE` file. Individual sources carry headers attributing
them to NRI (MIT) and AmortizedCausalDiscovery (MIT). Treat the vendored code as research
code under those terms, cite the ICML 2025 paper, and confirm licensing with the authors
before redistribution.
