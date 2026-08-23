# PREREG.md

Every threshold, formula, and computed integer in `docs/phase1.md` (v2, END-sentinel notation). Per §8: this file, once complete, is never edited — only appended to via `CALIBRATION_LOG.md` for the bounded §4.4 adjustments.

**Status of this file: incomplete.** Several entries are formulas whose numeric value is not knowable until pilot seeds actually train. Those are marked `PENDING`. Nothing in this file may be treated as frozen until every entry reads `FIXED`.

The rank-timing circularity noted in the previous version of this file (§2.1 pointing to §3.3, which runs after the training §2.1 was needed for) is resolved in the current doc via §3.1.5. No entry is `BLOCKED` any more.

---

## 1. Task generator (§1)

| Quantity | Value | Status |
|---|---|---|
| Answer/query notation | Query = `start_key` only, no depth token. Answer = key `kL` or `ABSTAIN`, discovered by following the chain until `END` is retrieved (§1.1) | FIXED |
| Hard negative distance | Hamming distance 1 from query key | FIXED |
| Easy negative distance | Hamming distance 3 from query key | FIXED |
| Chain length axis | `L ∈ {1, 2, 3}` | FIXED |
| Generator config | `alphabet_size (|S|)`, `chain_length (L)`, `n_distractors`, `n_hard` — vary independently (§1.5) | FIXED (implemented: `generator/task_generator.py::GeneratorConfig`) |

### 1.3 Capacity bounds — computed, from `generator/capacity.py`

```
cache_bits  = n_entries × 6 × log2(|S|)
state_bits  = d_model × d_state × bits_per_element
trajectory_bits = 3 × log2(|S|) + n_trajectory_features × bits_per_element
```

Proposed constants and their computed values:

| Constant | Value |
|---|---|
| `n_entries` | 1024 |
| `|S|` | 64 |
| `d_model` | 64 |
| `d_state` | 8 |
| `bits_per_element` | 16 (fp16) |
| `n_trajectory_features` | 4 (margin, displacement, gate state, integration count) |
| `cache_bits` | 36,864 |
| `state_bits` | 8,192 |
| `trajectory_bits` | 82 |
| **Upper bound ratio** (`cache_bits / state_bits`) | **4.5** — must be `≥ 4` — **PASSES** |
| **Lower bound ratio** (`state_bits / trajectory_bits`) | **~99.9** — must be `≥ 16` — **PASSES** |

Status: **FIXED**, contingent on `d_model=64, d_state=8` actually being the real-seed architecture (§2.1) — if that changes, re-run `generator/capacity.py::check_upper_bound` / `check_lower_bound` and update this table before training. Note this is the **real-seed** rank's `d_model`/`d_state` — §3.1.5's pilots train full-rank and are not subject to this bound in the same way (see below).

---

## 2. Model (§2)

| Quantity | Value | Status |
|---|---|---|
| Query projection rank (real seeds) | Fixed by the pilot measurement in §3.1.5, **before** real seeds train. §3.3 verifies it afterward on the real seeds; it does not produce it. **Value: 36** (§3.1.5, `max(query_PR)` across the 3 stable pilots, rounded to next multiple of 4) | FIXED (procedure and value); **the §3.1.7 throwaway pilot must confirm competence survives this bottleneck before it's used on any of the six real-seed runs** — no pilot to date has ever trained with the rank actually applied |
| Integration write | `state ← state + g_t · W_v(value)` (additive residual) | FIXED |
| Gate inputs | Quantile-transformed (`p_ret`, margin percentile; `p_set`, displacement percentile), CDF fit once on the calibration split, never updated | FIXED (procedure); the fitted CDF itself is `PENDING` (needs the calibration split to exist) |
| Supervision regimes | Regime 1: arms A, D only, never sees a terminal mid-follow. Regime 2: all arms, B1/B2/C → `ABSTAIN` | FIXED |
| ABSTAIN mechanism (§2.4) | Separate binary head — forced, not preferred: `retrieve()` scores against per-item content-addressed `cache_keys`, no fixed answer vocabulary exists to add a token to. Trained on the same state that feeds the step's query, at the step the model would otherwise emit. §2.2's gate quadrant stays measured and untrained throughout — training a signal to compete with it would reintroduce the "learned prediction about its own epistemic state" failure mode the quadrant exists to avoid | **FIXED, implemented and dry-run verified** — `model/task_model.py::forward_train_regime2`, `model/data.py` (generalized `n_steps`/`is_abstain`/`rho` fields, derived mechanically from `item.rho`/`item.answer` for all five arms, no arm-specific branching). 16-20 step dry runs confirmed: per-batch `abstain_class_balance` correctly alternates 0.0 (A/D batches) / 1.0 (B1/B2/C batches) since batches are single-arm; pure arm-C batches (`rho=0` for every item) correctly reduce `loss` to exactly `abstain_loss`, confirming retrieval-loss and count-penalty masking both zero out as designed. Full 31-test generator suite still passes |
| ABSTAIN class balance and loss weight (§2.4) | Regime 2 samples all five arms at equal frequency (`arms=(A,B1,B2,C,D)`, same round-robin mechanism as Regime 1's `(A,D)`) — **nominal** class balance 40% negative (A,D) / 60% positive (B1,B2,C). Loss weight `λ=1`, abstention BCE added unweighted to retrieval loss + count penalty | FIXED (as defaults, per §8); **realized class balance and per-component loss magnitudes must be measured and reported for Regime 2 runs, not assumed to match the nominal figures** — arm-specific rejection-sampling behavior in the generator could shift the actual ratio |
| Regime 1 vs. Regime 2 data confound (§2.4) | Regime 2 trains on strictly more information than Regime 1 (all five arms vs. two) — any Phase 3 behavioral contrast between regimes is confounded with data exposure to B1/B2/C's structures, not purely with the presence of a supervised abstention target | FIXED (disclosure requirement, not a number) |
| Integration count target `ρ` | `ρ = (number of hops walked) − 1`, computed by following the item's own chain to termination. `ρ(A) = ρ(D) = L−1`, `ρ(C) = 0`, `ρ(B1) = j−1` (fails to retrieve at hop `j`), `ρ(B2) = j` (hop `j` retrieves the terminal, commits it, fails at the *next* query — one integration longer than B1 at the same break) | FIXED (implemented and tested: `test_rho_matches_walk_semantics`, `test_rho_not_conflated_between_b1_and_b2_at_same_hop`) |
| Count penalty scope | `(Σ g_t − ρ)²` applied on **arm A only**. Deliberate: sidesteps whether committing a dead end is "correct," which is what §2.4's Regime-1-on-B2 generalization test is meant to observe rather than supervise | FIXED |
| Seeds | 3 minimum per condition | FIXED |
| Total real-seed commitment (§2.6) | 2 supervision regimes × 3 seeds = **6 runs**, budget is per-run: `6 × 102,400 = 614,400` optimizer steps total, before any Phase 2-4 instrumentation | **FIXED** — §3.1.7's throwaway pilot resolved the provisional status: `S*=15,900`, below 25,600, so `102,400` stands unchanged. Wall-clock estimate revised after fixing the generator (below): **~16 hours for all six**, not the original ~70 |

---

## 3. Phase 1 — Competence gate (§3)

| Quantity | Value | Status |
|---|---|---|
| Pilot seeds | 3, discarded, **train full-rank** (no query-projection bottleneck) — required so pilot-measured PR isn't capped by the very guess it's meant to produce | FIXED |
| Training budget | `budget = min(4 × max(S* over 3 pilots), 200,000 steps)` = `min(4×25600, 200000)` = **102,400 optimizer steps**. `S* = [25600, 21100, 21600]`, spread **4,500** | **FIXED** — measured 2026-08-22, cosine-LR pilots run to the full 30,000-step cap with no early stopping, `runs/pilot_summary.json`. All three reached genuinely stable criterion (holds at every evaluation from `S*` through the terminal checkpoint) — a first for this project; every earlier attempt (constant LR, first-touch `S*`) either wasn't stable or wasn't measured for stability at all. Spread is real (25,600 vs ~21,000) but far tighter, relatively, than the 76× seen under the broken recipe — consistent with three seeds finding compatible solutions at different speeds, not three different outcomes. |
| Early stopping | No validation improvement for 20% of budget; may only shorten, never extend | FIXED |
| Competence gate | `≥95%` exact-match accuracy on arm A, at every `L ∈ {1,2,3}`, held-out, in all 3 real seeds | FIXED |
| Seed-shopping | Prohibited — 2/3 passing is reported as the result, not retried with fresh seeds | FIXED |
| Query-projection rank (§3.1.5) | `rank = ceil(max over 3 pilots of **query PR**)`, rounded to next multiple of 4 = **36**. `query_PR = [32.54, 32.53, 31.57]`, spread **0.97** | **FIXED** — measured 2026-08-22 on each pilot's terminal (cap-reached, stable-since-`S*`) checkpoint, `runs/pilot_summary.json`. Tightly clustered across all three seeds (spread <3% of the mean) — a real contrast with the pre-recipe-fix measurement, where query PR varied by 14+ and moved in the *opposite direction* from causal PR on the outlier. Stable training produces a stable geometry measurement; the earlier instability wasn't just a competence problem, it was also corrupting the thing §3.1.5 exists to measure. |
| Causal PR, for §6.1 (not rank-setting) | `[30.10, 30.10, 29.17]`, spread **0.93** | **FIXED**, informational — measured 2026-08-22, same checkpoints as above. Also tightly clustered, unlike the pre-fix measurement's wide, direction-flipping spread. §6.1's causal-subspace claims can now be made against a stable-across-seeds quantity rather than one that depended on which convergence regime a seed happened to land in. |
| Causal rank verification (§3.3) | **Corrected criterion**, replacing an ill-posed earlier version that compared causal PR (retrieval-logit Jacobian, whole-state output sensitivity) against `rank` — a bound on the query projection specifically, a different object; causal PR can legitimately exceed `rank` without anything being clipped. Now: **query PR** on all 3 passing real seeds. If query PR saturates at `rank`: the bottleneck binds — report as a finding and a bound on §6.1's claims, **do not retrain at a larger rank**. If meaningfully below `rank`: the bottleneck has slack. Causal PR is still measured and reported per seed (feeds §6.1) but is no longer compared against `rank` at all. Confirmed on all 3 hard-gate-validated pilots (rows above) that causal PR and query PR are not just numerically different but can diverge in *direction* on the same checkpoint — the strongest evidence yet that conflating them would misdiagnose a real seed | FIXED (procedure, `model/geometry.py::measure_query_participation_ratio`); **numeric value PENDING** real seeds |
| Query PR reference point, pre-real-seed | **`27.99` against `rank=36`** (§3.1.7's throwaway pilot, single-seed, informal — not a real-seed measurement, but the only data point that exists before real seeds run). Meaningfully below `rank`: the bottleneck has headroom, not saturation, at this one seed. A real seed landing near `35.9` would mean something structurally different from one landing near `28` — logged now specifically so that contrast is visible when the real numbers arrive, not reconstructed from memory afterward | FIXED (reference value, not a real-seed result) |
| Known limitation, for `RESULTS.md` | Pilots train full-rank, real seeds train at the fixed reduced rank — `S*` (and thus the training budget) is measured under a slightly different model than the one it's budgeting for. The `4×` multiplier is intended to absorb this; it is not corrected for, only disclosed | FIXED (as a disclosure requirement, not a number) |
| Rank-bottleneck throwaway pilot (§3.1.7) | One discarded pilot, trained with the query projection actually fixed at `rank=36`. Same §3.2 terminal-checkpoint criterion, judged only from the complete history — **`S* = 15,900`** (stable from that step through the terminal checkpoint at 29,999, held-out accuracy 1.0/1.0/1.0), **causal PR = 26.31, query PR = 27.99** (below rank — the bottleneck did not saturate at 36, some headroom rather than maximal clipping) | **FIXED, PASS** — measured 2026-08-22, `runs/rank_bottleneck_pilot.pt`, `runs/rank_bottleneck_pilot_result.json`. `S*` below 25,600 → budget stays at 102,400, no re-derivation triggered |
| Canary run, real seed 0, Regime 1 (§2.6 launch order) | First of the six real-seed runs, arms A+D, full budget (`102,400` steps), launched alone before the remaining five specifically to surface any defect that only shows up at full scale before committing five more runs' compute to it. **`S* = 23,800`** (stable from that step through the terminal checkpoint at 102,399, held-out accuracy 1.0/1.0/1.0 at every `L`), **causal PR = 25.35, query PR = 26.99** — close to §3.1.7's `27.99` reference, not near `35.9`: headroom at the bottleneck, consistent with the throwaway pilot rather than diverging from it. Wall time 7,370s (~2.05h, under the 3h estimate); disk footprint for the full run ~1.4MB (resume checkpoint is one atomically-overwritten file, not one-per-checkpoint, so it does not grow with checkpoint count); process exited cleanly | **FIXED, canary PASS** — measured 2026-08-22, `runs/real_seed_r1_0.pt`, `runs/real_seed_r1_0_result.json`. `S*` at real budget (23,800) came in higher than the pilot's 15,900 — see `RESULTS.md` — but still leaves >3x `S*` of headroom before the 102,400 cap. Clears every concern §2.6 flagged for the canary: stability at 7x pilot length, disk growth, resume across ~100 checkpoint cycles. Remaining five runs may now launch in the pinned interleaved order |
| Real seed 0, Regime 2 (§2.6 launch order, second run) | First real use of `forward_train_regime2` (the abstain head, variable-length teacher forcing) at full scale — arms A/B1/B2/C/D, rank=36, full budget (`102,400` steps). **`S* = 7,500`** (stable from that step through the terminal checkpoint at 102,399, held-out accuracy 1.0/1.0/1.0 at every `L`), **causal PR = 26.92, query PR = 28.58** — again close to the `27-28` cluster the other three real-data points sit in, not near `35.9`. Abstain head: BCE loss and per-batch accuracy both saturated (loss ≈0, accuracy 1.0) from roughly step 700 onward and held through the terminal checkpoint. Wall time 8,821s (~2.45h); process exited cleanly | **FIXED, PASS** — measured 2026-08-23, `runs/real_seed_r2_0.pt`, `runs/real_seed_r2_0_result.json`. **`S*` came in 3.2x faster than R1seed0's 23,800** — logged as a finding in `RESULTS.md`, not assumed to be regime-general from an n=1-per-regime comparison |
| Real seed 1, Regime 1 (§2.6 launch order, third run) | Second Regime 1 seed, arms A+D, rank=36, full budget (`102,400` steps). **`S* = 23,800`** — the identical checkpoint index as `real_seed_r1_0`, verified genuine (distinct loss trajectories throughout, distinct checkpoint file hashes; see `RESULTS.md` finding 9). **Causal PR = 25.36, query PR = 26.99** — nearly indistinguishable from seed 0's 25.35/26.99, consistent with the already-established tight PR clustering across seeds (finding 5). Held-out accuracy 1.0/1.0/1.0 at every `L`; process exited cleanly | **FIXED, PASS** — measured 2026-08-23, `runs/real_seed_r1_1.pt`, `runs/real_seed_r1_1_result.json`. **`S*` exactly matching seed 0's is not yet interpretable with n=2** — `real_seed_r1_2` (the third Regime 1 seed) is what determines whether `23,800` is a structural boundary or a coincidence |

### §3.1.6 evidence record — accuracy vs. `L` and `P(m|L)`, hard-gated

The shortcut hypothesis had four checks; the census and per-step trace were reported earlier. These two are the ones that positively demonstrate traversal rather than merely fail to demonstrate a shortcut — recorded here as the finding's cause of death, not just a passing verdict. `N=2,000` held-out items per `L`, gate hardened at the same 0.5 threshold used for the competence gate itself, `model/diagnose_slope.py`.

| Pilot | `L` | `ρ` | accuracy | `P(m\|L)` (bin = integration count) | mean `m` | errors |
|---|---|---|---|---|---|---|
| 0 | 1 | 0 | 1.0000 | `[2000,0,0]` | 0.000 | 0 |
| 0 | 2 | 1 | 1.0000 | `[0,2000,0,0]` | 1.000 | 0 |
| 0 | 3 | 2 | 0.9965 | `[0,0,1993,7,0]` | 2.003 | 7 |
| 1 | 1 | 0 | 1.0000 | `[2000,0,0]` | 0.000 | 0 |
| 1 | 2 | 1 | 1.0000 | `[0,2000,0,0]` | 1.000 | 0 |
| 1 | 3 | 2 | 0.9940 | `[0,0,1988,2,10]` | 2.011 | 12 |
| 2 | 1 | 0 | 1.0000 | `[2000,0,0]` | 0.000 | 0 |
| 2 | 2 | 1 | 0.9340 | `[132,1868,0,0]` | 0.934 | 132 |
| 2 | 3 | 2 | 0.9240 | `[143,0,1848,0,0,0,1,8]` | 1.879 | 152 |

**What this shows, not just that it "passes":**

- **`m` tracks `ρ` almost exactly wherever the model is correct.** Pilots 0/1 at `L=3`: 1993/1988 out of 2000 items land at exactly `m=2`. This is the direct evidence the census and per-step trace couldn't give on their own — integration count scales with hop count, per item, not just as an aggregate correlation.
- **The real cost is concentrated at `m=0`, not spread across the tail.** Every error for pilot 2, and nearly every error for pilots 0/1, is an item where the gate never opened at all — not an item that over-integrated or wandered. The failure mode is "declined to start," not "got lost partway."
- **The `m=0` error count is roughly `L`-independent, not `L`-scaling, for pilot 2** (132 at `L=2` vs. 143 at `L=3` — close in absolute terms despite `L=3` needing one more hop). That is a per-item first-hop reliability problem, not a problem that compounds with chain length — a different diagnosis than "traversal gets harder with more hops," and a more specific one.
- **This is exactly the profile the original shortcut hypothesis predicted would be absent.** A shortcut model's `m` would sit near 0 regardless of `L`, uncorrelated with `ρ`. What's measured instead is `m ≈ ρ` for the large majority of items at every `L`, on every pilot — including pilot 2, whose accuracy is worse but whose *successful* items still show the hop-count-matching pattern.

---

## 4. Phase 2 — Integrity checks (§4)

| Quantity | Value | Status |
|---|---|---|
| Leakage test | Shuffled cache from a *different* item, not zeroed | FIXED |
| Chance accuracy | `1 / |keys present in this item's cache|` — **not** `1/|S|³`; the output space under END-notation is the set of cache keys, which varies per item | FIXED |
| Counterfactual arms | clean / wrong-value (valid key `k2'` with its own entry, equal norm) / clamped (`g₁=0`) | FIXED |
| Margin AUROC criterion | `≥ 0.70`, predicting top-1 retrieval correctness | FIXED |
| Top-1 retrieval accuracy criterion | within `[0.60, 0.98]` on arm A | FIXED |
| Calibration budget | At most 3 adjustments; knobs `n_hard` and `|S|` only; evaluated on the calibration split | FIXED |

---

## 5. Phase 3 — Dead-end representation, Q2 (§5)

| Quantity | Value | Status |
|---|---|---|
| Probe positions | P1: after integrating `kj`, before the next lookup resolves. P2: after the next lookup returns low margin | FIXED |
| P2 requirement | Must carry information beyond the raw margin value — regress out margin, re-probe | FIXED |
| Probe split | By generated graph, not by item | FIXED |
| Cross-condition transfer combos | **Pinned, exactly two — not a sweep.** (1) train `L=2` → test `L=3`. (2) train `n_distractors=1021` (target scale) → test `n_distractors=256` (pinned low band). 256 chosen as comfortably above the `n_hard·L + MIN_END_DECOYS` generator floor at current settings while being meaningfully shorter than the target — a distinct input-length regime, not an arbitrary number | FIXED |
| Layer sweep | **= all 8 fixed recursion steps (§6.1's `max_steps`), not a separately-chosen width.** Read from the same per-step trace one rollout already produces (`model/raw_records.py`); does not require additional rollout passes. Same for P1/P2 above — two positions within one already-logged trace | FIXED |
| Abstention threshold `τ` | Percentile of the frozen calibration distribution; fixed on validation before any test-set evaluation. **Target corrected**: within 5 percentage points of a first measured floor (false-non-integration rate on arm A at the most permissive defensible gate setting), not an absolute 5% ceiling — pilot evidence (below) shows the floor itself can exceed 5%, which a threshold choice cannot fix | FIXED (procedure, corrected target); **numeric value PENDING** validation-split measurement, including the floor itself |
| First-hop reliability floor | **Resolved for full-rank pilots, not yet for real seeds.** Under the unstable constant-LR recipe, pilot 2 showed `m=0` on arm A at rates up to 6.6% (`L=2`) and 7.2% (`L=3`). Re-measured 2026-08-22 on the stable cosine-schedule, full-rank pilots: `m=0` at `L=2`/`L=3` is 0.6%/0.5% (pilot 0), 0%/0.1% (pilot 1), 0.35%/0.65% (pilot 2) — all comfortably under the original flat 5% target, confirming the floor was a training-instability symptom, not an architectural ceiling. **But every pilot to date is full-rank (§3.1.5); the rank-36 bottleneck real seeds actually train under has never been measured against this floor and could reintroduce it.** §5.3's floor-relative form stays — costs nothing if the floor stays near zero on real seeds, catches it immediately if it doesn't. Do not skip the real-seed re-measurement on the strength of the full-rank result | FIXED (procedure); **numeric value re-opens** at the §3.1.7 throwaway pilot and again on real seeds |
| Q2 falsification | Cross-condition transfer AUROC, arm A vs. arm B2 at position P2, `< 0.65` in Regime 1 → Q2 fails | FIXED |

---

## 6. Phase 4 — Recursion, Q3 (§6)

| Quantity | Value | Status |
|---|---|---|
| Recursion step budget | `max_steps = 4 × (L_max − 1)` **= 8** at `L_max = 3` — every required integration plus three recurse-without-integrate steps per hop. Budget exhaustion is reported, not raised. **One fixed value, not a sweep** — §5.2's `raw/*.jsonl` volume accounting assumes exactly this | FIXED |
| Phase 4 pass structure | 2 arms (constrained/free) × 3 chain lengths = **6 combinations per run**, step budget fixed (above), not additionally swept | FIXED |
| Integration count distribution — expectation | Mode at `ρ`, with **`P(m = ρ) ≥ 0.80`** on held-out arm A. Below that: report a gate failure, not a count finding | FIXED |
| Integration count distribution — deviation semantics | `m > ρ`: gate over-fires, drifting toward always-integrate. `m < ρ`: gate skipped a hop — **not distinguishable from the count alone**; requires both the §1.3 capacity arithmetic and a §4.1 shuffled-cache check on those specific items before a "synthesis" interpretation is claimed. All `m ≠ ρ` findings logged, never filtered | FIXED |

---

## 7. Deliverables (§7)

| Quantity | Value | Status |
|---|---|---|
| `raw/*.jsonl` scope | **Resolved, was ambiguous** — "per-item, per-step" didn't say which passes. Training-loop per-step logging would be enormous (six runs × up to 102,400 steps × 64 items × a recursion budget) and analytically useless. Resolved: evaluation/instrumentation passes only — §3.2 terminal held-out check, §4.4/§5.3 calibration, §3.1.5/§3.3 geometry. Per-step record while an item is active (gate open/closed, margin, displacement, integration count) plus one final record per item (prediction, target, correctness) | FIXED (procedure, `model/raw_records.py`) |
| `raw/*.jsonl` volume, Phase 1 | Measured (not guessed) at ~157 bytes/record, ~4.3 records/item → **~8.1MB per run, ~48.4MB for all six** at realistic pass sizes (2,000 items/`L` for held-out and calibration, 30 for geometry). Small enough that the earlier concern (matching the 54GB pool estimate) doesn't apply here — the two are unrelated in scale | FIXED — estimated before generation, per the requirement, not after |
| `raw/*.jsonl` volume, Phase 3-4 projected | **Corrected — the first projection (~2.8GB) over-counted.** P1/P2 and the layer sweep are reads of one already-logged per-step trace, not separate rollout passes — they don't multiply volume. What actually multiplies it: new arms Phase 1 never logged (B1/B2/C/D — Phase 1 is arm A only), and the pinned `n_distractors=256` cross-condition band. Phase 3: `5 arms × 3 L × 2 bands = 30` combos, 3 already reused from Phase 1 → **27 new passes ≈ 36.5MB/run**. Phase 4: `2 arms × 3 L = 6` combos, step budget fixed not swept → **≈ 8.1MB/run**. **Corrected total ≈ 52.7MB/run, ≈ 317MB for all six** — still a projection (Phase 3/4 aren't run yet) but now grounded in which factors are genuinely new passes vs. re-reads of existing data, not a guessed multiplier | FIXED (as an order-of-magnitude projection, explicitly not a measurement; supersedes the ~2.8GB figure, which is a corrected value not a re-estimate) |
| `raw/*.jsonl` git handling | Kept out of git (`raw/*.jsonl` in `.gitignore`) — regenerable from a checkpoint plus the seeds this file logs. A small, git-tracked manifest (`model/raw_records.py::write_manifest`) stands in for the data: paths, per-split record/byte counts, run identity | FIXED |

---

## Summary

**FIXED now, everything Phase 1 pilots can settle:** generator structure and invariants, capacity bounds (computed), gate mechanics, supervision regimes, `ρ` definition (including the B1/B2 off-by-one) and its scoping to arm A, competence gate threshold (judged at the terminal checkpoint, never first pass), seed-shopping prohibition, the corrected §3.1.5/§3.3 rank division of labor, all of §4's integrity-check criteria, Q2 probe protocol and falsification threshold, the §6.1 step-budget formula, the §6.2 expected-distribution target and deviation semantics, the `MIN_END_DECOYS=8` generator requirement (§3.1.6), the corrected `S*` definition, the `4×` budget multiplier, the cosine LR schedule (`RECIPE_LOG.md`), and — final, measured 2026-08-22 on genuinely stable pilots — the training budget (**102,400 steps**) and query-projection rank (**36**).

**PENDING (needs a real-seed run to produce a number):** gate CDF fit, abstention threshold τ, query-PR verification against the real-seed rank (§3.3).

**BLOCKED:** none.

**Note on the path to these numbers, for `RESULTS.md`:** five separate problems, each caught before the next round of numbers got written down as final, not after.

1. An initial pilot run used a shrunk cache (`n_distractors=64`) purely for generator speed, without rechecking §1.3's bounds at that scale — it violated the upper bound (state larger than the cache) and was discarded before writing anything here.
2. The corrected run, at the target scale, hit an unrelated infrastructure failure: a `ProcessPoolExecutor` used to parallelize the CPU-bound generator was created after the model had already touched CUDA, which deadlocks forked worker processes on a futex — silently, no exception, ran for 2.5 hours doing nothing. Fixed with a `spawn` multiprocessing context instead of the default `fork`.
3. The run after that produced a 600/24 pair — genuinely at the right scale, genuinely fast (all three pilots passed within 100–200 steps, ~70s each) — but fast enough to be suspicious. §3.1.6's check found that every generated item had exactly one entry with value `END`, a one-lookup shortcut around the entire chain at any length. A per-step retrieval trace on pilot 0 showed this *specific* checkpoint had not taken it (0/256 held-out L=3 items found END before step 3, matching genuine traversal) — but the shortcut being unexploited by one seed is not evidence it stays unexploited by the next, so the generator was fixed (`MIN_END_DECOYS=8`) regardless of that finding.
4. Re-run against the fixed generator: 900/24→28 (two sub-measurements), both numbers moved (higher budget, higher rank, wider PR spread) rather than staying put — evidence the fix changed something real rather than being a no-op.
5. Asked to report the accuracy-vs-`L` slope and `P(m|L)` as positive evidence rather than "clear," not just a rounded-sum proxy — which surfaced a second, unrelated bug: `forward_eval_autoregressive` never actually hardened the gate at inference, despite §2.2 requiring it ("trained soft, hardened at inference"). On the same checkpoint and held-out batch, soft-gated accuracy read 96%, hard-gated read 77%. Every prior "reached criterion" determination, across every pilot run so far, had been measured under the wrong (soft) evaluation. Fixed (`gate_threshold` param, defaults to 0.5) and re-run: pilots 0 and 1 still converge in 100 steps and now show `m=ρ` almost exactly at every `L`; pilot 2 took 7,600 steps — 76× longer — with a stretch where hard-gated accuracy *fell* before recovering. That spread (`S*` 100/100/7,600; PR 19/20/29) is the real, final measurement, and it is a materially different result from every number that preceded it in this list, not a refinement of the same one.

All five are logged here because they're the kind of failure invisible in a results table but real to the process that produced it.
