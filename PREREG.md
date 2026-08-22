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
| Query projection rank (real seeds) | Fixed by the pilot measurement in §3.1.5, **before** real seeds train. §3.3 verifies it afterward on the real seeds; it does not produce it. | FIXED (procedure); **numeric value PENDING** pilot runs |
| Integration write | `state ← state + g_t · W_v(value)` (additive residual) | FIXED |
| Gate inputs | Quantile-transformed (`p_ret`, margin percentile; `p_set`, displacement percentile), CDF fit once on the calibration split, never updated | FIXED (procedure); the fitted CDF itself is `PENDING` (needs the calibration split to exist) |
| Supervision regimes | Regime 1: arms A, D only, never sees a terminal mid-follow. Regime 2: all arms, B1/B2/C → `ABSTAIN` | FIXED |
| Integration count target `ρ` | `ρ = (number of hops walked) − 1`, computed by following the item's own chain to termination. `ρ(A) = ρ(D) = L−1`, `ρ(C) = 0`, `ρ(B1) = j−1` (fails to retrieve at hop `j`), `ρ(B2) = j` (hop `j` retrieves the terminal, commits it, fails at the *next* query — one integration longer than B1 at the same break) | FIXED (implemented and tested: `test_rho_matches_walk_semantics`, `test_rho_not_conflated_between_b1_and_b2_at_same_hop`) |
| Count penalty scope | `(Σ g_t − ρ)²` applied on **arm A only**. Deliberate: sidesteps whether committing a dead end is "correct," which is what §2.4's Regime-1-on-B2 generalization test is meant to observe rather than supervise | FIXED |
| Seeds | 3 minimum per condition | FIXED |

---

## 3. Phase 1 — Competence gate (§3)

| Quantity | Value | Status |
|---|---|---|
| Pilot seeds | 3, discarded, **train full-rank** (no query-projection bottleneck) — required so pilot-measured PR isn't capped by the very guess it's meant to produce | FIXED |
| Training budget | `budget = min(4 × max(S* over 3 pilots), 200,000 steps)` (formula corrected: §3.1 now uses 4×, not 3× — the 3× multiplier was calibrated against first-pass variance, since shown to be the wrong scale) | **PENDING re-measurement, superseded number below kept for the record only.** The 22,800 figure (`3×7600`) was measured under the constant-LR=3e-3 recipe and the first-touch `S*` definition — both since superseded. §3.1.6-adjacent diagnostics found that under constant LR, *none* of the three pilots ever held §3.2 criterion at every subsequent evaluation (including pilot 2, whose own 7,600-step pass didn't survive to the next check), and a disconfirming test showed a cosine LR decay resolves this. All three pilots are re-running from scratch under the corrected schedule and the corrected (onset-of-stability) `S*` definition; budget will be set from that run. |
| Early stopping | No validation improvement for 20% of budget; may only shorten, never extend | FIXED |
| Competence gate | `≥95%` exact-match accuracy on arm A, at every `L ∈ {1,2,3}`, held-out, in all 3 real seeds | FIXED |
| Seed-shopping | Prohibited — 2/3 passing is reported as the result, not retried with fresh seeds | FIXED |
| Query-projection rank (§3.1.5) | `rank = ceil(max over 3 pilots of **query PR**)`, rounded to next multiple of 4, measured on each pilot's **terminal** (cap-reached, stable) checkpoint — never a first-pass or mid-transition checkpoint | **PENDING re-measurement.** A value of 48 (`query_PR=[46.33,47.54,33.27]`) was measured 2026-08-21 on pilots 0/1 at their step-100 first pass and pilot 2 at its (since shown non-stable) step-7,600 pass — i.e., on three checkpoints none of which represent stable competence under the corrected `S*` definition. Kept here for the record, not as a candidate value. Real number comes from the terminal checkpoint of the in-progress re-run. **Independent of the specific number**, the derivation method itself is settled and will not change: causal PR and query PR are confirmed to move in *opposite directions* on the same checkpoint (highest causal PR ≠ highest query PR), so `rank` is derived from query PR only, never causal PR — causal PR remains informational for §6.1. |
| Causal PR, for §6.1 (not rank-setting) | `[19.13, 20.45, 29.00]`, spread 9.87 | FIXED, informational. See the row above: the spread isn't just wide, its *relationship* to query PR flips sign on the outlier — a stronger caveat on §6.1 than "seeds vary" alone. Causal dimensionality may depend on which convergence regime a seed fell into, not be a fixed property of the task the way §3.1.5 originally assumed. |
| Causal rank verification (§3.3) | **Corrected criterion**, replacing an ill-posed earlier version that compared causal PR (retrieval-logit Jacobian, whole-state output sensitivity) against `rank` — a bound on the query projection specifically, a different object; causal PR can legitimately exceed `rank` without anything being clipped. Now: **query PR** on all 3 passing real seeds. If query PR saturates at `rank`: the bottleneck binds — report as a finding and a bound on §6.1's claims, **do not retrain at a larger rank**. If meaningfully below `rank`: the bottleneck has slack. Causal PR is still measured and reported per seed (feeds §6.1) but is no longer compared against `rank` at all. Confirmed on all 3 hard-gate-validated pilots (rows above) that causal PR and query PR are not just numerically different but can diverge in *direction* on the same checkpoint — the strongest evidence yet that conflating them would misdiagnose a real seed | FIXED (procedure, `model/geometry.py::measure_query_participation_ratio`); **numeric value PENDING** real seeds |
| Known limitation, for `RESULTS.md` | Pilots train full-rank, real seeds train at the fixed reduced rank — `S*` (and thus the training budget) is measured under a slightly different model than the one it's budgeting for. The `3×` multiplier is intended to absorb this; it is not corrected for, only disclosed | FIXED (as a disclosure requirement, not a number) |

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
| Abstention threshold `τ` | Percentile of the frozen calibration distribution; fixed on validation before any test-set evaluation. **Target corrected**: within 5 percentage points of a first measured floor (false-non-integration rate on arm A at the most permissive defensible gate setting), not an absolute 5% ceiling — pilot evidence (below) shows the floor itself can exceed 5%, which a threshold choice cannot fix | FIXED (procedure, corrected target); **numeric value PENDING** validation-split measurement, including the floor itself |
| First-hop reliability floor, pre-flagged | Pilot 2 (hard-gated, §3.1.6): `m=0` on arm A at rates up to **6.6%** at `L=2` (132/2000) and **7.2%** at `L=3` (143/2000) — the model declines to open the gate on the first hop at all, on answerable items, with no threshold involved. Pilots 0/1 show no comparable rate (0/2000, 0/2000). Not yet known whether real seeds inherit pilot 2's specific failure mode or whether it is a training-instability artifact tied to the same slow convergence — but it is the reason §5.3's target needed restating before calibration, not after | FIXED (as a pre-flagged risk with numbers attached, not a real-seed finding yet) |
| Q2 falsification | Cross-condition transfer AUROC, arm A vs. arm B2 at position P2, `< 0.65` in Regime 1 → Q2 fails | FIXED |

---

## 6. Phase 4 — Recursion, Q3 (§6)

| Quantity | Value | Status |
|---|---|---|
| Recursion step budget | `max_steps = 4 × (L_max − 1)` **= 8** at `L_max = 3` — every required integration plus three recurse-without-integrate steps per hop. Budget exhaustion is reported, not raised | FIXED |
| Integration count distribution — expectation | Mode at `ρ`, with **`P(m = ρ) ≥ 0.80`** on held-out arm A. Below that: report a gate failure, not a count finding | FIXED |
| Integration count distribution — deviation semantics | `m > ρ`: gate over-fires, drifting toward always-integrate. `m < ρ`: gate skipped a hop — **not distinguishable from the count alone**; requires both the §1.3 capacity arithmetic and a §4.1 shuffled-cache check on those specific items before a "synthesis" interpretation is claimed. All `m ≠ ρ` findings logged, never filtered | FIXED |

---

## Summary

**FIXED now:** generator structure and invariants, capacity bounds (computed), gate mechanics, supervision regimes, `ρ` definition (including the B1/B2 off-by-one) and its scoping to arm A, competence gate threshold (judged at the terminal checkpoint, never first pass), seed-shopping prohibition, the corrected §3.1.5/§3.3 rank division of labor (query PR sets `rank`, causal PR is informational only — the derivation *method*, independent of any specific number), all of §4's integrity-check criteria, Q2 probe protocol and falsification threshold, the §6.1 step-budget formula, the §6.2 expected-distribution target and deviation semantics, the `MIN_END_DECOYS=8` generator requirement (§3.1.6), the corrected `S*` definition (onset of stable competence, no window parameter), the `4×` budget multiplier, and the cosine LR schedule (`RECIPE_LOG.md`).

**PENDING (awaiting the in-progress pilot re-run — `SESSION_LOG.md` §4-5 for how we got here):** training budget, query-projection rank, causal PR (informational). All three numbers previously in this file were measured on checkpoints since shown not to represent stable competence (first-pass or mid-oscillation) and are superseded, not refined.

**PENDING (needs a real-seed run to produce a number):** gate CDF fit, abstention threshold τ, query-PR verification against the real-seed rank (§3.3).

**BLOCKED:** none.

**Note on the path to these numbers, for `RESULTS.md`:** five separate problems, each caught before the next round of numbers got written down as final, not after.

1. An initial pilot run used a shrunk cache (`n_distractors=64`) purely for generator speed, without rechecking §1.3's bounds at that scale — it violated the upper bound (state larger than the cache) and was discarded before writing anything here.
2. The corrected run, at the target scale, hit an unrelated infrastructure failure: a `ProcessPoolExecutor` used to parallelize the CPU-bound generator was created after the model had already touched CUDA, which deadlocks forked worker processes on a futex — silently, no exception, ran for 2.5 hours doing nothing. Fixed with a `spawn` multiprocessing context instead of the default `fork`.
3. The run after that produced a 600/24 pair — genuinely at the right scale, genuinely fast (all three pilots passed within 100–200 steps, ~70s each) — but fast enough to be suspicious. §3.1.6's check found that every generated item had exactly one entry with value `END`, a one-lookup shortcut around the entire chain at any length. A per-step retrieval trace on pilot 0 showed this *specific* checkpoint had not taken it (0/256 held-out L=3 items found END before step 3, matching genuine traversal) — but the shortcut being unexploited by one seed is not evidence it stays unexploited by the next, so the generator was fixed (`MIN_END_DECOYS=8`) regardless of that finding.
4. Re-run against the fixed generator: 900/24→28 (two sub-measurements), both numbers moved (higher budget, higher rank, wider PR spread) rather than staying put — evidence the fix changed something real rather than being a no-op.
5. Asked to report the accuracy-vs-`L` slope and `P(m|L)` as positive evidence rather than "clear," not just a rounded-sum proxy — which surfaced a second, unrelated bug: `forward_eval_autoregressive` never actually hardened the gate at inference, despite §2.2 requiring it ("trained soft, hardened at inference"). On the same checkpoint and held-out batch, soft-gated accuracy read 96%, hard-gated read 77%. Every prior "reached criterion" determination, across every pilot run so far, had been measured under the wrong (soft) evaluation. Fixed (`gate_threshold` param, defaults to 0.5) and re-run: pilots 0 and 1 still converge in 100 steps and now show `m=ρ` almost exactly at every `L`; pilot 2 took 7,600 steps — 76× longer — with a stretch where hard-gated accuracy *fell* before recovering. That spread (`S*` 100/100/7,600; PR 19/20/29) is the real, final measurement, and it is a materially different result from every number that preceded it in this list, not a refinement of the same one.

All five are logged here because they're the kind of failure invisible in a results table but real to the process that produced it.
