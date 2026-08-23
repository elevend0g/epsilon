# Experimental Protocol — SSM-Driven Gated Cache with Latent Recursion

**Status:** design, pre-code. v2 (END-sentinel notation).
**Relationship to prior work:** fresh project. Does not reuse `elevend0g/beta` checkpoints, task, or eval sets. Reuses lessons only.
**Scope claim:** a mechanism testbed on a synthetic symbolic task, using one small model class. It establishes whether specific signals exist and are readable. It does **not** license claims about language models. Every writeup must state this in the abstract.

**This document is the preregistration.** Anything discussed elsewhere and not written here does not exist. Every threshold, formula, and arm definition must be in this file before the first training run.

---

## 0. What is being tested — stated narrowly

Three questions in strict dependency order. Each phase gates the next.

**Q1 (competence).** Can a small SSM-driver + external-cache model learn multi-hop symbolic lookup to high accuracy?
*If no, nothing downstream is interpretable.* The prior project lacked this gate, and its absence made every subsequent null uninformative.

**Q2 (dead-end representation).** After following a chain into a dead end, does the recurrent state represent the dead end **as a dead end** rather than as another step to take — and does that representation arise without being trained to report it?

**Q3 (recursion).** Does latent iteration constrained to the causal subspace outperform unconstrained iteration, and does the integration gate exhibit the predicted chain-linked structure?

### 0.1 What this task does and does not stress

**It does not test SSM multi-step memory degradation.** Chain lengths here are short; a recurrent state at this scale holds an entire chain trivially. The saw-toothed decay of facts over many sequential hops is not exercised, and the motivation must not claim it is.

**It does test one-shot content-addressable selection over a long shuffled context compressed into fixed state.** The hard step is absorbing ~1k shuffled entries *before* the query is known, then pulling the matching entry from compressed state. That is a genuine SSM weakness and the architecture is a clean answer to it.

**Chain length earns its place for a different reason:** it creates the sequential dependency that makes gate interleaving testable at all. Not because it stresses memory decay.

**Q2 is not broad epistemic uncertainty.** It is one specific, mechanistic species of "I can't derive this": chain exhaustion under a follow-until-END rule. Write it that way in the abstract. The overclaim-from-a-toy failure is the one the standing rules exist to prevent.

**Explicitly not tested:** basin geometry as a halt criterion. Retired by preregistered stopping rule in the prior project. Do not reintroduce it in any form.

---

## 1. Task generator

### 1.1 Structure — follow until END

```
[MEMORY]  <cache entries, shuffled>
[QUERY]   <start_key>
[ANSWER]  <key or ABSTAIN>
```

- **Keys** are 3-tuples over a discrete symbol alphabet `S`: `(s1, s2, s3)`.
- **Values** are 3-tuples over the same alphabet, or the reserved symbol `END`.
- A chain is `k1 -> k2`, `k2 -> k3`, ..., `kL -> END`.
- **The query contains `start_key` only.** No depth token.
- **Emit rule:** the state carries the current key. Integration advances it. On retrieving `END`, emit the current key.

**Why depth is out of the query.** With depth in the query, the count penalty `(Σ g_t − ρ)²` is satisfiable by embedding the depth token and counting down — a label-copy that has nothing to do with detecting whether the state needs a fact, and which pre-loads the §6.2 result. With `END`, the model cannot read the target count; landing on it requires following the actual chain.

**Why END also fixes the broken arm.** Without a terminator, a chain broken at hop *j* is structurally identical to a complete chain of length *j−1* — both end at a non-key value, with no marker that continuation was expected. With END, a complete chain terminates in a *present* END entry and a broken chain does not. The two are distinguishable by what the terminal lookup returns, never by a label.

This also makes the recursion loop exact fixed-point iteration: state carries the current key, integration advances the pointer, END is the fixpoint.

### 1.2 The core invariant

**The query segment carries zero bits about intermediate keys or about terminal status.** Both must come from the cache.

Enforced by:
- **Per-item graph resampling.** The key→value mapping is redrawn for every example. Never a fixed global graph; that would move the task into weights and make the cache decorative.
- **Capacity bounds** (§1.3).

These conditions do the work jointly. Neither alone suffices. Verify empirically in §4.1 rather than asserting by construction.

### 1.3 Capacity arithmetic — a generator assertion, not a paragraph

The prior version of this document asserted that bulk-loading the cache into state was "blocked by counting bits" without doing the count. Do the count.

```
cache_bits  = n_entries × 6 × log2(|S|)
state_bits  = d_model × d_state × bits_per_element      # conservative upper bound
```

At `n_entries=1024, |S|=64`: cache ≈ 36.9 kbit. At `d_model=256, d_state=16`, fp16: state ≈ 65.5 kbit — **larger than the entire cache.** The asymmetry does not exist at that config. At `d_model=128` it is roughly a coin flip.

**Two bounds, both asserted in the generator's unit tests and both written into `PREREG.md` as computed integers:**

- **Upper:** `cache_bits / state_bits ≥ 4`. The table must not fit in state.
- **Lower:** `state_bits ≥ 16 × trajectory_bits`, where `trajectory_bits` is the information needed to carry the current key plus a handful of scalar trajectory features. The state must have ample room for more than a pointer (see §2.1).

Raw element width overstates true recurrent capacity — effective capacity is far lower, as the prior project's participation ratios suggest. That cuts in favor of these bounds, not against them: if even the conservative bound says the table fits, the asymmetry cannot be claimed.

### 1.4 Condition arms

| Arm | Construction | Terminal event | Ground truth |
|---|---|---|---|
| **A — answerable** | complete chain, `kL -> END` | END retrieved | `kL` |
| **B1 — missing key** | mid-chain entry replaced by a decoy entry | low margin at hop *j* | ABSTAIN |
| **B2 — dead end** | `kj -> x`, where `x` is drawn like a key but has **no cache entry** | high margin at hop *j*, low margin at *j+1* | ABSTAIN |
| **C — key absent** | `start_key` not in cache | low margin at hop 1 | ABSTAIN |
| **D — length control** | answerable, cache padded to B/C token count | END retrieved | `kL` |

**B2 is the thesis arm.** Its dead end arrives *after* a high-confidence retrieval, and the only dead-end cue is absence from the cache — `x` is distributionally identical to a key. This is where confident ignorance lives mechanistically, and no other arm produces it.

**B1 is deletion-free.** Removing an entry changes token count and breaks twin identity, so B1 *replaces* the entry with a decoy. B2 needs no such patch: it changes one entry's value, leaving cache size and token count identical.

**Twin identity is mandatory.** Each B item is generated from an A item and is identical in distractor count, Hamming distribution, cache size, token count, and entry ordering, differing only by the specified change.

**Detection modality note.** B1 and B2 both terminate in a low-margin query — the "is `x` a key?" check *is* the next follow, not a separate lookup. B2's distinctiveness is therefore about the **gate's** behaviour, not about a different detection mechanism: the question is whether the gate opens on `x` and commits a dead end to state. Say this explicitly; do not imply B2 introduces a second query per hop.

**Pin the probe against a length shortcut.** B1 and C differ mainly in hops-to-failure, which is a length feature. Any probe separating them must be shown not to be counting integrations.

### 1.5 Difficulty axes, orthogonal by construction

- **chain length** `L ∈ {1, 2, 3}` — sequential depth
- **n_distractors** — input length
- **n_hard** — retrieval difficulty

Must vary independently. In the prior project difficulty and length were the same knob, which is why the halt head learned length. Assert orthogonality in generator unit tests.

### 1.6 Hard negatives

Proximity must be **generator-controlled**, not emergent from learned embeddings. Composite keys give the lever: shared components force shared embedding structure.

- **Hard negative:** cache entry at Hamming distance 1 from the true query key.
- **Easy negative:** Hamming distance 3.
- `n_hard` is preregistered and swept.

Requirements:

1. **Generate hard negatives per hop.** Distractors clustered only around `start_key` leave later hops saturated — and those are the hops the design is about.
2. **Reject any distance-1 distractor whose value continues the active chain** (alternate path through the distractor generator).
3. **B1 decoys may not** map into the chain or collide with any hop query.
4. **Terminals in B2 drawn from the key distribution** so cache-absence is the only dead-end cue.
5. **Assert no duplicate keys with differing values** — the only condition that makes an item genuinely ambiguous.
6. **Shuffle cache order**; verify no ordinal-position signal.
7. **Verify follow-path uniqueness.** The followed path must be unique.
8. **Minimum decoy END entries: ≥ 8 per item, drawn on keys not on the followed path.** Without this, if the chain terminus is the *only* entry mapping to `END`, the answer is retrievable in a single lookup — "find the END entry" — and the chain is never traversed at any `L`. This would silently invalidate ρ, §6.2, and both B arms. Assert the count in generator tests; it is not optional and "multiple ENDs are fine" is not a sufficient statement of it.

---

## 2. Model

Small enough to train several seeds to convergence in hours on one GPU.

### 2.1 Components — and what the state must be allowed to hold

- **SSM driver.** Recurrent state, full `[batch, d_model, d_state]` exposed at every step. No pooling anywhere in the forward path an epistemic readout will consume.
- **Cache.** Content-addressable store over the memory segment. Retrieval by similarity against a query projected from the causal state.
- **Query projection.** Low-rank: causal state → key space. Rank is fixed **before the real runs** by the pilot measurement in §3.1.5. §3.3 verifies it after the fact; it does not produce it.
- **Integration write — additive residual, stated explicitly:**
  `state ← state + g_t · W_v(value)`
  This is the only form under which the §4.2 equal-norm intervention is well-posed, and the only one under which displacement decomposes cleanly into a recursion component and a commitment component.

**The pointer hazard, and the wrong fix.** Under follow-until-END, the state's minimum job is to carry the current key. If the architecture *constrains* it to that, dead-end status has no home in state — it lives only in the transient outcome of the next retrieval — and a null Q2 probe would be vacuous rather than informative.

**The fix is capacity, not a status register.** A tempting response is to build an explicit register holding hop count, last margin, and value-validity, and point the probe at it. **Do not.** That decides Q2 by construction: a probe that finds coverage in a channel built to carry coverage has measured the architect, not the model. It would destroy Regime 1 (§2.4), whose entire point is whether the representation arises unbidden.

Instead: **impose no pointer-only constraint, and enforce the §1.3 lower bound** so the state has ample room to retain trajectory features beyond the pointer. Whether it *does* retain them is then the measured question. If Q2 returns null under a state with 16× the room it needs for a pointer, that is an informative null — the model had the capacity and did not use it.

### 2.2 Gating — query always, gate integration

**Retrieval is unconditional at every step. Integration is gated.**

This is the central design decision. Gating the *query* creates a counterfactual dead end: an unexecuted read yields no gradient about whether reading would have helped. Unconditional retrieval guarantees a gradient at every step. What is expensive is *committing* — writing a retrieved value into state changes the trajectory.

**Scale caveat, mandatory in any restatement:** the look-is-cheap/commit-is-expensive asymmetry holds **at testbed cache size only**. Against a production-scale cache, unconditional lookup is O(N) key comparisons per step — attention's cost structure with the expensive half swapped. This is not an architectural argument about large models and must not be quoted as one.

Two **measured** quantities drive control flow. Neither is a learned prediction about the model's own epistemic state — that is the failure mode the prior project documented.

- **Retrieval margin** — top-1 similarity minus top-2.
- **Causal displacement** — `‖Δh‖` within the causal subspace over the last step. *Not* basin width; requires no perturbation.

**Both enter the gate as quantiles, not raw values.** Each is mapped through its empirical CDF, estimated once on a frozen calibration split and never updated. Raw margin scales with the similarity temperature and raw displacement with state norm, so a gate on raw values is a temperature artifact rather than a controller. Quantile inputs are scale-invariant by construction.

| | high margin quantile | low margin quantile |
|---|---|---|
| **displacement high** | integrate | recurse |
| **displacement low** | emit | **abstain / external call** |

The bottom-right cell is the target phenomenon: state settled, nothing in memory supporting it. A stability-based halt would emit here. It is a **routing trigger on a measured condition**, not a deterministic hallucination catch — whether the condition covers the failure mode is the empirical question.

Gate trained soft (sigmoid over quantile inputs), hardened at inference.

### 2.3 Readout ablation — build both from day one

Every epistemic readout in two variants, trained in parallel: **pooled** (channel-averaged state) and **subspace** (low-rank projection of the full state). On a competent model this comparison is meaningful; in the prior project both arms sat at chance, which made the null uninformative about the readout-bottleneck hypothesis.

### 2.4 Supervision regimes — two models, and this is the experiment

- **Regime 1 (unsupervised).** Trained on arms A and D only. Never sees an abstention target, and **never sees a terminal mid-follow.**
- **Regime 2 (supervised abstention).** Trained on all arms; B1, B2, C map to ABSTAIN.

Regime 2 tells you whether a trained abstention signal is real or a surface correlate. Regime 1 tells you whether dead-end representation emerges in a network never taught to report it. The contrast is the direct answer to "is ignorance acknowledgment trainable or architectural."

**Regime 1 on B2 is a preregistered generalization test.** Its gate learned "open iff the retrieved value is a key, hold at END" from complete chains alone. At test time on B2, does it open on `x` — treating a dead end as another step? Higher gate confidence is *worse* here: an open gate commits the terminal to state, and the abstention decision must then fight its own committed garbage. Preregister the prediction and the direction.

**ABSTAIN mechanism, pinned before the code exists.** Three structural decisions, all binding across all three Regime 2 seeds:

- **A separate binary head — forced, not merely preferred.** `retrieve()` (`model/cache_bank.py`) scores the query against `cache_keys`, a per-item content-addressed embedding of that item's own N entries, rebuilt fresh every forward pass. There is no fixed-size answer vocabulary anywhere in the model — the retrieval "softmax" is over whichever entries this item happens to have. An ABSTAIN token cannot be added to it; there is no stable dimension to add it to, only per-item similarity scores. A separate head is the only structurally available option, not a preference among several. It is also the more useful one: trained on the same state that feeds the step's query, at the step the model would otherwise emit (Regime 2 target: B1/B2/C → 1, A/D → 0), it keeps "retrieved right but also abstained" askable — which folding ABSTAIN into a shared distribution, were one to exist, would have foreclosed by construction (§5.4 needs this open).
- **The §2.2 gate quadrant stays measured and untrained.** It already exists to avoid "a learned prediction about the model's own epistemic state" — the prior project's documented failure mode. A trained ABSTAIN signal competing with the quadrant's routing decision would reintroduce exactly that. Regime 2 trains the head; the quadrant's inputs (margin, displacement) are never a training target. The head is the answer-level output; the quadrant is the routing signal that was already there — two distinct things, not one renamed.
- **Arm-sampling ratio and loss weight, fixed as parameters, not inherited from whatever the dataloader happens to do.** Regime 2 samples all five arms at equal frequency (`arms=(A,B1,B2,C,D)`, the existing round-robin combo mechanism already used for Regime 1's `(A,D)` — no new sampling code, just a different tuple), giving the head a nominal class balance of 40% negative (A, D) / 60% positive (B1, B2, C) — close enough to balanced that no reweighting is applied by default. **This is a nominal figure, not a measured one** — arm-specific rejection-sampling behavior in the generator could shift it in practice, and the actual realized balance must be measured and reported for Regime 2 runs, not assumed to match. Loss weight: `λ=1`, the abstention BCE term added unweighted to the existing retrieval loss + count penalty (§8: simplest version, logged). Per-component loss magnitudes must be logged separately for Regime 2 (extending the existing `logs` dict pattern in `forward_train`) specifically so a scale imbalance between terms — one term numerically dominating an unweighted sum even with balanced classes — is visible rather than silently assumed away.

**Phase 3 treats agreement between the head and the quadrant as an empirical question, not an assumption (§5.4).** A trained head that fires without the gate ever reaching the abstain quadrant (bottom-right, §2.2) is tracking B1/B2/C's structural signature — count shortfall, decoy pattern — rather than the coverage representation Q2 is asking about. Both are reported; neither stands in for the other.

**Regime 1 vs. Regime 2 is not a pure supervision ablation — say so now so it isn't misread later.** Regime 1 trains on arms A and D only; Regime 2 trains on all five. Any behavioral difference observed between the two regimes at Phase 3 is confounded with data exposure — Regime 2 has seen B1/B2/C's decoy, dead-end, and absent-key structures at all, Regime 1 never has — not purely with the presence or absence of a supervised abstention target. The honest framing: Regime 2 has strictly more information than Regime 1, and the trained-vs-emergent contrast (§2.4's opening question) is read through that confound, not around it.

### 2.5 Integration count supervision

Let **ρ = number of key-to-key hops** = chain length − 1 under END notation. The END retrieval **does not open the gate** (this is a design commitment, not an emergent property — state it).

Per arm, where the break occurs at hop *j*:
- **C:** ρ = 0.
- **B1** (entry for `kj` replaced by a decoy): the follow reaches and commits `kj`, then the query on `kj` returns low margin. **ρ = j − 1.**
- **B2** (`kj -> x`, `x` terminal): the follow commits `kj`, retrieves `x`, **commits `x`**, then the query on `x` returns low margin. **ρ = j.** B2 is one integration longer than B1 at the same break point, because a dead end can only be discovered by committing it and querying it.
- **A and D:** ρ = `L − 1`.

**The count penalty is applied on arm A only.** This is deliberate: it sidesteps the question of whether committing a dead end is "correct" behaviour, which is exactly what §2.4's generalization test is trying to observe rather than supervise.

Penalty `(Σ g_t − ρ)²` on arm A. A gate that always integrates becomes attention and overshoots ρ; one that never integrates becomes a pure SSM and undershoots. No MoE-style load balancing needed.

**No temporal regularizer.** Premature firing is not penalized because it is useless — the next key does not exist in any addressable form before the previous integration (§1.2). Ordering is enforced by the data distribution, not a tuned loss term.

### 2.6 Seeds

**Three seeds minimum per condition.** Every result in the prior project was a single-checkpoint result; that is its broadest quiet limitation and it is cheap to fix at this scale.

**The commitment is six runs, not three.** §2.4 requires two supervision regimes, each with three seeds, and the training budget (§3.1) is per-run: `6 × 102,400 = 614,400` optimizer steps, total, before any Phase 2-4 instrumentation. Written here so it cannot be missed at launch time — the failure mode is discovering the true cost halfway through and cutting seeds to fit, and one regime with three seeds is interpretable while two regimes with one seed each is not. Confirm the full six-run cost is affordable before launching any of them, not after the first one or two are already running.

**Affordability, updated after fixing the generator (not the pool) instead of accepting the number.** The original ~70-hour estimate was measured against a generator later found to be reducible: bulk numpy generation, one `.tolist()` conversion instead of per-element access, reusing the item's own distractor pool for arm C instead of rebuilding one, and (caught in the same pass) a buffered-RNG bug that briefly made things worse by thrashing between two different alphabet sizes on one shared buffer. Clean, uncontended measurement after all four: **4.4× faster** (311ms/batch vs. 1,370ms baseline). Revised six-run estimate: **~16 hours**, not ~70 — same aggregate CPU budget, no 10-hour pool build, no 54GB artifact.

**`102,400` is confirmed, not provisional.** §3.1.7's throwaway pilot resolved this: `S*=15,900`, under the full-rank pilots' 25,600, so no re-derivation triggers. Competence survives the bottleneck.

**Launch order: one full-budget run first, then the remaining five interleaved by regime.** `S*=15,900` came from a 30,000-step pilot; real runs go to 102,400 — nearly 7× further, and nothing in this project has trained that long. Disk growth, resume behavior across ~100 checkpoints instead of ~30, and terminal-checkpoint evaluation at full length are all untested at that scale. One run costs under three hours at the fixed generator's throughput; a defect found in run one instead of run six is the difference between losing one run's compute and most of the six-run budget.

**Once the canary clears: alternate regimes, don't block them.** `Regime1-seed0, Regime2-seed0, Regime1-seed1, Regime2-seed1, Regime1-seed2, Regime2-seed2` — not three of one regime followed by three of the other. If launched as a block and something interrupts the sequence partway, a block ordering leaves three seeds of one regime and none of the other — the least useful possible partial result, since §2.4's entire comparison needs both regimes represented. Alternating means any prefix of the sequence — one run, three runs, five — is a balanced (if seed-thin) version of the same comparison.

---

## 3. Phase 1 — Competence gate

### 3.1 Training budget — set by pilots, then frozen

**Optimizer schedule, fixed before this section's pilots run: Adam, cosine LR `3e-3 → 3e-5`, no warmup, decaying across the full pilot cap.** Not a free implementation detail — the first pilot run at constant `3e-3` never held §3.2 criterion at any checkpoint through 8,100 steps, oscillating 85-97% on a fixed validation set (so the model's own behavior, not sampling noise). A disconfirming test (`RECIPE_LOG.md`) continued a pilot's honest checkpoint with everything unchanged except a cosine decay to `3e-5`: oscillation amplitude went from 0.04-0.06 to 0.000, locked to perfect accuracy, tracking the LR cut rather than coinciding with it. Recipe bug, not architecture — but it means every pilot number measured under the constant-LR recipe is stale and is re-measured under this schedule before being trusted.

**Anchoring rule: `S*` is the onset of stable competence, never the first touch.** The pilots demonstrated a transient spike — criterion met at step ~100, then a sustained multi-thousand-step regression to 0.55–0.70 before recovery. A first-pass anchor would have set a real-seed budget of ~300 steps and frozen every seed mid-collapse.

A "N consecutive passes" window is the wrong fix: a spike can outlast any fixed N, and choosing N moves the problem rather than solving it. Define it without a window instead.

1. Train **three pilot seeds** (discarded — never used downstream, never in any result, not among the three real seeds).
2. **Train every pilot to a fixed cap regardless of when criterion is first met.** Cap = 30,000 steps, roughly 4× where the slowest pilot stabilized.
3. Record `S*` per pilot = **the first step after which the §3.2 criterion holds at every subsequent evaluation**, or the cap if that never occurs. No window parameter; the transient spike is excluded by construction. **Report all three and the spread.**
4. **Budget for all real runs = `4 × max(S*)`**, hard cap 200k steps. Write the integer into `PREREG.md` before the first real seed. The margin is 4× rather than 3× because the original 3× was calibrated against first-pass variance, which is now known to be the wrong scale; recovery-time variance is observed on only a handful of seeds.

**No early stopping.** Removed deliberately, not omitted. On a procedural generator every held-out item is a fresh graph, so there is no finite-corpus overfitting for early stopping to protect against — its only function here is risk. Coupled to a short budget it is actively dangerous: a 20%-of-budget patience window under a first-pass anchor would have fired mid-collapse and frozen the model at its worst. Fixed budget, gate at the terminal checkpoint.

Three pilots rather than one because a single fast pilot sets a tight ceiling that could fail a real seed on budget rather than on competence — a fundamentally different result.

**Extension rule for parked pilots.** Pilots already trained past first pass are *continued*, never restarted: the only missing information — where a spiking seed re-clears and whether it holds — comes solely from continuing. Continue until the criterion holds on two checks ≥500 steps apart after the first re-clear, or +8,000 additional steps with no re-clear, whichever comes first. **If a pilot never re-clears within that, report it as a finding — spike-seeds do not recover within ~2× their valley length — not as a nuisance.** Verify the LR schedule under continuation matches what real seeds will run; a schedule that decayed for a short run makes the continued trajectory unrepresentative and `S*` untransferable.

If no pilot reaches criterion within its cap, the honest report is that the architecture did not learn the task under this budget. State the budget; do not extend it.

### 3.1.5 Query-projection rank — derived from the pilots, fixed before the real runs

The rank cannot come from §3.3: that runs after training, and training needs a rank. The pilots break the circle, since they already exist and are already discarded.

- **The rank is derived from QUERY PR, not causal PR.** These are different objects — the query projection bottlenecks the state→key-space map, while causal PR describes output sensitivity across the whole state — and pilot data showed them diverging in both magnitude and direction. Sizing a query bottleneck from causal PR silently clips the map it is meant to fit. This was the same conflation §3.3 was corrected for; it must not be reintroduced here.
- **Pilots train with a full-rank query projection** — no bottleneck. A pilot trained at rank 16 cannot exhibit a participation ratio above 16, and would circularly confirm whatever guess was used.
- **Measured at the pilot's terminal checkpoint**, never at first pass (§3.1, §3.3). Pre-transition query PR describes a solution the model subsequently abandons — pilot values of ~46–47 before the transition converged to ~33 after it, a difference of a third in the final rank.
- **`rank = ceil(max over pilots of query PR)`, rounded up to the next multiple of 4.** Applied once, written to `PREREG.md` before the first real seed launches. The number falls out; it is not selected.
- **No iteration.** If the rank later looks wrong, that is a finding to report, not a value to revise.

**Known wrinkle, accepted and reported:** pilots train full-rank while real seeds train at reduced rank, so `S*` is measured under a slightly different model. The 4× multiplier in §3.1 is intended to absorb this. State it in `RESULTS.md` rather than correcting for it.

### 3.1.6 Pilot-stage integrity checks — validate the generator before spending real runs

The pilots are discarded, so running integrity checks on them contaminates nothing and costs almost nothing. Run these on pilot 0 **before** launching any real seed. They validate the *task*, not the model.

- **§4.1 shuffled cache.** If a pilot retains accuracy with another item's cache, the generator leaks and no amount of real-seed training will fix it.
- **Integration count vs. `L`.** Record `P(m | L)` on the pilot. If `L=3` items solve with one integration, the model found a shortcut and is not traversing the chain.
- **Accuracy vs. `L` slope.** Perfectly flat accuracy across chain length is a warning sign, not a success. Genuine multi-hop traversal should cost something at `L=3` relative to `L=1`.
- **END-entry census.** Count `END`-valued entries per item and confirm §1.6.8.

Any failure here is a generator defect. Fix the generator and re-run the pilots; this is not a threshold move and does not consume the §4.4 calibration budget.

### 3.1.7 Rank-bottleneck throwaway pilot — validate the bottleneck before spending real runs

**One discarded pilot, trained with the query projection actually bottlenecked to `rank` (§3.1.5), before any of the six real-seed runs launch.** Same logic as §3.1.6, applied to the model instead of the task: every §3.1-§3.1.6 pilot is mandated full-rank by construction, so the low-rank projection is a component nothing has ever exercised. The first real seed would otherwise be the first time the bottleneck exists at all — an expensive way to discover it doesn't fit.

- Same architecture, same generator config, same cosine schedule as the official pilots, with the query projection fixed at `rank` instead of full-rank.
- Judged against the same §3.2 criterion, at the same terminal-checkpoint standard — does competence survive the bottleneck, not just "does loss go down." **The same trap §3.2 was rewritten to close applies here too: a streak of passing checks mid-run is not the verdict.** Pilots 0 and 1 looked identical to this at step 100, before four thousand steps at 0.55-0.70. Nothing about this pilot is read until it reaches its terminal checkpoint (or the cap) and `S*` is computed from the complete history, same as §3.1's official pilots.
- **Pass:** rank 36 is survivable; proceed to the six-run matrix. **Fails or plateaus below criterion:** that is a finding about the rank itself, to report before committing real-seed budget to it, not a hyperparameter to quietly bump.
- **This pilot is also a budget check, not just a survivability check.** §3.1's `102,400` was derived from full-rank pilots; §3.1's "known wrinkle" only *hopes* the 4× multiplier absorbs the bottleneck's cost, it doesn't verify it. Compute this pilot's `S*` under §3.1's identical stable-competence definition. **If it exceeds 25,600 (the official pilots' `S*_max`), re-derive the budget as `4 × max(25600, rank-pilot S*)` before any real seed launches.** This is the pilot doing its stated job, not a threshold moved after seeing results — and it only ever moves the budget up, never down.
- Discarded regardless of outcome — this is a validation run, not a seed.

### 3.2 The gate

**≥95% exact-match accuracy on arm A, at every chain length 1–3, on held-out items, in all three real seeds.**

**Judged at the terminal checkpoint, and at every evaluation in the final quartile of the budget.** Never at first pass, and no snapshotting of the first passing step. The checkpoint that Phases 2–4 instrument is the budget-exhaustion checkpoint, so that is the one whose competence must be stable. A seed that passes at step 100 and sits at 0.60 at step 300 **fails the gate**, and that is correct — instrumenting it would mean measuring a collapsing network.

**All three must pass.** If 2 of 3 pass, that is the result. Drawing fresh seeds until three pass is seed-shopping and is prohibited.

**A failing seed is reported as budget-limited or competence-limited, not merely as a failure.** Log its full accuracy trajectory and accuracy at budget exhaustion. Still climbing at 94% and plateaued at 60% are different facts. Neither justifies extending that run.

### 3.3 Causal rank verification — one shot, mechanical, dedicated split

Runs **immediately on gate pass**, before any Phase 2–4 work. This **verifies** the rank fixed in §3.1.5; it does not produce it.

- **Checkpoint:** the **terminal** checkpoint only — the same one Phases 2–4 instrument. Never a first-pass checkpoint. Pilot data showed the participation ratios converging across all three seeds *while one was still at 0.88 accuracy*: representational geometry stabilizes before behaviour does, so PR agreement is not evidence of competence and must not be read as a substitute for the §3.2 gate.
- **Data:** the **geometry split**, carved out at generation time, used for nothing else. Not validation (§3.1, §5.3), not calibration (§2.2, §4.4), not test.
- **Quantity:** participation ratio of the gradient-derived causal subspace — unconditional logit Jacobian over the full output vocabulary, no label reference.
- **Per seed:** computed independently on all three passing seeds. Report all three.
- **Verification criterion:** real-seed **query** PR should not exceed the rank fixed in §3.1.5. If it does, the projection is clipping structure the model wanted — **report it as a finding and a bound on §6.1's claims. Do not retrain at a larger rank.**
- **No iteration.** If the number looks wrong, that is a result, not a value to revise.

**Report the seed spread as a standalone result.** If PR varies widely across seeds trained identically to the same accuracy, causal dimensionality is not a stable property of the task — which directly bounds what §6.1 is allowed to claim.

---

## 4. Phase 2 — Integrity checks

All must pass before any coverage or recursion claim.

### 4.1 Leakage — shuffled cache, not zeroed cache

Pair each query with a **different item's** cache. Same distribution, same statistics, zero information about this item's chain.

Do **not** zero the cache; an all-zeros cache is off-distribution and may fail for unrelated reasons, making the test vacuous.

**Chance is computed over the key vocabulary, not `1/|S|³`.** Under END notation the answer is always a cache key, so the output space is the set of keys present in the item's cache. Compute chance accordingly and write the formula into `PREREG.md`.

Above-chance accuracy under shuffled cache halts the run for state leakage.

### 4.2 Counterfactual content flow — three arms, three predictions

| Arm | Intervention | Prediction if the pathway carries content |
|---|---|---|
| **clean** | none | next step retrieves `k2`'s target |
| **wrong-value** | integrate a **valid key** `k2'` that has its own cache entry, equal norm | next step retrieves **`k2'`'s target** — margin stays **high**, target **substitutes** |
| **clamped** | force `g₁ = 0` | next step retrieves nothing coherent, margin collapses |

The signature is **retrieval-target substitution**, not margin collapse.

**The substituted value must itself be a key with an entry.** Substituting a terminal would predict "dead end detected" instead of substitution — a different arm, not a clean intervention.

The clamped arm alone is confounded: it removes content *and* perturbs the trajectory. All three are required.

### 4.3 Retrieval margin dynamic range — scale-free criterion

Raw margin is **not** preregisterable: top-1 minus top-2 scales with the similarity temperature, so bounds on it can be manufactured by rescaling without any behavioural change.

Preregister instead:
- **margin AUROC for predicting top-1 retrieval correctness ≥ 0.70**
- **top-1 retrieval accuracy within [0.60, 0.98]** on arm A

Both scale-invariant. Collapse and saturation appear in these naturally.

### 4.4 Calibration budget — bounded, logged

If §4.3 fails: **at most 3 adjustments**, knobs `n_hard` and `|S|` **only** (not embedding dimension — an architecture change makes attempts incomparable), evaluated on a **calibration split** separate from validation, geometry, and test. **Every attempt logged and reported, including failures.**

Without a budget this is hyperparameter search wearing a preregistration's clothes, and each pass is a peek.

---

## 5. Phase 3 — Dead-end representation (Q2)

### 5.1 Probe positions — two, answering different questions

B2 makes the timing matter. Preregister both:

- **P1 — after integrating `kj`, before the next lookup resolves.** High margin, content committed, dead end not yet proved. Does the state encode that the committed value is not itself a key?
- **P2 — after the next lookup returns low margin.** Dead end proved. Does the state encode terminal status rather than merely reflecting the raw margin?

P2 must be shown to carry information **beyond** the current margin value — regress it out and re-probe. Otherwise the probe has rediscovered the retrieval statistic.

### 5.2 Probe protocol

Linear probe on frozen state, predicting answerable vs. not.

- **Split by generated graph, not by item.** A probe that memorizes graphs is not a coverage probe.
- **Report cross-condition transfer as the headline, exactly two combos, pinned:** train `L=2` → test `L=3`; train `n_distractors=1021` (the target scale) → test `n_distractors=256` (the pinned low band — comfortably above the `n_hard·L + MIN_END_DECOYS` floor at current settings, meaningfully shorter than the target). Not a sweep over bands or `L`-pairs; these two, decided now rather than however many "a few more configurations" turns into later.
- **Layer sweep = every one of the 8 fixed recursion steps (§6.1's `max_steps`), not a separately swept width.** Probed from the same per-step trace a single rollout already produces (`model/raw_records.py`) — this reads existing per-step records at different step indices, it does not require additional rollout passes. Same for §5.1's P1/P2: two positions within one already-logged trace, not two passes. Neither multiplies `raw/*.jsonl` volume; only new arms or new data configs (below) do.
- **Arm D is the discriminating test.** A probe that fires on the length control learned length, and the length-matched arms cannot reveal this.
- **B1-vs-C separation must be shown not to be hops-to-failure** (§1.4).
- **Pass accounting for `raw/*.jsonl` (§7), corrected:** Phase 1 logged arm A only (§3.2's gate is defined on arm A alone). Phase 3 needs all five arms, at all 3 chain lengths, at both `n_distractors` bands — `5 × 3 × 2 = 30` arm/`L`/band combinations, of which 3 (arm A, both existing `L`s... all 3 `L`s, at `n_distractors=1021`) are already logged by Phase 1 and reused. **27 new passes, ~2,000 items each (matching Phase 1's own convention) ≈ 36.5MB new data per run.** Phase 4: 2 arms (constrained/free) × 3 chain lengths = 6 combinations, step budget fixed at 8 per §6.1 — **not** swept, no additional multiplier — ≈ 8.1MB per run. **Corrected total: ≈ 52.7MB per run, ≈ 317MB for all six** — the earlier ~2.8GB projection over-counted by treating the layer sweep and P1/P2 as separate rollout passes rather than reads of data already logged once.

### 5.3 Abstention threshold

Displacement enters as a quantile (§2.2), so τ is a percentile of the frozen calibration distribution, fixed on **validation before any test evaluation**.

**Target: 5 percentage points above a measured floor, not a flat 5%.** The floor is the model's structural false-abstention rate on arm A with no τ involved — the `m=0` "declined to start" rate. Under the unstable constant-LR recipe, pilots showed this at 6.6–7.2%; under the stable cosine-schedule recipe it dropped to 0-0.65%, confirming the floor was a training-instability symptom rather than an architectural ceiling. **That measurement was on full-rank pilots (§3.1.5) — do not treat it as resolved for real seeds.** A rank-36 bottleneck is a component the floor has never been measured against, and could reintroduce it; re-measure on real seeds before trusting the near-zero figure. The floor-relative form of this target costs nothing if the floor stays near zero and catches it immediately if it doesn't. **If the floor dominates the budget, report the floor itself as the finding** rather than presenting a calibrated τ that was never the binding constraint.

**Re-examine displacement distributions under END notation before applying one τ across arms.** Pointer advances are discrete jumps, so "settled" may confound with "stopped" — and a chain that ends and a chain that stalls can both show low displacement. Verify the distributions are on comparable scales across arms and report the check. If they are not, a single τ is not defensible and that must be stated.

### 5.4 Falsification

Q2 fails if cross-condition transfer AUROC for arm A vs. arm B2 at position P2 is below 0.65 in Regime 1.

**A null here is a positive finding**, provided the §1.3 lower bound held: if the state had 16× the capacity it needed for a pointer and still did not retain dead-end status, that is the argument that coverage must be architected in rather than trained for.

**The secondary Q2 result is the gate, not the probe.** If the model abstains off count shortfall alone, with no difference in gate behaviour at the terminal step (§2.4), then the abstention signal is a counter rather than a status. Report that explicitly either way.

---

## 6. Phase 4 — Recursion (Q3)

Only after Phases 1–3.

### 6.1 Constrained vs. free latent iteration

**Step budget: `max_steps = 4 × (L_max − 1)` = 8 at `L_max = 3`.** A formula, not a magic number: it allows every required integration plus three recurse-without-integrate steps per hop. Fixed in `PREREG.md`. If a condition systematically exhausts the budget, report the exhaustion rate rather than raising it. **One value, not a sweep** — §5.2's volume accounting assumes this explicitly; a later temptation to compare a few step-budget values would both move a preregistered threshold and multiply `raw/*.jsonl` volume beyond what's estimated.

Fixed step budget, no learned halt. Compare recursion updates **constrained to the per-item causal subspace** against unconstrained updates.

Motivation: in the prior project, causal-subspace perturbation preserved ~98% of trajectories where isotropic perturbation destroyed ~38%, suggesting the causal subspace behaves like a tangent space to the manifold of valid computations. Drift into degeneracy is the known failure of latent-reasoning approaches.

**This is an untested hypothesis, not a finding.** Report it as such regardless of outcome, and bound the claim by the §3.3 seed spread.

**The causal subspace must be shown to be a property of the learned computation, not of the rank-36 projection alone (`RESULTS.md` finding 10).** Untrained, randomly-initialized models already measure causal PR in the 15-17 range and query PR in the 21-22 range on this architecture — the projection's fixed dimensionality supplies a floor before any training happens. Training moves both quantities up by a further, fairly consistent 5-11 points across every real seed measured so far. §6.1's constrained-vs-free comparison, and any claim built on "the causal subspace," should be read against that floor: the untrained baseline is what a causal-subspace-constrained perturbation would preserve from architecture alone, and the interesting claim is about the *increment* above it, not the raw measured PR. Report both the raw PR and the floor-relative increment when characterizing the causal subspace in Phase 4, not the raw number alone.

### 6.2 Integration count distribution

Record `P(m integrations | generator chain length L)` — **conditioned on the generator's `L`, not on any depth label**, which no longer exists in the input.

This is a **prediction about a learned gate, not an architectural axiom.** **Preregistered expectation:** mode at ρ, with **P(m = ρ) ≥ 0.80 on held-out arm A**. Below that, the gate did not learn the follow-until-END rule and §6.2 reports a gate failure rather than a count finding.

Deviations mean different things and are reported separately:
- **m > ρ** — the gate over-fires; it is drifting toward always-integrate (attention).
- **m < ρ** — the gate skipped a hop. Either the model synthesized it internally, or it shortcut using chain fragments held in state. **These are not distinguishable from the count alone**; the §1.3 arithmetic and a §4.1 shuffled-cache check on those specific items must both be brought to bear before the synthesis interpretation is claimed. If `L=3` chains consistently solve in two integrations, the model may have synthesized a hop internally — but that interpretation must be restated against the §1.3 arithmetic before it is claimed, since a state large enough to hold chain fragments could shortcut without any synthesis. Findings of `m ≠ ρ` are logged, never filtered.

---

## 7. Deliverables

1. `PREREG.md` — every threshold, formula, and computed integer in this document, fixed before the first training run. Never edited.
2. `generator_tests/` — asserting: axis orthogonality, twin identity, follow-path uniqueness, no duplicate keys, no ordinal signal, per-hop hard negatives, B1 decoy legality, B2 terminal distribution, **and both §1.3 capacity bounds**.
3. `run_manifest.json` — seeds, config, git SHA, environment, per run.
4. `raw/*.jsonl` — **scope resolved, was ambiguous.** "Per-item, per-step" originally didn't say which passes. Per-step records *during training* would be enormous (up to 102,400 steps × 64 items × a recursion budget, per run, × six runs) and analytically useless — nobody reads mid-optimization per-step traces at this scale. Resolved: **evaluation and instrumentation passes only** — the §3.2 terminal held-out check, §4.4/§5.3 calibration, and §3.1.5/§3.3 geometry passes. One record per active item per step (gate open/closed, margin, displacement, integration count so far) plus one final record per item (prediction, target, correctness) — `model/raw_records.py`. Measured (not guessed) at ~8.1MB per run, ~48.4MB for all six, **at Phase 1's pass count only.**

   **Phase 3/4 multiply this, by roughly the factors those phases' own designs already name.** Phase 3: 2 probe positions (§5.1) × a layer/recursion-step sweep (§5.2, ~8 points at `L_max=3`'s step budget) × a handful of cross-condition transfer combos (§5.2, train/test `L` and `n_distractors` pairs, ~3-4) ≈ **~48×**. Phase 4: constrained-vs-free (§6.1, 2×) × a small recursion-budget sweep (~4 configs) ≈ **~8×**. Rough total: `8.1 × (1 + 48 + 8) ≈ 462MB per run`, **`~2.8GB for all six`** — plausible, not precise (exact Phase 3/4 pass counts aren't preregistered yet), but the right order of magnitude to size the manifest scheme for now rather than re-discover at Phase 3 launch. Still small enough to stay off git with the same manifest pattern; nothing about the scheme needs to change, only the expectation of what it'll hold.
5. `CALIBRATION_LOG.md` — every §4.4 adjustment, including failures.
6. `RESULTS.md` — verdict against each preregistered condition, written **before** interpretation.

---

## 8. Standing rules

- Report nulls as results. The prior project's most useful output was a preregistered null.
- Nothing in a summary or abstract that was not measured. Distinguish findings from hypotheses by sentence.
- No metric introduced after seeing data. No threshold moved after seeing test data.
- Where a design decision is ambiguous, implement the simplest version, log it in the manifest, flag it in `RESULTS.md`.
- The document is the preregistration. Chat agreements are not protocol until they are prose here.
- One small symbolic model class. Say so every time.