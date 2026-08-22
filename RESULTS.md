# RESULTS.md

Per §7: verdict against each preregistered condition, written before interpretation. That table doesn't exist yet — no real seed has trained. This file currently holds only the methodological findings from Phase 1 pilots that belong here rather than buried in `PREREG.md`'s incident log, because of what kind of failures they were, not just that they happened.

---

## Methodological findings, logged as they occurred (pre-real-seed)

### 1. Gate-hardening bug (2026-08-21) — corrupted a decision, not a measurement

§2.2 specifies the gate is "trained soft, hardened at inference." The evaluation path (`forward_eval_autoregressive`) never did this — it used the soft gate value throughout, including at the §3.2 competence-gate check. This is a different class of bug from the others found this session (a shrunk cache, a multiprocessing deadlock, a structural shortcut): those corrupted *numbers*. This one corrupted a *decision* — whether a pilot counted as "reached criterion" at all, which everything downstream (S*, the training budget, which checkpoints the causal/query PR were even measured on) was built on top of.

**Magnitude, same checkpoint and held-out batch:** soft-gated accuracy 96%, hard-gated accuracy 77%, at `L=3`. Not a rounding difference — a checkpoint that looked like it had passed had actually failed by 19 points.

**It surfaced only because a stronger check was insisted on.** The immediate task was reporting the accuracy-vs-`L` slope and `P(m|L)` as evidence, not "clear." Computing `P(m|L)` correctly required hardening the gate to get a discrete integration count instead of a soft sum that could hide partial integrations. The bug was found as a side effect of building a more rigorous diagnostic, not by looking for it directly — worth recording as a case for building the stronger check even when the weaker one already reads "pass."

**Consequence:** every "reached criterion" determination in every pilot run this session, before the fix, was measured under the wrong evaluation. All were re-run under the corrected one; only the hard-gated numbers appear in `PREREG.md`.

### 2. Causal-PR / query-PR conflation, in both directions (2026-08-21)

§3.3's original verification criterion compared causal PR (retrieval-logit Jacobian, whole-state output sensitivity) against `rank` (a bound on the query projection specifically) — different objects, confirmed empirically to diverge (pilot 0: causal PR 19.1, query PR 46.3, same checkpoint). Fixing §3.3 exposed that §3.1.5 — which sets `rank` in the first place — had made the identical error in the opposite direction: sizing the rank *from* causal PR, on the stated intent of "measure the dimensionality this mechanism needs and give it that many dimensions," when causal PR was never that quantity for the query projection specifically.

**Both were written by the same process, in the same document, before either was caught.** The self-consistency check that would have caught this earlier — do the two PR measurements on the same pilot actually agree in direction? — wasn't run until asked for. On pilot 2 (the slow, unstable one) they don't just disagree in magnitude, they invert: highest causal PR, lowest query PR of the three pilots. A model that used *more* of its output-sensitive dimensionality used a *narrower* query to do it.

**Consequence:** rank corrected from a wrongly-derived 32 (causal PR) to 48 (query PR, the right quantity) — itself later superseded again (see finding 4) once it turned out none of the checkpoints this was measured on were stable. Final rank: 36. Not a refinement at any step — a different number from a different measurement, twice.

### 3. First-hop reliability floor may exceed §5.3's original target (pre-flagged, not yet a real-seed finding)

Pilot 2, hard-gated, **under the unstable constant-LR recipe**: 6.6-7.2% of arm-A items never open the gate at all (`m=0`), on items that are answerable with no ambiguity. §5.3 originally set a flat 5% false-abstention target with no accounting for a reliability floor independent of calibration. Restated in `docs/phase1.md` §5.3 before real seeds run: target is now 5 points *above a measured floor*, not an absolute ceiling, and the floor itself must be reported as a finding if it turns out to dominate.

**Re-checked against the stable, cosine-schedule pilots (2026-08-22): the floor mostly evaporated.** `m=0` rate at `L=2`/`L=3`: pilot 0, 0.6%/0.5%; pilot 1, 0%/0.1%; pilot 2, 0.35%/0.65%. All comfortably under the original flat 5% target — the 6.6-7.2% figure was itself a symptom of the training instability (finding 4), not an architectural ceiling on first-hop reliability. The floor-relative restatement in §5.3 stays as the more defensible general principle (it would have caught this instead of just describing it after the fact), but the specific number it was written in response to no longer applies to the stable recipe.

### 4. Constant-LR oscillation — none of three pilots ever stably passed (2026-08-22)

Extending pilots past their first criterion pass, to test the causal/query-PR migration in finding 2, surfaced something the migration question didn't anticipate: under `S*`'s corrected definition ("holds at every subsequent evaluation"), **none of the three pilots ever qualified** — including pilot 2, whose own pass at step 7,600 didn't survive to the next check (step 7,700: L2 0.93→dropped, L3 0.96→0.88). All three oscillated 85-97% on a *fixed* validation set indefinitely (same 128 items every check, so the oscillation is the model's own behavior changing, not sampling noise).

**Disconfirming test, not a fix applied on faith:** continued pilot 2's honest (non-peak) checkpoint with everything unchanged except a cosine LR decay, `3e-3 → 3e-5`. Two named predictions before running it — damping tracks the LR cut (recipe) vs. oscillation persists at the floor (architecture). Result: amplitude `0.04-0.06 → 0.000`, locked to perfect accuracy, damping visibly tracking the schedule rather than merely coinciding with training running longer. Recipe confirmed (`RECIPE_LOG.md` has the full test and numbers).

**Batch size (64) checked and ruled out before touching anything** — not the tiny-batch bounce scenario — so the test isolated one variable.

**Consequence, and why this is the second time "measure it again on a competent model" mattered more than the number it replaced:** every S*/rank/budget number in this document before 2026-08-22 was measured on checkpoints now known not to represent stable competence. The re-run under the corrected schedule produced the first genuinely stable pilots this project has had — criterion holding at every evaluation from `S*` to the terminal checkpoint — and both PR measurements *tightened* by an order of magnitude in cross-seed spread (query PR: 14.3 → 0.97; causal PR: 9.9 → 0.93) compared to the unstable measurement. The earlier instability wasn't only a competence problem; it was corrupting the geometry measurement §3.1.5/§3.3 depend on, silently, in a way that produced numbers confident enough to nearly get written into `PREREG.md` as final twice.

### 5. The instability was corrupting the geometry, not just the accuracy — the fourth instance of this pattern

Worth its own line rather than a closing sentence on finding 4, because it's a recurring failure mode across this project, not a one-off: **a measurement kept turning out to be of something other than what it named.** The prior project's original participation-ratio estimate (~15) was taken at chance accuracy — the reason "measure again on a competent model" was this project's founding instinct rather than an afterthought. This session found the same shape three more times: causal PR measured and reported as if it were query PR, and the reverse (finding 2); "reached criterion" measured under a soft gate when §2.2 specifies a hardened one (finding 1); and now, `S*`, causal PR, and query PR all measured on a network that hadn't finished reorganizing itself.

The numbers make this one unambiguous rather than merely suspected: query PR's cross-seed spread went from 14.3 to 0.97, causal PR's from 9.9 to 0.93 — roughly an order of magnitude in both, purely from letting the network reach a stable solution before measuring it. Both `S*` and `rank` — the two quantities the entire real-seed launch depends on — were being read off checkpoints mid-reorganization, twice, before this was caught. Neither prior measurement was a rough approximation of the right answer; both were confident, precise readings of the wrong thing.

**Resolved, not open anymore:** §3.1.7's throwaway pilot ran to completion — `S*=15,900`, well below the official pilots' 25,600, so the training budget (102,400) stands unchanged. Competence survives the rank-36 bottleneck.

### 6. Generator speedup: two working fixes, one that looked right and thrashed anyway

Profiling found the generator's ~1,370ms/batch dominated by `random.randrange()` overhead (58%) and `_generate_arm_c` rebuilding an independent ~1,020-item pool the item already had a perfectly good one for (28%). First fix attempt (buffer symbol draws through numpy) made things *slower* (2,910ms/batch) — numpy's speed is in vectorized bulk operations, and per-element access from a Python loop pays numpy's C→Python boxing on top of the Python-loop overhead it was meant to remove. Root cause correctly diagnosed, reverted, tried again: bulk-generate through numpy, convert the *entire* buffer to a plain Python list with one `.tolist()` call, then consume via ordinary list indexing — no numpy touched after the refill. That version worked, ~3.7x faster at the primitive level.

Integrating it back into the full pipeline surfaced a second bug the isolated benchmark couldn't show: the same buffered RNG serves both symbol draws (alphabet size 64) and position selection in `_flip_positions` (alphabet size 3, added while fixing that function's own `random.sample()` cost), and a single buffer keyed to "the last alphabet size used" thrashed — invalidating and refilling on every alternation between the two call sites, worse than either `random.randrange()` or `random.sample()` alone (2,758ms/batch, the worst of any version tried). Keying the buffer by alphabet size instead of holding one shared buffer fixed it. Final, clean, uncontended measurement: **311ms/batch, a 4.4x improvement** — verified only after specifically re-testing end-to-end in the full pipeline, not trusting the isolated micro-benchmark that looked clean.

Three consecutive "should be faster" changes, one net regression caught and reverted, one net regression caught and fixed in place, before landing on a real 4.4x. Worth its own line for the same reason finding 4 was: an isolated benchmark reading "faster" does not mean the change is faster where it actually runs.

**This is the fifth instance of the pattern finding 5 named, and the clearest one yet.** The `_flip_positions` fix was verified correct on its own — the isolated microbenchmark for bulk-generate-plus-`.tolist()` genuinely showed ~3.7x. It was the *integrated* path, two call sites sharing one buffer keyed to a single alphabet size, that did something the component-level test had no way to show. Same shape as the gate-hardening bug (finding 1): a piece verified correct in isolation while the assembled system quietly did something else. The general lesson, stated plainly so it doesn't need re-deriving next time: **never accept a component-level speedup — or any component-level result — without re-measuring the whole pipeline it feeds.** A clean microbenchmark is evidence about the microbenchmark, not about the system.

### 7. Canary run clears, but S* came in 50% higher than the throwaway pilot predicted (2026-08-22)

The first real seed (Regime 1, arms A+D, rank=36, full 102,400-step budget) is the first genuinely apples-to-apples test of `S*=15,900` from §3.1.7's throwaway pilot — same rank, same criterion, but a real seed instead of a discarded one, and 7x the step budget available to reach it. It landed at **`S*=23,800`**, not near 15,900.

This is not a repeat of findings 1-6 — nothing was measured wrong. Both numbers are honest onset-of-stable-competence reads on their own runs; they just disagree, by about 50%, on a single seed each. The gap is worth naming rather than letting `15,900` stand unqualified as "the" number, because it's exactly the kind of single-seed variance §2.6's move to 3 seeds per regime exists to characterize. `S*` still leaves more than 3x headroom before the 102,400 cap either way, so nothing about the budget itself is threatened — but treat `15,900` as what it always was, a single discarded pilot's number, not a prediction of where real seeds will land.

Query PR came in closer: `26.99` against the pilot's `27.99`, both comfortably below `rank=36` — the bottleneck-headroom finding replicated even though `S*` didn't.

---

## Verdict against preregistered conditions

Not yet — one real seed (real_seed_r1_0, Regime 1) has trained and passed its canary check. The five-run remainder of the matrix has not launched.
