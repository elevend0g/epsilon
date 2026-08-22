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

**Consequence:** rank corrected from a wrongly-derived 32 (causal PR) to 48 (query PR, the right quantity). Not a refinement — a different number from a different measurement.

### 3. First-hop reliability floor may exceed §5.3's original target (pre-flagged, not yet a real-seed finding)

Pilot 2, hard-gated: 6.6-7.2% of arm-A items never open the gate at all (`m=0`), on items that are answerable with no ambiguity. §5.3 originally set a flat 5% false-abstention target with no accounting for a reliability floor independent of calibration. Restated in `docs/phase1.md` §5.3 before real seeds run: target is now 5 points *above a measured floor*, not an absolute ceiling, and the floor itself must be reported as a finding if it turns out to dominate.

---

## Verdict against preregistered conditions

Not yet — no real seed has trained.
