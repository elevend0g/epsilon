# SESSION_LOG.md

Chronological record of this session's work on the SSM-driven gated-cache project, from an empty repo through the design document, the generator, the model, and the pilot-training saga currently running in the background. `PREREG.md` holds thresholds, `RESULTS.md` holds methodological findings, `RECIPE_LOG.md` holds training-recipe changes — this file holds the sequence connecting them, for anyone trying to understand how the current state was reached rather than just what it is.

---

## 1. Document: `docs/phase1.md`

**v1 (depth-in-query notation).** Started from an existing draft: query carried `(start_key, depth)`, arms were A / B (terminal) / B2 (missing-entry) / C / D. Several rounds of review and fixes against it: an answer-output-grammar decision (position-wise categoricals), worked capacity-bound arithmetic in place of an asserted claim, a rank-timing circularity in §2.1→§3.3 (the query-projection rank was specified as coming from a measurement that could only run after training needed that same rank — fixed via a pilot-measurement step), four "gate pins" written into §2.2 (percentile inputs, recursion-only displacement, a slot mask, a controller-vs-readout distinction), and a corrected per-arm `ρ` (integration-count target) formula.

**v2 (END-sentinel notation) — full rewrite, superseding v1.** Query carries `start_key` only, no depth token; a chain is `k1→k2→…→kL→END`; the model emits by following the chain and reporting the current key when `END` is retrieved. Rationale logged in the doc itself: depth-in-query lets the count penalty be satisfied by reading and counting down a label rather than by detecting whether the state needs a fact, and it pre-loads the §6.2 result. Arms became A / B1 (missing key, deletion-free — decoy substitution, not removal) / B2 (dead end, the thesis arm) / C / D.

Fixes made to v2 over the course of the session, roughly in order:
- The same rank-timing circularity reappeared (v2 was a fresh rewrite that didn't carry v1's fix forward) — re-fixed the same way, via §3.1.5.
- §3.1.6 added: a single findable `END` entry per item is a one-lookup shortcut around the entire chain at any length. Fixed by requiring `MIN_END_DECOYS = 8` unreachable `END`-valued entries per item, asserted in `generator_tests`.
- §5.3's abstention target corrected from a flat 5% false-abstention rate to 5 points above a *measured floor* — pilot evidence showed the floor itself (the gate declining to open on the first hop, with no threshold involved) can exceed 5%, which no τ choice fixes.
- §3.1.5/§3.3 conflated causal PR (retrieval-logit Jacobian, whole-state output sensitivity) with query PR (the query projection's own output Jacobian) — the rank was being sized from the wrong quantity. Corrected to derive `rank` from query PR specifically; causal PR is still measured (feeds §6.1) but no longer sets or verifies the bottleneck.
- §3.1 rewritten around a redefined `S*`: not first touch, but "the first step after which criterion holds at every subsequent evaluation" — no window-length parameter, so a transient spike can't be mistaken for stable competence. Pilot cap set empirically to 30,000 steps; budget formula changed from `3×max(S*)` to `4×max(S*)` (the original 3× was calibrated against first-pass variance, which turned out to be the wrong scale). Early stopping removed as actively dangerous under a short budget. §3.2 correspondingly judges the *terminal* checkpoint and the final quartile of the budget, never first pass.
- §3.1 gained an optimizer-schedule line once the constant-LR recipe was shown inadequate (see §4 below): cosine `3e-3 → 3e-5`, no warmup, decaying across the full pilot cap.

---

## 2. Generator: `generator/`

Built to v1's spec first (arms A/B/B2/C/D, depth-in-query), then rebuilt for v2 (END-sentinel, arms A/B1/B2/C/D, `walk()`-based mechanical derivation of both the answer and the integration-count target `ρ` from following the item's own chain — not asserted per arm). `generator/capacity.py` added to compute §1.3's two capacity bounds as real integers rather than an asserted paragraph.

One vulnerability found and fixed mid-session: the generator produced exactly one `END`-valued entry per item, making "find the entry whose value is END, emit its key" a valid one-lookup answer at any chain length. A per-step retrieval trace on a trained pilot showed that specific checkpoint had *not* taken the shortcut — but a structural shortcut being unexploited by one model isn't evidence it stays unexploited by the next, so the generator was fixed regardless (`MIN_END_DECOYS = 8`, §3.1.6 above). 31/31 `generator_tests` pass against the current generator.

---

## 3. Model: `model/`

A minimal PyTorch implementation, not a literal Mamba/S4: `embeddings.py` (shared key/value tuple encoder, so a retrieved value embeds identically to how it would as a key — needed for §4.2's counterfactuals later), `ssm.py` (diagonal decay + additive injection, no parallel scan — sequences here are short enough not to need one), `cache_bank.py` (feedforward-embedded content-addressable retrieval, built once per item rather than ingested through the recurrence — the state's job is the pointer trajectory, not memorizing the cache), `gate.py` (quantile-ish soft gate via running batch statistics, a pilot-scope stand-in for the real calibration-split CDF), `task_model.py` (wires it together: teacher-forced training forward pass, autoregressive eval forward pass), `geometry.py` (causal PR and query PR, both via per-item Jacobian → SVD → participation ratio), `data.py` (batching plus a `BatchPrefetcher` that parallelizes the CPU-bound generator across worker processes, since generation — not the GPU step — is the bottleneck at the ~1024-entry target cache size).

---

## 4. The pilot-training saga

Five materially different pilot runs, each superseding the last for a specific, logged reason:

1. **`n_distractors=64`.** Fast, clean pass — and wrong: the state (8,192 bits) turned out larger than the cache (2,412 bits) at that scale, violating §1.3's own upper bound. A model *could* have just memorized the cache rather than doing genuine content-addressable retrieval. Discarded before writing any number into `PREREG.md`.
2. **`n_distractors=1021`** (the real target, ~1024 entries). Ran for 2.5 hours doing nothing: a `ProcessPoolExecutor` parallelizing the generator was created after the model had already touched CUDA, which deadlocks forked worker processes on a futex — silently, no exception. Diagnosed via `ps -eo stat,wchan` showing every process blocked. Fixed with a `spawn` multiprocessing context instead of the default `fork`.
3. **Same config, fork fixed.** All three pilots passed in under 300 steps each — fast enough to be suspicious. §3.1.6's census confirmed the single-`END`-entry vulnerability existed; a per-step trace refuted the specific hypothesis that *this* checkpoint was exploiting it (genuine multi-step traversal, zero early-step resolutions). Generator fixed anyway (`MIN_END_DECOYS=8`); this run's numbers superseded regardless of the refutation.
4. **Post-generator-fix.** Passed again, similarly fast. Asked to report the accuracy-vs-`L` slope and `P(m|L)` as positive evidence rather than "clear" — computing `P(m|L)` correctly (a hardened, discrete integration count instead of a soft sum that can hide partial integrations) surfaced a second, larger bug: `forward_eval_autoregressive` had never actually hardened the gate at inference, despite §2.2 requiring it. Same checkpoint, same held-out batch: 96% soft-gated, 77% hard-gated. Every prior "reached criterion" call in this session had been measured under the wrong evaluation.
5. **Hard-gated, correctly.** Pilots 0/1 passed fast (`S*`=100 each); pilot 2 took 7,600 steps — 76× longer, with a multi-thousand-step stretch where accuracy *fell* before recovering. Forcing pilots 0/1 to continue to 7,600 anyway (rather than trusting their early pass) showed their causal/query PR migrating to match pilot 2's almost exactly — one trajectory at different speeds, not two regimes. Getting there also exposed that §3.1.5 had sized the rank from causal PR, the same causal-vs-query conflation §3.3 was independently found to have — rank corrected from a wrongly-derived 32 (causal PR) to 48 (query PR). Continuing all three to check whether the pass would *hold* found that none of them did — including pilot 2, whose own pass at 7,600 didn't survive to the very next check. All three oscillate 85-97% on a fixed validation set indefinitely, never settling.

**The oscillation turned out to be the training recipe, not the architecture.** Constant LR=3e-3 (plain Adam, no schedule) was the suspect — a disconfirming test (`RECIPE_LOG.md`) continued pilot 2's honest, unstabilized checkpoint with everything else identical except a cosine decay to `3e-5`: oscillation amplitude went from 0.04–0.06 to exactly 0.000, locked to perfect accuracy, tracking the LR cut rather than merely coinciding with it. Recipe bug, confirmed; not an architecture finding.

**Completed** (ran ~6.8 hours, faster than the ~10-hour estimate): all three pilots, from scratch, under the cosine schedule, to the full 30,000-step cap with no early stopping. All three reached genuinely stable competence — criterion holding at every evaluation from `S*` onward, not just at one checkpoint — for the first time this project has produced that. `S* = [25600, 21100, 21600]` → training budget `4×25600=102,400`; query PR `[32.54, 32.53, 31.57]` → rank `36`; causal PR `[30.10, 30.10, 29.17]`, informational. Both PR spreads tightened by roughly an order of magnitude relative to the unstable measurement — the instability wasn't only a competence problem, it was corrupting the geometry measurements too.

---

## 5. What's real right now vs. what's pending

**Real, verified, final, in `PREREG.md`/`docs/phase1.md`:** generator invariants and their tests, both capacity bounds (computed), the gate mechanics and per-arm `ρ` (including the B1/B2 off-by-one), the corrected §3.1.5/§3.3 causal-vs-query PR split, the corrected §5.3 floor-relative abstention target, the cosine LR schedule and the evidence for it — and, as of the pilot re-run completing 2026-08-22, real numbers: `S* = [25600, 21100, 21600]`, training budget **102,400** steps, query PR `[32.54, 32.53, 31.57]` (rank **36**), causal PR `[30.10, 30.10, 29.17]`. All three pilots hold criterion at every evaluation from `S*` through the terminal checkpoint — the first genuinely stable pilot run this project has produced — and both PR measurements are tightly clustered across seeds (spreads under 1, versus 14+ and 9+ respectively on the unstable pre-recipe-fix measurement). `RESULTS.md` finding 4 has the full account of how instability was silently corrupting the geometry measurements too, not just the competence numbers.

**Repo pushed to GitHub, public: `https://github.com/elevend0g/epsilon`.** `runs/` (checkpoints, per-step logs) stays gitignored — regenerable from the scripts and the seeds already logged in `PREREG.md`; a second commit with final pilot artifacts is a reasonable next step now that a stable run exists to commit.

**Not started**: real-seed training (needs the low-rank query projection added to the model — pilots only ever run full-rank), Phase 2 integrity checks, Phase 3 coverage probes, Phase 4 recursion.
