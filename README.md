# epsilon

An SSM-driven gated-cache architecture with latent recursion, tested against a preregistered protocol. **Project closed 2026-08-24 — no further measurement planned.** This file is a self-contained summary; the full record is in the documents linked below.

## What this is

A small (`d_model=64, d_state=8`, rank-36 query bottleneck) diagonal-decay SSM that queries a content-addressable cache unconditionally at every step but gates *integration* — writing a retrieved value into state — on two measured quantities (retrieval margin, causal displacement), never on a learned prediction about its own epistemic state. Trained on a synthetic multi-hop key-chain task with an explicit END sentinel and five arms (A: clean chain, B1: missing-key dead end, B2: fake-terminal dead end — the thesis arm, C: immediate dead end, D: length-matched clean control).

Three preregistered questions, gated in strict order:
- **Q1 — competence** (§3): does the model reliably follow chains of length 1–3?
- **Q2 — coverage** (§5): does the recurrent state represent "this is a dead end" separably from raw retrieval confidence, in a network never trained to report it?
- **Q3 — recursion** (§6): does constraining latent-iteration updates to a per-item causal subspace preserve computation better than an equal-rank unconstrained update?

## Documents

- **`docs/phase1.md`** — the locked preregistration. Never edited after the fact; all changes are logged separately.
- **`PREREG.md`** — every threshold, formula, and computed constant, each marked FIXED (with the measurement) or, if never resolved, said so plainly.
- **`RESULTS.md`** — the findings log, numbered chronologically, nulls and bugs included alongside positive results. Ends in a closing verdict section.
- **`AMENDMENTS.md`** — every place preregistered text was ambiguous, a design choice was made to resolve it, and the choice was logged *before* being measured against.
- **`SESSION_LOG.md`**, **`RECIPE_LOG.md`** — process narrative: how the document and training recipe reached their final form.
- **`generator/`**, **`model/`**, **`generator_tests/`**, **`model_tests/`** — the actual implementation and its test suite (39 tests, passing).

## Final status

**Q1 (competence): PASS, unconditionally.** All six real seeds (2 supervision regimes × 3 seeds), full commitment, held-out accuracy criterion met and stable from convergence through the terminal checkpoint.

**Q2 (coverage): no verdict — deliberately.** Q2's formal falsification threshold is never triggered. Three independent methods ran to completion across all six seeds:
- A **direct gate-behavior measurement** is a solid positive finding: the model's abstention signal is a counter (integration count falling short), not a status representation — demonstrated directly, not probe-mediated, not subject to any capacity or instrument caveat.
- A **linear probe** and a **trained abstention head** both return a real, complete, six-seed-consistent null — but it rests on an instrument-sensitivity question a flawed positive control failed to close, and on a capacity bound that doesn't clear its own preregistered floor once measured correctly. The null is real; the strongest available reading of it is not earned.
- The larger sweep that would have extended this comparison across all five arms was never run — held pending review, closed out unrun.

**Q3 (recursion): not confirmed, as measured.** The integration-count gate passes its preregistered threshold, but the threshold turns out to be one-directional — it can't distinguish the intended policy from a simpler one the model actually appears to run. The causal-subspace-vs-random-subspace comparison returns a clean, well-resolved null (indistinguishable to within ≈0.4 points against a ≈3-point destructive range) — but only in the two data cells where the underlying phenomenon (recursion without integration) occurs at all, and both are first-query situations structurally incapable of exercising the multi-step drift the hypothesis is actually about.

**Also found, not just unmeasured:** the gate's specified calibration mechanism (a frozen, empirical-CDF quantile transform fit on a dedicated calibration split) was never implemented for the whole project — the actual trained gate uses `BatchNorm1d` throughout, all six seeds, discovered only when checked directly. Computing the specified quantities post-hoc, on the existing checkpoints, produced a degenerate abstention threshold (an artifact of how the state is initialized, not a usable boundary) — reported as found, not patched into something more convenient.

## What's genuinely unresolved

- The 30-combo Q2 sweep (5 arms × 3 chain lengths × 2 distractor bands) — never executed.
- The post-hoc abstention threshold's degeneracy — whether excluding each item's structurally-zero first step would produce a usable value was deliberately left open, not decided.
- The gate's displacement input was never projected onto the causal subspace, as specified — a separate, still-unaddressed gap from the calibration-CDF one above.
- A "branching generator," referenced in this project's own closing working notes but never specified anywhere in the preregistration or findings — unstarted, uncharacterized.

## Why this record looks the way it does

Every number in `RESULTS.md` that looked clean, convenient, or too good was checked before it was trusted — that habit is what caught a gate-hardening bug that had silently corrupted every "reached criterion" decision, a causal/query-PR conflation that had sized the model's own capacity from the wrong measurement, a preregistered criterion that turned out arithmetically unsatisfiable, a positive control that was circular by construction, a specification that had never actually been implemented, and a preregistered threshold that computes to a structurally degenerate value the moment it's actually calculated. None of those were found by looking for problems — they were found by not stopping at the first number that looked like an answer.
