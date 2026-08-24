# AMENDMENTS.md

Per §8: "Where a design decision is ambiguous, implement the simplest version, log it in the manifest, flag it in `RESULTS.md`." This file is the record of every such ambiguity — what was ambiguous, what triggered noticing it, and which simplest-version reading was applied. Kept separate from `RESULTS.md`'s measured findings because these are resolutions to gaps in the preregistration text itself, not results of measurement.

---

## 1. §5.4's 0.65 threshold: raw or margin-residualized AUROC? (2026-08-24)

**The ambiguity.** §5.4 fixes: "Q2 fails if cross-condition transfer AUROC for arm A vs. arm B2 at position P2 is below 0.65 in Regime 1." Written before residualization existed as a measured quantity — §5.1's "regress out margin, re-probe" only became a concrete instruction once the P2 probe was actually implemented (Phase 3 canary run, `model/phase3_probe.py`, 2026-08-24). The text names one AUROC; measurement now produces two, raw and margin-residualized, and only one of them is anywhere near the boundary.

**What triggered noticing it.** The first canary run of the Phase 3 P2 probe (Regime 1, `real_seed_r1_0`) measured combo1's (L=2→3) residualized AUROC at `0.6460` — four thousandths below `0.65`. Raw AUROC for the same combo was `0.9977`, nowhere close. Deciding, in that moment, which reading §5.4 actually meant would have meant interpreting a genuinely ambiguous text right after seeing a number sitting at the boundary of it — the same shape of problem as moving a threshold after seeing test data, even though nothing about the *threshold itself* was in question, only which quantity it applies to.

**Resolution — simplest version, per §8: §5.4 reads on raw AUROC, exactly as written. The margin-residualized number is a separate, complementary condition from §5.1, reported alongside raw always, but does not itself carry a numeric falsification gate.** §5.4's text names one quantity, "cross-condition transfer AUROC," with no mention of residualization — the textually conservative reading applies the threshold to exactly what was written, not to a refinement invented after the fact. §5.1's margin-regression instruction stands on its own terms: report it, always, alongside raw AUROC — it is diagnostic information about how much of P2's separability survives once a raw retrieval statistic (margin) is accounted for, informative regardless of where it falls, but no threshold is retroactively assigned to it here.

**Consequence for the canary result.** Regime 1's raw AUROC (`0.9977` combo1, `0.9987` combo2) is nowhere near `0.65` in either combo — under this resolution, §5.4 is nowhere close to falsifying on this seed. This is not a pass verdict either: no falsification-or-confirmation verdict is drawn from a single seed, full stop — see `RESULTS.md`, and §2.6/§3.2's three-seeds-per-regime commitment, which exists precisely so no one run carries an interpretation on its own.
