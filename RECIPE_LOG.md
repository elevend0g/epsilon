# RECIPE_LOG.md

Training-recipe changes, logged separately from `CALIBRATION_LOG.md` (§4.4's calibration budget covers task knobs — `n_hard`, `|S|` — not the training recipe itself). Same transparency requirement, different category: a recipe change alters what "the model" means; a calibration adjustment alters what the task looks like to a fixed model.

---

## 1. Constant LR (3e-3, plain Adam) — oscillation discovered, decay under test

**2026-08-22.** All three pilots, extended to step 8,100 under the "onset of stable competence" S* definition, failed to hold §3.2 criterion at every subsequent evaluation — including pilot 2, whose original pass at step 7,600 (L2=0.992, L3=0.961) did not survive to the very next check (step 7,700: L2=0.93, L3=0.883). All three oscillate in roughly an 85-97% band on the *same fixed validation set* at every check, so this is the model's own behavior changing checkpoint to checkpoint, not evaluation-sample noise.

**Suspected cause:** `pilot_train.py` trains with plain Adam at a constant LR=3e-3, no decay schedule — a "simplest version" choice (§8), logged here as demonstrably inadequate rather than silently changed. Batch size (64) checked and ruled out as the obvious confound before touching anything — not a tiny-batch bounce scenario.

**Discriminator test (`model/lr_discriminator.py`), not yet a fix:** pilot 2's honest 8,100-step checkpoint (not its peak), continued with everything else identical except a cosine LR decay, 3e-3 → 3e-5 over 2,000 steps, held at the floor for 500 more. Batch size deliberately untouched, so this isolates one variable.

- **Damping as LR drops** → recipe confirmed. Next: write the final schedule (shape, start, floor, warmup) here as the fixed recipe, re-run all three pilots under it to stable hold, set S* from that run.
- **Same amplitude at floor LR** → architecture: the driver doesn't settle under any reasonable recipe. That is a Phase 1 finding, not a bug to iron out further.
- **Damps but plateaus below 95%** (e.g., 93-94%) → informative middle: marginal L=3 capacity, not instability. Addressed via §4.4's task knobs (`n_hard`, `|S|`), not a recipe or architecture change.

**Result: recipe confirmed.** `model/lr_discriminator.py`, pilot 2 continued from its honest step-8,100 checkpoint, cosine 3e-3→3e-5 over 2,000 steps then held 500 more:

- Early-third amplitude (L2/L3): 0.039 / 0.055 — matches the oscillation already observed.
- Late-third amplitude: **0.000 / 0.000** — locked to 1.0/1.0/1.0 for 9 consecutive checks (900 steps) as LR approached the floor.
- The lock-in (first of many consecutive perfect checks) occurred around step 9,300–9,700, while LR was still descending (1.06e-3 → 1.92e-4) — damping visibly tracked the schedule, not a coincidence of timing.

Damping that tracks the LR cut is the recipe signature, not the architecture signature. Constant 3e-3 was the bug; the SSM driver settles fine once given the chance to.

**Final schedule, written into `PREREG.md` and `docs/phase1.md` §3.1**: cosine, `3e-3 → 3e-5`, no warmup, decaying across the full pilot cap (30,000 steps) rather than a fixed window — the discriminator's 2,000-step window was sized for a test starting from an already-far-along checkpoint, not for a from-scratch run. No warmup because the original constant-LR runs showed fast, stable early progress (near-0% to ~90%+ within 100-300 steps) starting cold at 3e-3; nothing in the failure mode implicated the start of training.

**Next**: all three pilots re-run from scratch under this schedule, to the full cap, `S*` computed from that run's history — not deformed from the discriminator's continuation, which answers "does decay change the picture" and nothing about what `S*` actually is.

**Re-run complete, 2026-08-22.** All three pilots trained from scratch to the full 30,000-step cap under the cosine schedule, no early stopping. Result — a genuinely different shape from every prior attempt:

| Pilot | `S*` (onset of stable competence) | causal PR | query PR |
|---|---|---|---|
| 0 | 25,600 | 30.10 | 32.54 |
| 1 | 21,100 | 30.10 | 32.53 |
| 2 | 21,600 | 29.17 | 31.57 |

All three genuinely stable — criterion holds at every evaluation from `S*` through the terminal checkpoint, the first time this project has produced that. `S*` spread (4,500, ~20% relative) is real but far tighter than the 76× seen under constant LR. Both PR measurements cluster within ~3% of their mean across all three seeds — a striking contrast with the pre-fix measurement's wide, direction-flipping spread (§3.1.5 in `PREREG.md`). Confirms the diagnosis directly: the earlier instability wasn't only a competence problem, it was corrupting the geometry measurement too. `S*=25,600` → training budget `4×25,600=102,400`; `max(query PR)=32.54` → rank `36`. Both written into `PREREG.md` as final.
