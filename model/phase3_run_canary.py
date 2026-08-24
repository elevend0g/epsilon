"""Phase 3 canary: one Regime 1 checkpoint (real_seed_r1_0), end to end
-- probe corpus, state capture, P2 probe with margin regression, P1,
arm D disqualifier. Confirms the pipeline produces interpretable
numbers before spending the other five checkpoints (same staging logic
as the real-seed canary run, §2.6).

Scope, deliberately narrower than the full 30-combo sweep: arm A and
B2 at the training config (n_distractors=1021, L=2), both cross-
condition test configs (L=3 same band; n_distractors=256 same L), and
arm D at the training config for the disqualifier. B1/C and the L=1
condition are part of the full sweep, not this validation pass.
"""

from __future__ import annotations

import torch

from model.phase3_probe import (
    N_PER_COMBO, auroc, bootstrap_auroc_ci, build_combo, capture_pos1_pos2, fit_linear_probe,
    load_model, probe_score, residualize,
)

CHECKPOINT = "runs/real_seed_r1_0.pt"


def report_transfer(name: str, train_pos: dict, test_pos: dict, position: str,
                     regress_key: str | None, regress_label: str) -> dict:
    """regress_key: the covariate to residualize against (e.g. "pos2_margin"
    for P2, "pos1_integration_count" for P1), or None to report raw only.
    Each position gets the covariate that is actually well-defined for it --
    P2 has "the current margin value" (§5.1); P1 has no single well-defined
    margin (the last successful retrieval's margin isn't captured), but does
    have a well-defined integration count (§2.5's rho), which is the leading
    shortcut candidate for P1 specifically."""
    x_key = position  # "pos1" or "pos2"
    train_x, train_y = train_pos[x_key], train_pos["label"]
    test_x, test_y = test_pos[x_key], test_pos["label"]

    probe = fit_linear_probe(train_x, train_y)
    train_score = probe_score(probe, train_x)
    test_score = probe_score(probe, test_x)

    train_auroc = auroc(train_score, train_y)
    raw_auroc = auroc(test_score, test_y)

    result = {"train_auroc": train_auroc, "raw_auroc": raw_auroc, "probe": probe}

    if regress_key is not None:
        test_cov = test_pos[regress_key].float()
        test_resid = residualize(test_score, test_cov)
        resid_auroc, resid_lo, resid_hi = bootstrap_auroc_ci(test_resid, test_y)
        cov_only_auroc = auroc(test_cov, test_y)
        print(f"{name} [{position}]: train_AUROC={train_auroc:.4f}  "
              f"test_raw_AUROC={raw_auroc:.4f}  test_{regress_label}_only_AUROC={cov_only_auroc:.4f}  "
              f"test_{regress_label}_residualized_AUROC={resid_auroc:.4f}  [95% CI {resid_lo:.4f}-{resid_hi:.4f}]",
              flush=True)
        result.update({"resid_auroc": resid_auroc, "resid_ci": (resid_lo, resid_hi), "cov_only_auroc": cov_only_auroc})
    else:
        print(f"{name} [{position}]: train_AUROC={train_auroc:.4f}  "
              f"test_raw_AUROC={raw_auroc:.4f}  (no regression covariate available at this position)", flush=True)
    return result


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT, device)
    print(f"loaded {CHECKPOINT}", flush=True)

    print("building probe corpus (7 combos, N={} each)...".format(N_PER_COMBO), flush=True)
    combos = {
        ("A", 1021, 2): build_combo(1021, 2, "A", device=device),
        ("B2", 1021, 2): build_combo(1021, 2, "B2", device=device),
        ("A", 1021, 3): build_combo(1021, 3, "A", device=device),
        ("B2", 1021, 3): build_combo(1021, 3, "B2", device=device),
        ("A", 256, 2): build_combo(256, 2, "A", device=device),
        ("B2", 256, 2): build_combo(256, 2, "B2", device=device),
        ("D", 1021, 2): build_combo(1021, 2, "D", device=device),
    }

    print("capturing pos1/pos2 states (one instrumentation pass per combo)...", flush=True)
    CHUNK = 512
    FIELDS = ["pos1", "pos2", "pos2_margin", "pos1_integration_count", "pos2_is_end", "label"]
    captured = {}
    for key, batch in combos.items():
        arm = key[0]
        chunks = {f: [] for f in FIELDS}
        for start in range(0, N_PER_COMBO, CHUNK):
            end = min(start + CHUNK, N_PER_COMBO)
            chunk = {k: (v[start:end] if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = capture_pos1_pos2(model, chunk, arm)
            for f in FIELDS:
                chunks[f].append(out[f])
        captured[key] = {f: torch.cat(chunks[f]) for f in FIELDS}
    print("done capturing.", flush=True)

    def combine(*keys):
        return {f: torch.cat([captured[k][f] for k in keys]) for f in FIELDS}

    train = combine(("A", 1021, 2), ("B2", 1021, 2))
    test_L3 = combine(("A", 1021, 3), ("B2", 1021, 3))
    test_256 = combine(("A", 256, 2), ("B2", 256, 2))

    print("\n=== P2 (after the next lookup returns low margin) ===", flush=True)
    p2_L3 = report_transfer("combo1 (L=2->3)", train, test_L3, "pos2", "pos2_margin", "margin")
    p2_256 = report_transfer("combo2 (1021->256)", train, test_256, "pos2", "pos2_margin", "margin")
    print(f"P2 residualized spread: combo1(L transfer)={p2_L3['resid_auroc']:.4f}  "
          f"combo2(band transfer)={p2_256['resid_auroc']:.4f}  -- "
          f"{'L transfer is lower, same direction as P1' if p2_L3['resid_auroc'] < p2_256['resid_auroc'] else 'L transfer is NOT lower -- does not match P1 pattern'}",
          flush=True)

    print("\n=== P1 (after integrating kj, before the next lookup resolves) ===", flush=True)
    p1_L3 = report_transfer("combo1 (L=2->3)", train, test_L3, "pos1", "pos1_integration_count", "count")
    p1_256 = report_transfer("combo2 (1021->256)", train, test_256, "pos1", "pos1_integration_count", "count")

    print("\n=== Arm D disqualifier (probe trained on A vs B2, applied to D) ===", flush=True)
    d_data = captured[("D", 1021, 2)]
    d_score = probe_score(p2_L3["probe"], d_data["pos2"])
    d_pred_answerable = (torch.sigmoid(d_score) > 0.5)
    d_frac_correct = d_pred_answerable.float().mean().item()
    print(f"P2 probe (trained on A vs B2, combo1's train set) applied to arm D: "
          f"predicted 'answerable' on {d_frac_correct:.4f} of D items (should be ~1.0 if the "
          f"probe learned semantic content; a value near 0 would mean it learned token count instead, "
          f"since D matches C's token count while being semantically answerable like A)", flush=True)


if __name__ == "__main__":
    main()
