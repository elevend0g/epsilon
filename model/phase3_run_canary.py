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


def report_transfer(name: str, train_pos: dict, test_pos: dict, position: str, with_margin_regression: bool) -> None:
    """with_margin_regression: only meaningful for pos2, where §5.1 defines
    "the current margin value" precisely (the margin of the failing lookup
    itself). Pos1 has no analogous well-defined single margin -- the last
    successful retrieval's margin isn't captured, and reusing pos2's margin
    as a stand-in would residualize against the wrong quantity, so pos1 is
    reported as raw AUROC only, not silently given a misleading number."""
    x_key = position  # "pos1" or "pos2"
    train_x, train_y = train_pos[x_key], train_pos["label"]
    test_x, test_y = test_pos[x_key], test_pos["label"]

    probe = fit_linear_probe(train_x, train_y)
    train_score = probe_score(probe, train_x)
    test_score = probe_score(probe, test_x)

    train_auroc = auroc(train_score, train_y)
    raw_auroc = auroc(test_score, test_y)

    if with_margin_regression:
        train_margin, test_margin = train_pos["pos2_margin"], test_pos["pos2_margin"]
        test_resid = residualize(test_score, test_margin)
        resid_auroc, resid_lo, resid_hi = bootstrap_auroc_ci(test_resid, test_y)
        margin_only_auroc = auroc(test_margin, test_y)
        print(f"{name} [{position}]: train_AUROC={train_auroc:.4f}  "
              f"test_raw_AUROC={raw_auroc:.4f}  test_margin_only_AUROC={margin_only_auroc:.4f}  "
              f"test_residualized_AUROC={resid_auroc:.4f}  [95% CI {resid_lo:.4f}-{resid_hi:.4f}]", flush=True)
    else:
        print(f"{name} [{position}]: train_AUROC={train_auroc:.4f}  "
              f"test_raw_AUROC={raw_auroc:.4f}  (no margin regression -- no well-defined "
              f"single 'current margin' at this position)", flush=True)
    return probe


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
    captured = {}
    for key, batch in combos.items():
        arm = key[0]
        pos1_chunks, pos2_chunks, margin_chunks, label_chunks = [], [], [], []
        for start in range(0, N_PER_COMBO, CHUNK):
            end = min(start + CHUNK, N_PER_COMBO)
            chunk = {k: (v[start:end] if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = capture_pos1_pos2(model, chunk, arm)
            pos1_chunks.append(out["pos1"])
            pos2_chunks.append(out["pos2"])
            margin_chunks.append(out["pos2_margin"])
            label_chunks.append(out["label"])
        captured[key] = {
            "pos1": torch.cat(pos1_chunks), "pos2": torch.cat(pos2_chunks),
            "pos2_margin": torch.cat(margin_chunks), "label": torch.cat(label_chunks),
        }
    print("done capturing.", flush=True)

    def combine(*keys):
        return {
            "pos1": torch.cat([captured[k]["pos1"] for k in keys]),
            "pos2": torch.cat([captured[k]["pos2"] for k in keys]),
            "pos2_margin": torch.cat([captured[k]["pos2_margin"] for k in keys]),
            "label": torch.cat([captured[k]["label"] for k in keys]),
        }

    train = combine(("A", 1021, 2), ("B2", 1021, 2))
    test_L3 = combine(("A", 1021, 3), ("B2", 1021, 3))
    test_256 = combine(("A", 256, 2), ("B2", 256, 2))

    print("\n=== P2 (after the next lookup returns low margin) ===", flush=True)
    p2_probe_L3 = report_transfer("combo1 (L=2->3)", train, test_L3, "pos2", with_margin_regression=True)
    p2_probe_256 = report_transfer("combo2 (1021->256)", train, test_256, "pos2", with_margin_regression=True)

    print("\n=== P1 (after integrating kj, before the next lookup resolves) ===", flush=True)
    report_transfer("combo1 (L=2->3)", train, test_L3, "pos1", with_margin_regression=False)
    report_transfer("combo2 (1021->256)", train, test_256, "pos1", with_margin_regression=False)

    print("\n=== Arm D disqualifier (probe trained on A vs B2, applied to D) ===", flush=True)
    d_data = captured[("D", 1021, 2)]
    d_score = probe_score(p2_probe_L3, d_data["pos2"])
    d_pred_answerable = (torch.sigmoid(d_score) > 0.5)
    d_frac_correct = d_pred_answerable.float().mean().item()
    print(f"P2 probe (trained on A vs B2, combo1's train set) applied to arm D: "
          f"predicted 'answerable' on {d_frac_correct:.4f} of D items (should be ~1.0 if the "
          f"probe learned semantic content; a value near 0 would mean it learned token count instead, "
          f"since D matches C's token count while being semantically answerable like A)", flush=True)


if __name__ == "__main__":
    main()
