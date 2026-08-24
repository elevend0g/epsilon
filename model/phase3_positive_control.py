"""Post-hoc methodological control (§8 simplest-version rule; explicitly
NOT an arm, NOT a seed, NEVER used in any Q2 result -- see AMENDMENTS.md
for the logged implementation choice and reasoning).

§2.1: "A tempting response is to build an explicit register holding hop
count, last margin, and value-validity, and point the probe at it. Do
not... a probe that finds coverage in a channel built to carry coverage
has measured the architect, not the model." This control deliberately
builds exactly that forbidden channel and points the identical probe
pipeline at it, on an already-trained checkpoint (real_seed_r1_0, no
new training) -- reads cleanly means the P2 null is about the model;
reads at chance would mean the null is partly about the instrument.

Implementation choice, logged: the register is appended as extra probe
INPUT FEATURES (pos2_margin, pos2_is_end, pos2_integration_count,
already captured by model/phase3_probe.py, concatenated onto pos2
state) rather than built into a separately-trained model. §2.1's own
concern is that a channel BUILT TO CARRY coverage would let a probe
"measure the architect, not the model" -- that concern is about the
INSTRUMENT's ability to find planted signal, not about training
dynamics, so post-hoc concatenation isolates exactly the question this
control is for without the added cost/confound of new training.
"""

from __future__ import annotations

import torch

from model.phase3_probe import (
    N_PER_COMBO, auroc, bootstrap_auroc_ci, build_combo, capture_pos1_pos2, fit_linear_probe,
    load_model, probe_score, residualize_multi,
)

CHECKPOINT = "runs/real_seed_r1_0.pt"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT, device)
    print(f"positive control (§2.1's forbidden register, appended post-hoc): {CHECKPOINT}", flush=True)

    combos = {
        ("A", 1021, 2): build_combo(1021, 2, "A", device=device),
        ("B2", 1021, 2): build_combo(1021, 2, "B2", device=device),
        ("A", 1021, 3): build_combo(1021, 3, "A", device=device),
        ("B2", 1021, 3): build_combo(1021, 3, "B2", device=device),
    }
    FIELDS = ["pos2", "pos2_margin", "pos2_integration_count", "pos2_is_end", "label"]
    CHUNK = 512
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

    def combine(*keys):
        return {f: torch.cat([captured[k][f] for k in keys]) for f in FIELDS}

    def augmented(d):
        """[pos2_state ; margin ; is_end ; integration_count] -- the
        register §2.1 forbids, appended directly to what the probe sees."""
        return torch.cat([
            d["pos2"], d["pos2_margin"].unsqueeze(1),
            d["pos2_is_end"].float().unsqueeze(1), d["pos2_integration_count"].unsqueeze(1),
        ], dim=1)

    train = combine(("A", 1021, 2), ("B2", 1021, 2))
    test_L3 = combine(("A", 1021, 3), ("B2", 1021, 3))

    train_x, train_y = augmented(train), train["label"]
    test_x, test_y = augmented(test_L3), test_L3["label"]

    probe = fit_linear_probe(train_x, train_y)
    train_score = probe_score(probe, train_x)
    test_score = probe_score(probe, test_x)
    train_auroc = auroc(train_score, train_y)
    raw_auroc = auroc(test_score, test_y)

    covs = [test_L3["pos2_margin"], test_L3["pos2_is_end"], test_L3["pos2_integration_count"]]
    joint_resid = residualize_multi(test_score, covs)
    joint_auroc, joint_lo, joint_hi = bootstrap_auroc_ci(joint_resid, test_y)

    print(f"augmented-feature probe (combo1, L=2->3): train_AUROC={train_auroc:.4f}  "
          f"test_raw_AUROC={raw_auroc:.4f}  JOINT_resid={joint_auroc:.4f}[{joint_lo:.3f},{joint_hi:.3f}]", flush=True)
    print("expected: raw_AUROC near 1.0 (pipeline detects planted signal cleanly); "
          "JOINT_resid near chance is EXPECTED here, not concerning -- the register's whole "
          "signal IS the residualized-out covariates by construction.", flush=True)

    # Also report the plain (unaugmented) state probe on the same items for direct contrast.
    plain_probe = fit_linear_probe(train["pos2"], train_y)
    plain_raw = auroc(probe_score(plain_probe, test_L3["pos2"]), test_y)
    print(f"for contrast, plain (unaugmented) state probe on the same items: raw_AUROC={plain_raw:.4f}", flush=True)


if __name__ == "__main__":
    main()
