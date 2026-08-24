"""Phase 3 canary, shared across regimes -- one function, checkpoint as a
parameter, so both regimes always run through identical analysis code.
Superseded model/phase3_run_canary.py and model/phase3_run_canary_r2.py
(kept for the historical record, not maintained further): those were two
independently-written near-duplicate scripts, and the divergence between
them is exactly how the END-flag control ended up scoped to Regime 2 only
even though the mechanism behind it turned out to be task-structural, not
regime-specific. One shared function can't drift that way.

P2 control set, applied identically to both regimes: margin (§5.1
original), END-flag identity (§5.1 Regime-2 addendum -- now understood to
be structural, not regime-specific, and run on both), and integration
count (the same §2.5 rho asymmetry already shown to fully explain P1's
raw signal, checked at P2 too since nothing about the mechanism is
position-specific). Reported both as three separate single-covariate
residuals (diagnostic, shows which confound does how much work) and as
one joint three-way residual (the actual question: does anything survive
once all three known confounds are removed at once).

P1 control set: integration count only (no well-defined single margin or
END-flag identity exists at pos1 itself -- see model/phase3_probe.py).
"""

from __future__ import annotations

import json

import torch

from model.phase3_probe import (
    N_PER_COMBO, auroc, bootstrap_auroc_ci, build_combo, capture_pos1_pos2, fit_linear_probe,
    load_model, probe_score, residualize, residualize_multi,
)

P2_COVARIATES = [("pos2_margin", "margin"), ("pos2_is_end", "endflag"), ("pos2_integration_count", "count")]
P1_COVARIATES = [("pos1_integration_count", "count")]
FIELDS = ["pos1", "pos2", "pos2_margin", "pos1_integration_count", "pos2_integration_count", "pos2_is_end", "label"]


def report_transfer(name: str, train_pos: dict, test_pos: dict, position: str,
                     regress: list[tuple[str, str]]) -> dict:
    x_key = position
    train_x, train_y = train_pos[x_key], train_pos["label"]
    test_x, test_y = test_pos[x_key], test_pos["label"]

    probe = fit_linear_probe(train_x, train_y)
    train_score = probe_score(probe, train_x)
    test_score = probe_score(probe, test_x)

    train_auroc = auroc(train_score, train_y)
    raw_auroc = auroc(test_score, test_y)

    result = {"train_auroc": train_auroc, "raw_auroc": raw_auroc, "probe": probe, "residuals": {}}
    line = f"{name} [{position}]: train_AUROC={train_auroc:.4f}  test_raw_AUROC={raw_auroc:.4f}"

    for regress_key, regress_label in regress:
        test_cov = test_pos[regress_key].float()
        test_resid = residualize(test_score, test_cov)
        resid_auroc, resid_lo, resid_hi = bootstrap_auroc_ci(test_resid, test_y)
        cov_only_auroc = auroc(test_cov, test_y)
        line += (f"  {regress_label}_only={cov_only_auroc:.4f}  "
                 f"{regress_label}_resid={resid_auroc:.4f}[{resid_lo:.3f},{resid_hi:.3f}]")
        result["residuals"][regress_label] = {
            "resid_auroc": resid_auroc, "resid_ci": (resid_lo, resid_hi), "cov_only_auroc": cov_only_auroc,
        }

    if len(regress) > 1:
        covs = [test_pos[k] for k, _ in regress]
        test_joint_resid = residualize_multi(test_score, covs)
        joint_auroc, joint_lo, joint_hi = bootstrap_auroc_ci(test_joint_resid, test_y)
        joint_label = "+".join(lbl for _, lbl in regress)
        line += f"  JOINT[{joint_label}]_resid={joint_auroc:.4f}[{joint_lo:.3f},{joint_hi:.3f}]"
        result["joint_residual"] = {"resid_auroc": joint_auroc, "resid_ci": (joint_lo, joint_hi), "label": joint_label}

    if not regress:
        line += "  (no regression covariate available at this position)"
    print(line, flush=True)
    return result


def report_abstain_head(name: str, model, test_pos: dict, regress: list[tuple[str, str]]) -> dict:
    """§2.4: 'Regime 2 tells you whether a trained abstention signal is
    real or a surface correlate.' The frozen-state probe answers this for
    state; the trained head is a separate object and needs its own check
    against the same three confounds. Score = -abstain_head(pos2), negated
    so that high score means 'predicted answerable' -- matching the frozen-
    state probe's convention (label=True for A/D) so the numbers sit in
    the same table without a sign flip to remember."""
    device = next(model.abstain_head.parameters()).device
    pos2 = test_pos["pos2"].to(device)
    with torch.no_grad():
        raw_logit = model.abstain_head(pos2).squeeze(-1)
    score = (-raw_logit).cpu()
    test_y = test_pos["label"]

    raw_auroc = auroc(score, test_y)
    covs = [test_pos[k] for k, _ in regress]
    joint_resid = residualize_multi(score, covs)
    joint_auroc, joint_lo, joint_hi = bootstrap_auroc_ci(joint_resid, test_y)
    joint_label = "+".join(lbl for _, lbl in regress)
    print(f"{name} [abstain_head]: raw_AUROC={raw_auroc:.4f}  "
          f"JOINT[{joint_label}]_resid={joint_auroc:.4f}[{joint_lo:.3f},{joint_hi:.3f}]", flush=True)
    return {"raw_auroc": raw_auroc, "joint_residual": {"resid_auroc": joint_auroc, "resid_ci": (joint_lo, joint_hi)}}


def run_canary(checkpoint: str, regime_label: str, report_abstain: bool = False) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device)
    print(f"=== {regime_label}: {checkpoint} ===", flush=True)

    combos = {
        ("A", 1021, 2): build_combo(1021, 2, "A", device=device),
        ("B2", 1021, 2): build_combo(1021, 2, "B2", device=device),
        ("A", 1021, 3): build_combo(1021, 3, "A", device=device),
        ("B2", 1021, 3): build_combo(1021, 3, "B2", device=device),
        ("A", 256, 2): build_combo(256, 2, "A", device=device),
        ("B2", 256, 2): build_combo(256, 2, "B2", device=device),
        ("D", 1021, 2): build_combo(1021, 2, "D", device=device),
    }

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

    train = combine(("A", 1021, 2), ("B2", 1021, 2))
    test_L3 = combine(("A", 1021, 3), ("B2", 1021, 3))
    test_256 = combine(("A", 256, 2), ("B2", 256, 2))

    print("--- P2 (all three controls: margin, endflag, count -- separate and joint) ---", flush=True)
    p2_L3 = report_transfer(f"{regime_label} combo1(L=2->3)", train, test_L3, "pos2", P2_COVARIATES)
    p2_256 = report_transfer(f"{regime_label} combo2(1021->256)", train, test_256, "pos2", P2_COVARIATES)

    abstain_L3 = abstain_256 = None
    if report_abstain:
        print("--- Abstain head (§2.4: real signal or surface correlate? -- same 3 controls) ---", flush=True)
        abstain_L3 = report_abstain_head(f"{regime_label} combo1(L=2->3)", model, test_L3, P2_COVARIATES)
        abstain_256 = report_abstain_head(f"{regime_label} combo2(1021->256)", model, test_256, P2_COVARIATES)

    print("--- P1 (count only -- no well-defined margin/endflag at this position) ---", flush=True)
    p1_L3 = report_transfer(f"{regime_label} combo1(L=2->3)", train, test_L3, "pos1", P1_COVARIATES)
    p1_256 = report_transfer(f"{regime_label} combo2(1021->256)", train, test_256, "pos1", P1_COVARIATES)

    d_data = captured[("D", 1021, 2)]
    d_score = probe_score(p2_L3["probe"], d_data["pos2"])
    d_frac = (torch.sigmoid(d_score) > 0.5).float().mean().item()
    print(f"--- Arm D disqualifier: predicted 'answerable' on {d_frac:.4f} of D items ---", flush=True)

    result = {"p2_L3": p2_L3, "p2_256": p2_256, "p1_L3": p1_L3, "p1_256": p1_256, "d_frac": d_frac,
              "abstain_L3": abstain_L3, "abstain_256": abstain_256}

    def strip(d):
        if d is None:
            return None
        return {k: v for k, v in d.items() if k != "probe"}

    out_path = f"runs/phase3_canary_{regime_label.lower()}_result.json"
    with open(out_path, "w") as f:
        json.dump({"checkpoint": checkpoint, "regime_label": regime_label,
                    "p2_L3": strip(p2_L3), "p2_256": strip(p2_256),
                    "p1_L3": strip(p1_L3), "p1_256": strip(p1_256),
                    "abstain_L3": abstain_L3, "abstain_256": abstain_256,
                    "d_frac": d_frac}, f, indent=2)
    print(f"wrote {out_path}", flush=True)

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        run_canary(sys.argv[1], sys.argv[2], report_abstain=(sys.argv[2].startswith("R2")))
    else:
        r1 = run_canary("runs/real_seed_r1_0.pt", "R1")
        print(flush=True)
        r2 = run_canary("runs/real_seed_r2_0.pt", "R2", report_abstain=True)
