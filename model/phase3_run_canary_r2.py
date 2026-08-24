"""Phase 3 canary: one Regime 2 checkpoint (real_seed_r2_0), end to end
-- probe corpus, state capture, P2 probe with BOTH regress-out controls
built in from the start (margin, per §5.1's original requirement, and
END-flag identity, per §5.1's Regime-2-specific second control fixed
in the prior commit, from finding 11's END-guess-rate asymmetry), P1,
arm D disqualifier.

Same probe corpus (same seeds, same items) as the Regime 1 canary
(model/phase3_run_canary.py) -- only the checkpoint differs, so the two
runs' numbers are directly comparable. Scope is the same 7-combo subset
as the R1 canary, same reasons.
"""

from __future__ import annotations

import torch

from model.phase3_probe import (
    N_PER_COMBO, auroc, bootstrap_auroc_ci, build_combo, capture_pos1_pos2, fit_linear_probe,
    load_model, probe_score, residualize,
)

CHECKPOINT = "runs/real_seed_r2_0.pt"


def report_transfer(name: str, train_pos: dict, test_pos: dict, position: str,
                     regress: list[tuple[str, str]]) -> dict:
    """regress: list of (covariate_key, label) pairs, each residualized
    SEPARATELY (not jointly) against the probe score -- e.g. [("pos2_margin",
    "margin")] for Regime 1's P2, or [("pos2_margin", "margin"),
    ("pos2_is_end", "endflag")] for Regime 2's P2 (§5.1's second control,
    finding 11). Empty list reports raw only. Each position gets whichever
    covariates are actually well-defined for it -- P2 has margin (and, for
    Regime 2, END-flag identity); P1 has no single well-defined margin, but
    does have integration count (§2.5's rho), the leading P1 shortcut
    candidate."""
    x_key = position  # "pos1" or "pos2"
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
        line += (f"  test_{regress_label}_only_AUROC={cov_only_auroc:.4f}  "
                 f"test_{regress_label}_residualized_AUROC={resid_auroc:.4f} [95% CI {resid_lo:.4f}-{resid_hi:.4f}]")
        result["residuals"][regress_label] = {
            "resid_auroc": resid_auroc, "resid_ci": (resid_lo, resid_hi), "cov_only_auroc": cov_only_auroc,
        }

    if not regress:
        line += "  (no regression covariate available at this position)"
    print(line, flush=True)
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

    print("\n=== P2 (after the next lookup returns low margin) — both controls built in ===", flush=True)
    P2_CONTROLS = [("pos2_margin", "margin"), ("pos2_is_end", "endflag")]
    p2_L3 = report_transfer("combo1 (L=2->3)", train, test_L3, "pos2", P2_CONTROLS)
    p2_256 = report_transfer("combo2 (1021->256)", train, test_256, "pos2", P2_CONTROLS)
    p2_L3_resid = p2_L3["residuals"]["margin"]["resid_auroc"]
    p2_256_resid = p2_256["residuals"]["margin"]["resid_auroc"]
    print(f"P2 margin-residualized spread: combo1(L transfer)={p2_L3_resid:.4f}  "
          f"combo2(band transfer)={p2_256_resid:.4f}  -- "
          f"{'L transfer is lower, same direction as R1' if p2_L3_resid < p2_256_resid else 'L transfer is NOT lower'}",
          flush=True)
    print(f"P2 endflag-residualized: combo1={p2_L3['residuals']['endflag']['resid_auroc']:.4f}  "
          f"combo2={p2_256['residuals']['endflag']['resid_auroc']:.4f}  "
          f"(endflag_only_AUROC: combo1={p2_L3['residuals']['endflag']['cov_only_auroc']:.4f}  "
          f"combo2={p2_256['residuals']['endflag']['cov_only_auroc']:.4f})", flush=True)

    print("\n=== P1 (after integrating kj, before the next lookup resolves) ===", flush=True)
    report_transfer("combo1 (L=2->3)", train, test_L3, "pos1", [("pos1_integration_count", "count")])
    report_transfer("combo2 (1021->256)", train, test_256, "pos1", [("pos1_integration_count", "count")])

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
