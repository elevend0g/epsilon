"""§2.2 / §5.3, instantiated post-hoc for the first time (`AMENDMENTS.md`
#10) -- both were `PENDING` in `PREREG.md` since it was written, and
`RESULTS.md` finding 29 found §2.2's gate had never even been
implemented as specified (BatchNorm, not a calibration-split CDF).
This module does NOT fix that -- the trained gate's weights were fit
assuming BatchNorm-normalized inputs, permanent and unfixable without
retraining. What this module does: build the calibration split §2.2
and §5.3 both presuppose (never previously reserved in
`model/seed_registry.py` -- also new here), fit the frozen empirical
CDF §2.2 specifies, measure §5.3's abstention floor and `tau` on real
seeds, check §5.3's displacement-scale-comparability requirement before
using one `tau` across arms, and recompute §2.2's quadrant occupancy
against the real fitted boundaries instead of finding 23's per-arm
median-split stand-in.

Every quantity computed here is POST-HOC, ANALYSIS-TIME ONLY -- it
never touches the trained gate's own weights or behavior (`g_soft`,
`g_hard` at the standing `gate_threshold=0.5`, are exactly what they
were everywhere else in this project). This module answers "what would
§2.2's quadrant look like under its own specified boundaries," not
"what does the trained model actually do" -- that remains whatever
BatchNorm-conditioned behavior findings 23/26/27/28 already measured.

Design choices, logged here and in AMENDMENTS.md #10 per §8:
- Calibration split: arm A only, all three L, POOLED into one frozen
  distribution per checkpoint (§2.2 says "a frozen calibration split,"
  singular -- not per-L). N=1024/L, matching N_VAL's convention.
  Captured autoregressively (hard-gated at the standing threshold 0.5),
  every active step contributes a (margin, displacement) pair -- this
  is the actual operating distribution the gate sees at inference, not
  a teacher-forced approximation of it.
- Margin's own quadrant boundary: §5.3 only specifies `tau` for
  DISPLACEMENT ("Displacement enters as a quantile (§2.2), so tau is a
  percentile of the frozen calibration distribution"). Nothing in
  docs/phase1.md names a special calibration target for margin's own
  high/low split. Resolved: margin's boundary is the plain 50th
  percentile (median) of its own frozen calibration-split CDF -- no
  special target-setting logic, since margin isn't the axis tied to the
  abstention floor the way displacement is.
"""

from __future__ import annotations

import json

import torch

from model.cache_bank import build_cache, margin_from_logits, retrieve
from model.data import make_batch_cpu
from model.phase3_gate_secondary import build_arm_batch, capture_terminal_gate
from model.phase3_probe import auroc, load_model
from model.pilot_train import ALPHABET_SIZE, N_HARD
from model.phase4_count import GATE_THRESHOLD, MAX_STEPS

CALIBRATION_SEED_BASE = 840_000_000  # registered in model/seed_registry.py
CHAIN_LENGTHS = (1, 2, 3)
N_CALIB = 1024  # matches N_VAL convention
N_DISTRACTORS_REF = 1021
FLOOR_TARGET_MARGIN_PP = 5.0  # §5.3: "5 percentage points above a measured floor"

CHECKPOINTS = [
    ("R1seed0", "runs/real_seed_r1_0.pt", "R1"),
    ("R1seed1", "runs/real_seed_r1_1.pt", "R1"),
    ("R1seed2", "runs/real_seed_r1_2.pt", "R1"),
    ("R2seed0", "runs/real_seed_r2_0.pt", "R2"),
    ("R2seed1", "runs/real_seed_r2_1.pt", "R2"),
    ("R2seed2", "runs/real_seed_r2_2.pt", "R2"),
]


def build_calibration_batch(L: int, device) -> dict:
    seed = CALIBRATION_SEED_BASE + L
    batch = make_batch_cpu(
        alphabet_size=ALPHABET_SIZE, chain_length=L, n_distractors=N_DISTRACTORS_REF,
        n_hard=N_HARD, batch_size=N_CALIB, seed=seed, arm="A",
    )
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def capture_operating_stream(model, batch: dict, max_steps: int = MAX_STEPS,
                              gate_threshold: float = GATE_THRESHOLD) -> dict:
    """Autoregressive, hard-gated (matches every other inference pass in
    this project) -- collects (margin, displacement) at every step an
    item is still active, freezing after its own END-retrieval (same
    convention as model/phase4_count.py)."""
    cache_keys, cache_values = build_cache(model.encoder, batch["keys"], batch["values"], batch["is_end"])
    start_embed = model.encoder.encode_tuples(batch["start_key"])
    state = model.ssm.init_state(start_embed)
    prev_state = state
    B = cache_keys.shape[0]
    done = torch.zeros(B, dtype=torch.bool, device=state.device)

    margins, displacements = [], []
    for _ in range(max_steps):
        active = ~done
        query = model._query(state)
        logits = retrieve(query, cache_keys)
        top1 = logits.argmax(dim=-1)
        margin = margin_from_logits(logits)
        displacement = (state - prev_state).flatten(1).norm(dim=-1)
        g_soft = model.gate(margin, displacement)
        g_hard = (g_soft > gate_threshold).float()
        is_end_now = batch["is_end"].gather(1, top1.unsqueeze(1)).squeeze(1)
        g_eff = torch.where(is_end_now, torch.zeros_like(g_hard), g_hard) * active.float()

        margins.append(margin[active].cpu())
        displacements.append(displacement[active].cpu())

        newly_done = is_end_now & active
        retrieved_value = cache_values.gather(
            1, top1.view(-1, 1, 1).expand(-1, 1, cache_values.shape[-1])
        ).squeeze(1)
        new_state = model.ssm.step(state, g_eff.unsqueeze(-1) * model.value_proj(retrieved_value))
        active_mask = active.view(-1, 1, 1)
        prev_state = torch.where(active_mask, state, prev_state)
        state = torch.where(active_mask, new_state, state)
        done = done | newly_done
        if bool(done.all()):
            break

    return {"margin": torch.cat(margins), "displacement": torch.cat(displacements)}


class FrozenCDF:
    """A frozen empirical CDF: fit once (sorted reference array), never
    updated. value_at_quantile(q) looks up the reference value at
    percentile q -- this is what §2.2 specifies and gate.py's BatchNorm
    stand-in never was (finding 29)."""

    def __init__(self, values: torch.Tensor):
        self.sorted = torch.sort(values).values

    def value_at_quantile(self, q: float) -> float:
        idx = min(len(self.sorted) - 1, max(0, int(round(q * (len(self.sorted) - 1)))))
        return self.sorted[idx].item()

    def quantile_of(self, value: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(self.sorted, value)
        return idx.float() / (len(self.sorted) - 1)


def fit_calibration(model, device) -> dict:
    """Pooled across all three L, per §8 choice logged in the module
    docstring. Returns frozen CDFs for margin and displacement."""
    all_margin, all_disp = [], []
    for L in CHAIN_LENGTHS:
        batch = build_calibration_batch(L, device)
        out = capture_operating_stream(model, batch)
        all_margin.append(out["margin"])
        all_disp.append(out["displacement"])
    margin_cdf = FrozenCDF(torch.cat(all_margin))
    disp_cdf = FrozenCDF(torch.cat(all_disp))
    return {"margin_cdf": margin_cdf, "displacement_cdf": disp_cdf,
            "n_pooled": len(margin_cdf.sorted)}


def read_floor_from_finding26(checkpoint_name: str) -> dict:
    """§5.3's floor ('the m=0 declined-to-start rate on arm A, no tau
    involved') is not re-run here -- it is a direct read-off of
    RESULTS.md finding 26 / model/phase4_count.py's already-complete
    result: P(m<rho) was measured at 0.0000 for L in {2,3}, all six
    checkpoints, 18,432 items pooled, zero exceptions. Since m=0 < rho
    whenever rho>=1 (L>=2), P(m<rho)=0 mechanically implies P(m=0)=0 for
    those L. At L=1, rho=0 so m=0 IS the correct behavior, not a
    decline -- no floor concept applies there, reported as N/A, not 0."""
    with open("runs/phase4_count_result.json") as f:
        data = json.load(f)
    for r in data["results"]:
        if r["name"] == checkpoint_name:
            return {
                "L1": None,  # not applicable -- rho=0, m=0 is correct behavior
                "L2": r["per_L"]["2"]["P(m<rho)"],
                "L3": r["per_L"]["3"]["P(m<rho)"],
            }
    raise KeyError(checkpoint_name)


def displacement_scale_check(a_data: dict, b2_data: dict) -> dict:
    """§5.3: 'Re-examine displacement distributions under END notation
    before applying one tau across arms... Verify the distributions are
    on comparable scales across arms and report the check.' Terminal-
    step displacement (same population finding 23 used), arm A vs B2,
    same checkpoint. AUROC-based separability (reusing the project's
    existing rank statistic, no new machinery) as the comparability
    signal: near 0.5 = heavily overlapping/comparable scales; far from
    0.5 = the two arms' displacement distributions occupy different
    ranges, and a single boundary fit on one may misclassify the other."""
    a_d, b2_d = a_data["term_displacement"], b2_data["term_displacement"]
    label = torch.cat([torch.ones(len(a_d), dtype=torch.bool), torch.zeros(len(b2_d), dtype=torch.bool)])
    sep_auroc = auroc(torch.cat([a_d, b2_d]), label)
    return {
        "a_mean": a_d.mean().item(), "a_median": a_d.median().item(),
        "b2_mean": b2_d.mean().item(), "b2_median": b2_d.median().item(),
        "separability_auroc": sep_auroc,
    }


def recompute_quadrant(margin: torch.Tensor, displacement: torch.Tensor,
                        margin_median: float, tau: float) -> dict:
    """§2.2's quadrant, using the FROZEN calibration-split boundaries
    (margin_median, tau) -- identical boundary applied to whichever
    population is passed in, unlike finding 23's quadrant_table() which
    re-derived a fresh median from each arm's own population separately
    (RESULTS.md finding 23's correction)."""
    high_m = margin >= margin_median
    high_d = displacement >= tau
    n = len(margin)
    return {
        "integrate (high m, high d)": float((high_m & high_d).sum()) / n,
        "recurse (low m, high d)": float((~high_m & high_d).sum()) / n,
        "emit (high m, low d)": float((high_m & ~high_d).sum()) / n,
        "abstain (low m, low d)": float((~high_m & ~high_d).sum()) / n,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_batch = build_arm_batch("A", device)
    b2_batch = build_arm_batch("B2", device)

    results = {}
    for name, path, regime in CHECKPOINTS:
        print(f"=== {name} ({regime}): {path} ===", flush=True)
        model = load_model(path, device)

        cal = fit_calibration(model, device)
        margin_cdf, disp_cdf = cal["margin_cdf"], cal["displacement_cdf"]
        zero_disp_frac = (disp_cdf.sorted == 0).float().mean().item()
        margin_median = margin_cdf.value_at_quantile(0.5)

        floor = read_floor_from_finding26(name)
        floor_l2, floor_l3 = floor["L2"], floor["L3"]
        floor_pct = max(floor_l2, floor_l3) * 100  # both are 0.0 in practice; max is the conservative choice
        tau_quantile = (floor_pct + FLOOR_TARGET_MARGIN_PP) / 100.0
        tau = disp_cdf.value_at_quantile(tau_quantile)

        print(f"  calibration: n_pooled={cal['n_pooled']}  margin_median={margin_median:.4f}  "
              f"frac_exactly_zero_displacement={zero_disp_frac:.4f}", flush=True)
        print(f"  floor: L1=N/A(rho=0)  L2={floor_l2:.4f}  L3={floor_l3:.4f}  "
              f"tau_quantile={tau_quantile:.4f}  tau={tau:.4f}", flush=True)

        a_data = capture_terminal_gate(model, a_batch)
        b2_data = capture_terminal_gate(model, b2_batch)
        scale_check = displacement_scale_check(a_data, b2_data)
        print(f"  displacement scale check: A mean/median={scale_check['a_mean']:.2f}/{scale_check['a_median']:.2f}  "
              f"B2 mean/median={scale_check['b2_mean']:.2f}/{scale_check['b2_median']:.2f}  "
              f"separability_AUROC={scale_check['separability_auroc']:.4f}", flush=True)

        a_quad_fitted = recompute_quadrant(a_data["term_margin"], a_data["term_displacement"], margin_median, tau)
        b2_quad_fitted = recompute_quadrant(b2_data["term_margin"], b2_data["term_displacement"], margin_median, tau)
        print(f"  A quadrant (fitted CDF/tau):  {a_quad_fitted}", flush=True)
        print(f"  B2 quadrant (fitted CDF/tau): {b2_quad_fitted}", flush=True)

        results[name] = {
            "regime": regime, "n_pooled": cal["n_pooled"], "margin_median": margin_median,
            "frac_exactly_zero_displacement": zero_disp_frac,
            "floor": {"L1": None, "L2": floor_l2, "L3": floor_l3},
            "tau_quantile": tau_quantile, "tau": tau,
            "displacement_scale_check": scale_check,
            "a_quadrant_fitted": a_quad_fitted, "b2_quadrant_fitted": b2_quad_fitted,
        }

    with open("runs/gate_calibration_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote runs/gate_calibration_result.json", flush=True)


if __name__ == "__main__":
    main()
