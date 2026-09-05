"""England and Wales analysis stage: register analysis, robustness, validation,
the cross-fitted doubly robust threshold estimator, and Figures 2 and 3.

Figure 4 is drawn by ``map_figure.make_map`` and Figure 5 by
``figures.fig_france_reclassification``; both are sequenced by
``scripts/reproduce_all.py``.
"""
from __future__ import annotations

from .analysis import run_analysis
from .config import OUT_FIGURES, OUT_TABLES
from .figures import make_figures
from .markov import run_markov
from .robustness import run_robustness
from .validation import (heldout_validation, latest_frame_characteristics,
                         policy_proxy_subsets, representativeness_assessment)


def run_optional_mae_rescue(enabled: bool = True) -> bool:
    """Run the Supplementary 1.18 analysis unless a diagnostic run omits it."""
    if not enabled:
        print("  Supplementary 1.18 MAE-rescue analysis skipped (--skip-mae-rescue).")
        return False
    from .mae_rescue import main as run_mae_rescue, validate_mae_rescue_inputs
    validate_mae_rescue_inputs()
    run_mae_rescue()
    return True


def main(*, run_mae_rescue_stage: bool = True) -> None:
    print("  register and fixed-accounting analyses ...", flush=True)
    results = run_analysis(n_boot=1000)
    print("  reassessment-throughput sensitivity ...", flush=True)
    markov = run_markov()
    print(f"  markov gap (EPC-B, threshold expected-state): "
          f"{markov['epcb_gap_p10']} -> {markov['epcb_gap_p100']} pp (10% -> 100% coverage)")

    print("  held-out bridge validation ...", flush=True)
    heldout_validation()             # bridge prediction accuracy on the calibration sample
    print("  latest-register characteristics ...", flush=True)
    latest_frame_characteristics()   # latest-certificate frame characteristics
    print("  robustness suite ...", flush=True)
    run_robustness()                 # certificate validity, England/Wales split, stricter no-works
    print("  policy-proxy subsets ...", flush=True)
    policy_proxy_subsets()           # policy-relevant register subsets
    print("  ND-NEED representativeness assessment ...", flush=True)
    representativeness_assessment()  # comparison and raking to independent ND-NEED margins

    from .bridge_analysis import run_bridge_analysis
    print("  cross-fitted bridge and transport analyses ...", flush=True)
    run_bridge_analysis()            # cross-fitted doubly robust estimator and expenditure scale

    # Same-methodology reassessment noise floor (Supplementary 1.18).
    print("  Supplementary 1.18 MAE-rescue analysis ...", flush=True)
    run_optional_mae_rescue(run_mae_rescue_stage)

    from .dr_estimator import DR_TOLERANCE_PERCENT, run_dr_estimator
    print("  independent doubly robust recomputation ...", flush=True)
    dr = run_dr_estimator()          # independent recomputation of the Stage-1 DR estimator
    primary = dr["comparison"].query("model == 'C' and threshold == 'EPC-B'")
    deltas = dict(zip(primary.quantity, primary.percent_difference.abs()))
    print(f"  doubly robust reproduction gate passed (limits: "
          f"count {DR_TOLERANCE_PERCENT['count']:.2f}%, "
          f"area {DR_TOLERANCE_PERCENT['area_Mm2']:.2f}%); primary estimate: "
          f"count {deltas.get('count', float('nan')):.2f}%, "
          f"area {deltas.get('area_Mm2', float('nan')):.2f}% "
          f"(full comparison: {OUT_TABLES / 'dr_estimator_vs_reported.csv'})")

    print("  Figures 2 and 3 ...", flush=True)
    make_figures()                   # Figures 2 and 3

    identity = results["identity_trends"]
    ratios = results["emissions_ratio_validation"]
    fixed = results["fixed_factor_recompute"]
    print("England and Wales analysis complete.")
    print(f"  identity          : asset rating = {identity['identity_k']} x BER/SER "
          f"(r = {identity['identity_r']})")
    print(f"  emissions ratio   : electric {ratios.get('observed_electric_ratio')} observed "
          f"vs {ratios['predicted_elec_ratio']} predicted")
    print(f"  below-B           : {fixed['belowB_observed_%']:.1f}% observed -> "
          f"{fixed['belowB_fixedfactor_%']:.1f}% constant accounting")
    print(f"  reached-B on paper: {fixed['n_reached_B_on_paper']:,}   "
          f"cleared legal floor on paper: {fixed['n_cleared_FG_on_paper']:,}")
    print(f"  tables  -> {OUT_TABLES}")
    print(f"  figures -> {OUT_FIGURES}")


if __name__ == "__main__":
    main()
