"""Reproduce every computed result in the manuscript: result tables, the main-text
tables, and Figures 2 to 5.

    python scripts/reproduce_all.py                  # everything
    python scripts/reproduce_all.py --skip-france    # England and Wales + typology + map
    python scripts/reproduce_all.py --skip-map       # omit the geopandas map figure
    python scripts/reproduce_all.py --skip-mae-rescue # omit Supplementary 1.18

Stages
    1. England and Wales carbon-accounting analysis. Register trends, the within-UPRN
       straddling cohort, the no-recorded-works calibration, the fixed-accounting
       bridge, the Markov throughput sensitivity, the robustness and validation
       suites, the cross-fitted doubly robust threshold estimator and the
       expenditure-equivalent scale. Draws Figures 2 and 3.
    2. International structural-exposure typology. Draws Figure 4.
    3. France DPE 2026 coefficient-reform replication. Draws Figure 5.
    4. Main-text Tables 1 to 3.

Requires the validated input bundle under ``EPC_DATA_DIR`` (see
``docs/data_manifest.md``). Generated results and disposable caches are kept under
``EPC_OUTPUT_DIR`` and ``EPC_SCRATCH_DIR`` respectively.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-france", action="store_true",
                        help="skip the France replication and Figure 5")
    parser.add_argument("--skip-map", action="store_true",
                        help="skip Figure 4, which needs geopandas and pyogrio")
    parser.add_argument("--france-max-records", type=int, default=None,
                        help=("download and validate a disposable ADEME sample only; "
                              "Figure 5 and Table 3 are not produced"))
    mae_group = parser.add_mutually_exclusive_group()
    mae_group.add_argument(
        "--skip-mae-rescue", dest="run_mae_rescue", action="store_false",
        help=("omit the Supplementary 1.18 MAE-rescue analysis from a diagnostic run"))
    mae_group.add_argument(
        "--run-mae-rescue", dest="run_mae_rescue", action="store_true",
        help=argparse.SUPPRESS)
    parser.set_defaults(run_mae_rescue=True)
    args = parser.parse_args()
    if args.skip_france and args.france_max_records is not None:
        parser.error("--skip-france and --france-max-records cannot be combined")

    publication_france = not args.skip_france and args.france_max_records is None
    from epc_artefact.preflight import validate_release_inputs
    validate_release_inputs(
        include_france=publication_france,
        include_map=not args.skip_map,
        include_mae_rescue=args.run_mae_rescue,
    )
    print("Release input validation passed.")
    started = time.time()

    print("\n[1/4] England and Wales carbon-accounting analysis ...")
    from epc_artefact.run import main as run_england_wales
    run_england_wales(run_mae_rescue_stage=args.run_mae_rescue)

    print("\n[2/4] International structural-exposure typology ...")
    from epc_artefact.typology import run_typology
    run_typology()
    if args.skip_map:
        print("  Figure 4 skipped (--skip-map).")
    else:
        try:
            from epc_artefact.map_figure import make_map
            make_map()
            print("  Figure 4 written.")
        except ImportError:
            print("  Figure 4 skipped: geopandas/pyogrio are missing from the environment.")

    if args.skip_france:
        print("\n[3/4] France replication skipped (--skip-france).")
    elif args.france_max_records is not None:
        print("\n[3/4] France disposable download smoke test ...")
        from epc_artefact.config import FRANCE_CACHE_PARQUET
        from epc_artefact.france import build_frame, download_window, summarise
        sample_path = download_window(
            out=FRANCE_CACHE_PARQUET, max_records=args.france_max_records)
        sample_frame = build_frame(sample_path)
        sample_summary = summarise(sample_frame)
        print(f"  Validated {len(sample_frame):,} sampled rows at {sample_path}.")
        print(f"  Diagnostic sample exits: {sample_summary['exits']:,}; "
              "publication Figure 5 and Table 3 intentionally suppressed.")
    else:
        print("\n[3/4] France DPE 2026 coefficient-reform replication ...")
        from epc_artefact.france import run_france
        summary = run_france(download=False)["summary"]
        print(f"  France exits F/G: {summary['exits']:,} "
              f"({summary['exit_pct_passoires']}% of pre-reform passoires)")
        from epc_artefact.figures import fig_france_reclassification
        fig_france_reclassification()
        print("  Figure 5 written.")

    print("\n[4/4] Main-text tables ...")
    from epc_artefact.paper_tables import make_paper_tables
    if not publication_france:
        print("  Tables 1 and 2 only; Table 3 needs the France stage.")
        from epc_artefact.paper_tables import PAPER_DIR, table1_mechanism, table2_thresholds
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        table1_mechanism()
        table2_thresholds()
    else:
        make_paper_tables()
        from epc_artefact.config import OUT_TABLES
        print(f"  Tables 1 to 3 written to {OUT_TABLES / 'paper'}.")

    from epc_artefact.config import OUT_FIGURES, OUT_TABLES
    print(f"\nDone in {time.time() - started:.0f}s. "
          f"Tables -> {OUT_TABLES}, figures -> {OUT_FIGURES}.")


if __name__ == "__main__":
    main()
