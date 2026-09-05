"""Main-text tables of the manuscript.

Assembles Tables 1 to 3 from the result files written by ``analysis.run_analysis``,
``bridge_analysis.run_bridge_analysis`` and ``france.run_france``. Nothing is
recomputed here: this module only reformats already-validated quantities into the
layout used in the manuscript, writing one CSV per table plus a combined Markdown
rendering under ``outputs/tables/paper/``.

Supplementary tables are not reassembled here; each is written directly by the
module that computes it (see docs/output_manifest.md).
"""
from __future__ import annotations

import json

import pandas as pd

from .config import DR_ESTIMATES_DIR, ELEC_FACTOR_NEW, ELEC_FACTOR_OLD, OUT_TABLES

PAPER_DIR = OUT_TABLES / "paper"

# Published National Calculation Methodology fuel-emission factors (kgCO2/kWh).
# NCM 2013 modelling guide Table 27; NCM 2021 modelling guide Tables 29-30. The
# electricity value is the representative October NCM 2021 monthly factor.
GAS_FACTOR_OLD, GAS_FACTOR_NEW = 0.216, 0.210
FUEL_ORDER = ["Electric", "Gas", "Other"]
FRANCE_HEATING_LABELS = {
    "Wood/biomass": "Wood or biomass",
    "Oil/other-fossil": "Oil/other",
}


def _load(name: str, reported: bool = False):
    path = (DR_ESTIMATES_DIR if reported else OUT_TABLES) / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run the England & Wales and France analyses first")
    return json.load(open(path)) if path.suffix == ".json" else pd.read_csv(path)


def _closed_percentages(values: pd.Series, decimals: int = 1) -> pd.Series:
    """Apportion rounded shares so displayed categories sum exactly to 100%."""
    values = pd.Series(values, dtype=float).reset_index(drop=True)
    if values.empty or values.isna().any() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("Percentage shares require finite, non-negative values with a positive sum")
    scale = 10 ** decimals
    exact_units = 100 * scale * values / values.sum()
    units = exact_units.astype(int)
    unallocated = int(100 * scale - units.sum())
    priority = (exact_units - units).sort_values(ascending=False, kind="stable").index
    for index in priority[:unallocated]:
        units.loc[index] += 1
    return units / scale


def table1_mechanism() -> pd.DataFrame:
    """Table 1: no-recorded-works mechanism diagnostic and fuel-specific inputs."""
    comp = _load("multiplier_components.csv").set_index("fuel")
    windfall = _load("no_measure_change_straddlers.csv").set_index("fuel")
    implied = {"Electric": f"{ELEC_FACTOR_NEW / ELEC_FACTOR_OLD:.3f}",
               "Gas": f"{GAS_FACTOR_NEW / GAS_FACTOR_OLD:.3f}",
               "Other": "-"}
    factor = {"Electric": f"{ELEC_FACTOR_OLD:.3f} -> {ELEC_FACTOR_NEW:.3f}",
              "Gas": f"{GAS_FACTOR_OLD:.3f} -> {GAS_FACTOR_NEW:.3f}",
              "Other": "mixed"}
    rows = []
    for fuel in FUEL_ORDER:
        c = comp.loc[fuel]
        rows.append({
            "Heating fuel": fuel,
            "Counts": int(c["n"]),
            "Carbon factor (kgCO2/kWh)": factor[fuel],
            "Implied BER ratio": implied[fuel],
            "Observed BER ratio": round(float(c["BER_ratio"]), 3),
            "SER ratio": round(float(c["SER_ratio"]), 3),
            "Multiplier m_f": round(float(c["AR_multiplier"]), 3),
            "Rating windfall (pts)": round(-float(windfall.loc[fuel, "mean"]), 1),
        })
    out = pd.DataFrame(rows)
    out.to_csv(PAPER_DIR / "table1_mechanism_diagnostic.csv", index=False)
    return out


def table2_thresholds() -> pd.DataFrame:
    """Table 2: policy-threshold reclassification under constant pre-revision accounting."""
    ff = _load("fixed_factor_recompute.json")
    ci = ff["bootstrap_CI"]
    frozen = _load("bridge_stage1_frozen.csv", reported=True)
    primary = frozen[frozen.role == "PRIMARY"].iloc[0]
    boot = _load("bridge_bootstrap_draws.csv", reported=True)

    def pct(key, dp=1):
        lo, _, hi = ci[key]
        return f"{ff[key]:.{dp}f} ({lo:.{dp}f}-{hi:.{dp}f})"

    def count(key):
        lo, _, hi = ci[key]
        return f"{ff[key]:,} ({int(round(lo)):,}-{int(round(hi)):,})"

    def dr(series, dp=0):
        lo, hi = series.quantile(0.025), series.quantile(0.975)
        fmt = (lambda v: f"{v:,.0f}") if dp == 0 else (lambda v: f"{v:,.{dp}f}")
        return lo, hi, fmt

    lo_n, hi_n, fmt_n = dr(boot["count"])
    lo_a, hi_a, fmt_a = dr(boot["area_Mm2"], 1)
    rows = [
        {"Panel": "A - Fixed-accounting bridge", "Quantity": "Below-EPC-B threshold rate (%)",
         "Current basis": f"{ff['belowB_observed_%']:.1f}",
         "Constant-accounting estimate": pct("belowB_fixedfactor_%")},
        {"Panel": "A - Fixed-accounting bridge", "Quantity": "F/G threshold rate (%)",
         "Current basis": f"{ff['FG_observed_%']:.2f}",
         "Constant-accounting estimate": pct("FG_fixedfactor_%")},
        {"Panel": "A - Fixed-accounting bridge",
         "Quantity": "Entries at EPC-B or better reclassified below B (n)",
         "Current basis": "-", "Constant-accounting estimate": count("n_reached_B_on_paper")},
        {"Panel": "A - Fixed-accounting bridge",
         "Quantity": "Entries at EPC-E or better reclassified into F/G (n)",
         "Current basis": "-", "Constant-accounting estimate": count("n_cleared_FG_on_paper")},
        {"Panel": "B - Primary cross-fitted doubly robust",
         "Quantity": "Expected entries reclassified below EPC-B (n)", "Current basis": "-",
         "Constant-accounting estimate":
             f"{int(primary.expected_count):,} ({fmt_n(lo_n)}-{fmt_n(hi_n)})"},
        {"Panel": "B - Primary cross-fitted doubly robust",
         "Quantity": "Expected affected floor area (Mm2)", "Current basis": "-",
         "Constant-accounting estimate":
             f"{float(primary.affected_area_Mm2):.1f} ({fmt_a(lo_a)}-{fmt_a(hi_a)})"},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(PAPER_DIR / "table2_policy_threshold_reclassification.csv", index=False)
    return out


def table3_france() -> pd.DataFrame:
    """Table 3: F/G exits by main heating energy in the France DPE counterfactual."""
    heat = _load("france_by_heating.csv")
    out = pd.DataFrame({
        "Heating energy": heat.heat_group.replace(FRANCE_HEATING_LABELS),
        "Certificates": heat.n.astype(int),
        "Flow share (%)": _closed_percentages(heat.n, decimals=1),
        "Pre-reform passoires": heat.passoires.astype(int),
        "Exits": heat.exits.astype(int),
        "Exit-rate passoires (%)": heat["exit_rate_passoires_%"].round(2),
        "Share of all exits (%)": heat["share_of_exits_%"].round(2),
    })
    out.to_csv(PAPER_DIR / "table3_france_exits_by_heating.csv", index=False)
    return out


CAPTIONS = {
    1: ("Table 1: No-recorded-works mechanism diagnostic and fuel-specific counterfactual "
        "inputs across the 2022 methodology revision."),
    2: ("Table 2: Policy-threshold reclassification of the latest-register frame under "
        "constant pre-revision accounting."),
    3: ("Table 3: F/G exits by main heating energy in the France DPE factor-revision "
        "counterfactual."),
}


def make_paper_tables() -> dict[int, pd.DataFrame]:
    """Write Tables 1 to 3 as CSV plus a combined Markdown rendering."""
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    tables = {1: table1_mechanism(), 2: table2_thresholds(), 3: table3_france()}
    parts = ["# Main-text tables\n"]
    for number, frame in tables.items():
        parts.append(f"\n## {CAPTIONS[number]}\n")
        parts.append(frame.to_markdown(index=False))
        parts.append("")
    (PAPER_DIR / "paper_tables.md").write_text("\n".join(parts) + "\n")
    return tables


if __name__ == "__main__":
    for number, frame in make_paper_tables().items():
        print(f"\n{CAPTIONS[number]}\n")
        print(frame.to_string(index=False))
