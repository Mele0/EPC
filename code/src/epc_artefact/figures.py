"""Main-text result figures for the EPC carbon-accounting study.

Reads the result tables written by ``analysis.run_analysis`` and
``france.run_france`` and writes Figures 2, 3 and 5 as PDF and PNG under
``OUT_FIGURES``. Figure 4 is drawn by ``map_figure.make_map``.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch, Polygon, Rectangle

from .config import (
    DR_ESTIMATES_DIR,
    EXTERNAL_DIR,
    OUT_FIGURES,
    OUT_TABLES,
)

NAVY, TEAL, RUST, GREY, PURP = "#1E3A5F", "#0D9488", "#B45309", "#94A3B8", "#6D5BA6"


BLUE, ORANGE, BAND = "#2563EB", "#EA580C", "#BBD0EA"


def _clean_style():
    """Sans-serif, Illustrator-editable text (TrueType, not Type-3 paths)."""
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9.5, "axes.titlesize": 12, "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.unicode_minus": False, "figure.facecolor": "white",
                         "savefig.facecolor": "white"})


def _save(fig, name, png=False):
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURES / f"{name}.pdf", bbox_inches="tight")
    if png:
        fig.savefig(OUT_FIGURES / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


BAND_COLORS = {
    "A/B": "#2E887D",
    "C": "#A8BC63",
    "D": "#F0B93A",
    "E": "#E97837",
    "F/G": "#C8432C",
}


FUEL_COLORS = {"Electric": BLUE, "Gas": ORANGE, "Other": PURP}


def _share_label(value: float) -> str:
    """Format a manuscript share without spurious trailing decimals."""
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _cohort_blocks(cf: dict, decomposition: dict) -> list[dict]:
    """Return the four sequential Figure 2c denominators and their partitions."""
    total = int(cf["n_buildings"])
    repeat = int(cf["n_reassessed_buildings"])
    straddled = int(cf["n_straddle_2022"])
    improved = int(cf["straddle_improved"])
    accounting = round(improved * float(decomposition["improved_majority_formula_%"]) / 100)
    return [
        {"stage": "Full Register", "total": total,
         "selected": ("Repeat-Certified", repeat),
         "remainder": ("Single-Certified", total - repeat)},
        {"stage": "Repeat-Certified", "total": repeat,
         "selected": ("Straddled", straddled),
         "remainder": ("Not Straddled", repeat - straddled)},
        {"stage": "Straddled\n2022 Update", "total": straddled,
         "selected": ("Improved", improved),
         "remainder": ("Not Improved", straddled - improved)},
        {"stage": "Improved on\nReassessment", "total": improved,
         "selected": ("Accounting", accounting),
         "remainder": ("Residual", improved - accounting)},
    ]


def _asset_rating_change(frame: pd.DataFrame) -> pd.Series:
    """Return the signed change used in Figure 3b (negative means improvement)."""
    return pd.to_numeric(frame["mean_dAR"], errors="raise")


def fig_panel1_overview():
    """Draw manuscript Figure 2: register decomposition and validation series."""
    _clean_style()
    with (OUT_TABLES / "fixed_factor_recompute.json").open() as stream:
        ff = json.load(stream)
    with (OUT_TABLES / "cohort_funnel.json").open() as stream:
        cf = json.load(stream)
    with (OUT_TABLES / "section_percentage.json").open() as stream:
        sp = json.load(stream)["decomposition"]
    m = pd.read_csv(OUT_TABLES / "metered_vs_epc_validation.csv")
    gas = pd.read_csv(EXTERNAL_DIR / "metered_nondomestic_gas_EW.csv")
    gas_base = gas.loc[gas.year == 2015, "nd_gas_mean_kWh_per_meter"]
    if gas_base.empty or float(gas_base.iloc[0]) <= 0:
        raise ValueError("Metered gas series requires a positive 2015 baseline")
    gas["idx"] = 100 * gas.nd_gas_mean_kWh_per_meter / float(gas_base.iloc[0])

    fig = plt.figure(figsize=(11.6, 9.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.88, 1.22], hspace=0.30, wspace=0.23)
    axd = fig.add_subplot(gs[0, 0])
    axm = fig.add_subplot(gs[0, 1])
    axs = fig.add_subplot(gs[1, :])

    # (a) Observed and fixed-accounting band distributions. The manuscript uses
    # aligned 100% bars so that each threshold movement can be read directly.
    bands = ["A/B", "C", "D", "E", "F/G"]
    distributions = [
        ("Observed", [float(ff["band_observed_%"][b]) for b in bands], 1),
        ("Constant\nAccounting", [float(ff["band_constant_accounting_%"][b]) for b in bands], 0),
    ]
    for row_label, values, y in distributions:
        left = 0.0
        for band, value in zip(bands, values):
            axd.barh(y, value, left=left, height=0.56, color=BAND_COLORS[band],
                     edgecolor="#4A4A4A", linewidth=0.65, zorder=3)
            if value >= 3.5:
                axd.text(left + value / 2, y, _share_label(value), ha="center", va="center",
                         color="white", fontsize=9, fontweight="bold", zorder=4)
            left += value
        axd.text(-3.0, y, row_label, ha="right", va="center", fontsize=8.5,
                 color="#5A5A5A", linespacing=0.9)
    axd.set_xlim(0, 100)
    axd.set_ylim(-0.62, 1.76)
    axd.set_yticks([])
    axd.set_xticks([0, 25, 50, 75, 100])
    axd.set_xlabel("Share of Register Entries (%)")
    axd.spines[["left", "top", "right"]].set_visible(False)
    axd.legend(
        handles=[Patch(facecolor=BAND_COLORS[b], edgecolor="#4A4A4A", label=b) for b in bands],
        frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.57, 1.09),
        fontsize=8, handlelength=1.0, handletextpad=0.35, columnspacing=0.7,
    )
    axd.text(-0.18, 1.10, "a", transform=axd.transAxes, fontsize=15,
             fontweight="bold", va="top")

    # (b) Modelled CO2 versus metered energy.
    axm.axvline(2022, ls="--", color="#C8C8C8", lw=1.1, zorder=1)
    axm.plot(m.year, m.epc_CO2_idx, marker="o", color=BLUE, lw=2.3, ms=4.5, label="EPC-modelled CO$_2$")
    axm.plot(m.year, m.metered_idx, marker="o", color=PURP, lw=2.3, ms=4.5,
             label="Metered Electricity/meter")
    axm.plot(gas.year, gas.idx, marker="o", color=TEAL, lw=2.3, ms=4.5,
             label="Metered Gas/meter")
    axm.annotate("Methodology\nUpdate", xy=(2021.75, 42), xytext=(2020.4, 48),
                 ha="center", va="center", fontsize=7.4, color=NAVY, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2,
                                 connectionstyle="arc3,rad=0.20"))
    axm.set_ylabel("Index")
    axm.set_xlabel("Year")
    axm.grid(axis="y", color="#DDE2E8", linewidth=0.75, zorder=0)
    axm.set_axisbelow(True)
    axm.set_ylim(20, max(110, float(m[["epc_CO2_idx", "metered_idx"]].max().max()) + 4))
    axm.legend(frameon=False, fontsize=7.6, loc="lower left", handlelength=1.0,
               handletextpad=0.35)
    axm.text(-0.18, 1.10, "b", transform=axm.transAxes, fontsize=15,
             fontweight="bold", va="top")

    # (c) Sequential cohort construction, each bar normalized to its own stage.
    blocks = _cohort_blocks(cf, sp)
    selected_colours = ["#82BFE5", "#82BFE5", "#82BFE5", "#E9AF8A"]
    remainder_colours = ["#E3E5E8", "#E3E5E8", "#E3E5E8", "#91D99A"]
    x = np.arange(len(blocks), dtype=float)
    width = 0.64
    selected_pct = [100 * row["selected"][1] / row["total"] for row in blocks]
    for i in range(len(blocks) - 1):
        axs.add_patch(Polygon([
            (x[i] + width / 2, 0),
            (x[i] + width / 2, selected_pct[i]),
            (x[i + 1] - width / 2, 100),
            (x[i + 1] - width / 2, 0),
        ], closed=True, facecolor="#DDEAF5", edgecolor="none", alpha=0.8, zorder=0))
    for i, row in enumerate(blocks):
        selected_name, selected_n = row["selected"]
        remainder_name, remainder_n = row["remainder"]
        pct = selected_pct[i]
        rem_pct = 100 - pct
        axs.bar(x[i], pct, width, color=selected_colours[i], edgecolor="#4A4A4A",
                linewidth=0.8, zorder=3)
        axs.bar(x[i], rem_pct, width, bottom=pct, color=remainder_colours[i],
                edgecolor="#4A4A4A", linewidth=0.8, zorder=3)
        axs.text(x[i], pct / 2, f"{selected_name}\n{selected_n:,}", ha="center", va="center",
                 fontsize=8.6, color="#1F2937", zorder=4)
        axs.text(x[i], pct + rem_pct / 2, f"{remainder_name}\n{remainder_n:,}",
                 ha="center", va="center", fontsize=8.6, color="#1F2937", zorder=4)
        axs.text(x[i] + width / 2 - 0.01, max(2.4, pct - 1.1), f"{pct:.0f}%",
                 ha="right", va="top", fontsize=7.8, color="#273444", zorder=5)
        axs.text(x[i] + width / 2 - 0.01, min(98.0, pct + 1.1), f"{rem_pct:.0f}%",
                 ha="right", va="bottom", fontsize=7.8, color="#273444", zorder=5)
        axs.text(x[i], 101.5, f"{row['total']:,}", ha="center", va="bottom",
                 fontsize=9.5, fontweight="bold")
    axs.set_xlim(-0.55, len(blocks) - 0.45)
    axs.set_ylim(0, 108)
    axs.set_ylabel("Share of cohort (%)")
    axs.set_xticks(x)
    axs.set_xticklabels([row["stage"] for row in blocks], fontsize=9)
    axs.set_yticks([0, 20, 40, 60, 80, 100])
    axs.grid(axis="y", color="#DDE2E8", linewidth=0.75, zorder=-1)
    axs.set_axisbelow(True)
    axs.text(-0.08, 1.08, "c", transform=axs.transAxes, fontsize=15,
             fontweight="bold", va="top")

    # Preserve the plotted cohort values independently of the artwork.
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"stage": row["stage"].replace("\n", " "), "denominator": row["total"],
         "selected_label": row["selected"][0], "selected_n": row["selected"][1],
         "selected_share_pct": 100 * row["selected"][1] / row["total"],
         "remainder_label": row["remainder"][0], "remainder_n": row["remainder"][1]}
        for row in blocks
    ]).to_csv(OUT_TABLES / "figure_2_data.csv", index=False)
    _save(fig, "figure_2", png=True)


def fig_panel2_thresholds_capex():
    """Draw manuscript Figure 3: threshold and expenditure-equivalent results."""
    _clean_style()
    with (OUT_TABLES / "fixed_factor_recompute.json").open() as stream:
        ff = json.load(stream)
    strad = pd.read_csv(OUT_TABLES / "within_building_straddle_by_fuel.csv")
    fuels = ["Electric", "Gas", "Other"]

    fig = plt.figure(figsize=(11.6, 9.6))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc, axe = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # ── (a) threshold crossings by fuel ───────────────────────────────────────
    comps = [("EPC-B\nThreshold", ff["reachedB_fuel_composition"], "#E8EFF8"),
             ("F/G Rating\nThreshold", ff["clearedFG_fuel_composition"], "#E9F3EC")]
    gw, x0 = 0.24, np.array([0.0, 1.25])
    for gi, (lab, comp, back) in enumerate(comps):
        tot = sum(comp.values())
        axa.bar(x0[gi], tot, 0.86, color=back, zorder=1)
        axa.text(x0[gi], tot + 1400, f"{tot:,}", ha="center", fontweight="bold", fontsize=10)
        for fi, fg in enumerate(fuels):
            xb = x0[gi] + (fi - 1) * gw
            axa.bar(xb, comp[fg], gw * 0.92, color=FUEL_COLORS[fg], zorder=3,
                    label=fg if gi == 0 else None)
            axa.text(xb, comp[fg] + 900, f"{comp[fg]:,}", ha="center", fontsize=7.6)
    axa.set_xticks(x0); axa.set_xticklabels([c[0] for c in comps])
    axa.set_ylabel("Register Entries"); axa.legend(frameon=False, fontsize=8.5, loc="upper right")
    axa.set_title("a", loc="left", fontsize=14, fontweight="bold")

    # (b) Mean within-UPRN change, retaining the manuscript sign convention. A
    # decline in the asset-rating number is an improvement and is therefore negative.
    periods = [("Both pre-2022", "Both $pre$-2022"), ("Straddle 2022", "$Straddle$ 2022"),
               ("Both post-2022", "Both $post$-2022")]
    gw = 0.26
    strad = strad.assign(plot_change=_asset_rating_change(strad))
    for pi, (grp, _label) in enumerate(periods):
        sub = strad[strad.group == grp].set_index("fuel")
        for fi, fg in enumerate(fuels):
            xb = pi + (fi - 1) * gw
            v = float(sub.loc[fg, "plot_change"])
            axb.bar(xb, v, gw * 0.92, color=FUEL_COLORS[fg], label=fg if pi == 0 else None)
    axb.axhline(0, color="black", lw=0.8)
    axb.set_xticks([0, 1, 2])
    axb.set_xticklabels(["Both\npre-2022", "Straddle\n2022", "Both\npost-2022"],
                        fontsize=9, style="italic")
    axb.set_ylabel("Mean asset-rating change")
    ymin = min(-40.0, float(strad[strad.fuel.isin(fuels)].plot_change.min()) * 1.08)
    axb.set_ylim(ymin, 0)
    axb.grid(axis="y", color="#DDE2E8", linewidth=0.75, zorder=0)
    axb.set_axisbelow(True)
    axb.legend(frameon=False, fontsize=8.5, loc="lower right")
    axb.set_title("b", loc="left", fontsize=14, fontweight="bold")

    # ══ economic panels: reconciled source-of-truth values ════════════════════
    # Primary EPC-B affected area = cross-fitted DR Model C original-sample point
    # (Stage-1 frozen); bootstrap gives the interval only (never the centre).
    frz = pd.read_csv(DR_ESTIMATES_DIR / "bridge_stage1_frozen.csv")
    frz = frz[frz.frame == "complete_eligible_EPCB"]
    aC = float(frz[(frz.model == "C") & (frz.estimator == "doubly_robust_crossfitted")]
               .iloc[0].affected_area_Mm2)                       # 57.24 Mm2  (plotted centre)
    C_M2 = 100.0
    fbn = aC * C_M2 / 1e3                                        # £5.72bn generic full-area
    boot = pd.read_csv(DR_ESTIMATES_DIR / "bridge_bootstrap_draws.csv")
    aLo, aHi = np.percentile(boot.area_Mm2, [2.5, 97.5])         # 50.7 / 62.5 Mm2
    # IPF-covered sector DR cost functionals (Model C), sum -> covered total
    sc = pd.read_csv(DR_ESTIMATES_DIR / "bridge_sectorcost_dr.csv")
    sc = sc[sc.model == "C"].set_index("sector")
    sect_bn = {s: float(sc.loc[s, "dr_GBPbn"]) for s in ["Offices", "Factories", "Warehouses"]}
    cov_bn = sum(sect_bn.values())                              # £6.46bn
    cLo, cHi = np.percentile(boot.cov_cost_GBPbn, [2.5, 97.5])  # £5.59 / 7.24bn
    # ND-NEED sector+size+region composition sensitivity
    lad = pd.read_csv(DR_ESTIMATES_DIR / "bridge_stage2_ndneed_ladder.csv")
    aND = float(lad[lad.margins == "sector_benchmark+size_benchmark+region_benchmark"]
                .iloc[0].dr_area_Mm2)                            # 43.9 Mm2
    ndbn = aND * C_M2 / 1e3                                      # £4.39bn
    # NDPRS IA capital+installation £4.6bn (2018 prices) rebased to 2024-25
    dfl = pd.read_csv(EXTERNAL_DIR / "gdp_deflator_hmt_jun2026.csv").set_index("financial_year")
    reb = float(dfl.loc["2024-25", "gdp_deflator_index_2025_26_base"] /
                dfl.loc["2017-18", "gdp_deflator_index_2025_26_base"])
    ndprs_bn = 4.6 * reb                                         # £5.97bn

    # (c) Same matched office/factory/warehouse area under the generic benchmark
    # and IPF archetype rates. The intervening bands make the matched-area bridge
    # explicit and reproduce the manuscript rather than adding unmatched sectors.
    sector_colours = {"Offices": "#4397AA", "Factories": "#C75B63",
                      "Warehouses": "#80AD98"}
    sar = pd.read_csv(DR_ESTIMATES_DIR / "bridge_sector_area_dr.csv").set_index("sector")["dr_area_Mm2"]
    sar = sar * (aC / sar.sum())                                # reconcile to plotted centre 57.24
    matched = ["Offices", "Factories", "Warehouses"]
    cov_area = float(sar[matched].sum())                        # 30.15 Mm2
    cov_share = cov_area / aC                                   # 0.526
    gen_bn = {s: float(sar[s]) * C_M2 / 1e3 for s in matched}
    cov_bn_generic = sum(gen_bn.values())                       # £3.01bn
    x1, x2, bw = 0.0, 1.58, 0.68
    left_base = right_base = 0.0
    for sector in matched:
        left_top = left_base + gen_bn[sector]
        right_top = right_base + sect_bn[sector]
        axc.add_patch(Polygon([
            (x1 + bw / 2, left_base), (x1 + bw / 2, left_top),
            (x2 - bw / 2, right_top), (x2 - bw / 2, right_base),
        ], closed=True, facecolor=sector_colours[sector], edgecolor="none", alpha=0.92,
            zorder=1))
        axc.bar(x1, gen_bn[sector], bw, bottom=left_base,
                color=sector_colours[sector], edgecolor="#4B5563", linewidth=0.55, zorder=3)
        axc.bar(x2, sect_bn[sector], bw, bottom=right_base,
                color=sector_colours[sector], edgecolor="#4B5563", linewidth=0.55, zorder=3)
        left_base, right_base = left_top, right_top

    def _component_label(value: float) -> str:
        decimals = 2 if value < 1 else 1
        return f"£{value:.{decimals}f}".rstrip("0").rstrip(".") + "bn"

    left_base = right_base = 0.0
    for sector in matched:
        axc.text(x1, left_base + gen_bn[sector] / 2, _component_label(gen_bn[sector]),
                 ha="center", va="center", fontsize=8, color="white", zorder=4)
        axc.text(x2, right_base + sect_bn[sector] / 2, _component_label(sect_bn[sector]),
                 ha="center", va="center", fontsize=8, color="white", zorder=4)
        left_base += gen_bn[sector]
        right_base += sect_bn[sector]

    axc.text(x1, cov_bn_generic + 0.15, f"£{cov_bn_generic:.2f}bn", ha="center",
             fontweight="bold", fontsize=9.2)
    axc.text(x2, cov_bn + 0.15, f"£{cov_bn:.2f}bn", ha="center",
             fontweight="bold", fontsize=9.2)
    axc.annotate("Matched\nFloor Area", xy=(1.05, 2.08), xytext=(0.86, 1.20),
                 ha="center", va="center", fontsize=8.1, color="white", zorder=5,
                 arrowprops=dict(arrowstyle="->", color="white", lw=1.5,
                                 connectionstyle="arc3,rad=0.35"))
    axc.set_xticks([x1, x2])
    axc.set_xticklabels(["Full affected area\nGeneric £100/m$^2$",
                         "IPF-covered sectors\nArchetype-based rates"], fontsize=8.2)
    axc.set_xlim(-0.55, 2.13); axc.set_ylim(0, 7.0)
    axc.set_ylabel("Expenditure-equivalent scale (£bn)")
    axc.grid(axis="y", color="#DDE2E8", linewidth=0.75, zorder=0)
    axc.set_axisbelow(True)
    axc.legend(handles=[Patch(facecolor=sector_colours[s], edgecolor="#4B5563", label=s)
                        for s in reversed(matched)], frameon=False, fontsize=8.2,
               loc="upper left", handlelength=0.9, handletextpad=0.3)
    axc.set_title("c", loc="left", fontsize=14, fontweight="bold")

    # ── (d) generic expenditure-equivalent sensitivity, £25–200/m2 ────────────
    DARK, MID, RIB = "#1F3B63", "#5B9BD5", "#DCE8F5"
    grid = np.arange(25, 201, 5)
    axe.grid(axis="y", color="#E6E6E6", lw=0.8, zorder=0)
    axe.fill_between(grid, aLo * grid / 1e3, aHi * grid / 1e3, color=RIB, zorder=1)
    axe.plot(grid, aC * grid / 1e3, color=DARK, lw=3.0, zorder=3,
             label=f"Primary DR ({aC:.1f} Mm$^2$)")
    axe.plot(grid, aND * grid / 1e3, color=MID, lw=3.0, zorder=3,
             label=f"ND-NEED composition ({aND:.1f} Mm$^2$)")
    axe.axhline(ndprs_bn, color="#9AA0A6", lw=1.6, ls="--", zorder=2)
    axe.text(150, ndprs_bn - 0.28, "NDPRS IA capital\n+ installation", ha="left", va="top",
             fontsize=8.2, fontweight="bold", color="#9AA0A6")
    axe.plot(100, fbn, "o", color=DARK, ms=8, zorder=4)
    axe.annotate(f"£{fbn:.2f}bn at £100/m$^2$", xy=(100, fbn), xytext=(60, fbn + 1.15),
                 fontsize=8.4, color=DARK, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))
    axe.plot(100, ndbn, "o", color=MID, ms=8, zorder=4)
    axe.annotate(f"£{ndbn:.2f}bn at £100/m$^2$", xy=(100, ndbn), xytext=(92, ndbn - 1.5),
                 fontsize=8.4, color=MID, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=MID, lw=1.4))
    axe.set_xlabel("Assumed unit value (£/m$^2$)")
    axe.set_ylabel("Expenditure-equivalent scale (£bn)")
    axe.set_xlim(25, 200); axe.set_ylim(0, 12.5)
    axe.set_axisbelow(True)
    axe.legend(frameon=False, fontsize=8.6, loc="upper left", handlelength=1.3)
    axe.set_title("d", loc="left", fontsize=14, fontweight="bold")
    _save(fig, "figure_3", png=True)

    # machine-readable panel-data table
    panel = pd.DataFrame([
        ("c_bar1_matched_area_generic100", "Offices_GBPbn", gen_bn["Offices"]),
        ("c_bar1_matched_area_generic100", "Factories_GBPbn", gen_bn["Factories"]),
        ("c_bar1_matched_area_generic100", "Warehouses_GBPbn", gen_bn["Warehouses"]),
        ("c_bar1_matched_area_generic100", "total_GBPbn", cov_bn_generic),
        ("c_bar1_matched_area_generic100", "covered_area_Mm2", cov_area),
        ("c_bar1_matched_area_generic100", "covered_area_share", cov_share),
        ("c_bar2_ipf_covered", "Offices_GBPbn", sect_bn["Offices"]),
        ("c_bar2_ipf_covered", "Factories_GBPbn", sect_bn["Factories"]),
        ("c_bar2_ipf_covered", "Warehouses_GBPbn", sect_bn["Warehouses"]),
        ("c_bar2_ipf_covered", "total_GBPbn", cov_bn),
        ("c_bar2_ipf_covered", "boot_lo_GBPbn", cLo),
        ("c_bar2_ipf_covered", "boot_hi_GBPbn", cHi),
        ("d_sensitivity", "primary_area_Mm2", aC),
        ("d_sensitivity", "boot_area_lo_Mm2", aLo),
        ("d_sensitivity", "boot_area_hi_Mm2", aHi),
        ("d_sensitivity", "ndneed_area_Mm2", aND),
        ("d_sensitivity", "primary_at_100_GBPbn", fbn),
        ("d_sensitivity", "ndneed_at_100_GBPbn", ndbn),
        ("d_sensitivity", "ndprs_rebased_GBPbn", ndprs_bn),
    ], columns=["panel", "series", "value"])
    panel.to_csv(OUT_TABLES / "figure_3_data.csv", index=False)


FR_BAND = {"A": "#61A83E", "B": "#A3C741", "C": "#C9C23F", "D": "#EDBC32",
           "E": "#FAAD2D", "F": "#F16F27", "G": "#DD3740"}
FR_ELEC, FR_OTHER, FR_GREY = "#00485A", "#749596", "#595959"
FR_HEAT_LABELS = {"Electricity": "Electricity", "Gas": "Gas",
                  "Wood/biomass": "Biomass", "Heat network": "Heat\nNetwork",
                  "Oil/other-fossil": "Oil/Other"}


def fig_france_reclassification():
    """Figure 5: France's 2026 primary-energy coefficient reform reclassifies DPE labels.

    (a) Same-certificate DPE label distributions under the pre-reform electricity
    primary-energy coefficient of 2.3 and the post-reform coefficient of 1.9, holding
    final energy, greenhouse-gas emissions, dwelling surface area and all other
    diagnostic inputs fixed. (b) Certificates moving from F/G under 2.3 to E or better
    under 1.9, grouped by reported main heating energy.

    Both panels are read from the tables written by ``france.run_france``: panel (a)
    margins are the row and column sums of the surface-adjusted transition matrix, and
    panel (b) is the by-heating exit decomposition reported in main-text Table 3.
    """
    letters = list("ABCDEFG")
    trans = pd.read_csv(OUT_TABLES / "france_transition_matrix.csv", index_col=0)
    trans = trans.loc[letters, letters]
    T = trans.to_numpy(dtype=float)
    if not np.allclose(np.triu(T, 1), 0):
        raise ValueError("France transition matrix is not lower triangular: "
                         "the reform cannot worsen a label")

    n_total = int(T.sum())
    dist_23 = pd.Series(T.sum(axis=1), index=letters)
    dist_19 = pd.Series(T.sum(axis=0), index=letters)
    sh_23, sh_19 = 100 * dist_23 / n_total, 100 * dist_19 / n_total
    fg23, fg19 = float(sh_23[["F", "G"]].sum()), float(sh_19[["F", "G"]].sum())
    exits_total = int(T[5, :5].sum() + T[6, :5].sum())
    exit_rate = 100 * exits_total / float(dist_23[["F", "G"]].sum())

    heat = pd.read_csv(OUT_TABLES / "france_by_heating.csv")
    heat = heat[heat.heat_group.isin(FR_HEAT_LABELS)].copy()
    heat["label"] = heat.heat_group.map(FR_HEAT_LABELS)
    heat = heat.sort_values("exits", ascending=False).reset_index(drop=True)
    heat["share"] = 100 * heat.exits / heat.exits.sum()
    if int(heat.exits.sum()) != exits_total:
        raise ValueError(f"by-heating exits {int(heat.exits.sum())} do not sum to the "
                         f"matrix exit total {exits_total}")

    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "axes.linewidth": 0.9})
    fig = plt.figure(figsize=(11.6, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.30,
                          left=0.085, right=0.975, top=0.90, bottom=0.10)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    for title, sub, sh, y in [("Coefficient 2.3", "(pre-reform)", sh_23, 1.0),
                              ("Coefficient 1.9", "(post-reform)", sh_19, 0.0)]:
        left = 0.0
        for b in letters:
            v = float(sh[b])
            axa.barh(y, v, 0.52, left=left, color=FR_BAND[b], edgecolor="black",
                     lw=0.7, zorder=3)
            if v >= 3.6:
                axa.text(left + v / 2, y, f"{v:.1f}", ha="center", va="center",
                         fontsize=8.2 if v >= 6 else 7.0, fontweight="bold",
                         color="white", zorder=5)
            else:
                axa.annotate(f"{v:.1f}", xy=(left + v / 2, y + 0.26),
                             xytext=(left + v / 2 - 3.0, y + 0.56),
                             ha="center", va="bottom", fontsize=7.0,
                             fontweight="bold", color="#222222",
                             arrowprops=dict(arrowstyle="-", lw=0.6, color="#222222",
                                             shrinkA=0, shrinkB=1))
            left += v
        axa.text(-2.0, y + 0.10, title, ha="right", va="center", fontsize=8.6, color="black")
        axa.text(-2.0, y - 0.10, f"{sub}\n{n_total:,} certificates", ha="right", va="top",
                 fontsize=6.6, color=FR_GREY, linespacing=1.25)

    lx = 26.0
    for b in letters:
        axa.add_patch(Rectangle((lx, 1.66), 2.6, 0.13, facecolor=FR_BAND[b],
                                edgecolor="black", lw=0.6, clip_on=False, zorder=4))
        axa.text(lx + 3.4, 1.725, b, ha="left", va="center", fontsize=8.0)
        lx += 10.4
    axa.set_xlim(0, 100)
    axa.set_ylim(-1.52, 1.95)
    axa.set_xticks([0, 25, 50, 75, 100])
    axa.set_yticks([])
    for s in ["top", "right", "left"]:
        axa.spines[s].set_visible(False)
    axa.spines["bottom"].set_position(("data", -0.42))
    axa.tick_params(axis="x", labelsize=8.5)
    axa.text(50, -0.80, "Share of Certificates", ha="center", va="center", fontsize=10)

    fg_start = float(sh_19[:"E"].sum())
    ybr, ybox_c = -0.66, -1.16
    axa.plot([fg_start, fg_start, 100.0, 100.0], [ybr + 0.09, ybr, ybr, ybr + 0.09],
             color="black", lw=0.8, clip_on=False, zorder=4)
    xmid = (fg_start + 100.0) / 2
    axa.plot([xmid, xmid], [ybr, ybox_c], color="black", lw=0.8, clip_on=False, zorder=4)
    axa.add_patch(FancyBboxPatch((6, -1.39), 76, 0.46,
                                 boxstyle="round,pad=0.02,rounding_size=0.10",
                                 linewidth=0.9, edgecolor="black", facecolor="white",
                                 clip_on=False, zorder=4))
    axa.annotate("", xy=(82.4, ybox_c), xytext=(xmid, ybox_c),
                 arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black"), zorder=5)
    axa.text(44, -1.07, f"F/G share falls from {fg23:.2f}% to {fg19:.2f}%",
             ha="center", va="center", fontsize=9.2, zorder=5)
    axa.text(44, -1.27,
             f"{exits_total:,} certificates exit F/G ({exit_rate:.2f}% of pre-reform F/G)",
             ha="center", va="center", fontsize=6.9, zorder=5)
    axa.text(-0.085, 1.02, "a", transform=axa.transAxes, fontsize=15, fontweight="bold")

    y = np.arange(len(heat))[::-1]
    axb.barh(y, heat.exits, 0.62, color=[FR_ELEC] + [FR_OTHER] * (len(heat) - 1), zorder=3)
    for yy, n, sh in zip(y, heat.exits, heat.share):
        axb.text(n + 2200, yy, f"{n:,}\n({sh:.1f}%)", va="center", ha="left",
                 fontsize=7.8, linespacing=1.25)
    axb.set_yticks(y)
    axb.set_yticklabels(heat.label, fontsize=8.4, style="italic")
    axb.set_xlim(0, 120000)
    axb.set_xticks(np.arange(0, 120001, 20000))
    axb.set_xticklabels([f"{v:,}" for v in np.arange(0, 120001, 20000)], fontsize=8.0)
    axb.set_xlabel("Certificates exiting F/G", fontsize=10, labelpad=4)
    axb.grid(axis="x", ls=":", lw=0.6, color="#BFBFBF", zorder=0)
    axb.set_axisbelow(True)
    for s in ["top", "right"]:
        axb.spines[s].set_visible(False)
    axb.text(-0.22, 1.02, "b", transform=axb.transAxes, fontsize=15, fontweight="bold")

    _save(fig, "figure_5", png=True)
    pd.DataFrame({"label": letters,
                  "coef_2p3_count": dist_23.astype(int).values,
                  "coef_1p9_count": dist_19.astype(int).values,
                  "coef_2p3_share_pct": sh_23.round(3).values,
                  "coef_1p9_share_pct": sh_19.round(3).values}
                 ).to_csv(OUT_TABLES / "figure_5_data.csv", index=False)


def make_figures() -> None:
    """Draw the England and Wales result figures (Figures 2 and 3).

    Figure 4, the European structural-exposure map, is drawn by
    ``map_figure.make_map``; Figure 5 by ``fig_france_reclassification`` once the
    France tables exist. Figure 1 of the manuscript is a schematic with no computed
    content and is not produced by this code.
    """
    fig_panel1_overview()
    fig_panel2_thresholds_capex()
