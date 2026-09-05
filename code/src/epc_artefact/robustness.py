"""Pre-submission robustness checks for the EPC carbon-accounting artefact study.

Three reviewer-requested analyses, all reproducing the headline national-policy
outputs (observed vs constant-accounting below-B and F/G, on-paper EPC-B and
legal-floor crossings, floor area and capex scale) under alternative samples:

    1. certificate_validity_robustness -- latest-stock frame under EPC validity
       cut-offs (all latest; lodged within 10 years; post-2022 only;
       >1000 m2 and within 10 years).
    2. country_split_robustness -- England-only and Wales-only national metrics,
       plus the within-building electric-gas straddler contrast and the
       no-recorded-works multiplier estimated separately by country.
    3. stricter_no_works -- the fuel-specific re-pricing multiplier under
       progressively stricter no-recorded-works definitions (adding unchanged
       fuel/air-conditioning, payback mix and recommendation-code set).

Run:  python -m epc_artefact.robustness   (also run as part of scripts/reproduce_all.py)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import OUT_TABLES, CUT
from .data import build_extended_frame, with_dates
from .analysis import within_building_pairs, compute_multipliers

# EPC validity is 10 years; the "as-of" date anchors the validity window.
ASOF = pd.Timestamp("2026-06-15")
VALID_FROM = ASOF - pd.DateOffset(years=10)
FUELS = ["Electric", "Gas", "Other"]


def _stock_metrics(ar, post, fuel, fa, M, unit_cost=100.0) -> dict:
    """Observed vs constant-accounting stock metrics for a latest-stock subset."""
    ar = np.asarray(ar, float); fa = np.asarray(fa, float)
    mult = np.array([M.get(x, M["Other"]) for x in fuel])
    arc = np.where(post, ar * mult, ar)
    reachB = (ar <= 50) & (arc > 50)
    clearFG = (ar <= 125) & (arc > 125)
    return {"n": int(len(ar)),
            "belowB_obs_%": round(100 * (ar > 50).mean(), 1),
            "belowB_const_%": round(100 * (arc > 50).mean(), 1),
            "FG_obs_%": round(100 * (ar > 125).mean(), 2),
            "FG_const_%": round(100 * (arc > 125).mean(), 1),
            "reachedB_n": int(reachB.sum()),
            "reachedB_Mm2": round(fa[reachB].sum() / 1e6, 1),
            "reachedB_capex_bn": round(fa[reachB].sum() * unit_cost / 1e9, 2),
            "clearedFG_n": int(clearFG.sum()),
            "clearedFG_Mm2": round(fa[clearFG].sum() / 1e6, 1),
            "clearedFG_capex_bn": round(fa[clearFG].sum() * unit_cost / 1e9, 2)}


def _latest_stock(dfx: pd.DataFrame):
    d = with_dates(dfx)
    latest = d.groupby("uprn").last().copy()
    return latest


def certificate_validity_robustness(dfx: pd.DataFrame | None = None,
                                    P: pd.DataFrame | None = None) -> pd.DataFrame:
    dfx = build_extended_frame() if dfx is None else dfx
    P = within_building_pairs(dfx) if P is None else P
    M = compute_multipliers(P)
    latest = _latest_stock(dfx)
    ar = latest.asset_rating.values.astype(float)
    post = (latest.insp_dt >= CUT).values
    fuel = latest.fuelgrp.values
    fa = latest.floor_area.values.astype(float)
    lodge = pd.to_datetime(latest.lodgement_date)
    valid10 = (lodge >= VALID_FROM).values

    subsets = [("All latest certificates", np.ones(len(ar), bool)),
               ("Lodged within 10 years", valid10),
               ("Post-2022 latest only", post),
               (">1000 m2 and within 10 years", valid10 & (fa > 1000))]
    rows = []
    for lab, mask in subsets:
        r = _stock_metrics(ar[mask], post[mask], fuel[mask], fa[mask], M)
        rows.append({"subset": lab, **r})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "robustness_certificate_validity.csv", index=False)
    return out


def country_split_robustness(dfx: pd.DataFrame | None = None,
                             P: pd.DataFrame | None = None) -> dict:
    dfx = build_extended_frame() if dfx is None else dfx
    P = within_building_pairs(dfx) if P is None else P
    M = compute_multipliers(P)
    d = with_dates(dfx)
    latest = d.groupby("uprn").last().copy()
    ar = latest.asset_rating.values.astype(float)
    post = (latest.insp_dt >= CUT).values
    fuel = latest.fuelgrp.values
    fa = latest.floor_area.values.astype(float)
    country = latest.country.values

    nat_rows = []
    for c in ["England", "Wales"]:
        m = country == c
        nat_rows.append({"country": c, **_stock_metrics(ar[m], post[m], fuel[m], fa[m], M)})
    nat = pd.DataFrame(nat_rows)
    nat.to_csv(OUT_TABLES / "robustness_country_national.csv", index=False)

    # within-building contrast + no-works multiplier by country
    uprn_country = d.groupby("uprn").country.first()
    Pc = P.copy()
    Pc["country"] = Pc.index.map(uprn_country)
    wb_rows = []
    for c in ["England", "Wales"]:
        S = Pc[Pc.straddle & (Pc.country == c)]
        e, g = S[S.fuel == "Electric"].dAR, S[S.fuel == "Gas"].dAR
        nw = Pc[Pc.straddle & (Pc.drec == 0) & (Pc.dfa < 0.02) & (Pc.country == c)]

        def _mult(fg):
            r = nw[nw.fuel == fg].ar_f / nw[nw.fuel == fg].ar_l
            r = r[(r > 0) & np.isfinite(r)]
            return round(float(r.median()), 3) if len(r) else np.nan, int(len(r))
        (mE, nE), (mG, nG) = _mult("Electric"), _mult("Gas")
        wb_rows.append({"country": c, "straddlers_n": int(len(S)),
                        "electric_dAR": round(e.mean(), 1), "gas_dAR": round(g.mean(), 1),
                        "elec_minus_gas": round(e.mean() - g.mean(), 1),
                        "noworks_n": int(len(nw)), "mult_electric": mE, "mult_gas": mG,
                        "noworks_n_electric": nE, "noworks_n_gas": nG})
    wb = pd.DataFrame(wb_rows)
    wb.to_csv(OUT_TABLES / "robustness_country_within_building.csv", index=False)
    return {"national": nat, "within_building": wb}


def stricter_no_works(dfx: pd.DataFrame | None = None,
                      P: pd.DataFrame | None = None) -> pd.DataFrame:
    dfx = build_extended_frame() if dfx is None else dfx
    P = within_building_pairs(dfx) if P is None else P
    d = with_dates(dfx)
    g = d.groupby("uprn")
    # ac_f already comes from within_building_pairs (identical definition), so it is not
    # recomputed here; recreating it collides on the join.
    agg = g.agg(fuel_f=("fuelgrp", "first"), fuel_l=("fuelgrp", "last"),
                ac_l=("aircon_present_clean", "last"),
                pay_f=("payback_triple", "first"), pay_l=("payback_triple", "last"),
                code_f=("rec_codes", "first"), code_l=("rec_codes", "last"))
    Px = P.join(agg)

    def _eq(a, b):
        return Px[a].astype(str).values == Px[b].astype(str).values

    strad_nw = Px.straddle.values & (Px.drec.values == 0) & (Px.dfa.values < 0.02)
    same_fuelac = _eq("fuel_f", "fuel_l") & _eq("ac_f", "ac_l")
    same_pay = _eq("pay_f", "pay_l")
    same_code = _eq("code_f", "code_l")
    defs = [("Baseline (drec=0, dFA<2%)", strad_nw),
            ("+ unchanged fuel and air-conditioning", strad_nw & same_fuelac),
            ("+ unchanged payback mix", strad_nw & same_fuelac & same_pay),
            ("+ identical recommendation-code set", strad_nw & same_fuelac & same_pay & same_code)]
    rows = []
    for lab, mask in defs:
        s = Px[mask]
        row = {"definition": lab, "n": int(len(s))}
        for fg in FUELS:
            r = s[s.fuel == fg].ar_f / s[s.fuel == fg].ar_l
            r = r[(r > 0) & np.isfinite(r)]
            row[f"mult_{fg.lower()}"] = round(float(r.median()), 3) if len(r) else np.nan
            row[f"n_{fg.lower()}"] = int(len(r))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "robustness_stricter_no_works.csv", index=False)
    return out


def run_robustness() -> dict:
    dfx = build_extended_frame()
    P = within_building_pairs(dfx)
    return {"certificate_validity": certificate_validity_robustness(dfx, P),
            "country_split": country_split_robustness(dfx, P),
            "stricter_no_works": stricter_no_works(dfx, P)}


if __name__ == "__main__":
    r = run_robustness()
    print("=== certificate validity ===")
    print(r["certificate_validity"].to_string(index=False))
    print("\n=== country: national ===")
    print(r["country_split"]["national"].to_string(index=False))
    print("\n=== country: within-building ===")
    print(r["country_split"]["within_building"].to_string(index=False))
    print("\n=== stricter no-works ===")
    print(r["stricter_no_works"].to_string(index=False))
    print("\nWrote robustness CSVs to", OUT_TABLES)
