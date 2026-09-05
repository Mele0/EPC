"""Statistical analyses for the EPC carbon-accounting artefact study.

Each function writes one or more result tables to ``OUT_TABLES``;
``run_analysis`` orchestrates the full pipeline. The analyses quantify how much
of the measured 2012-2025 improvement in the England & Wales non-domestic EPC
asset rating is carbon-factor re-accounting rather than physical building change,
using a within-building natural experiment around the June-2022 Part L 2021 /
SAP 10 electricity carbon-factor cut and DESNZ metered-energy ground truth.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import (OUT_TABLES, EXTERNAL_DIR, CUT, ELEC_FACTOR_OLD, ELEC_FACTOR_NEW,
                     BROAD, RANDOM_SEED, DESNZ_ELEC_URL, DESNZ_GAS_URL,
                     DESNZ_ELEC_XLSX, DESNZ_GAS_XLSX, ensure_dirs)
from .data import (broad_band, with_dates, build_analysis_frame,
                   download_if_missing, desnz_nd_per_meter)


# ── 1. Rating identity + register year trend ──────────────────────────────────
def identity_and_trends(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["asset_rating", "building_emissions", "standard_emissions"])
    d = d[d.standard_emissions > 0]
    ratio = d.building_emissions / d.standard_emissions
    k = (d.asset_rating / ratio).replace([np.inf, -np.inf], np.nan).median()
    r = float(d.asset_rating.corr(ratio))
    yr = (df.groupby("inspection_year")
            .agg(n=("asset_rating", "size"), AR=("asset_rating", "mean"),
                 BER=("building_emissions", "mean"), SER=("standard_emissions", "mean"),
                 pe=("primary_energy_value", "median")).reset_index())
    yr.to_csv(OUT_TABLES / "register_year_trend.csv", index=False)
    s = yr.set_index("inspection_year")
    return {"identity_k": round(float(k), 2), "identity_r": round(r, 4),
            "AR_2012": float(s.AR[2012]), "AR_2025": float(s.AR[2025])}


# ── 2. Within-building pairs ──────────────────────────────────────────────────
def within_building_pairs(df: pd.DataFrame) -> pd.DataFrame:
    d = with_dates(df)
    cnt = d.groupby("uprn")["asset_rating"].transform("size")
    rep = d[cnt >= 2]
    P = rep.groupby("uprn").agg(
        ar_f=("asset_rating", "first"), ar_l=("asset_rating", "last"),
        ber_f=("building_emissions", "first"), ber_l=("building_emissions", "last"),
        dt_f=("insp_dt", "first"), dt_l=("insp_dt", "last"),
        fuel=("fuelgrp", "first"), rec_f=("n_recommendations", "first"),
        rec_l=("n_recommendations", "last"), fa_f=("floor_area", "first"),
        fa_l=("floor_area", "last"), sector=("property_type_clean", "first"),
        ac_f=("aircon_present_clean", "first"),
        txn_l=("transaction_type_clean", "last"), n=("asset_rating", "size"))
    P["dAR"] = P.ar_l - P.ar_f
    P["gap"] = (P.dt_l - P.dt_f).dt.days / 365.25
    P = P[P.gap > 0.25]
    P["straddle"] = (P.dt_f < CUT) & (P.dt_l >= CUT)
    P["both_pre"] = (P.dt_f < CUT) & (P.dt_l < CUT)
    P["both_post"] = (P.dt_f >= CUT) & (P.dt_l >= CUT)
    P["drec"] = P.rec_l - P.rec_f
    P["dfa"] = (P.fa_l - P.fa_f).abs() / P.fa_f
    P["ber_ratio"] = P.ber_l / P.ber_f
    return P


def straddle_by_fuel(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grp, lab in [("straddle", "Straddle 2022"), ("both_pre", "Both pre-2022"),
                     ("both_post", "Both post-2022")]:
        for fg in ["Gas", "Electric", "Other", "All"]:
            s = P[P[grp]] if fg == "All" else P[P[grp] & (P.fuel == fg)]
            if len(s):
                rows.append([lab, fg, len(s), round(s.dAR.mean(), 2), round(s.gap.mean(), 2)])
    out = pd.DataFrame(rows, columns=["group", "fuel", "n", "mean_dAR", "mean_gap"])
    out.to_csv(OUT_TABLES / "within_building_straddle_by_fuel.csv", index=False)
    return out


def no_works_windfall(P: pd.DataFrame) -> dict:
    """Per-fuel 2022 accounting windfall from no-recorded-works straddlers (rating points)."""
    nw = P[P.straddle & (P.drec == 0) & (P.dfa < 0.02)]
    w = {fg: round(-nw[nw.fuel == fg].dAR.mean(), 2) for fg in ["Electric", "Gas", "Other"]}
    nw.groupby("fuel")["dAR"].agg(["size", "mean"]).round(2).to_csv(
        OUT_TABLES / "no_measure_change_straddlers.csv")
    return w


# ── 3. Within-building event study (electric vs gas, building FE) ──────────────
def event_study(df: pd.DataFrame) -> pd.DataFrame:
    d = with_dates(df)
    d = d[d.fuelgrp.isin(["Electric", "Gas"])]
    cnt = d.groupby("uprn")["asset_rating"].transform("size")
    d = d[cnt >= 2].copy()
    d["electric"] = (d.fuelgrp == "Electric").astype(float)
    yrs = [y for y in range(2012, 2026) if y != 2021]
    D = pd.get_dummies(d.inspection_year.astype(int), prefix="y")[[f"y_{y}" for y in yrs]].astype(float)
    DE = D.mul(d.electric.values, axis=0)
    DE.columns = [c + "_xElec" for c in D.columns]
    X = pd.concat([D, DE], axis=1)
    X.index = d.index
    grp = d.uprn.values
    Xd = X - X.groupby(grp).transform("mean")
    y = d.asset_rating.astype(float)
    yd = y - y.groupby(grp).transform("mean")
    m = sm.OLS(yd.values, Xd.values).fit(cov_type="cluster", cov_kwds={"groups": grp})
    coef = dict(zip(Xd.columns, m.params)); se = dict(zip(Xd.columns, m.bse))
    rows = [[yv, round(coef[f"y_{yv}_xElec"], 2), round(se[f"y_{yv}_xElec"], 2)] for yv in yrs]
    ev = pd.DataFrame(rows, columns=["year", "elec_gas_differential", "se"])
    ev.loc[len(ev)] = [2021, 0.0, 0.0]
    ev = ev.sort_values("year").reset_index(drop=True)
    ev.to_csv(OUT_TABLES / "event_study_electric_gas.csv", index=False)
    return ev


# ── 4. Emissions-ratio validation: within-building emissions ratio vs the factor cut ──
def emissions_ratio_validation(P: pd.DataFrame) -> dict:
    """Compare the predicted electricity re-pricing ratio (new/old carbon factor) with the
    observed within-building modelled-emissions ratio among no-recorded-works straddlers."""
    nw = P[P.straddle & (P.drec == 0) & (P.dfa < 0.02)]
    out = {"predicted_elec_ratio": round(ELEC_FACTOR_NEW / ELEC_FACTOR_OLD, 3)}
    for fg in ["Electric", "Gas", "Other"]:
        s = nw[nw.fuel == fg].ber_ratio
        s = s[(s > 0) & (s < 5)]
        out[f"observed_{fg.lower()}_ratio"] = round(float(s.median()), 3) if len(s) else None
    json.dump(out, open(OUT_TABLES / "emissions_ratio_validation.json", "w"), indent=2)
    return out


def decomposition_shares(P: pd.DataFrame) -> dict:
    """Decompose each straddler's measured asset-rating change into a formula component
    (per-fuel carbon re-pricing, calibrated on no-recorded-works buildings) and a residual
    building component, and report the aggregate improvement-point shares."""
    mult = compute_multipliers(P)
    S = P[P.straddle & P.fuel.isin(["Electric", "Gas", "Other"]) & (P.ar_l > 0)].copy()
    S["m"] = S.fuel.map(mult)
    S["formula_comp"] = S.ar_l - S.ar_l * S.m          # observed post minus old-accounting post
    S["real_comp"] = S.ar_l * S.m - S.ar_f             # remainder (physical + unmodelled)

    def split(sub: pd.DataFrame) -> dict:
        tot = -float(sub.dAR.sum())
        return {"n": int(len(sub)), "mean_dAR": round(float(sub.dAR.mean()), 2),
                "formula_share_%": round(100 * (-float(sub.formula_comp.sum())) / tot, 1) if tot else None,
                "building_share_%": round(100 * (-float(sub.real_comp.sum())) / tot, 1) if tot else None}

    changed, improved = S[S.dAR.abs() >= 1], S[S.dAR < 0]
    imp_pct = (100 * improved.formula_comp / improved.dAR).replace([np.inf, -np.inf], np.nan)
    decomp = {"multiplier_per_fuel": {k: round(v, 3) for k, v in mult.items()},
              "all_straddlers": split(S), "changed_only": split(changed), "improved_only": split(improved),
              "by_fuel_changed": {fg: split(changed[changed.fuel == fg]) for fg in ["Electric", "Gas", "Other"]},
              "improved_median_formula_pct_per_bldg": round(float(imp_pct.median()), 1),
              "improved_entirely_by_formula_%": round(100 * (improved.real_comp >= 0).mean(), 1),
              "improved_majority_formula_%": round(100 * (imp_pct >= 50).mean(), 1)}
    out = {"decomposition": decomp}
    json.dump(out, open(OUT_TABLES / "section_percentage.json", "w"), indent=2)
    return out


# ── 5. Dose-response (electricity exposure) ───────────────────────────────────
def dose_response(df: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    d = with_dates(df)
    has_ac = d.aircon_present_clean.astype(str).str.lower().str.contains("y|true|1|present", regex=True)
    ac_first = has_ac.groupby(d.uprn).first()
    S = P[P.straddle & P.fuel.isin(["Gas", "Electric"])].copy()
    S["ac"] = S.index.map(ac_first).fillna(False)
    S["exposure"] = np.select(
        [(S.fuel == "Gas") & (~S.ac), (S.fuel == "Gas") & (S.ac),
         (S.fuel == "Electric") & (~S.ac), (S.fuel == "Electric") & (S.ac)],
        ["1 Gas, no AC", "2 Gas + AC", "3 Electric, no AC", "4 Electric + AC"], default="0 other")
    dr = S[S.exposure != "0 other"].groupby("exposure")["dAR"].agg(["size", "mean"]).round(2)
    dr.to_csv(OUT_TABLES / "dose_response_exposure.csv")
    return dr


# ── 6. Redistribution of compliance progress ──────────────────────────────────
def redistribution(df: pd.DataFrame, P: pd.DataFrame) -> dict:
    latest = with_dates(df).groupby("uprn").last()
    st = P[P.straddle].copy()
    st["into_B"] = (st.ar_f > 50) & (st.ar_l <= 50)
    out = {"n_into_B": int(st.into_B.sum()),
           "pct_electric_all_stock": round(100 * (latest.fuelgrp == "Electric").mean(), 1),
           "pct_electric_among_into_B": round(100 * (st[st.into_B].fuel == "Electric").mean(), 1),
           "pct_electric_among_straddlers": round(100 * (st.fuel == "Electric").mean(), 1)}
    json.dump(out, open(OUT_TABLES / "redistribution.json", "w"), indent=2)
    return out


# ── 7. Corrected 2030 stress test (recertification Markov) ────────────────────
def _transition_matrices(d: pd.DataFrame):
    d = d.copy()
    g = d.groupby("uprn")
    d["prev_ar"] = g.asset_rating.shift(1)
    d["prev_dt"] = g.insp_dt.shift(1)
    tr = d.dropna(subset=["prev_ar"]).copy()
    tr["prev_band"] = broad_band(tr.prev_ar.values)
    tr["cur_band"] = broad_band(tr.asset_rating.values)
    clean = tr[(tr.prev_dt >= CUT) & (tr.insp_dt >= CUT)]

    def mat(sub):
        m = pd.DataFrame(1.0, index=BROAD, columns=BROAD)   # Laplace smoothing
        for (s, t), c in sub.groupby(["prev_band", "cur_band"]).size().items():
            if s in m.index and t in m.columns:
                m.loc[s, t] += c
        return m.div(m.sum(axis=1), axis=0)
    return mat(tr), mat(clean)


def corrected_stress_test(df: pd.DataFrame, windfall: dict) -> pd.DataFrame:
    d = with_dates(df)
    hot = d[d.property_type_clean.astype(str).str.contains("Hotel", case=False)]
    latest = hot.groupby("uprn").last().copy()
    head = pd.Series(broad_band(latest.asset_rating.values)).value_counts(normalize=True).reindex(BROAD).fillna(0)
    post = latest.insp_dt >= CUT
    ar_corr = latest.asset_rating + np.where(post, latest.fuelgrp.map(windfall).fillna(windfall["Other"]).values, 0.0)
    corr = pd.Series(broad_band(ar_corr.values)).value_counts(normalize=True).reindex(BROAD).fillna(0)
    hT_all, _ = _transition_matrices(hot)
    _, T_post = _transition_matrices(d)          # register-wide, windfall-free engine
    hT_all.round(4).to_csv(OUT_TABLES / "transition_matrix_contaminated.csv")
    T_post.round(4).to_csv(OUT_TABLES / "transition_matrix_clean_steadystate.csv")

    def project(stock, T, p):
        return 100 * sum(stock[b] * ((1 - p) * (0 if b == "A/B" else 1)
                         + p * T.loc[b, ["C", "D", "E", "F/G"]].sum()) for b in BROAD)
    rows = [[p, round(project(head, hT_all, p), 1), round(project(corr, T_post, p), 1)]
            for p in [0.10, 0.157, 0.20, 0.35, 0.50, 0.75, 1.0]]
    proj = pd.DataFrame(rows, columns=["recert_throughput_p", "headline_2030_belowB_%", "corrected_2030_belowB_%"])
    proj["gap_pp"] = (proj["corrected_2030_belowB_%"] - proj["headline_2030_belowB_%"]).round(1)
    proj.to_csv(OUT_TABLES / "corrected_2030_stress_test.csv", index=False)
    json.dump({"hotel_belowB_headline": round(100 * (latest.asset_rating > 50).mean(), 1),
               "hotel_belowB_corrected": round(100 * (ar_corr > 50).mean(), 1),
               "AB_headline": round(100 * head["A/B"], 1), "AB_corrected": round(100 * corr["A/B"], 1)},
              open(OUT_TABLES / "corrected_stress_summary.json", "w"), indent=2)
    return proj


def commensurate_stress_test(df: pd.DataFrame, P: pd.DataFrame) -> dict:
    """Illustrative reassessment-throughput sensitivity with every band state on
    a single accounting basis within each scenario (rescuing the Markov exercise
    from mixing bases). Two scenarios project the below-EPC-B share of the
    latest-stock frame as the reassessed share p rises:

      Observed / current-accounting -- starting distribution from observed latest
        certificates and transition matrix from observed post-2022 reassessments
        (both on the reported post-revision basis).
      Constant-accounting -- every rating in the starting distribution and every
        previous/current rating in the transition matrix converted to the pre-2022
        basis (pre-2022 certificates unchanged; post-2022 rescaled by the
        fixed-accounting multiplier) before the transition matrix is estimated.

    Illustrative only: not a forecast of legal compliance or physical retrofit."""
    m = compute_multipliers(P)
    d = with_dates(df)
    latest = d.groupby("uprn").last()
    ar = latest.asset_rating.values.astype(float)
    post = (latest.insp_dt >= CUT).values
    mult = latest.fuelgrp.where(latest.fuelgrp.isin(FUELS), "Other").map(m).values

    def band_dist(a):
        return pd.Series(broad_band(np.asarray(a, float))).value_counts(normalize=True).reindex(BROAD).fillna(0)

    s0_obs = band_dist(ar)
    s0_cc = band_dist(ar * np.where(post, mult, 1.0))

    d2 = d.sort_values("insp_dt")
    g = d2.groupby("uprn")
    pairs = pd.DataFrame({"prev": g.asset_rating.shift(1), "prev_dt": g.insp_dt.shift(1),
                          "cur": d2.asset_rating, "cur_dt": d2.insp_dt,
                          "fuel": g.fuelgrp.transform("first")}).dropna(subset=["prev"])
    pairs["fm"] = pairs.fuel.where(pairs.fuel.isin(FUELS), "Other").map(m)

    def tmat(pb, cb):
        T = pd.DataFrame(1.0, index=BROAD, columns=BROAD)
        for s, t in zip(pb, cb):
            T.loc[s, t] += 1
        return T.div(T.sum(axis=1), axis=0)

    po = pairs[(pairs.prev_dt >= CUT) & (pairs.cur_dt >= CUT)]
    T_obs = tmat(broad_band(po.prev.values), broad_band(po.cur.values))
    prev_cc = pairs.prev.values * np.where(pairs.prev_dt.values >= CUT, pairs.fm.values, 1.0)
    cur_cc = pairs.cur.values * np.where(pairs.cur_dt.values >= CUT, pairs.fm.values, 1.0)
    T_cc = tmat(broad_band(prev_cc), broad_band(cur_cc))
    T_obs.round(4).to_csv(OUT_TABLES / "stress_transition_observed_basis.csv")
    T_cc.round(4).to_csv(OUT_TABLES / "stress_transition_constant_basis.csv")

    def project(stock, T, p):
        return 100 * sum(stock[b] * ((1 - p) * (0 if b == "A/B" else 1)
                         + p * T.loc[b, ["C", "D", "E", "F/G"]].sum()) for b in BROAD)

    rows = []
    for p in [0.10, 0.157, 0.20, 0.35, 0.50, 0.75, 1.0]:
        o, c = project(s0_obs, T_obs, p), project(s0_cc, T_cc, p)
        rows.append([p, round(o, 1), round(c, 1), round(c - o, 1)])
    proj = pd.DataFrame(rows, columns=["reassessment_throughput_p", "observed_basis_belowB_%",
                                       "constant_accounting_belowB_%", "gap_pp"])
    proj.to_csv(OUT_TABLES / "commensurate_stress_test.csv", index=False)

    diag = pd.DataFrame([
        {"scenario": "Observed / current-accounting", "starting_distribution_basis": "observed",
         "transition_matrix_basis": "observed (post-2022 pairs)", "commensurate": "yes"},
        {"scenario": "Constant-accounting", "starting_distribution_basis": "pre-2022 (fixed-accounting)",
         "transition_matrix_basis": "pre-2022 (both ratings converted)", "commensurate": "yes"}])
    diag.to_csv(OUT_TABLES / "commensurate_stress_diagnostic.csv", index=False)
    return {"projection": proj, "diagnostic": diag}


# ── 8. National policy consequence + bootstrap CIs ────────────────────────────
def national_consequence(df: pd.DataFrame, P: pd.DataFrame, windfall: dict, n_boot: int = 1000) -> dict:
    d = with_dates(df)
    latest = d.groupby("uprn").last()
    ar = latest.asset_rating.values
    post = (latest.insp_dt >= CUT).values
    fcode = latest.fuelgrp.map({"Electric": 0, "Gas": 1, "Other": 2}).fillna(2).astype(int).values
    w0 = np.array([windfall["Electric"], windfall["Gas"], windfall["Other"]])
    ar_old = ar + np.where(post, w0[fcode], 0.0)
    point = {"n_latest_units": int(len(latest)),
             "FG_substandard_headline_%": round(100 * (ar > 125).mean(), 2),
             "FG_substandard_corrected_%": round(100 * (ar_old > 125).mean(), 2),
             "belowB_headline_%": round(100 * (ar > 50).mean(), 1),
             "belowB_corrected_%": round(100 * (ar_old > 50).mean(), 1),
             "n_exited_legal_FG_via_accounting": int(((ar <= 125) & (ar_old > 125)).sum()),
             "n_reached_EPC_B_via_accounting": int(((ar <= 50) & (ar_old > 50)).sum())}
    json.dump(point, open(OUT_TABLES / "national_policy_consequence.json", "w"), indent=2)
    nw = P[P.straddle & (P.drec == 0) & (P.dfa < 0.02) & P.fuel.isin(["Electric", "Gas", "Other"])]
    by = {fg: nw[nw.fuel == fg].dAR.values for fg in ["Electric", "Gas", "Other"]}
    rng = np.random.default_rng(RANDOM_SEED)
    out = []
    for _ in range(n_boot):
        w = np.array([-rng.choice(by["Electric"], size=len(by["Electric"])).mean(),
                      -rng.choice(by["Gas"], size=len(by["Gas"])).mean(),
                      -rng.choice(by["Other"], size=len(by["Other"])).mean()])
        a = ar + np.where(post, w[fcode], 0.0)
        out.append([100 * (a > 125).mean(), 100 * (a > 50).mean(),
                    int(((ar <= 125) & (a > 125)).sum()), int(((ar <= 50) & (a > 50)).sum())])
    out = np.array(out)
    labels = ["FG_substandard_corrected_%", "belowB_corrected_%", "n_exited_legal_FG", "n_reached_EPC_B"]
    ci = {l: [round(np.percentile(out[:, i], 2.5), 1), round(np.percentile(out[:, i], 50), 1),
              round(np.percentile(out[:, i], 97.5), 1)] for i, l in enumerate(labels)}
    json.dump(ci, open(OUT_TABLES / "national_consequence_bootstrap_CI.json", "w"), indent=2)
    return {"point": point, "ci": ci}


# ── 9. Metered ground-truth validation (DESNZ) ────────────────────────────────
def metered_validation() -> pd.DataFrame:
    elec_csv = EXTERNAL_DIR / "metered_nondomestic_electricity_EW.csv"
    gas_csv = EXTERNAL_DIR / "metered_nondomestic_gas_EW.csv"
    if elec_csv.is_file() and gas_csv.is_file():
        # The release Data Asset contains the exact derived series used by the
        # manuscript.  Prefer those immutable snapshots to mutable live workbooks.
        elec = pd.read_csv(elec_csv)
        gas = pd.read_csv(gas_csv)
    else:
        download_if_missing(DESNZ_ELEC_URL, DESNZ_ELEC_XLSX)
        download_if_missing(DESNZ_GAS_URL, DESNZ_GAS_XLSX)
        elec = desnz_nd_per_meter(DESNZ_ELEC_XLSX, "electricity")
        gas = desnz_nd_per_meter(DESNZ_GAS_XLSX, "gas")
        elec.to_csv(elec_csv, index=False)
        gas.to_csv(gas_csv, index=False)
    required_elec = {"year", "nd_electricity_mean_kWh_per_meter"}
    required_gas = {"year", "nd_gas_mean_kWh_per_meter"}
    if not required_elec.issubset(elec.columns) or not required_gas.issubset(gas.columns):
        raise ValueError("Frozen DESNZ metered series have an unexpected schema")
    epc = pd.read_csv(OUT_TABLES / "register_year_trend.csv")
    m = epc.merge(elec, left_on="inspection_year", right_on="year")
    b = int(m.year.min())
    g = lambda col, yr: m.loc[m.year == yr, col].iloc[0]
    m["metered_idx"] = 100 * m["nd_electricity_mean_kWh_per_meter"] / g("nd_electricity_mean_kWh_per_meter", b)
    m["epc_CO2_idx"] = 100 * m.BER / g("BER", b)
    m["epc_AR_idx"] = 100 * m.AR / g("AR", b)
    m["epc_PE_idx"] = 100 * m.pe / g("pe", b)
    m.to_csv(OUT_TABLES / "metered_vs_epc_validation.csv", index=False)
    return m


# ── 10b. Repeat-certificate cohort funnel ──────────────────────────────────────
def cohort_funnel(df: pd.DataFrame, P: pd.DataFrame) -> dict:
    """Counts through the repeat-certificate cohort: certificates -> unique
    buildings -> buildings with >=2 valid certificates >=0.25yr apart (the
    within-building analysis cohort) -> pre/straddle/post-2022 split, and the
    straddle split into improved/worsened/unchanged asset rating."""
    st = P[P.straddle]
    out = {"n_certificates": int(len(df)), "n_buildings": int(df.uprn.nunique()),
           "n_reassessed_buildings": int(len(P)), "n_both_pre2022": int(P.both_pre.sum()),
           "n_straddle_2022": int(P.straddle.sum()), "n_both_post2022": int(P.both_post.sum()),
           "straddle_improved": int((st.dAR < 0).sum()), "straddle_worsened": int((st.dAR > 0).sum()),
           "straddle_nochange": int((st.dAR == 0).sum())}
    json.dump(out, open(OUT_TABLES / "cohort_funnel.json", "w"), indent=2)
    return out


# ── 10c. Fixed-factor recomputation (per-fuel re-pricing multiplier) ──────────
FUELS = ["Electric", "Gas", "Other"]


def _no_works_ratios(P: pd.DataFrame, dfa_tol: float = 0.02) -> dict:
    """Per-fuel arrays of the no-recorded-works re-pricing ratio AR_pre/AR_post.

    The calibration sample is the set of buildings that straddle the June-2022
    accounting change, gained no recommendations and changed floor area by less
    than ``dfa_tol``. Their asset-rating ratio isolates the carbon re-pricing
    from physical change; the per-fuel median is the fixed-factor multiplier."""
    nw = P[P.straddle & (P.drec == 0) & (P.dfa < dfa_tol) & (P.ar_l > 0)
           & P.fuel.isin(FUELS)]
    out = {}
    for fg in FUELS:
        r = (nw.loc[nw.fuel == fg, "ar_f"] / nw.loc[nw.fuel == fg, "ar_l"])
        out[fg] = r[(r > 0) & np.isfinite(r)].values
    return out


def compute_multipliers(P: pd.DataFrame, dfa_tol: float = 0.02) -> dict:
    """Fixed-factor multipliers: the per-fuel median AR_pre/AR_post among
    no-recorded-works straddlers. Computed from the register, not hard-coded."""
    return {fg: float(np.median(r)) if len(r) else float("nan")
            for fg, r in _no_works_ratios(P, dfa_tol).items()}


def fixed_factor_recompute(df: pd.DataFrame, P: pd.DataFrame, n_boot: int = 1000) -> dict:
    """National stock re-priced under constant (pre-2022) carbon accounting via the
    per-fuel median re-pricing multiplier: band distribution, EPC-B / legal-floor
    on-paper crossings, their fuel composition, the within-building crossings that
    dissolve, and multiplicative bootstrap intervals consistent with the point
    estimate (resampling the no-recorded-works calibration sample within fuel)."""
    ratios = _no_works_ratios(P)
    mult = {fg: float(np.median(ratios[fg])) for fg in FUELS}
    pd.DataFrame({"fuel": FUELS,
                  "multiplier_median_AR_pre_over_post": [round(mult[f], 4) for f in FUELS],
                  "n_calibration": [len(ratios[f]) for f in FUELS]}
                 ).to_csv(OUT_TABLES / "fixed_factor_multipliers.csv", index=False)

    d = with_dates(df)
    latest = d.groupby("uprn").last().copy()
    ar = latest.asset_rating.values.astype(float)
    post = (latest.insp_dt >= CUT).values
    fcode = latest.fuelgrp.map({f: i for i, f in enumerate(FUELS)}).fillna(2).astype(int).values
    m = np.array([mult[f] for f in FUELS])[fcode]
    ar_const = np.where(post, ar * m, ar)
    obs_band = pd.Series(broad_band(ar)).value_counts(normalize=True).reindex(BROAD).fillna(0) * 100
    const_band = pd.Series(broad_band(ar_const)).value_counts(normalize=True).reindex(BROAD).fillna(0) * 100
    reachedB = (ar <= 50) & (ar_const > 50)
    clearedFG = (ar <= 125) & (ar_const > 125)
    fuel = latest.fuelgrp.values

    def fuel_comp(mask):
        s = pd.Series(fuel[mask]).value_counts()
        return {fg: int(s.get(fg, 0)) for fg in FUELS}

    # Within-building EPC-B crossings that dissolve under constant accounting.
    cr = P[P.straddle & (P.ar_f > 50) & (P.ar_l <= 50)]
    cr_m = cr.fuel.map(mult).fillna(mult["Other"]).values
    cr_dissolve = cr.ar_l.values * cr_m > 50

    # Multiplicative bootstrap: resample the calibration sample within fuel,
    # re-estimate the median multiplier, recompute the national quantities.
    rng = np.random.default_rng(RANDOM_SEED)
    boot = []
    for replicate in range(n_boot):
        mv = np.array([np.median(rng.choice(ratios[f], size=len(ratios[f]))) for f in FUELS])
        a = np.where(post, ar * mv[fcode], ar)
        boot.append([replicate, *mv, 100 * (a > 125).mean(), 100 * (a > 50).mean(),
                     int(((ar <= 125) & (a > 125)).sum()),
                     int(((ar <= 50) & (a > 50)).sum())])
    ci_labels = ["FG_fixedfactor_%", "belowB_fixedfactor_%", "n_cleared_FG_on_paper", "n_reached_B_on_paper"]
    boot_columns = ["replicate", *(f"multiplier_{f}" for f in FUELS), *ci_labels]
    boot_frame = pd.DataFrame(boot, columns=boot_columns)
    boot_frame.to_csv(OUT_TABLES / "fixed_factor_bootstrap_draws.csv", index=False)
    boot_values = boot_frame[ci_labels].to_numpy()
    ci = {l: [round(np.percentile(boot_values[:, i], 2.5), 1 if i < 2 else 0),
              round(np.percentile(boot_values[:, i], 50), 1 if i < 2 else 0),
              round(np.percentile(boot_values[:, i], 97.5), 1 if i < 2 else 0)]
          for i, l in enumerate(ci_labels)}

    out = {"multipliers": {f: round(mult[f], 4) for f in FUELS},
           "n_calibration": {f: len(ratios[f]) for f in FUELS},
           "n_stock": int(len(latest)),
           "band_observed_%": obs_band.round(2).to_dict(),
           "band_constant_accounting_%": const_band.round(2).to_dict(),
           "belowB_observed_%": round(100 * (ar > 50).mean(), 2),
           "belowB_fixedfactor_%": round(100 * (ar_const > 50).mean(), 2),
           "FG_observed_%": round(100 * (ar > 125).mean(), 2),
           "FG_fixedfactor_%": round(100 * (ar_const > 125).mean(), 2),
           "n_reached_B_on_paper": int(reachedB.sum()),
           "n_cleared_FG_on_paper": int(clearedFG.sum()),
           "n_within_building_crossings": int(len(cr)),
           "n_crossings_dissolving": int(cr_dissolve.sum()),
           "pct_crossings_dissolving": round(100 * cr_dissolve.mean(), 1),
           "reachedB_fuel_composition": fuel_comp(reachedB),
           "clearedFG_fuel_composition": fuel_comp(clearedFG),
           "bootstrap": {"replicates": int(n_boot), "seed": int(RANDOM_SEED),
                         "rng": "numpy.random.default_rng",
                         "resampling": "within fuel with replacement"},
           "bootstrap_CI": ci}
    with (OUT_TABLES / "fixed_factor_recompute.json").open("w", encoding="utf-8") as stream:
        json.dump(out, stream, indent=2)
    return out


def capex_equivalent(df: pd.DataFrame, P: pd.DataFrame,
                     cost_per_m2: float = 100.0, ia_snpv_bn: float = 4.7) -> dict:
    """Capex-equivalent of the on-paper threshold crossings: the floor area that
    would have to be physically upgraded to genuinely reach each threshold, priced
    at an assumed band-upgrade benchmark (default £100/m2). Floor areas are reported
    per fuel for the stacked-bar figure. The reached-EPC-B and cleared-legal-floor
    sets are disjoint (an asset rating at or below the legal floor cannot also sit at
    or below the EPC-B line and cross only one), so their union equals their sum."""
    mult = compute_multipliers(P)
    d = with_dates(df)
    latest = d.groupby("uprn").last().copy()
    ar = latest.asset_rating.values.astype(float)
    post = (latest.insp_dt >= CUT).values
    fa = pd.to_numeric(latest.floor_area, errors="coerce").values
    fuel = latest.fuelgrp.values
    m = latest.fuelgrp.map(mult).fillna(mult["Other"]).values
    ar_const = np.where(post, ar * m, ar)
    sets = {"reached_B": (ar <= 50) & (ar_const > 50),
            "cleared_floor": (ar <= 125) & (ar_const > 125)}
    out = {"cost_per_m2": cost_per_m2, "IA_SNPV_bn": ia_snpv_bn,
           "overlap_n": int((sets["reached_B"] & sets["cleared_floor"]).sum())}
    for lab, mask in sets.items():
        by_m2 = {fg: float(np.nansum(fa[mask & (fuel == fg)])) for fg in FUELS}
        tot = float(np.nansum(fa[mask]))
        out[lab] = {"n": int(mask.sum()), "m2_by_fuel": by_m2, "m2_total": tot,
                    "capex_bn_by_fuel": {fg: by_m2[fg] * cost_per_m2 / 1e9 for fg in FUELS},
                    "capex_bn": tot * cost_per_m2 / 1e9}
    out["combined_capex_bn"] = out["reached_B"]["capex_bn"] + out["cleared_floor"]["capex_bn"]
    json.dump(out, open(OUT_TABLES / "capex_equivalent.json", "w"), indent=2)
    return out


def multiplier_robustness(df: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    """Re-estimate the fixed-factor counterfactual under progressively more granular
    multiplier definitions (Supplementary robustness table). Each specification maps
    every latest-stock building to a multiplier; sparse cells (< ``MIN_CELL``
    calibration observations) fall back to the fuel-level multiplier. Reports the
    within-building crossings that dissolve and the latest-stock on-paper counts."""
    MIN_CELL = 50
    d = with_dates(df)
    latest = d.groupby("uprn").last().copy()
    ar = latest.asset_rating.values.astype(float)
    post = (latest.insp_dt >= CUT).values
    cr = P[P.straddle & (P.ar_f > 50) & (P.ar_l <= 50)].copy()

    def _tally(stock_mult: np.ndarray, cross_mult: np.ndarray) -> list:
        a = np.where(post, ar * stock_mult, ar)
        dissolve = cr.ar_l.values * cross_mult > 50
        return [round(100 * dissolve.mean(), 1),
                int(((ar <= 50) & (a > 50)).sum()),
                int(((ar <= 125) & (a > 125)).sum())]

    def _fa_bin(fa):
        return pd.cut(fa, [0, 500, 2000, np.inf], labels=["s", "m", "l"])

    def _ac(frame):
        return (frame.get("aircon_present_clean", pd.Series(index=frame.index, dtype=object))
                .astype(str).str.lower().str.contains("y|true|1|present", regex=True).fillna(False))

    stock_fuel = latest.fuelgrp.where(latest.fuelgrp.isin(FUELS), "Other")
    cross_fuel = cr.fuel.where(cr.fuel.isin(FUELS), "Other")
    base = compute_multipliers(P)

    def _grouped(keyfn_stock, keyfn_cross):
        """Estimate multipliers on fuel x extra-key cells of the calibration sample."""
        nw = P[P.straddle & (P.drec == 0) & (P.dfa < 0.02) & (P.ar_l > 0) & P.fuel.isin(FUELS)].copy()
        nw_key = keyfn_cross(nw)
        r = nw.ar_f / nw.ar_l
        r = r.where((r > 0) & np.isfinite(r))
        cell = r.groupby([nw.fuel, nw_key]).agg(["median", "size"])
        table = {k: (v["median"] if v["size"] >= MIN_CELL else base[k[0]]) for k, v in cell.iterrows()}

        def lookup(fuels, keys):
            return np.array([table.get((f, k), base.get(f, base["Other"])) for f, k in zip(fuels, keys)])
        return lookup

    rows = [["Baseline (fuel only)"] + _tally(stock_fuel.map(base).values, cross_fuel.map(base).values)]

    lk = _grouped(None, lambda x: x.get("sector", x.get("property_type_clean")))
    rows.append(["Fuel x sector"] + _tally(
        lk(stock_fuel.values, latest.get("property_type_clean")),
        lk(cross_fuel.values, cr.get("sector"))))

    lk = _grouped(None, lambda x: _fa_bin(x.fa_f if "fa_f" in x else x.floor_area))
    rows.append(["Fuel x floor-area bin"] + _tally(
        lk(stock_fuel.values, _fa_bin(latest.floor_area)),
        lk(cross_fuel.values, _fa_bin(cr.fa_f))))

    ac_stock = _ac(latest)
    nwP = P[P.straddle & (P.drec == 0) & (P.dfa < 0.02) & (P.ar_l > 0) & P.fuel.isin(FUELS)]
    # AC status is not carried on P; approximate via the latest-certificate flag by uprn.
    ac_first = _ac(d).groupby(d.uprn).first()
    lk = _grouped(None, lambda x: x.index.map(ac_first).fillna(False))
    rows.append(["Fuel x air-conditioning status"] + _tally(
        lk(stock_fuel.values, ac_stock.values),
        lk(cross_fuel.values, cr.index.map(ac_first).fillna(False).values)))

    for tol, lab in [(0.01, "1%"), (0.05, "5%")]:
        m = compute_multipliers(P, dfa_tol=tol)
        rows.append([f"No-works tolerance |dFA|<{lab}"] + _tally(
            stock_fuel.map(m).values, cross_fuel.map(m).values))

    # Cross-fitted: estimate multipliers on a random half of the calibration sample
    # and apply out of sample, repeated over 200 deterministic splits; report the
    # median and 2.5-97.5 percentile range (a single split is seed-sensitive).
    rr = (nwP.ar_f / nwP.ar_l).values
    rvalid = (rr > 0) & np.isfinite(rr)
    fvals = nwP.fuel.values
    cf_stack = []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        half = rng.random(len(nwP)) < 0.5
        m_cf = {}
        for fg in FUELS:
            sel = half & rvalid & (fvals == fg)
            m_cf[fg] = float(np.median(rr[sel])) if sel.any() else base[fg]
        cf_stack.append(_tally(stock_fuel.map(m_cf).values, cross_fuel.map(m_cf).values))
    cf_stack = np.array(cf_stack, dtype=float)
    cf_med = np.median(cf_stack, axis=0)
    cf_lo = np.percentile(cf_stack, 2.5, axis=0)
    cf_hi = np.percentile(cf_stack, 97.5, axis=0)
    rows.append(["Cross-fitted (200 splits, median)",
                 round(cf_med[0], 1), int(round(cf_med[1])), int(round(cf_med[2]))])
    # companion interval row (percentile band) for the supplementary caption
    cf_interval = {"crossings_dissolving_%": [round(cf_lo[0], 1), round(cf_hi[0], 1)],
                   "reached_B_n": [int(cf_lo[1]), int(cf_hi[1])],
                   "cleared_floor_n": [int(cf_lo[2]), int(cf_hi[2])]}
    json.dump(cf_interval, open(OUT_TABLES / "multiplier_robustness_crossfit_ci.json", "w"), indent=2)

    # Confounded: the multiplier is estimated per pre-revision rating band, but a
    # building's pre-revision band is exactly the counterfactual we are trying to
    # recover, so it is proxied by the current band -- keying both stock and
    # crossings on the current rating. Regression to the mean then collapses the
    # estimate, which is why this specification is invalid.
    _band = lambda v: pd.cut(v, [0, 50, 100, np.inf], labels=["ab", "cd", "efg"])
    lk = _grouped(None, lambda x: _band(x.ar_f))
    rows.append(["Fuel x pre-rating band (confounded)"] + _tally(
        lk(stock_fuel.values, _band(ar)),
        lk(cross_fuel.values, _band(cr.ar_l))))

    # Additive windfall (de-step) alternative.
    w = no_works_windfall(P)
    a_stock = ar + np.where(post, stock_fuel.map(w).values, 0.0)
    a_cross = cr.ar_l.values + cross_fuel.map(w).values
    rows.append(["Additive windfall (alternative)",
                 round(100 * (a_cross > 50).mean(), 1),
                 int(((ar <= 50) & (a_stock > 50)).sum()),
                 int(((ar <= 125) & (a_stock > 125)).sum())])

    out = pd.DataFrame(rows, columns=["specification", "crossings_dissolving_%",
                                      "reached_B_n", "cleared_floor_n"])
    out.to_csv(OUT_TABLES / "multiplier_robustness.csv", index=False)
    return out


def multiplier_components(P: pd.DataFrame) -> pd.DataFrame:
    """Supplementary: components of the per-fuel re-pricing multiplier on the
    no-recorded-works calibration sample (N, AR multiplier, BER ratio, SER ratio).
    Uses AR = 50*BER/SER so SER ratio = BER ratio * (AR_pre/AR_post)."""
    nw = P[P.straddle & (P.drec == 0) & (P.dfa < 0.02) & (P.ar_l > 0)].copy()
    nw["m"] = nw.ar_f / nw.ar_l
    nw = nw[(nw.m > 0) & np.isfinite(nw.m)]
    nw["ser_ratio"] = nw.ber_ratio * nw.m
    rows = []
    for fg in FUELS:
        s = nw[nw.fuel == fg]
        ber = s.ber_ratio[(s.ber_ratio > 0) & (s.ber_ratio < 5)]
        rows.append([fg, int(len(s)), round(float(s.m.median()), 3),
                     round(float(ber.median()), 3) if len(ber) else np.nan,
                     round(float(s.ser_ratio.median()), 3)])
    out = pd.DataFrame(rows, columns=["fuel", "n", "AR_multiplier", "BER_ratio", "SER_ratio"])
    out.to_csv(OUT_TABLES / "multiplier_components.csv", index=False)
    return out


def local_cutoff(P: pd.DataFrame) -> pd.DataFrame:
    """Supplementary: within-building dAR by fuel for straddlers whose two
    certificates both fall within +/- k months of the June-2022 cut."""
    S = P[P.straddle].copy()
    S["days_pre"] = (CUT - S.dt_f).dt.days
    S["days_post"] = (S.dt_l - CUT).dt.days
    rows = []
    for k in [3, 6, 12, 24]:
        w = S[(S.days_pre <= k * 30.4) & (S.days_post <= k * 30.4)]
        e, g = w[w.fuel == "Electric"].dAR, w[w.fuel == "Gas"].dAR
        rows.append([k, int(len(w)),
                     round(e.mean(), 1), int(len(e)),
                     round(g.mean(), 1), int(len(g)),
                     round(e.mean() - g.mean(), 1)])
    out = pd.DataFrame(rows, columns=["window_months", "n", "electric_dAR", "n_elec",
                                      "gas_dAR", "n_gas", "elec_minus_gas"])
    out.to_csv(OUT_TABLES / "local_cutoff.csv", index=False)
    return out


def band_transition_and_rounding(df: pd.DataFrame, P: pd.DataFrame) -> dict:
    """Supplementary: full seven-band observed->constant transition matrix on the
    latest-stock frame, plus threshold rounding sensitivity at AR=50 and AR=125."""
    m = compute_multipliers(P)
    d = with_dates(df)
    latest = d.groupby("uprn").last().copy()
    ar = latest.asset_rating.values.astype(float)
    post = (latest.insp_dt >= CUT).values
    mult = latest.fuelgrp.where(latest.fuelgrp.isin(FUELS), "Other").map(m).values
    arc = np.where(post, ar * mult, ar)

    def band7(a):
        return np.select([a <= 25, a <= 50, a <= 75, a <= 100, a <= 125, a <= 150],
                         list("ABCDEF"), "G")
    ct = (pd.crosstab(pd.Series(band7(ar), name="observed"),
                      pd.Series(band7(arc), name="constant"))
          .reindex(index=list("ABCDEFG"), columns=list("ABCDEFG")).fillna(0).astype(int))
    ct.to_csv(OUT_TABLES / "band_transition_observed_to_constant.csv")

    reachB = (ar <= 50) & (arc > 50)
    clearFG = (ar <= 125) & (arc > 125)
    arcr = np.round(arc)
    rounding = {
        "reachB_n": int(reachB.sum()),
        "reachB_within_1pt_of_50_%": round(100 * (reachB & (np.abs(arc - 50) <= 1)).sum() / reachB.sum(), 1),
        "reachB_n_rounded": int(((ar <= 50) & (arcr > 50)).sum()),
        "clearFG_n": int(clearFG.sum()),
        "clearFG_within_1pt_of_125_%": round(100 * (clearFG & (np.abs(arc - 125) <= 1)).sum() / clearFG.sum(), 1),
        "clearFG_n_rounded": int(((ar <= 125) & (arcr > 125)).sum()),
    }
    json.dump(rounding, open(OUT_TABLES / "rounding_sensitivity.json", "w"), indent=2)

    # Broad-band (A/B..F/G) observed->constant transition with the fuel composition
    # of downgraded buildings under the constant-accounting counterfactual.
    ob, cb = broad_band(ar), broad_band(arc)
    bct = (pd.crosstab(pd.Series(ob), pd.Series(cb))
           .reindex(index=BROAD, columns=BROAD).fillna(0).astype(int))
    fuel = latest.fuelgrp.where(latest.fuelgrp.isin(FUELS), "Other").values
    down = cb != ob
    broad = {
        "n_stock": int(len(ar)),
        "observed_total": {b: int((ob == b).sum()) for b in BROAD},
        "transition": {b: {c: int(bct.loc[b, c]) for c in BROAD} for b in BROAD},
        # per-cell fuel composition (for the reclassification figure's fuel peaks)
        "transition_fuel": {b: {c: {f: int(((ob == b) & (cb == c) & (fuel == f)).sum())
                                    for f in FUELS} for c in BROAD} for b in BROAD},
        "downgraded_total": int(down.sum()),
        "downgraded_fuel": {f: int(((fuel == f) & down).sum()) for f in FUELS},
        "downgraded_fuel_by_observed": {
            b: {f: int(((ob == b) & down & (fuel == f)).sum()) for f in FUELS} for b in BROAD},
        # per-observed-group downgrade rate (share of that group that drops)
        "downgrade_rate_by_group": {
            b: round(100 * ((ob == b) & down).sum() / max((ob == b).sum(), 1), 1) for b in BROAD},
    }
    fa = latest.floor_area.values.astype(float)
    reachB = (ar <= 50) & (arc > 50)
    clearFG = (ar <= 125) & (arc > 125)
    # policy-threshold crossing sets: counts, fuel composition, floor area, capex
    broad["reachB"] = {
        "n": int(reachB.sum()), "Mm2": round(fa[reachB].sum() / 1e6, 1),
        "capex_bn": round(fa[reachB].sum() * 100 / 1e9, 2),
        "fuel": {f: int((reachB & (fuel == f)).sum()) for f in FUELS}}
    broad["clearFG"] = {
        "n": int(clearFG.sum()), "Mm2": round(fa[clearFG].sum() / 1e6, 1),
        "capex_bn": round(fa[clearFG].sum() * 100 / 1e9, 2),
        "fuel": {f: int((clearFG & (fuel == f)).sum()) for f in FUELS}}
    broad["combined_capex_bn"] = round((fa[reachB].sum() + fa[clearFG].sum()) * 100 / 1e9, 2)
    # full A-G precision note: within-broad-group drops the broad view omits
    a2b = int(((band7(ar) == "A") & (band7(arc) == "B")).sum())
    f2g = int(((band7(ar) == "F") & (band7(arc) == "G")).sum())
    full_down = int(down.sum()) + a2b + f2g
    broad["full_band"] = {"A_to_B": a2b, "F_to_G": f2g, "downgraded": full_down,
                          "pct": round(100 * full_down / len(ar), 1)}
    # within-building validation: repeat-certified EPC-B crossers that dissolve
    into = P[P.straddle & (P.ar_f > 50) & (P.ar_l <= 50)]
    im = into.fuel.where(into.fuel.isin(FUELS), "Other").map(m).values
    dissolve = into.ar_l.values * im > 50
    broad["within_building_crossers"] = {
        "n": int(len(into)), "n_dissolve": int(dissolve.sum()),
        "pct": round(100 * dissolve.mean(), 1)}
    json.dump(broad, open(OUT_TABLES / "band_transition_broad.json", "w"), indent=2)
    return {"transition": ct, "rounding": rounding, "broad": broad}


def capex_by_size(df: pd.DataFrame, P: pd.DataFrame, unit_cost: float = 100.0) -> pd.DataFrame:
    """Supplementary: capex-equivalent for the EPC-B and legal-floor crossing sets,
    overall and restricted to buildings above 1,000 m2 (the 2031 target size band)."""
    m = compute_multipliers(P)
    d = with_dates(df)
    latest = d.groupby("uprn").last().copy()
    ar = latest.asset_rating.values.astype(float)
    post = (latest.insp_dt >= CUT).values
    fa = latest.floor_area.values.astype(float)
    mult = latest.fuelgrp.where(latest.fuelgrp.isin(FUELS), "Other").map(m).values
    arc = np.where(post, ar * mult, ar)
    reachB = (ar <= 50) & (arc > 50)
    clearFG = (ar <= 125) & (arc > 125)
    rows = []
    for lab, mask in [("Reached EPC-B", reachB), ("Reached EPC-B (>1000 m2)", reachB & (fa > 1000)),
                      ("Cleared legal floor", clearFG), ("Cleared legal floor (>1000 m2)", clearFG & (fa > 1000))]:
        rows.append([lab, int(mask.sum()), round(fa[mask].sum() / 1e6, 2),
                     round(fa[mask].sum() * unit_cost / 1e9, 2)])
    out = pd.DataFrame(rows, columns=["set", "n", "floor_area_Mm2", f"capex_at_{int(unit_cost)}_bn"])
    out.to_csv(OUT_TABLES / "capex_by_size.csv", index=False)
    return out


# ── 10. International EPBD metric-exposure typology ────────────────────────────
# ── Orchestration ─────────────────────────────────────────────────────────────
def run_analysis(n_boot: int = 1000) -> dict:
    """Reproduce all artefact result tables from the register + DESNZ data."""
    ensure_dirs()
    df = build_analysis_frame()
    results = {"identity_trends": identity_and_trends(df)}
    P = within_building_pairs(df)
    straddle_by_fuel(P)
    windfall = no_works_windfall(P)
    results["windfall"] = windfall
    results["event_study"] = event_study(df).to_dict("records")
    results["emissions_ratio_validation"] = emissions_ratio_validation(P)
    results["decomposition"] = decomposition_shares(P)
    dose_response(df, P)
    results["redistribution"] = redistribution(df, P)
    results["corrected_stress"] = corrected_stress_test(df, windfall).to_dict("records")
    results["commensurate_stress"] = commensurate_stress_test(df, P)["projection"].to_dict("records")
    results["national"] = national_consequence(df, P, windfall, n_boot=n_boot)
    results["cohort_funnel"] = cohort_funnel(df, P)
    results["fixed_factor_recompute"] = fixed_factor_recompute(df, P, n_boot=n_boot)
    results["capex_equivalent"] = capex_equivalent(df, P)
    results["multiplier_robustness"] = multiplier_robustness(df, P).to_dict("records")
    results["multiplier_components"] = multiplier_components(P).to_dict("records")
    results["local_cutoff"] = local_cutoff(P).to_dict("records")
    results["band_transition_rounding"] = band_transition_and_rounding(df, P)["rounding"]
    results["capex_by_size"] = capex_by_size(df, P).to_dict("records")
    results["metered_validation"] = metered_validation().to_dict("records")
    with (OUT_TABLES / "analysis_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, default=str)
    return results
