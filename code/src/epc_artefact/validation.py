"""Held-out validation of the constant-accounting bridge, and policy-relevant
subset analyses (reviewer-requested pre-submission additions).

The UK counterfactual is an *empirical bridge*: it predicts each post-2022
building's old-basis Asset Rating from its post-2022 rating using a per-fuel
multiplier (or additive windfall, or attribute-stratified multiplier). It does
not rerun SBEM. To test how well the bridge recovers a building's true old-basis
rating, we use the no-recorded-works straddlers, whose actual pre-2022 rating is
an observation of the old-basis rating: we split them into training and held-out
test sets, estimate the bridge on the training set, predict the held-out
buildings' pre-2022 ratings from their post-2022 ratings, and score the
predictions (error, calibration, threshold sensitivity/specificity, confusion).

    heldout_validation   -- bridge prediction accuracy (mult / additive / attr)
    policy_proxy_subsets -- national quantities for policy-relevant subsets
    latest_frame_characteristics -- Table S1 panel B (UPRN latest-certificate frame)

Run:  python -m epc_artefact.validation
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import (OUT_TABLES, CUT, RANDOM_SEED, BROAD, ND_NEED_2025_URL,
                     ND_NEED_GEO_2024_URL, ND_NEED_2025_XLSX,
                     ND_NEED_GEO_2024_XLSX)
from .data import (build_analysis_frame, with_dates, broad_band,
                   download_if_missing)
from .analysis import within_building_pairs, compute_multipliers

FUELS = ["Electric", "Gas", "Other"]
MIN_CELL = 50
SPECS = ["multiplicative", "additive", "floor_area", "sector", "aircon",
         "post_rating_band"]


# ── the three candidate bridges, fit on train and applied to test ─────────────
def _attribute_key(frame: pd.DataFrame, spec: str) -> pd.Series:
    if spec == "floor_area":
        return pd.cut(frame.fa_f, [0, 500, 2000, np.inf],
                      labels=["0-500", "500-2000", ">2000"]).astype(str)
    if spec == "sector":
        return frame.sector.fillna("Missing").astype(str)
    if spec == "aircon":
        ac = frame.get("ac_f", pd.Series("Missing", index=frame.index))
        yes = ac.astype(str).str.lower().str.contains("y|true|1|present", regex=True)
        return yes.map({True: "Present", False: "Not present"})
    if spec == "post_rating_band":
        return pd.Series(broad_band(frame.post.to_numpy()), index=frame.index)
    raise ValueError(f"Unknown attribute specification: {spec}")


def _fit(train: pd.DataFrame, spec: str):
    if spec == "additive":
        return {f: float((train[train.fuel == f].pre - train[train.fuel == f].post).mean())
                for f in FUELS}
    # Multiplicative baseline (median AR_pre/AR_post).
    base = {f: float((train[train.fuel == f].pre / train[train.fuel == f].post).median())
            for f in FUELS}
    if spec == "multiplicative":
        return base
    # Attribute-stratified multiplicative bridges; sparse cells fall back to
    # the fuel-only multiplier, consistently with the national sensitivity table.
    keyed = train.assign(key=_attribute_key(train, spec), r=train.pre / train.post)
    cell = keyed.groupby(["fuel", "key"], observed=True).agg(
        m=("r", "median"), n=("r", "size"))
    return {"base": base, "cell": cell, "spec": spec}


def _predict(test: pd.DataFrame, spec: str, params) -> np.ndarray:
    if spec == "additive":
        return test.post.values + np.array([params[f] for f in test.fuel])
    if spec == "multiplicative":
        return test.post.values * np.array([params[f] for f in test.fuel])
    base, cell = params["base"], params["cell"]
    keys = _attribute_key(test, params["spec"])
    out = []
    for f, key, po in zip(test.fuel, keys, test.post.values):
        m = (cell.loc[(f, key), "m"]
             if (f, key) in cell.index and cell.loc[(f, key), "n"] >= MIN_CELL
             else base[f])
        out.append(po * m)
    return np.array(out)


def _threshold_stats(pred, actual, thr):
    ap, pp = actual > thr, pred > thr
    tp, fp = int((ap & pp).sum()), int((~ap & pp).sum())
    fn, tn = int((ap & ~pp).sum()), int((~ap & ~pp).sum())
    sens = tp / max(ap.sum(), 1)
    spec = tn / max((~ap).sum(), 1)
    return sens, spec, (tp, fp, fn, tn)


def heldout_validation(P: pd.DataFrame | None = None, n_splits: int = 200,
                       train_frac: float = 0.5) -> dict:
    if P is None:
        P = within_building_pairs(build_analysis_frame())
    nw = P[P.straddle & (P.drec == 0) & (P.dfa < 0.02) & (P.ar_l > 0)
           & P.fuel.isin(FUELS)].copy()
    nw["pre"] = nw.ar_f.astype(float)
    nw["post"] = nw.ar_l.astype(float)
    nw["fa_f"] = nw.fa_f.astype(float)
    nw = nw[(nw.pre > 0) & (nw.post > 0)]
    nw = nw.reset_index()
    specs = SPECS

    # repeated splits -> stable summary metrics
    acc = {s: {k: [] for k in ["bias", "mae", "med", "slope", "intercept",
                               "B_sens", "B_spec", "FG_sens", "FG_spec"]} for s in specs}
    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(n_splits):
        m = rng.random(len(nw)) < train_frac
        tr, te = nw[m], nw[~m]
        for s in specs:
            pred = _predict(te, s, _fit(tr, s))
            a = te.pre.values
            err = pred - a
            sl, ic = np.polyfit(pred, a, 1)
            bs = _threshold_stats(pred, a, 50)
            fs = _threshold_stats(pred, a, 125)
            r = acc[s]
            r["bias"].append(err.mean()); r["mae"].append(np.abs(err).mean())
            r["med"].append(np.median(err)); r["slope"].append(sl); r["intercept"].append(ic)
            r["B_sens"].append(bs[0]); r["B_spec"].append(bs[1])
            r["FG_sens"].append(fs[0]); r["FG_spec"].append(fs[1])
    summary = pd.DataFrame([
        {"specification": s,
         "bias_pts": round(np.mean(acc[s]["bias"]), 2),
         "MAE_pts": round(np.mean(acc[s]["mae"]), 1),
         "median_error_pts": round(np.mean(acc[s]["med"]), 2),
         "calibration_slope": round(np.mean(acc[s]["slope"]), 3),
         "calibration_intercept": round(np.mean(acc[s]["intercept"]), 1),
         "B_sensitivity": round(np.mean(acc[s]["B_sens"]), 3),
         "B_specificity": round(np.mean(acc[s]["B_spec"]), 3),
         "FG_sensitivity": round(np.mean(acc[s]["FG_sens"]), 3),
         "FG_specificity": round(np.mean(acc[s]["FG_spec"]), 3)}
        for s in specs])
    summary.to_csv(OUT_TABLES / "validation_heldout_summary.csv", index=False)

    # One deterministic five-fold out-of-fold pass uses every calibration case
    # exactly once for the detailed fuel/rating diagnostics and confusion matrices.
    rng = np.random.default_rng(RANDOM_SEED)
    folds = np.zeros(len(nw), dtype=int)
    strata = (nw.fuel.astype(str) + "|"
              + pd.Series(broad_band(nw.post.to_numpy()), index=nw.index).astype(str))
    for idx in strata.groupby(strata).groups.values():
        idx = np.asarray(list(idx), dtype=int)
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % 5

    oof_predictions, perfuel_rows, fuel_rating_rows, conf_rows = {}, [], [], []
    for s in specs:
        pred = np.empty(len(nw), dtype=float)
        for fold in range(5):
            tr, te = nw[folds != fold], nw[folds == fold]
            pred[folds == fold] = _predict(te, s, _fit(tr, s))
        oof_predictions[s] = pred
        a = nw.pre.to_numpy()
        for f in FUELS + ["All"]:
            sel = np.ones(len(nw), bool) if f == "All" else (nw.fuel == f).values
            e = pred[sel] - a[sel]
            slope, intercept = np.polyfit(pred[sel], a[sel], 1)
            bs, fs = _threshold_stats(pred[sel], a[sel], 50), _threshold_stats(pred[sel], a[sel], 125)
            perfuel_rows.append({"specification": s, "fuel": f, "n_test": int(sel.sum()),
                                 "bias_pts": round(e.mean(), 2), "MAE_pts": round(np.abs(e).mean(), 1),
                                 "median_error_pts": round(float(np.median(e)), 2),
                                 "err_p10": round(np.percentile(e, 10), 1),
                                 "err_p90": round(np.percentile(e, 90), 1),
                                 "calibration_slope": round(float(slope), 3),
                                 "calibration_intercept": round(float(intercept), 1),
                                 "B_sensitivity": round(bs[0], 3), "B_specificity": round(bs[1], 3),
                                 "FG_sensitivity": round(fs[0], 3), "FG_specificity": round(fs[1], 3)})
        for threshold, value in [("EPC_B", 50), ("FG", 125)]:
            tp, fp, fn, tn = _threshold_stats(pred, a, value)[2]
            conf_rows.append({"specification": s, "threshold": threshold,
                              "n_test": len(nw), "TP": tp, "FP": fp, "FN": fn, "TN": tn})

    # Within-fuel/rating-range calibration. Rating range is
    # the observed post-revision range, which is available when the bridge is used.
    headline = nw.assign(pred=oof_predictions["multiplicative"])
    headline["error"] = headline.pred - headline.pre
    headline["post_rating_range"] = broad_band(headline.post.to_numpy())
    for (fuel, rating), g in headline.groupby(["fuel", "post_rating_range"], observed=True):
        slope = intercept = np.nan
        if len(g) >= 5 and g.pred.nunique() > 1:
            slope, intercept = np.polyfit(g.pred, g.pre, 1)
        fuel_rating_rows.append({
            "fuel": fuel, "post_rating_range": rating, "n_test": len(g),
            "observed_pre_mean": round(g.pre.mean(), 1),
            "predicted_pre_mean": round(g.pred.mean(), 1),
            "bias_pts": round(g.error.mean(), 1),
            "median_error_pts": round(g.error.median(), 1),
            "MAE_pts": round(g.error.abs().mean(), 1),
            "err_p10": round(g.error.quantile(.1), 1),
            "err_p90": round(g.error.quantile(.9), 1),
            "calibration_slope": round(float(slope), 3) if np.isfinite(slope) else np.nan,
            "calibration_intercept": round(float(intercept), 1) if np.isfinite(intercept) else np.nan,
        })

    pd.DataFrame(perfuel_rows).to_csv(OUT_TABLES / "validation_heldout_by_fuel.csv", index=False)
    pd.DataFrame(fuel_rating_rows).to_csv(
        OUT_TABLES / "validation_heldout_by_fuel_rating.csv", index=False)
    confusion = pd.DataFrame(conf_rows)
    confusion.to_csv(OUT_TABLES / "validation_confusion.csv", index=False)
    pd.DataFrame({"uprn": nw.uprn, "fuel": nw.fuel, "pre": nw.pre, "post": nw.post,
                  **{f"pred_{s}": p for s, p in oof_predictions.items()}}).to_csv(
        OUT_TABLES / "validation_oof_predictions.csv", index=False)
    return {"summary": summary, "by_fuel": pd.DataFrame(perfuel_rows),
            "by_fuel_rating": pd.DataFrame(fuel_rating_rows), "confusion": confusion}


# ── policy-relevant subset analyses ───────────────────────────────────────────
def _stock_metrics(latest, mask, M):
    ar = latest.asset_rating.values.astype(float)[mask]
    post = (latest.insp_dt >= CUT).values[mask]
    fuel = latest.fuelgrp.where(latest.fuelgrp.isin(FUELS), "Other").values[mask]
    fa = latest.floor_area.values.astype(float)[mask]
    arc = ar * np.where(post, np.array([M[f] for f in fuel]), 1.0)
    reachB = (ar <= 50) & (arc > 50)
    clearFG = (ar <= 125) & (arc > 125)
    return {"n": int(mask.sum()),
            "belowB_obs_%": round(100 * (ar > 50).mean(), 1),
            "belowB_const_%": round(100 * (arc > 50).mean(), 1),
            "FG_obs_%": round(100 * (ar > 125).mean(), 2),
            "FG_const_%": round(100 * (arc > 125).mean(), 1),
            "reachedB_n": int(reachB.sum()), "reachedB_Mm2": round(fa[reachB].sum() / 1e6, 1),
            "clearedFG_n": int(clearFG.sum())}


def policy_proxy_subsets(df=None, P=None) -> pd.DataFrame:
    from .data import build_extended_frame
    dfx = build_extended_frame() if df is None else df
    P = within_building_pairs(dfx) if P is None else P
    M = compute_multipliers(P)
    d = with_dates(dfx)
    latest = d.groupby("uprn").last().copy()
    fa = latest.floor_area.values.astype(float)
    txn = latest.transaction_type_clean.astype(str)
    lodge = pd.to_datetime(latest.lodgement_date)
    valid = (lodge >= (pd.Timestamp("2026-06-15") - pd.DateOffset(years=10))).values
    existing = ~txn.str.contains("construction", case=False).values
    to_let = txn.str.contains("let", case=False).values
    big = fa > 1000
    subsets = [
        ("All EPC-register UPRNs (measurement)", np.ones(len(latest), bool)),
        ("Valid (lodged within 10 years)", valid),
        ("Latest transaction is not construction", existing),
        ("Property-to-let (tenure proxy)", to_let),
        ("Above 1,000 m2", big),
        ("Policy intersection (valid, non-construction, to-let, >1000 m2)",
         valid & existing & to_let & big)]
    rows = [{"subset": lab, **_stock_metrics(latest, mask, M)} for lab, mask in subsets]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "policy_proxy_subsets.csv", index=False)
    return out


def latest_frame_characteristics(df=None) -> pd.DataFrame:
    """Table S1 panel B: the latest-certificate (UPRN) register frame, so the
    reader can compare the stock-style cross-section with the certificate-level
    counts rather than mixing denominators."""
    dfx = build_analysis_frame() if df is None else df
    latest = with_dates(dfx).groupby("uprn").last().copy()
    n = len(latest)
    rows = []

    def add(panel, cat, c):
        rows.append([panel, cat, int(c), round(100 * c / n, 1)])

    for f in FUELS:
        add("Heating fuel", f, (latest.fuelgrp == f).sum())
    for txn, count in latest.transaction_type_clean.value_counts(dropna=False).items():
        add("Transaction type", str(txn), count)
    ac = latest.aircon_present_clean.astype(str).str.strip().str.lower().eq("yes")
    add("Air conditioning", "Present", ac.sum())
    for lab, lo, hi in [("<100 m2", 0, 100), ("100-500 m2", 100, 500),
                        ("500-1000 m2", 500, 1000), (">1000 m2", 1000, np.inf)]:
        add("Floor area", lab, ((latest.floor_area >= lo) & (latest.floor_area < hi)).sum())
    yr = latest.inspection_year
    for lab, lo, hi in [("2012-2016", 2012, 2016), ("2017-2021", 2017, 2021),
                        ("2022-2025", 2022, 2025)]:
        add("Certificate age (inspection year)", lab, ((yr >= lo) & (yr <= hi)).sum())
    bands = pd.Series(broad_band(latest.asset_rating.values)).value_counts()
    for b in BROAD:
        add("Asset-rating band", b, int(bands.get(b, 0)))
    out = pd.DataFrame(rows, columns=["Panel", "Category", "n", "%"])
    out.to_csv(OUT_TABLES / "latest_frame_characteristics.csv", index=False)
    return out


# ── independent administrative benchmark and composition weighting ──────────
SIZE_BINS = [0, 50, 100, 250, 500, 1000, 5000, np.inf]
SIZE_LABELS = ["0-50 m2", ">50-100 m2", ">100-250 m2", ">250-500 m2",
               ">500-1,000 m2", ">1,000-5,000 m2", ">5,000 m2"]


def _nd_need_sector(series: pd.Series) -> np.ndarray:
    """Map register property types to the ten published ND-NEED use classes.

    The EPC category ``Offices and Workshop Businesses`` cannot be separated
    further and is assigned to Offices; this limitation is reported with the
    benchmark table rather than silently treated as an exact concordance.
    """
    s = series.fillna("").astype(str).str.lower()
    conditions = [
        s.str.contains("retail|financial and professional", regex=True),
        s.str.contains("office", regex=True),
        s.str.contains("restaurant|cafe|drinking|takeaway|hotel", regex=True),
        s.str.contains("storage|distribution", regex=True),
        s.str.contains("industrial", regex=True),
        s.str.contains("education|universit|residential school", regex=True),
        s.str.contains("hospital|care home|primary health", regex=True),
        s.str.contains("assembly|leisure|night club|theatre|community/day|libraries|museums|galleries",
                       regex=True),
        s.str.contains("emergency", regex=True),
    ]
    choices = ["Shops", "Offices", "Hospitality", "Warehouses", "Factories",
               "Education", "Health", "Arts, Community and Leisure",
               "Emergency Services"]
    return np.select(conditions, choices, default="Other")


def _rake_weights(frame: pd.DataFrame, targets: dict[str, pd.Series],
                  max_iter: int = 500, tol: float = 1e-8) -> tuple[np.ndarray, int]:
    """Iterative proportional fitting to published count-share margins."""
    weights = np.ones(len(frame), dtype=float)
    for iteration in range(max_iter):
        previous = weights.copy()
        for variable, target in targets.items():
            values = frame[variable].astype(str).to_numpy()
            for category, share in target.items():
                mask = values == str(category)
                if not mask.any():
                    raise ValueError(f"No EPC-frame observations for {variable}={category}")
                current = weights[mask].sum() / weights.sum()
                weights[mask] *= float(share) / current
        weights *= len(frame) / weights.sum()
        if np.max(np.abs(weights - previous)) < tol:
            return weights, iteration + 1
    return weights, max_iter


def representativeness_assessment(df=None, P=None) -> dict:
    """Compare the latest-certificate frame with the independent ND-NEED stock.

    Results are descriptive because the two sources do not have a record-level
    crosswalk and the EPC frame excludes missing/non-positive floor area. A
    raking sensitivity aligns the EPC frame to ND-NEED sector, known-area size,
    and region count margins; it does not turn the register into a census.
    """
    from .data import build_extended_frame

    download_if_missing(ND_NEED_2025_URL, ND_NEED_2025_XLSX)
    download_if_missing(ND_NEED_GEO_2024_URL, ND_NEED_GEO_2024_XLSX)
    dfx = build_extended_frame() if df is None else df
    P = within_building_pairs(dfx) if P is None else P
    latest = with_dates(dfx).drop_duplicates("uprn", keep="last").set_index("uprn").copy()
    latest["sector_benchmark"] = _nd_need_sector(latest.property_type_clean)
    latest["size_benchmark"] = pd.cut(
        latest.floor_area, SIZE_BINS, labels=SIZE_LABELS,
        include_lowest=True, right=True).astype(str)

    # Main 2025 margins (March 2025 stock).
    main_sector = pd.read_excel(ND_NEED_2025_XLSX, sheet_name="Table 1", header=5)
    main_size = pd.read_excel(ND_NEED_2025_XLSX, sheet_name="Table 2", header=6)
    bsector = main_sector[main_sector["Building use"] != "All"].set_index("Building use")
    bsize = main_size.set_index("Building size")
    benchmark_n = int(main_sector.loc[main_sector["Building use"] == "All",
                                      "Number of buildings "].iloc[0])

    # The 2024 geographical annex supplies both LA-to-region concordance and
    # published region number/floor-area margins.
    geo_n = pd.read_excel(ND_NEED_GEO_2024_XLSX, sheet_name="Table 1", header=6)
    geo_a = pd.read_excel(ND_NEED_GEO_2024_XLSX, sheet_name="Table 2", header=7)
    la = geo_n[geo_n["Local Authority"].notna()]
    code_region = dict(zip(la["Geographic Code"], la["Country or Region"]))
    name_region = dict(zip(la["Local Authority"], la["Country or Region"]))
    latest["region_benchmark"] = (latest.local_authority.map(code_region)
                                    .fillna(latest.local_authority_label.map(name_region))
                                    .replace({"East": "East of England"}))
    region_codes = [f"E1200000{i}" for i in range(1, 10)] + ["W92000004"]
    bregion = geo_n[(geo_n["Local Authority"].isna())
                    & geo_n["Geographic Code"].isin(region_codes)].set_index("Country or Region")
    bregion_area = geo_a[(geo_a["Local Authority"].isna())
                         & geo_a["Geographic Code"].isin(region_codes)].set_index("Country or Region")
    bregion_area = bregion_area.rename(index={"East": "East of England"})

    rows = []

    def add(dimension, category, epc_n, epc_share, benchmark_count=np.nan,
            benchmark_share=np.nan, epc_area_share=np.nan,
            benchmark_area_share=np.nan, basis=""):
        rows.append({
            "dimension": dimension, "category": str(category),
            "epc_frame_n": int(epc_n) if pd.notna(epc_n) else np.nan,
            "epc_frame_share_%": round(float(epc_share), 2) if pd.notna(epc_share) else np.nan,
            "benchmark_n": int(benchmark_count) if pd.notna(benchmark_count) else np.nan,
            "benchmark_share_%": round(float(benchmark_share), 2) if pd.notna(benchmark_share) else np.nan,
            "difference_pp": round(float(epc_share - benchmark_share), 2)
                             if pd.notna(epc_share) and pd.notna(benchmark_share) else np.nan,
            "epc_floor_area_share_%": round(float(epc_area_share), 2)
                                      if pd.notna(epc_area_share) else np.nan,
            "benchmark_floor_area_share_%": round(float(benchmark_area_share), 2)
                                            if pd.notna(benchmark_area_share) else np.nan,
            "benchmark_basis": basis,
        })

    total_area = latest.floor_area.sum()
    for category, b in bsector.iterrows():
        mask = latest.sector_benchmark == category
        add("Sector/use class", category, mask.sum(), 100 * mask.mean(),
            b["Number of buildings "], 100 * b["% of buildings "],
            100 * latest.loc[mask, "floor_area"].sum() / total_area,
            100 * b["% of total floor area "], "ND-NEED 2025, all buildings")

    add("Building size", "Missing floor area", 0, 0,
        bsize.loc["Missing", "Number of buildings "],
        100 * bsize.loc["Missing", "% of buildings "], basis="ND-NEED 2025, all buildings")
    for category in SIZE_LABELS:
        mask = latest.size_benchmark == category
        b = bsize.loc[category]
        add("Building size", category, mask.sum(), 100 * mask.mean(),
            b["Number of buildings "], 100 * b["% of buildings "],
            100 * latest.loc[mask, "floor_area"].sum() / total_area,
            100 * b["% of floor area "], "ND-NEED 2025, all buildings")

    matched = latest.region_benchmark.notna()
    matched_area = latest.loc[matched, "floor_area"].sum()
    for category, b in bregion.iterrows():
        mask = latest.region_benchmark == category
        ba = bregion_area.loc[category]
        add("Region", category, mask.sum(), 100 * mask.sum() / matched.sum(),
            b["All buildings: number of buildings"],
            100 * b["All buildings: share of building number"],
            100 * latest.loc[mask, "floor_area"].sum() / matched_area,
            100 * ba["All buildings: Share of floor area"],
            "ND-NEED 2024 geographical annex, matched geography")

    # Register-only descriptors requested by the reviewer but unavailable in
    # the independent benchmark.
    for label, lo, hi in [("2012-2016", 2012, 2016), ("2017-2021", 2017, 2021),
                          ("2022-2025", 2022, 2025)]:
        mask = latest.inspection_year.between(lo, hi)
        add("Certificate inspection year", label, mask.sum(), 100 * mask.mean(),
            basis="No independent ND-NEED equivalent")
    for category, count in latest.transaction_type_clean.value_counts(dropna=False).items():
        add("Transaction type", category, count, 100 * count / len(latest),
            basis="No independent ND-NEED equivalent")
    for category, count in latest.fuelgrp.value_counts(dropna=False).items():
        add("Recorded main heating fuel", category, count, 100 * count / len(latest),
            basis="No independent ND-NEED equivalent")

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT_TABLES / "representativeness_comparison.csv", index=False)

    valid = pd.to_datetime(latest.lodgement_date) >= (pd.Timestamp("2026-06-15")
                                                       - pd.DateOffset(years=10))
    overview = pd.DataFrame([
        ["Latest-certificate EPC-register UPRNs", len(latest), 100 * len(latest) / benchmark_n],
        ["Valid latest-certificate EPC-register UPRNs", int(valid.sum()),
         100 * valid.sum() / benchmark_n],
        ["ND-NEED 2025 non-domestic buildings", benchmark_n, 100.0],
    ], columns=["frame", "n", "relative_to_ND_NEED_%"])
    overview.to_csv(OUT_TABLES / "representativeness_overview.csv", index=False)

    # Composition sensitivity: rake only records that map to an E&W region.
    rake = latest[matched].copy()
    sector_target = bsector["Number of buildings "].astype(float)
    sector_target /= sector_target.sum()
    size_target = bsize.loc[SIZE_LABELS, "Number of buildings "].astype(float)
    size_target /= size_target.sum()  # conditional on known floor area
    region_target = bregion["All buildings: number of buildings"].astype(float)
    region_target /= region_target.sum()
    weights, iterations = _rake_weights(rake, {
        "sector_benchmark": sector_target,
        "size_benchmark": size_target,
        "region_benchmark": region_target,
    })

    multipliers = compute_multipliers(P)
    ar = rake.asset_rating.to_numpy(dtype=float)
    post = (rake.insp_dt >= CUT).to_numpy()
    fuel = rake.fuelgrp.where(rake.fuelgrp.isin(FUELS), "Other").to_numpy()
    bridge = np.array([multipliers[f] for f in fuel])
    constant = np.where(post, ar * bridge, ar)
    crossing_b = (ar <= 50) & (constant > 50)
    crossing_fg = (ar <= 125) & (constant > 125)
    floor = rake.floor_area.to_numpy(dtype=float)

    # Full-frame unweighted row (the raked row necessarily omits unmatched geography).
    ar0 = latest.asset_rating.to_numpy(dtype=float)
    post0 = (latest.insp_dt >= CUT).to_numpy()
    fuel0 = latest.fuelgrp.where(latest.fuelgrp.isin(FUELS), "Other").to_numpy()
    constant0 = np.where(post0, ar0 * np.array([multipliers[f] for f in fuel0]), ar0)
    b0 = (ar0 <= 50) & (constant0 > 50)
    fg0 = (ar0 <= 125) & (constant0 > 125)
    floor0 = latest.floor_area.to_numpy(dtype=float)
    raking_rows = [{
        "estimate": "Unweighted register frame", "n_frame": len(latest),
        "belowB_observed_%": round(100 * (ar0 > 50).mean(), 2),
        "belowB_constant_%": round(100 * (constant0 > 50).mean(), 2),
        "FG_observed_%": round(100 * (ar0 > 125).mean(), 2),
        "FG_constant_%": round(100 * (constant0 > 125).mean(), 2),
        "EPC_B_crossing_equivalent_n": int(b0.sum()),
        "FG_crossing_equivalent_n": int(fg0.sum()),
        "EPC_B_crossing_weighted_Mm2": round(float(floor0[b0].sum() / 1e6), 1),
        "FG_crossing_weighted_Mm2": round(float(floor0[fg0].sum() / 1e6), 1),
    }]
    w = weights
    raking_rows.append({
        "estimate": "Raked to ND-NEED count margins", "n_frame": len(rake),
        "belowB_observed_%": round(100 * np.average(ar > 50, weights=w), 2),
        "belowB_constant_%": round(100 * np.average(constant > 50, weights=w), 2),
        "FG_observed_%": round(100 * np.average(ar > 125, weights=w), 2),
        "FG_constant_%": round(100 * np.average(constant > 125, weights=w), 2),
        "EPC_B_crossing_equivalent_n": int(round(w[crossing_b].sum())),
        "FG_crossing_equivalent_n": int(round(w[crossing_fg].sum())),
        "EPC_B_crossing_weighted_Mm2": round(float((w[crossing_b] * floor[crossing_b]).sum() / 1e6), 1),
        "FG_crossing_weighted_Mm2": round(float((w[crossing_fg] * floor[crossing_fg]).sum() / 1e6), 1),
    })
    raking = pd.DataFrame(raking_rows)
    raking.to_csv(OUT_TABLES / "representativeness_raking.csv", index=False)
    diagnostics = {
        "iterations": iterations, "n_raked": len(rake),
        "n_unmatched_region_excluded": int((~matched).sum()),
        "weight_min": round(float(weights.min()), 3),
        "weight_p99": round(float(np.quantile(weights, .99)), 3),
        "weight_max": round(float(weights.max()), 3),
        "effective_sample_size": int(round(weights.sum() ** 2 / np.square(weights).sum())),
        "size_margin_is_conditional_on_known_floor_area": True,
    }
    with open(OUT_TABLES / "representativeness_raking_diagnostics.json", "w") as fh:
        json.dump(diagnostics, fh, indent=2)
    return {"comparison": comparison, "overview": overview,
            "raking": raking, "diagnostics": diagnostics}


def run_validation() -> dict:
    from .data import build_extended_frame
    dfx = build_extended_frame()
    P = within_building_pairs(dfx)
    return {"heldout": heldout_validation(P),
            "policy_proxy": policy_proxy_subsets(dfx, P),
            "latest_frame": latest_frame_characteristics(dfx),
            "representativeness": representativeness_assessment(dfx, P)}


if __name__ == "__main__":
    r = run_validation()
    print("=== held-out validation summary ===")
    print(r["heldout"]["summary"].to_string(index=False))
    print("\n=== per-fuel (multiplicative) ===")
    print(r["heldout"]["by_fuel"][r["heldout"]["by_fuel"].specification == "multiplicative"].to_string(index=False))
    print("\n=== policy-proxy subsets ===")
    print(r["policy_proxy"].to_string(index=False))
    print("\nWrote validation tables to", OUT_TABLES)
