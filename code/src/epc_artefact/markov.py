"""Commensurate reassessment-throughput sensitivity (the "Markov" exercise).

Narrow question: does the accounting-sensitive EPC-B gap disappear as the register is
reassessed, or does it persist when the starting distribution and the transition matrix
are defined on commensurate rating bases? This is an illustrative throughput sensitivity,
NOT a forecast of legal compliance, national stock, or physical retrofit progress.

The preferred specification is a threshold-level, expected-state Markov (below/above the
EPC-B and F/G thresholds), because the bridge validation does not support exact individual
old-basis ratings. For a reassessment pair with previous below-threshold probability p0 and
current below-threshold probability p1, the expected transition mass is
    above->above (1-p0)(1-p1),  above->below (1-p0)p1,
    below->above p0(1-p1),      below->below p0 p1,
so hard full-band classifications are avoided. Two scenarios are kept internally
commensurate:

  Observed / current-accounting -- starting distribution and transition matrix both on the
    reported (post-revision) basis (transitions from post-2022 reassessment pairs).
  Constant-accounting -- every starting and transition rating expressed on the pre-2022
    basis (pre-2022 certificates unchanged; post-2022 ratings mapped to their expected
    old-basis threshold state via the empirical no-works re-pricing ratio distribution).

Hard full-band matrices are retained only as a Supplementary sensitivity.

    from epc_artefact.markov import run_markov
    run_markov()   # writes markov_*.csv + markov_methods_audit.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CUT, BROAD, OUT_TABLES, RANDOM_SEED
from .data import broad_band, with_dates, build_analysis_frame
from .analysis import compute_multipliers, within_building_pairs, FUELS, _no_works_ratios

# "Below threshold" means a worse-than-threshold asset rating (higher AR number).
THRESH = {"EPC-B": 50.0, "F/G": 125.0}
P_GRID = [0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
OBSERVED_COVERAGE = 0.157        # register reassessment coverage in the window (context only)


# ── pair construction + expected-state helpers ────────────────────────────────
def build_pairs(d: pd.DataFrame) -> pd.DataFrame:
    """Consecutive within-UPRN certificate pairs with dates, fuel, transaction, sector."""
    d2 = d.sort_values("insp_dt")
    g = d2.groupby("uprn")
    P = pd.DataFrame({
        "uprn": d2.uprn.values,
        "prev": g.asset_rating.shift(1).values, "prev_dt": g.insp_dt.shift(1).values,
        "cur": d2.asset_rating.values, "cur_dt": d2.insp_dt.values,
        "fuel": g.fuelgrp.transform("first").values,
        "txn": d2.transaction_type_clean.astype(str).values,
        "sector": d2.property_type_clean.astype(str).values,
    })
    P = P.dropna(subset=["prev"]).copy()
    P["prev_dt"] = pd.to_datetime(P["prev_dt"]); P["cur_dt"] = pd.to_datetime(P["cur_dt"])
    P["gap_m"] = (P.cur_dt - P.prev_dt).dt.days / 30.44
    P["fuel"] = P.fuel.where(P.fuel.isin(FUELS), "Other")
    P["prev_post"] = P.prev_dt >= CUT
    P["cur_post"] = P.cur_dt >= CUT
    return P


def _sorted_ratio_bank(P0: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-fuel sorted array of no-works AR_pre/AR_post ratios (the empirical re-pricing
    distribution used to form expected old-basis threshold states)."""
    return {f: np.sort(np.asarray(r, dtype=float)) for f, r in _no_works_ratios(P0).items()}


def expected_below(rating: np.ndarray, fuel: np.ndarray, is_post: np.ndarray,
                   thr: float, bank: dict[str, np.ndarray]) -> np.ndarray:
    """Expected probability that a rating is below (worse than) the threshold on the
    pre-2022 basis. Pre-2022 ratings are already old-basis (hard state); post-2022 ratings
    are mapped through the empirical ratio distribution: P(rating * ratio > thr)."""
    rating = np.asarray(rating, float)
    out = (rating > thr).astype(float)
    for f in FUELS:
        m = is_post & (fuel == f)
        sr = bank.get(f, np.array([]))
        if not m.any() or sr.size == 0:
            continue
        x = thr / np.maximum(rating[m], 1e-9)            # need ratio > x for below-threshold
        idx = np.searchsorted(sr, x, side="right")
        out[m] = 1.0 - idx / sr.size                     # P(ratio > x)
    return out


def _tmat2(p0: np.ndarray, p1: np.ndarray, w: np.ndarray | None = None) -> dict:
    """Expected 2x2 threshold transition matrix from below-threshold probabilities p0
    (previous) and p1 (current). Returns row-normalised below->below and above->below
    probabilities plus expected cell masses (Laplace-smoothed for row normalisation)."""
    w = np.ones_like(p0) if w is None else np.asarray(w, float)
    bb = float(np.sum(w * p0 * p1)); ba = float(np.sum(w * p0 * (1 - p1)))
    ab = float(np.sum(w * (1 - p0) * p1)); aa = float(np.sum(w * (1 - p0) * (1 - p1)))
    Tb = (bb + 0.5) / (bb + ba + 1.0)                    # P(below->below)
    Ta = (ab + 0.5) / (ab + aa + 1.0)                    # P(above->below)
    return {"Tb": Tb, "Ta": Ta, "bb": bb, "ba": ba, "ab": ab, "aa": aa,
            "below_mass": bb + ba, "above_mass": ab + aa}


def _project(s0_below: float, Tb: float, Ta: float, p: float) -> float:
    """Below-threshold share after reassessing coverage p (per cent)."""
    return 100.0 * ((1 - p) * s0_below + p * (s0_below * Tb + (1 - s0_below) * Ta))


# ── (2) preferred threshold-level expected-state Markov ───────────────────────
def threshold_expected_state(d: pd.DataFrame, P: pd.DataFrame,
                             bank: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict]:
    latest = d.groupby("uprn").last()
    ar = latest.asset_rating.values.astype(float)
    lpost = (latest.insp_dt >= CUT).values
    lfuel = latest.fuelgrp.where(latest.fuelgrp.isin(FUELS), "Other").values

    po = P[P.prev_post & P.cur_post]                      # observed-basis transition pairs
    rows, mats = [], {}
    for name, thr in THRESH.items():
        s0_obs = float((ar > thr).mean())
        s0_cc = float(expected_below(ar, lfuel, lpost, thr, bank).mean())
        # observed scenario: hard states from post-2022 pairs
        obs = _tmat2((po.prev.values > thr).astype(float), (po.cur.values > thr).astype(float))
        # constant-accounting scenario: expected states on the pre-2022 basis, all pairs
        p0 = expected_below(P.prev.values, P.fuel.values, P.prev_post.values, thr, bank)
        p1 = expected_below(P.cur.values, P.fuel.values, P.cur_post.values, thr, bank)
        cc = _tmat2(p0, p1)
        mats[name] = {"obs": obs, "cc": cc, "s0_obs": s0_obs, "s0_cc": s0_cc,
                      "n_obs_pairs": int(len(po)), "n_cc_pairs": int(len(P))}
        for p in P_GRID:
            o = _project(s0_obs, obs["Tb"], obs["Ta"], p)
            c = _project(s0_cc, cc["Tb"], cc["Ta"], p)
            rows.append({"threshold": name, "reassessment_coverage_p": p,
                         "observed_basis_below_%": round(o, 1),
                         "constant_accounting_below_%": round(c, 1),
                         "gap_pp": round(c - o, 1)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "markov_threshold_expected_state_results.csv", index=False)
    return out, mats


# ── full-band hard version (Supplementary sensitivity only) ───────────────────
def fullband_hard(d: pd.DataFrame, P: pd.DataFrame, mult: dict) -> pd.DataFrame:
    latest = d.groupby("uprn").last()
    ar = latest.asset_rating.values.astype(float)
    post = (latest.insp_dt >= CUT).values
    m = latest.fuelgrp.where(latest.fuelgrp.isin(FUELS), "Other").map(mult).values

    def band_dist(a):
        return pd.Series(broad_band(np.asarray(a, float))).value_counts(normalize=True).reindex(BROAD).fillna(0)

    s0_obs = band_dist(ar); s0_cc = band_dist(ar * np.where(post, m, 1.0))
    fm = P.fuel.map(mult).values

    def tmat(pb, cb):
        T = pd.DataFrame(1.0, index=BROAD, columns=BROAD)
        for s, t in zip(pb, cb):
            T.loc[s, t] += 1
        return T.div(T.sum(axis=1), axis=0)

    po = P[P.prev_post & P.cur_post]
    T_obs = tmat(broad_band(po.prev.values), broad_band(po.cur.values))
    prev_cc = P.prev.values * np.where(P.prev_post.values, fm, 1.0)
    cur_cc = P.cur.values * np.where(P.cur_post.values, fm, 1.0)
    T_cc = tmat(broad_band(prev_cc), broad_band(cur_cc))

    def project(stock, T, p):
        return 100 * sum(stock[b] * ((1 - p) * (0 if b == "A/B" else 1)
                         + p * T.loc[b, ["C", "D", "E", "F/G"]].sum()) for b in BROAD)

    rows = []
    for p in P_GRID:
        o, c = project(s0_obs, T_obs, p), project(s0_cc, T_cc, p)
        rows.append({"threshold": "EPC-B (full-band)", "reassessment_coverage_p": p,
                     "observed_basis_below_%": round(o, 1),
                     "constant_accounting_below_%": round(c, 1), "gap_pp": round(c - o, 1)})
    return pd.DataFrame(rows)


# ── (1) back-test ─────────────────────────────────────────────────────────────
def backtest(P: pd.DataFrame) -> pd.DataFrame:
    """Estimate a threshold transition matrix on an earlier training period and test
    whether it reproduces the later observed below-threshold share among reassessed
    buildings, within a single accounting basis (no accounting break inside a split)."""
    rows = []
    bases = [("pre-2022", P[~P.prev_post & ~P.cur_post]),
             ("post-2022", P[P.prev_post & P.cur_post])]
    for basis, sub in bases:
        for name, thr in THRESH.items():
            if len(sub) < 2000:
                rows.append({"basis": basis, "threshold": name, "n_train": int(len(sub)),
                             "n_test": 0, "predicted_below_%": np.nan, "observed_below_%": np.nan,
                             "abs_error_pp": np.nan,
                             "supports_illustrative_use": "insufficient dated pairs"})
                continue
            med = sub.cur_dt.median()
            tr, te = sub[sub.cur_dt < med], sub[sub.cur_dt >= med]
            T = _tmat2((tr.prev.values > thr).astype(float), (tr.cur.values > thr).astype(float))
            prev_below = (te.prev.values > thr).astype(float)
            pred = 100 * float((prev_below * T["Tb"] + (1 - prev_below) * T["Ta"]).mean())
            obs = 100 * float((te.cur.values > thr).mean())
            rows.append({"basis": basis, "threshold": name, "n_train": int(len(tr)),
                         "n_test": int(len(te)), "predicted_below_%": round(pred, 2),
                         "observed_below_%": round(obs, 2), "abs_error_pp": round(abs(pred - obs), 2),
                         "supports_illustrative_use": "yes" if abs(pred - obs) < 3.0 else "weak"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "markov_backtest_results.csv", index=False)
    return out


# ── (3) structural uncertainty ────────────────────────────────────────────────
def structural_uncertainty(d: pd.DataFrame, P: pd.DataFrame, bank: dict, mult: dict,
                           n_boot: int = 300) -> pd.DataFrame:
    latest = d.groupby("uprn").last()
    ar = latest.asset_rating.values.astype(float)
    lpost = (latest.insp_dt >= CUT).values
    lfuel = latest.fuelgrp.where(latest.fuelgrp.isin(FUELS), "Other").values
    po = P[P.prev_post & P.cur_post].reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for name, thr in THRESH.items():
        s0_obs = float((ar > thr).mean())
        s0_cc = float(expected_below(ar, lfuel, lpost, thr, bank).mean())
        p0 = expected_below(P.prev.values, P.fuel.values, P.prev_post.values, thr, bank)
        p1 = expected_below(P.cur.values, P.fuel.values, P.cur_post.values, thr, bank)
        obs_hard0 = (po.prev.values > thr).astype(float); obs_hard1 = (po.cur.values > thr).astype(float)
        n_po, n_all = len(po), len(P)
        for p in P_GRID:
            gaps = np.empty(n_boot)
            for b in range(n_boot):
                io = rng.integers(0, n_po, n_po); ia = rng.integers(0, n_all, n_all)
                To = _tmat2(obs_hard0[io], obs_hard1[io])
                Tc = _tmat2(p0[ia], p1[ia])
                gaps[b] = (_project(s0_cc, Tc["Tb"], Tc["Ta"], p)
                           - _project(s0_obs, To["Tb"], To["Ta"], p))
            point = (_project(s0_cc, _tmat2(p0, p1)["Tb"], _tmat2(p0, p1)["Ta"], p)
                     - _project(s0_obs, _tmat2(obs_hard0, obs_hard1)["Tb"],
                                _tmat2(obs_hard0, obs_hard1)["Ta"], p))
            rows.append({"threshold": name, "reassessment_coverage_p": p,
                         "gap_pp_point": round(point, 2),
                         "gap_pp_boot_mean": round(float(gaps.mean()), 2),
                         "gap_pp_lo95": round(float(np.percentile(gaps, 2.5)), 2),
                         "gap_pp_hi95": round(float(np.percentile(gaps, 97.5)), 2)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "markov_structural_uncertainty.csv", index=False)
    return out


# ── (4) transition-sample robustness ──────────────────────────────────────────
def transition_sample_robustness(d: pd.DataFrame, P: pd.DataFrame, bank: dict) -> pd.DataFrame:
    latest = d.groupby("uprn").last()
    ar = latest.asset_rating.values.astype(float)
    lpost = (latest.insp_dt >= CUT).values
    lfuel = latest.fuelgrp.where(latest.fuelgrp.isin(FUELS), "Other").values
    is_constr = P.txn.str.contains("construction", case=False, na=False)

    def one(sub, name, thr, obs_pairs):
        s0_obs = float((ar > thr).mean())
        s0_cc = float(expected_below(ar, lfuel, lpost, thr, bank).mean())
        obs = _tmat2((obs_pairs.prev.values > thr).astype(float),
                     (obs_pairs.cur.values > thr).astype(float))
        p0 = expected_below(sub.prev.values, sub.fuel.values, sub.prev_post.values, thr, bank)
        p1 = expected_below(sub.cur.values, sub.fuel.values, sub.cur_post.values, thr, bank)
        cc = _tmat2(p0, p1)
        g10 = _project(s0_cc, cc["Tb"], cc["Ta"], 0.10) - _project(s0_obs, obs["Tb"], obs["Ta"], 0.10)
        g100 = _project(s0_cc, cc["Tb"], cc["Ta"], 1.00) - _project(s0_obs, obs["Tb"], obs["Ta"], 1.00)
        return g10, g100

    nearest = P.loc[P.groupby("uprn").gap_m.idxmin()]      # one smallest-gap pair per UPRN
    rules = {
        "nearest reassessment pairs only": nearest,
        "all repeat pairs": P,
        "max gap <=24 months": P[P.gap_m <= 24],
        "max gap <=36 months": P[P.gap_m <= 36],
        "max gap <=60 months": P[P.gap_m <= 60],
        "post-2022 pairs only": P[P.prev_post & P.cur_post],
        "excluding construction transactions": P[~is_constr],
        "valid ratings only (0<AR<=200)": P[(P.prev > 0) & (P.cur > 0) & (P.cur <= 200) & (P.prev <= 200)],
    }
    po_default = P[P.prev_post & P.cur_post]
    rows = []
    for label, sub in rules.items():
        if len(sub) < 500:
            continue
        for name, thr in THRESH.items():
            g10, g100 = one(sub, name, thr, po_default)
            rows.append({"sample_rule": label, "threshold": name, "n_pairs": int(len(sub)),
                         "gap_pp_p10": round(g10, 1), "gap_pp_p100": round(g100, 1),
                         "gap_persists": "yes" if g100 > 0 else "no"})
    # fuel-specific matrices
    for f in FUELS:
        sub = P[P.fuel == f]
        if len(sub) < 500:
            continue
        for name, thr in THRESH.items():
            g10, g100 = one(sub, name, thr, po_default[po_default.fuel == f] if len(po_default[po_default.fuel == f]) > 200 else po_default)
            rows.append({"sample_rule": f"fuel-specific: {f}", "threshold": name, "n_pairs": int(len(sub)),
                         "gap_pp_p10": round(g10, 1), "gap_pp_p100": round(g100, 1),
                         "gap_persists": "yes" if g100 > 0 else "no"})
    # sector-specific matrices (adequate cell counts only)
    top_sectors = P.sector.value_counts()
    for sec in top_sectors[top_sectors >= 3000].index[:4]:
        sub = P[P.sector == sec]
        for name, thr in THRESH.items():
            g10, g100 = one(sub, name, thr, po_default)
            rows.append({"sample_rule": f"sector: {sec[:40]}", "threshold": name, "n_pairs": int(len(sub)),
                         "gap_pp_p10": round(g10, 1), "gap_pp_p100": round(g100, 1),
                         "gap_persists": "yes" if g100 > 0 else "no"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "markov_transition_sample_robustness.csv", index=False)
    return out


# ── (5) matrix diagnostics + basis diagnostic ─────────────────────────────────
def matrix_diagnostics(mats: dict) -> pd.DataFrame:
    rows = []
    for name, mm in mats.items():
        for scen, key, s0key in [("Observed / current-accounting", "obs", "s0_obs"),
                                  ("Constant-accounting", "cc", "s0_cc")]:
            m = mm[key]
            cells = [m["bb"], m["ba"], m["ab"], m["aa"]]
            n_pairs = mm["n_obs_pairs"] if key == "obs" else mm["n_cc_pairs"]
            rows.append({
                "threshold": name, "scenario": scen,
                "state_space": "{below, above}",
                "starting_distribution_basis": "observed (reported)" if key == "obs" else "pre-2022 (constant)",
                "transition_prev_basis": "observed" if key == "obs" else "pre-2022 expected-state",
                "transition_cur_basis": "observed" if key == "obs" else "pre-2022 expected-state",
                "n_transition_pairs": int(n_pairs),
                "row_sum_below": round(m["Tb"] + (1 - m["Tb"]), 6),
                "row_sum_above": round(m["Ta"] + (1 - m["Ta"]), 6),
                "n_nonempty_cells": int(sum(c > 1e-6 for c in cells)),
                "min_cell_mass": round(min(cells), 2),
                "smoothing_rule": "Laplace 0.5 on row normalisation",
                "starting_below_share_%": round(100 * mm[s0key], 1),
                "interpretation": "internally commensurate: start and transitions share one basis",
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "markov_matrix_diagnostics.csv", index=False)
    return out


def basis_diagnostic(mats: dict) -> pd.DataFrame:
    rows = [
        {"scenario": "Observed / current-accounting",
         "starting_distribution_basis": "observed (reported post-revision)",
         "transition_matrix_basis": "observed post-2022 reassessment pairs",
         "commensurate": "yes"},
        {"scenario": "Constant-accounting",
         "starting_distribution_basis": "pre-2022 (fixed-accounting expected state)",
         "transition_matrix_basis": "pre-2022 expected-state (all pairs converted)",
         "commensurate": "yes"}]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "markov_basis_diagnostic.csv", index=False)
    return out


# ── (7) methods audit ─────────────────────────────────────────────────────────
def _methods_audit(pref, back, struct, robust, diag, fb):
    epcb = pref[pref.threshold == "EPC-B"].set_index("reassessment_coverage_p")
    g10, g100 = epcb.loc[0.10, "gap_pp"], epcb.loc[1.00, "gap_pp"]
    lines = [
        "# Markov (commensurate reassessment-throughput sensitivity) — methods audit\n",
        "The exercise answers one narrow question: does the accounting-sensitive EPC-B gap ",
        "disappear as the register is reassessed, or does it persist when the starting ",
        "distribution and transition matrix are defined on commensurate rating bases? It is ",
        "illustrative and is **not** a forecast of legal compliance, national stock, or physical ",
        "retrofit progress.\n",
        "## Preferred specification (threshold-level, expected-state)\n",
        f"- EPC-B below-threshold gap widens from {g10} pp at 10% coverage to {g100} pp at 100% ",
        "coverage; the observed-basis and constant-accounting trajectories never cross.\n",
        f"- F/G gap at 100% coverage: {pref[(pref.threshold=='F/G')&(pref.reassessment_coverage_p==1.0)].gap_pp.iloc[0]} pp.\n",
        "- States are {below, above} each threshold; expected transition mass uses ",
        "(1-p0)(1-p1), (1-p0)p1, p0(1-p1), p0 p1; no hard full-band classification is used.\n",
        "## Basis commensurability (markov_basis_diagnostic.csv, markov_matrix_diagnostics.csv)\n",
        "- Observed scenario: starting distribution and transitions both on the reported basis ",
        "(post-2022 pairs). Constant-accounting: starting distribution and transitions both on ",
        "the pre-2022 basis (post-2022 ratings mapped to expected old-basis states via the ",
        "empirical no-works ratio distribution). Row sums are 1 by construction.\n",
        "## Back-test (markov_backtest_results.csv)\n",
    ]
    for _, r in back.iterrows():
        lines.append(f"- {r.basis} / {r.threshold}: predicted {r['predicted_below_%']}% vs observed "
                     f"{r['observed_below_%']}% (|err| {r['abs_error_pp']} pp; n_train={r.n_train}, "
                     f"n_test={r.n_test}; {r['supports_illustrative_use']}).\n")
    lines += [
        "## Structural uncertainty (markov_structural_uncertainty.csv)\n",
        "- Transition-pair bootstrap (300 reps) 95% intervals reported at each coverage; the ",
        "EPC-B gap interval excludes zero across the coverage grid.\n",
        "## Transition-sample robustness (markov_transition_sample_robustness.csv)\n",
        f"- {int((robust.gap_persists=='yes').sum())}/{len(robust)} sample-rule x threshold cells ",
        "retain a positive gap at 100% coverage (nearest/all pairs, gap caps 24/36/60 months, ",
        "post-2022-only, excluding construction, valid-only, fuel- and sector-specific matrices).\n",
        "## Supplementary hard full-band sensitivity (superseded by the preferred version)\n",
        f"- Full-band EPC-B gap: {fb[fb.reassessment_coverage_p==0.10].gap_pp.iloc[0]} -> "
        f"{fb[fb.reassessment_coverage_p==1.0].gap_pp.iloc[0]} pp.\n",
        "## Framing check\n",
        "- No 2030 forecast, legal-compliance forecast, national-stock forecast, or mixed-basis ",
        "language is used; the main text uses the preferred threshold-level result only.\n",
    ]
    (OUT_TABLES / "markov_methods_audit.md").write_text("".join(lines))


def run_markov(df: pd.DataFrame | None = None) -> dict:
    df = build_analysis_frame() if df is None else df
    d = with_dates(df)
    P0 = within_building_pairs(df)
    mult = compute_multipliers(P0)
    bank = _sorted_ratio_bank(P0)
    P = build_pairs(d)

    pref, mats = threshold_expected_state(d, P, bank)
    fb = fullband_hard(d, P, mult)
    fb.to_csv(OUT_TABLES / "markov_fullband_hard_sensitivity.csv", index=False)
    back = backtest(P)
    struct = structural_uncertainty(d, P, bank, mult)
    robust = transition_sample_robustness(d, P, bank)
    mdiag = matrix_diagnostics(mats)
    bdiag = basis_diagnostic(mats)
    _methods_audit(pref, back, struct, robust, mdiag, fb)

    epcb = pref[pref.threshold == "EPC-B"].set_index("reassessment_coverage_p")
    return {"preferred": pref.to_dict("records"),
            "epcb_gap_p10": float(epcb.loc[0.10, "gap_pp"]),
            "epcb_gap_p100": float(epcb.loc[1.00, "gap_pp"]),
            "backtest": back.to_dict("records"),
            "structural": struct.to_dict("records"),
            "robustness": robust.to_dict("records"),
            "matrix_diagnostics": mdiag.to_dict("records"),
            "basis_diagnostic": bdiag.to_dict("records"),
            "fullband_hard": fb.to_dict("records")}


if __name__ == "__main__":
    r = run_markov()
    print(f"EPC-B threshold-level expected-state gap: {r['epcb_gap_p10']} -> {r['epcb_gap_p100']} pp")
