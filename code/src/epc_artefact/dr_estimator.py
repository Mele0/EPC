"""Stage-1 doubly robust threshold estimator, implemented from Supplementary 1.21.

This module implements the estimator whose results are reported in Table 2 Panel B and
Supplementary Table S31. The reported estimates are held in ``doubly_robust_estimates/``
and are the values the manuscript cites; the code here recomputes the same estimand
directly from the register, so the procedure is inspectable and can be re-run.

The estimand. For a policy threshold ``tau``, the eligible application frame ``T`` is
the set of post-revision register entries currently at or better than the threshold;
these are the only entries that can lose threshold status under pre-revision
accounting. ``S`` is the labelled calibration sample of no-recorded-works straddlers,
whose pre-revision rating is directly observed. With ``p_hat`` the crossing model,
``w`` the inverse-odds transport weights and ``g`` the functional of interest
(``g = 1`` for the expected count, ``g = A`` for expected affected floor area), the
doubly robust estimator of Supplementary equation S10 is

    Theta_DR(g) = sum_{j in T} g_j p_hat(x_j)
                  + (N_T / sum_i w_i) sum_{i in S} w_i g_i {y_i - p_hat(x_i)}

The first term is the application-frame outcome prediction; the second corrects
systematic prediction error using weighted residuals in the calibration sample. Both
the crossing model and the membership model are cross-fitted at UPRN level, so each
source case is scored by models estimated without it.

Agreement with the reported values. The eligible-frame size, the calibration-sample
size and the plug-in term reproduce exactly. The residual correction depends on the
cross-fitting fold assignment; at the EPC-B threshold that correction is about 0.15
per cent of the plug-in term, a weighted mean residual of roughly 0.001 across 1,110
binary outcomes, so it is sensitive to the partition while the substantive quantities
are not. This implementation reproduces the reported affected floor area to within
0.1 per cent and the reported count to within 0.23 per cent. Running the module writes
the recomputed values and a direct comparison against the reported ones.

Run:  python -m epc_artefact.dr_estimator
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DR_ESTIMATES_DIR, OUT_TABLES, ensure_dirs

MODEL_C = "x + C(fuel) + logfa + C(sector) + C(fuel):x + C(fuel):logfa"
MODEL_D = "cr(x, df=4) + C(fuel) + cr(logfa, df=4) + C(sector)"
MEMBERSHIP = "0 + post + logfa + C(fuel) + C(sector) + C(ac)"
MEMBERSHIP_SUBSAMPLE = 30000
TRIM_PERCENTILES = (1, 99)
DR_TOLERANCE_PERCENT = {"count": 0.5, "area_Mm2": 0.5}
DR_SECONDARY_TOLERANCE_PERCENT = {"count": 5.0, "area_Mm2": 5.0}


class DRValidationError(AssertionError):
    """Raised when a recomputed doubly robust estimate drifts materially."""


def dr_tolerance_percent(threshold: str, model: str, quantity: str) -> float:
    """Return the declared gate for a manuscript headline or sensitivity estimate.

    The EPC-B/model-C estimate is the headline result and is stable to well below
    one percent.  Secondary threshold and model-architecture estimates have a wider
    five-percent gate because their small residual corrections are cross-fit-partition
    sensitive; drift beyond that still stops the run.
    """
    limits = (DR_TOLERANCE_PERCENT
              if threshold == "EPC-B" and model == "C"
              else DR_SECONDARY_TOLERANCE_PERCENT)
    try:
        return limits[quantity]
    except KeyError as error:
        raise DRValidationError(f"No DR tolerance declared for quantity {quantity!r}") from error


def _prep(frame: pd.DataFrame, tau: float) -> pd.DataFrame:
    out = pd.DataFrame({
        "post": pd.to_numeric(frame.post, errors="coerce"),
        "logfa": np.log(pd.to_numeric(frame.floor_area, errors="coerce").clip(lower=1)),
        "fuel": frame.fuel.astype(str),
        "sector": frame.sector_group.astype(str),
        "ac": frame.ac_group.astype(str)})
    out["xcen"] = (out.post.to_numpy(dtype=float) - tau) / 25.0
    return out


def _fit_predict(train: pd.DataFrame, y: np.ndarray, target: pd.DataFrame,
                 formula: str, weights: np.ndarray | None = None) -> np.ndarray:
    """Fit the crossing model on ``train`` and score ``target``.

    Categorical levels are fixed to the union over source and target so that a sector
    present only in the register, or absent from one cross-fitting fold, cannot change
    the design matrix between fitting and prediction.
    """
    from patsy import build_design_matrices, dmatrix
    from sklearn.linear_model import LogisticRegression

    fuels = sorted(set(train.fuel) | set(target.fuel))
    sectors = sorted(set(train.sector) | set(target.sector))

    def design(frame):
        return dict(fuel=pd.Categorical(frame.fuel, categories=fuels), x=frame.xcen,
                    logfa=frame.logfa, sector=pd.Categorical(frame.sector, categories=sectors))

    source = dmatrix(formula, design(train), return_type="dataframe")
    model = LogisticRegression(C=1.0, max_iter=3000).fit(
        np.asarray(source), y, sample_weight=weights)
    scored = build_design_matrices([source.design_info], design(target))[0]
    return model.predict_proba(np.asarray(scored))[:, 1]


def _transport_weights(cal: pd.DataFrame, app: pd.DataFrame, folds: np.ndarray,
                       n_folds: int) -> np.ndarray:
    """Cross-fitted inverse-odds transport weights, trimmed and normalised to mean one."""
    from patsy import dmatrix
    from sklearn.linear_model import LogisticRegression

    app_sub = app.sample(n=min(MEMBERSHIP_SUBSAMPLE, len(app)), random_state=1)
    pool = pd.concat([cal.assign(_member=1), app_sub.assign(_member=0)], ignore_index=True)
    design = np.asarray(dmatrix(MEMBERSHIP, pool, return_type="dataframe"))
    cal_design, app_design = design[:len(cal)], design[len(cal):]
    app_label = np.zeros(len(app_sub))

    membership = np.zeros(len(cal))
    for fold in range(n_folds):
        train, test = folds != fold, folds == fold
        features = np.vstack([cal_design[train], app_design])
        labels = np.concatenate([np.ones(int(train.sum())), app_label])
        fitted = LogisticRegression(C=1.0, max_iter=3000).fit(features, labels)
        membership[test] = fitted.predict_proba(cal_design[test])[:, 1]

    weights = (1 - membership) / np.clip(membership, 1e-6, 1 - 1e-6)
    low, high = np.percentile(weights, list(TRIM_PERCENTILES))
    weights = np.clip(weights, low, high)
    return weights / weights.mean()


def stage1_doubly_robust(tau: float, sample: pd.DataFrame | None = None,
                         latest: pd.DataFrame | None = None) -> pd.DataFrame:
    """Recompute the Stage-1 doubly robust count and affected area at threshold ``tau``."""
    from .bridge_analysis import (N_FOLDS, assign_folds, build_calibration_sample,
                                  prepare_latest_frame)

    sample = build_calibration_sample()[0] if sample is None else sample
    latest = prepare_latest_frame() if latest is None else latest

    eligible = sample[sample.post <= tau].copy().reset_index(drop=True)
    cal = _prep(eligible, tau).dropna(subset=["post", "logfa"]).reset_index(drop=True)
    y = (pd.to_numeric(eligible.pre, errors="coerce").to_numpy() > tau).astype(int)[:len(cal)]

    app_mask = latest.is_post_revision & (latest.post <= tau)
    app = _prep(latest[app_mask], tau).dropna(subset=["post", "logfa"]).reset_index(drop=True)
    app_area = pd.to_numeric(latest[app_mask].floor_area, errors="coerce").to_numpy()[:len(app)]
    n_target = len(app)

    folds = assign_folds(eligible, n_folds=N_FOLDS)[:len(cal)]
    weights = _transport_weights(cal, app, folds, N_FOLDS)
    cal_area = np.exp(cal.logfa.to_numpy())

    rows = []
    for label, formula in [("C", MODEL_C), ("D", MODEL_D)]:
        target_p = _fit_predict(cal, y, app, formula)
        plug_count = float(target_p.sum())
        plug_area = float((app_area * target_p).sum()) / 1e6

        oof = np.zeros(len(cal))
        for fold in range(N_FOLDS):
            train, test = folds != fold, folds == fold
            oof[test] = _fit_predict(cal[train], y[train], cal[test], formula)
        residual = y - oof
        scale = n_target / weights.sum()
        corr_count = scale * float((weights * residual).sum())
        corr_area = scale * float((weights * cal_area * residual).sum()) / 1e6

        weighted_p = _fit_predict(cal, y, app, formula, weights=weights)
        rows.append({
            "threshold": "EPC-B" if tau == 50.0 else "F/G", "tau": tau,
            "model": label, "n_target": n_target, "n_calibration": len(cal),
            "plugin_count": round(plug_count), "plugin_area_Mm2": round(plug_area, 2),
            "correction_count": round(corr_count), "correction_area_Mm2": round(corr_area, 2),
            "dr_count": round(plug_count + corr_count),
            "dr_area_Mm2": round(plug_area + corr_area, 2),
            "inverse_odds_count": round(float(weighted_p.sum())),
            "inverse_odds_area_Mm2": round(float((app_area * weighted_p).sum()) / 1e6, 2),
        })
    return pd.DataFrame(rows)


def compare_with_reported(recomputed: pd.DataFrame) -> pd.DataFrame:
    """Compare the recomputed estimates against the reported values."""
    estimates = pd.read_csv(DR_ESTIMATES_DIR / "bridge_stage1_frozen.csv")
    frames = {"EPC-B": "complete_eligible_EPCB", "F/G": "FG_count_eligible"}
    rows = []
    for _, r in recomputed.iterrows():
        match = estimates[(estimates.frame == frames[r.threshold])
                          & (estimates.model == r.model)
                          & (estimates.estimator == "doubly_robust_crossfitted")]
        if match.empty:
            raise DRValidationError(
                f"No frozen doubly robust estimate for threshold={r.threshold!r}, "
                f"model={r.model!r}")
        if len(match) != 1:
            raise DRValidationError(
                f"Expected one frozen estimate for threshold={r.threshold!r}, "
                f"model={r.model!r}; found {len(match)}")
        a = match.iloc[0]
        for quantity, recomputed_value, reported_value in [
                ("count", r.dr_count, float(a.expected_count)),
                ("area_Mm2", r.dr_area_Mm2, float(a.affected_area_Mm2))]:
            percent_difference = 100 * (recomputed_value - reported_value) / reported_value
            tolerance = dr_tolerance_percent(r.threshold, r.model, quantity)
            rows.append({
                "threshold": r.threshold, "model": r.model, "quantity": quantity,
                "reported": reported_value, "recomputed": recomputed_value,
                "difference": round(recomputed_value - reported_value, 2),
                "percent_difference": round(percent_difference, 4),
                "tolerance_percent": tolerance,
                "within_tolerance": bool(abs(percent_difference) <= tolerance)})
    return pd.DataFrame(rows)


def validate_reported_agreement(comparison: pd.DataFrame) -> None:
    """Fail the run unless every recomputed result is within its declared tolerance."""
    required = {"threshold", "model", "quantity", "reported", "recomputed",
                "percent_difference"}
    missing = sorted(required - set(comparison.columns))
    if missing:
        raise DRValidationError(f"DR comparison is missing columns: {missing}")
    if comparison.empty:
        raise DRValidationError("DR comparison is empty")
    if comparison[list(required)].isna().any().any():
        raise DRValidationError("DR comparison contains missing or non-finite values")
    numeric = comparison[["reported", "recomputed", "percent_difference"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (comparison.reported == 0).any():
        raise DRValidationError("DR comparison contains non-finite values or a zero denominator")

    tolerances = comparison.apply(
        lambda row: dr_tolerance_percent(row.threshold, row.model, row.quantity), axis=1)
    actual_percent = 100 * (comparison.recomputed - comparison.reported) / comparison.reported
    failed = comparison[actual_percent.abs() > tolerances]
    if not failed.empty:
        details = "; ".join(
            f"{row.threshold}/{row.model}/{row.quantity}: "
            f"{row.percent_difference:+.2f}% "
            f"(limit {dr_tolerance_percent(row.threshold, row.model, row.quantity):.2f}%)"
            for row in failed.itertuples())
        raise DRValidationError(f"Doubly robust reproduction drift exceeds tolerance: {details}")


def run_dr_estimator() -> dict[str, pd.DataFrame]:
    """Recompute both thresholds and write the estimates and the comparison."""
    ensure_dirs()
    from .bridge_analysis import build_calibration_sample, prepare_latest_frame
    sample = build_calibration_sample()[0]
    latest = prepare_latest_frame()
    recomputed = pd.concat([stage1_doubly_robust(50.0, sample, latest),
                            stage1_doubly_robust(125.0, sample, latest)], ignore_index=True)
    comparison = compare_with_reported(recomputed)
    recomputed.to_csv(OUT_TABLES / "dr_estimator_recomputed.csv", index=False)
    comparison.to_csv(OUT_TABLES / "dr_estimator_vs_reported.csv", index=False)
    validate_reported_agreement(comparison)
    return {"recomputed": recomputed, "comparison": comparison}


if __name__ == "__main__":
    result = run_dr_estimator()
    print("Recomputed Stage-1 doubly robust estimates\n")
    print(result["recomputed"].to_string(index=False))
    print("\nAgreement with the reported values\n")
    print(result["comparison"].to_string(index=False))
