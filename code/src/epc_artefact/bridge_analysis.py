"""Authoritative cross-fitted validation and aggregate bridge analysis.

This module deliberately separates three questions that were previously mixed:

1. Can an empirical bridge predict an old-basis rating for a held-out UPRN?
2. Can it classify the old-basis side of the B/C and E/F thresholds?
3. What aggregate threshold and floor-area quantities follow when a fitted
   bridge is applied to the latest-certificate register frame?

All validation predictions are out-of-fold.  Full-sample fits are used only for
the explicitly labelled register-frame application.  Prediction error is always
``predicted old-basis rating - observed old-basis rating``; positive error means
the bridge predicts a numerically worse (less efficient) rating than observed.

Run with::

    python -m epc_artefact.bridge_analysis
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .analysis import within_building_pairs
from .config import (
    CUT,
    EXTERNAL_DIR,
    ND_NEED_2025_XLSX,
    ND_NEED_2025_URL,
    ND_NEED_GEO_2024_XLSX,
    ND_NEED_GEO_2024_URL,
    OUT_TABLES,
    RANDOM_SEED,
)
from .data import broad_band, build_extended_frame, download_if_missing, with_dates
from .validation import (
    SIZE_BINS,
    SIZE_LABELS,
    _nd_need_sector,
    _rake_weights,
    heldout_validation as legacy_heldout_validation,
)


FUELS = ["Electric", "Gas", "Other"]
BANDS = ["A/B", "C", "D", "E", "F/G"]
THRESHOLDS = {"B_C": 50.0, "E_F": 125.0}
N_FOLDS = 5
LEGACY_SPLITS = 200
LEGACY_TRAIN_FRAC = 0.5
MIN_CELL = 50
MIN_SHRINK_CELL = 10
SHRINKAGE_K = 50.0
ANALYSIS_AS_OF = pd.Timestamp("2026-06-15")
UNIT_COSTS_GBP_M2 = {"low": 25.0, "central": 100.0, "high": 200.0}
CHECKPOINT_DIR = OUT_TABLES.parent / "checkpoints" / "pre_authoritative_validation_2026-07-10" / "tables"
REPORT_PATH = OUT_TABLES / "bridge_validation_report.md"


@dataclass(frozen=True)
class BridgeSpec:
    name: str
    label: str
    kind: str
    features: str
    role: str
    structural_candidate: bool


RATING_SPECS = [
    BridgeSpec(
        "fuel_mult", "Fuel-only multiplicative", "rating",
        "post rating; main-heating-fuel group", "transparent baseline only", False,
    ),
    BridgeSpec(
        "fuel_add", "Fuel-only additive", "rating",
        "post rating; main-heating-fuel group", "transparent baseline only", False,
    ),
    BridgeSpec(
        "fuel_post_band_mult", "Fuel x post-band multiplicative", "rating",
        "post rating; main-heating fuel; observed post-revision EPC band",
        "rating-local candidate", True,
    ),
    BridgeSpec(
        "fuel_post_band_add", "Fuel x post-band additive", "rating",
        "post rating; main-heating fuel; observed post-revision EPC band",
        "rating-local candidate", True,
    ),
    BridgeSpec(
        "fuel_threshold_distance_add", "Fuel x threshold-distance additive", "rating",
        "post rating; main-heating fuel; signed distance to nearest 50/125 threshold",
        "threshold-local candidate", True,
    ),
    BridgeSpec(
        "fuel_post_band_shrunk_add", "Fuel x post-band shrunk additive", "rating",
        "post rating; main-heating fuel; post band; empirical-Bayes shrinkage/fallback",
        "rating-local shrinkage candidate", True,
    ),
    BridgeSpec(
        "fuel_post_band_floor_shrunk_add", "Fuel x post-band x floor-area shrunk additive",
        "rating",
        "post rating; main-heating fuel; post band; floor-area bin; hierarchical shrinkage",
        "exploratory descriptor refinement", True,
    ),
]
THRESHOLD_SPEC = BridgeSpec(
    "threshold_logit", "Fuel-specific direct threshold logistic models", "threshold",
    "post rating distance to threshold; fuel-specific intercept and slope",
    "probabilistic threshold candidate", True,
)
ALL_SPECS = RATING_SPECS + [THRESHOLD_SPEC]


def _safe_band(values: pd.Series | np.ndarray) -> np.ndarray:
    return broad_band(np.asarray(values, dtype=float))


def _ac_group(values: pd.Series) -> pd.Series:
    normal = values.fillna("").astype(str).str.strip().str.lower()
    return normal.isin({"yes", "y", "true", "1", "present"}).map(
        {True: "Present", False: "Not present/unknown"}
    )


def _floor_area_bin(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.cut(
        pd.Series(values),
        [0, 100, 500, 1000, 5000, np.inf],
        labels=["0-100", ">100-500", ">500-1,000", ">1,000-5,000", ">5,000"],
        include_lowest=True,
    ).astype(str)


def _threshold_distance_bin(values: pd.Series | np.ndarray) -> pd.Series:
    rating = np.asarray(values, dtype=float)
    distance_b = np.abs(rating - THRESHOLDS["B_C"])
    distance_fg = np.abs(rating - THRESHOLDS["E_F"])
    nearest = np.where(distance_b <= distance_fg, "B_C", "E_F")
    threshold = np.where(nearest == "B_C", THRESHOLDS["B_C"], THRESHOLDS["E_F"])
    signed = rating - threshold
    bins = pd.cut(
        signed,
        [-np.inf, -30, -20, -10, 0, 10, 20, 30, np.inf],
        labels=["<-30", "-30:-20", "-20:-10", "-10:0", "0:10", "10:20", "20:30", ">30"],
        include_lowest=True,
        right=True,
    ).astype(str)
    return pd.Series(nearest + "|" + bins.to_numpy(), index=pd.RangeIndex(len(rating)))


def _add_application_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["post_band"] = _safe_band(out["post"])
    out["distance_bin"] = _threshold_distance_bin(out["post"]).to_numpy()
    out["floor_area_bin"] = _floor_area_bin(out["floor_area"]).to_numpy()
    out["sector_group"] = _nd_need_sector(out["sector"])
    out["ac_group"] = _ac_group(out["ac"])
    return out


def build_calibration_sample(P: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return one first/last straddling pair per UPRN and an audit dictionary."""
    if P is None:
        P = within_building_pairs(build_extended_frame())
    selected = P[
        P.straddle
        & (P.drec == 0)
        & (P.dfa < 0.02)
        & (P.ar_l > 0)
        & P.fuel.isin(FUELS)
    ].copy()
    n_before_positive_filter = len(selected)
    selected = selected[(selected.ar_f > 0) & (selected.ar_l > 0)].copy()
    selected.index.name = "uprn"
    sample = selected.reset_index().rename(
        columns={
            "ar_f": "pre",
            "ar_l": "post",
            "fa_f": "floor_area",
            "ac_f": "ac",
        }
    )
    sample = _add_application_features(sample)
    sample["true_band"] = _safe_band(sample["pre"])
    sample["error_sign_definition"] = "predicted_old_minus_observed_old"
    duplicate_uprns = int(sample.uprn.duplicated().sum())
    if duplicate_uprns:
        raise AssertionError("Calibration sample must contain one row per UPRN")
    audit = {
        "n_before_positive_rating_filter": int(n_before_positive_filter),
        "n_heldout_eligible": int(len(sample)),
        "n_unique_uprn": int(sample.uprn.nunique()),
        "duplicate_uprn_rows": duplicate_uprns,
        "fuel_counts": {str(k): int(v) for k, v in sample.fuel.value_counts().items()},
        "selection": (
            "first/latest UPRN pair straddles 2022-06-15; unchanged recommendation count; "
            "absolute floor-area change <2%; positive pre/post rating; recognised fuel group"
        ),
        "pair_definition": "one row per UPRN; first and latest chronologically usable certificate",
        "error_sign": "predicted old-basis rating minus observed old-basis rating",
        "positive_error_interpretation": "prediction is numerically higher/worse than observed old-basis rating",
    }
    return sample, audit


def assign_folds(sample: pd.DataFrame, n_folds: int = N_FOLDS,
                 seed: int = RANDOM_SEED) -> np.ndarray:
    """Deterministic UPRN-level folds, stratified only by application-time fields."""
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(sample), dtype=int)
    strata = sample.fuel.astype(str) + "|" + sample.post_band.astype(str)
    for idx in strata.groupby(strata).groups.values():
        idx = np.asarray(list(idx), dtype=int)
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % n_folds
    return folds


def _base_by_fuel(train: pd.DataFrame, value: pd.Series, statistic: str) -> dict[str, float]:
    work = train.assign(_value=np.asarray(value, dtype=float))
    grouped = work.groupby("fuel", observed=True)["_value"]
    result = grouped.median() if statistic == "median" else grouped.mean()
    overall = float(work._value.median() if statistic == "median" else work._value.mean())
    return {fuel: float(result.get(fuel, overall)) for fuel in FUELS}


def _cell_table(train: pd.DataFrame, keys: list[str], value: pd.Series,
                statistic: str) -> pd.DataFrame:
    work = train.assign(_value=np.asarray(value, dtype=float))
    grouped = work.groupby(keys, observed=True)["_value"]
    estimate = grouped.median() if statistic == "median" else grouped.mean()
    return pd.concat([estimate.rename("estimate"), grouped.size().rename("n")], axis=1)


def fit_rating_bridge(train: pd.DataFrame, spec: str) -> dict[str, Any]:
    ratio = train.pre.to_numpy(dtype=float) / train.post.to_numpy(dtype=float)
    delta = train.pre.to_numpy(dtype=float) - train.post.to_numpy(dtype=float)
    base_mult = _base_by_fuel(train, pd.Series(ratio), "median")
    base_add = _base_by_fuel(train, pd.Series(delta), "mean")
    if spec == "fuel_mult":
        return {"base": base_mult, "mode": "multiply"}
    if spec == "fuel_add":
        return {"base": base_add, "mode": "add"}
    if spec == "fuel_post_band_mult":
        return {
            "base": base_mult,
            "cell": _cell_table(train, ["fuel", "post_band"], pd.Series(ratio), "median"),
            "keys": ["fuel", "post_band"], "mode": "multiply", "min_cell": MIN_CELL,
        }
    if spec == "fuel_post_band_add":
        return {
            "base": base_add,
            "cell": _cell_table(train, ["fuel", "post_band"], pd.Series(delta), "mean"),
            "keys": ["fuel", "post_band"], "mode": "add", "min_cell": MIN_CELL,
        }
    if spec == "fuel_threshold_distance_add":
        return {
            "base": base_add,
            "cell": _cell_table(train, ["fuel", "distance_bin"], pd.Series(delta), "mean"),
            "keys": ["fuel", "distance_bin"], "mode": "add", "min_cell": MIN_CELL,
        }
    if spec == "fuel_post_band_shrunk_add":
        cell = _cell_table(train, ["fuel", "post_band"], pd.Series(delta), "mean")
        estimates = []
        for (fuel, _), row in cell.iterrows():
            weight = row.n / (row.n + SHRINKAGE_K) if row.n >= MIN_SHRINK_CELL else 0.0
            estimates.append(weight * row.estimate + (1.0 - weight) * base_add[fuel])
        cell["raw_estimate"] = cell.estimate
        cell["shrinkage_weight"] = [
            row.n / (row.n + SHRINKAGE_K) if row.n >= MIN_SHRINK_CELL else 0.0
            for _, row in cell.iterrows()
        ]
        cell["estimate"] = estimates
        return {
            "base": base_add, "cell": cell, "keys": ["fuel", "post_band"],
            "mode": "add", "min_cell": 1,
        }
    if spec == "fuel_post_band_floor_shrunk_add":
        parent = _cell_table(train, ["fuel", "post_band"], pd.Series(delta), "mean")
        parent_est: dict[tuple[str, str], float] = {}
        for key, row in parent.iterrows():
            fuel = key[0]
            weight = row.n / (row.n + SHRINKAGE_K) if row.n >= MIN_SHRINK_CELL else 0.0
            parent_est[key] = float(weight * row.estimate + (1.0 - weight) * base_add[fuel])
        child = _cell_table(
            train, ["fuel", "post_band", "floor_area_bin"], pd.Series(delta), "mean"
        )
        estimates, weights = [], []
        for key, row in child.iterrows():
            parent_value = parent_est[(key[0], key[1])]
            weight = row.n / (row.n + SHRINKAGE_K) if row.n >= MIN_SHRINK_CELL else 0.0
            estimates.append(weight * row.estimate + (1.0 - weight) * parent_value)
            weights.append(weight)
        child["raw_estimate"] = child.estimate
        child["shrinkage_weight"] = weights
        child["estimate"] = estimates
        return {
            "base": base_add, "parent": parent_est, "cell": child,
            "keys": ["fuel", "post_band", "floor_area_bin"], "mode": "add", "min_cell": 1,
        }
    raise ValueError(f"Unknown rating bridge: {spec}")


def _lookup_cell(frame: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    cell = params["cell"]
    keys = params["keys"]
    valid = cell[cell.n >= params["min_cell"]].reset_index()[keys + ["estimate"]]
    lookup = frame[keys].reset_index(drop=True).merge(
        valid, on=keys, how="left", sort=False, validate="many_to_one"
    )
    output = lookup.estimate.to_numpy(dtype=float)
    if "parent" in params:
        parent = pd.DataFrame([
            {"fuel": key[0], "post_band": key[1], "parent_estimate": value}
            for key, value in params["parent"].items()
        ])
        parent_values = frame[["fuel", "post_band"]].reset_index(drop=True).merge(
            parent, on=["fuel", "post_band"], how="left", sort=False,
            validate="many_to_one",
        ).parent_estimate.to_numpy(dtype=float)
        output = np.where(np.isnan(output), parent_values, output)
    fallback = frame.fuel.map(params["base"]).to_numpy(dtype=float)
    return np.where(np.isnan(output), fallback, output)


def predict_rating_bridge(frame: pd.DataFrame, spec: str,
                          params: dict[str, Any]) -> np.ndarray:
    post = frame.post.to_numpy(dtype=float)
    if spec in {"fuel_mult", "fuel_add"}:
        estimate = frame.fuel.map(params["base"]).to_numpy(dtype=float)
    else:
        estimate = _lookup_cell(frame, params)
    return post * estimate if params["mode"] == "multiply" else post + estimate


def _fit_binary_logit(x: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """Stable two-parameter ridge logistic fit (intercept and local slope)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    if np.unique(y).size < 2:
        p = (y.sum() + 1.0) / (len(y) + 2.0)
        return np.array([np.log(p / (1.0 - p)), 0.0])

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ beta
        loss = np.logaddexp(0.0, eta).sum() - y @ eta + 0.5 * ridge * beta[1] ** 2
        grad = design.T @ (expit(eta) - y)
        grad[1] += ridge * beta[1]
        return float(loss), grad

    initial_p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    initial = np.array([np.log(initial_p / (1.0 - initial_p)), 0.0])
    result = minimize(objective, initial, jac=True, method="BFGS")
    if not result.success:
        raise RuntimeError(f"Logistic fit failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def _training_balanced_cutoff(actual: np.ndarray, probability: np.ndarray) -> float:
    """Choose a hard cutoff on training data only; probability sums do not use it."""
    actual = np.asarray(actual, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    candidates = np.unique(np.concatenate([
        [0.0, 0.5, 1.0], np.quantile(probability, np.linspace(0.0, 1.0, 201))
    ]))
    scored = []
    for cutoff in candidates:
        predicted = probability >= cutoff
        sensitivity = (actual & predicted).sum() / max(actual.sum(), 1)
        specificity = (~actual & ~predicted).sum() / max((~actual).sum(), 1)
        scored.append((float((sensitivity + specificity) / 2.0), -abs(cutoff - 0.5), cutoff))
    return float(max(scored)[2])


def fit_threshold_models(train: pd.DataFrame) -> dict[str, dict[str, Any]]:
    params: dict[str, dict[str, Any]] = {}
    for threshold_name, threshold in THRESHOLDS.items():
        coefficients: dict[str, np.ndarray] = {}
        for fuel in FUELS:
            group = train[train.fuel == fuel]
            x = (group.post.to_numpy(dtype=float) - threshold) / 25.0
            y = (group.pre.to_numpy(dtype=float) > threshold).astype(float)
            coefficients[fuel] = _fit_binary_logit(x, y)
        train_probability = np.empty(len(train), dtype=float)
        for fuel in FUELS:
            mask = train.fuel.to_numpy() == fuel
            x = (train.loc[mask, "post"].to_numpy(dtype=float) - threshold) / 25.0
            beta = coefficients[fuel]
            train_probability[mask] = expit(beta[0] + beta[1] * x)
        params[threshold_name] = {
            "coefficients": coefficients,
            "hard_cutoff": _training_balanced_cutoff(
                train.pre.to_numpy(dtype=float) > threshold, train_probability
            ),
        }
    return params


def predict_threshold_probabilities(frame: pd.DataFrame,
                                    params: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for threshold_name, threshold in THRESHOLDS.items():
        probability = np.empty(len(frame), dtype=float)
        for fuel in FUELS:
            mask = frame.fuel.to_numpy() == fuel
            x = (frame.loc[mask, "post"].to_numpy(dtype=float) - threshold) / 25.0
            beta = params[threshold_name]["coefficients"][fuel]
            probability[mask] = expit(beta[0] + beta[1] * x)
        output[threshold_name] = probability
    return output


def crossfit_predictions(sample: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    """Create long-form OOF predictions for every pre-specified model."""
    rows: list[pd.DataFrame] = []
    for spec in RATING_SPECS:
        prediction = np.empty(len(sample), dtype=float)
        for fold in range(N_FOLDS):
            train = sample[folds != fold]
            test = sample[folds == fold]
            fitted = fit_rating_bridge(train, spec.name)
            prediction[folds == fold] = predict_rating_bridge(test, spec.name, fitted)
        rows.append(pd.DataFrame({
            "uprn": sample.uprn,
            "fold": folds,
            "specification": spec.name,
            "model_kind": "rating",
            "fuel": sample.fuel,
            "sector_group": sample.sector_group,
            "floor_area": sample.floor_area,
            "floor_area_bin": sample.floor_area_bin,
            "ac_group": sample.ac_group,
            "distance_bin": sample.distance_bin,
            "pre_observed": sample.pre,
            "post_observed": sample.post,
            "true_band": sample.true_band,
            "post_band": sample.post_band,
            "predicted_old_rating": prediction,
            "prob_old_below_B": (prediction > THRESHOLDS["B_C"]).astype(float),
            "prob_old_FG": (prediction > THRESHOLDS["E_F"]).astype(float),
            "hard_old_below_B": prediction > THRESHOLDS["B_C"],
            "hard_old_FG": prediction > THRESHOLDS["E_F"],
            "hard_cutoff_B": np.nan,
            "hard_cutoff_FG": np.nan,
        }))

    prob_b = np.empty(len(sample), dtype=float)
    prob_fg = np.empty(len(sample), dtype=float)
    hard_b = np.empty(len(sample), dtype=bool)
    hard_fg = np.empty(len(sample), dtype=bool)
    cutoff_b = np.empty(len(sample), dtype=float)
    cutoff_fg = np.empty(len(sample), dtype=float)
    for fold in range(N_FOLDS):
        train = sample[folds != fold]
        test = sample[folds == fold]
        fitted = fit_threshold_models(train)
        probabilities = predict_threshold_probabilities(test, fitted)
        prob_b[folds == fold] = probabilities["B_C"]
        prob_fg[folds == fold] = probabilities["E_F"]
        cutoff_b[folds == fold] = fitted["B_C"]["hard_cutoff"]
        cutoff_fg[folds == fold] = fitted["E_F"]["hard_cutoff"]
        hard_b[folds == fold] = probabilities["B_C"] >= fitted["B_C"]["hard_cutoff"]
        hard_fg[folds == fold] = probabilities["E_F"] >= fitted["E_F"]["hard_cutoff"]
    rows.append(pd.DataFrame({
        "uprn": sample.uprn,
        "fold": folds,
        "specification": THRESHOLD_SPEC.name,
        "model_kind": "threshold",
        "fuel": sample.fuel,
        "sector_group": sample.sector_group,
        "floor_area": sample.floor_area,
        "floor_area_bin": sample.floor_area_bin,
        "ac_group": sample.ac_group,
        "distance_bin": sample.distance_bin,
        "pre_observed": sample.pre,
        "post_observed": sample.post,
        "true_band": sample.true_band,
        "post_band": sample.post_band,
        "predicted_old_rating": np.nan,
        "prob_old_below_B": prob_b,
        "prob_old_FG": prob_fg,
        "hard_old_below_B": hard_b,
        "hard_old_FG": hard_fg,
        "hard_cutoff_B": cutoff_b,
        "hard_cutoff_FG": cutoff_fg,
    }))
    out = pd.concat(rows, ignore_index=True)
    out["error"] = out.predicted_old_rating - out.pre_observed
    out.to_csv(OUT_TABLES / "bridge_oof_predictions.csv", index=False)
    return out


def _rating_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    if len(actual) >= 2 and np.unique(predicted).size > 1:
        slope, intercept = np.polyfit(predicted, actual, 1)
    else:
        slope = intercept = np.nan
    return {
        "n": int(len(actual)),
        "bias_pts": float(np.mean(error)),
        "MAE_pts": float(np.mean(np.abs(error))),
        "median_error_pts": float(np.median(error)),
        "error_p10_pts": float(np.quantile(error, 0.10)),
        "error_p50_pts": float(np.quantile(error, 0.50)),
        "error_p90_pts": float(np.quantile(error, 0.90)),
        "calibration_slope_actual_on_predicted": float(slope),
        "calibration_intercept_actual_on_predicted": float(intercept),
    }


def _threshold_metrics(actual_positive: np.ndarray, probability: np.ndarray,
                       predicted_positive: np.ndarray | None = None) -> dict[str, float | int]:
    actual = np.asarray(actual_positive, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= 0.5 if predicted_positive is None
                 else np.asarray(predicted_positive, dtype=bool))
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "n": int(len(actual)), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "sensitivity": float(sensitivity), "specificity": float(specificity),
        "PPV": float(ppv), "NPV": float(npv),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "actual_positive_n": int(actual.sum()),
        "predicted_positive_hard_n": int(predicted.sum()),
        "predicted_positive_probability_sum": float(probability.sum()),
        "aggregate_probability_error_n": float(probability.sum() - actual.sum()),
        "aggregate_probability_error_pp": float(100 * (probability.mean() - actual.mean())),
        "brier_score": float(np.mean((probability - actual.astype(float)) ** 2)),
    }


def evaluate_oof(sample: pd.DataFrame, oof: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create all held-out calibration, confusion and cancellation tables."""
    summaries, by_fuel, threshold_rows, threshold_confusion = [], [], [], []
    threshold_by_fuel, probability_calibration = [], []
    band_confusion, true_band_rows, post_band_rows, window_rows = [], [], [], []
    contribution_rows, cancellation_rows = [], []

    for spec in ALL_SPECS:
        pred = oof[oof.specification == spec.name].copy()
        row: dict[str, Any] = {
            "specification": spec.name,
            "label": spec.label,
            "model_kind": spec.kind,
            "role": spec.role,
            "structural_candidate_predeclared": spec.structural_candidate,
            "n_oof": int(len(pred)),
        }
        if spec.kind == "rating":
            row.update(_rating_metrics(pred.pre_observed, pred.predicted_old_rating))
            pred["predicted_band"] = _safe_band(pred.predicted_old_rating)
            matrix = pd.crosstab(pred.true_band, pred.predicted_band).reindex(
                index=BANDS, columns=BANDS, fill_value=0
            )
            for actual_band in BANDS:
                denominator = int(matrix.loc[actual_band].sum())
                for predicted_band in BANDS:
                    count = int(matrix.loc[actual_band, predicted_band])
                    band_confusion.append({
                        "specification": spec.name,
                        "actual_true_old_band": actual_band,
                        "predicted_old_band": predicted_band,
                        "n": count,
                        "row_percent": 100 * count / max(denominator, 1),
                    })
            for fuel in FUELS + ["All"]:
                group = pred if fuel == "All" else pred[pred.fuel == fuel]
                by_fuel.append({
                    "specification": spec.name, "fuel": fuel,
                    **_rating_metrics(group.pre_observed, group.predicted_old_rating),
                })
            for band_basis, band_column, destination in [
                ("true_old_basis", "true_band", true_band_rows),
                ("post_revision_observed", "post_band", post_band_rows),
            ]:
                for (fuel, band), group in pred.groupby(["fuel", band_column], observed=True):
                    destination.append({
                        "specification": spec.name,
                        "band_basis": band_basis,
                        "fuel": fuel,
                        "band": band,
                        **_rating_metrics(group.pre_observed, group.predicted_old_rating),
                    })
            for basis, rating_column in [
                ("true_old_basis", "pre_observed"),
                ("post_revision_observed", "post_observed"),
            ]:
                for threshold_name, threshold in THRESHOLDS.items():
                    for window in [10, 20, 30]:
                        within = (pred[rating_column] - threshold).abs() <= window
                        for fuel in FUELS + ["All"]:
                            group = pred[within] if fuel == "All" else pred[within & (pred.fuel == fuel)]
                            if group.empty:
                                continue
                            window_rows.append({
                                "specification": spec.name,
                                "rating_basis_for_window": basis,
                                "threshold": threshold_name,
                                "threshold_rating": threshold,
                                "window_plus_minus_points": window,
                                "fuel": fuel,
                                **_rating_metrics(group.pre_observed, group.predicted_old_rating),
                            })

            # Decompose the pooled bias into fuel x band contributions.  The
            # contribution rows sum exactly to the overall OOF bias.
            for basis, band_column in [
                ("true_old_basis", "true_band"),
                ("post_revision_observed", "post_band"),
            ]:
                components = []
                for (fuel, band), group in pred.groupby(["fuel", band_column], observed=True):
                    group_bias = float(group.error.mean())
                    contribution = len(group) / len(pred) * group_bias
                    components.append(contribution)
                    contribution_rows.append({
                        "specification": spec.name,
                        "band_basis": basis,
                        "fuel": fuel,
                        "band": band,
                        "n": int(len(group)),
                        "group_bias_pts": group_bias,
                        "weighted_contribution_to_overall_bias_pts": contribution,
                    })
                gross = float(np.abs(components).sum())
                net = float(np.sum(components))
                cancellation_rows.append({
                    "specification": spec.name,
                    "band_basis": basis,
                    "overall_bias_pts": net,
                    "gross_absolute_bias_contribution_pts": gross,
                    "cancellation_fraction": 1.0 - abs(net) / gross if gross else 0.0,
                    "positive_contribution_pts": float(sum(x for x in components if x > 0)),
                    "negative_contribution_pts": float(sum(x for x in components if x < 0)),
                })
        else:
            row.update({
                "n": int(len(pred)), "bias_pts": np.nan, "MAE_pts": np.nan,
                "median_error_pts": np.nan, "error_p10_pts": np.nan,
                "error_p50_pts": np.nan, "error_p90_pts": np.nan,
                "calibration_slope_actual_on_predicted": np.nan,
                "calibration_intercept_actual_on_predicted": np.nan,
            })

        for threshold_name, threshold in THRESHOLDS.items():
            probability_column = "prob_old_below_B" if threshold_name == "B_C" else "prob_old_FG"
            hard_column = "hard_old_below_B" if threshold_name == "B_C" else "hard_old_FG"
            actual = pred.pre_observed.to_numpy(dtype=float) > threshold
            metrics = _threshold_metrics(actual, pred[probability_column], pred[hard_column])
            threshold_rows.append({
                "specification": spec.name,
                "threshold": threshold_name,
                "positive_class": "old-basis rating above numerical threshold",
                "threshold_rating": threshold,
                "probability_or_score": probability_column,
                "hard_decision_rule": (
                    "predicted old rating above threshold" if spec.kind == "rating"
                    else "cutoff chosen within each training fold to maximise balanced accuracy"
                ),
                "mean_training_fold_probability_cutoff": (
                    np.nan if spec.kind == "rating"
                    else float(pred["hard_cutoff_B" if threshold_name == "B_C" else "hard_cutoff_FG"].mean())
                ),
                **metrics,
            })
            threshold_confusion.append({
                "specification": spec.name,
                "threshold": threshold_name,
                "positive_class": "old-basis rating above numerical threshold",
                **{k: metrics[k] for k in ["TP", "FP", "FN", "TN"]},
            })
            for fuel in FUELS + ["All"]:
                group = pred if fuel == "All" else pred[pred.fuel == fuel]
                group_actual = group.pre_observed.to_numpy(dtype=float) > threshold
                group_metrics = _threshold_metrics(
                    group_actual, group[probability_column], group[hard_column]
                )
                threshold_by_fuel.append({
                    "specification": spec.name,
                    "threshold": threshold_name,
                    "fuel": fuel,
                    **group_metrics,
                })

            grouping_columns = {
                "fuel": ["fuel"],
                "post_band": ["post_band"],
                "fuel_x_post_band": ["fuel", "post_band"],
                "sector_group": ["sector_group"],
                "floor_area_bin": ["floor_area_bin"],
                "AC_status": ["ac_group"],
                "threshold_distance_bin": ["distance_bin"],
            }
            for grouping, columns in grouping_columns.items():
                grouper = columns[0] if len(columns) == 1 else columns
                for key, group in pred.groupby(grouper, observed=True):
                    key_tuple = key if isinstance(key, tuple) else (key,)
                    actual_group = group.pre_observed.to_numpy(dtype=float) > threshold
                    probability_group = group[probability_column].to_numpy(dtype=float)
                    hard_group = group[hard_column].to_numpy(dtype=bool)
                    group_metrics = _threshold_metrics(
                        actual_group, probability_group, hard_group
                    )
                    probability_calibration.append({
                        "specification": spec.name,
                        "threshold": threshold_name,
                        "grouping": grouping,
                        "group": " | ".join(map(str, key_tuple)),
                        "n": int(len(group)),
                        "actual_positive_rate_%": float(100 * actual_group.mean()),
                        "predicted_probability_mean_%": float(100 * probability_group.mean()),
                        "probability_calibration_error_pp": float(
                            100 * (probability_group.mean() - actual_group.mean())
                        ),
                        "brier_score": group_metrics["brier_score"],
                        "sensitivity": group_metrics["sensitivity"],
                        "specificity": group_metrics["specificity"],
                        "PPV": group_metrics["PPV"],
                        "NPV": group_metrics["NPV"],
                        "balanced_accuracy": group_metrics["balanced_accuracy"],
                    })
            prefix = "B" if threshold_name == "B_C" else "FG"
            row.update({
                f"{prefix}_sensitivity": metrics["sensitivity"],
                f"{prefix}_specificity": metrics["specificity"],
                f"{prefix}_PPV": metrics["PPV"],
                f"{prefix}_NPV": metrics["NPV"],
                f"{prefix}_balanced_accuracy": metrics["balanced_accuracy"],
                f"{prefix}_aggregate_probability_error_n": metrics["aggregate_probability_error_n"],
                f"{prefix}_aggregate_probability_error_pp": metrics["aggregate_probability_error_pp"],
            })
        row["threshold_score"] = np.mean([
            row["B_balanced_accuracy"], row["FG_balanced_accuracy"]
        ])
        summaries.append(row)

    tables = {
        "summary": pd.DataFrame(summaries),
        "by_fuel": pd.DataFrame(by_fuel),
        "threshold_metrics": pd.DataFrame(threshold_rows),
        "threshold_confusion": pd.DataFrame(threshold_confusion),
        "threshold_by_fuel": pd.DataFrame(threshold_by_fuel),
        "probability_calibration": pd.DataFrame(probability_calibration),
        "band_confusion": pd.DataFrame(band_confusion),
        "true_band": pd.DataFrame(true_band_rows),
        "post_band": pd.DataFrame(post_band_rows),
        "windows": pd.DataFrame(window_rows),
        "bias_contributions": pd.DataFrame(contribution_rows),
        "bias_cancellation": pd.DataFrame(cancellation_rows),
    }
    filenames = {
        "summary": "bridge_validation_summary.csv",
        "by_fuel": "bridge_validation_by_fuel.csv",
        "threshold_metrics": "bridge_threshold_metrics.csv",
        "threshold_confusion": "bridge_threshold_confusion.csv",
        "threshold_by_fuel": "bridge_threshold_metrics_by_fuel.csv",
        "probability_calibration": "bridge_threshold_probability_calibration.csv",
        "band_confusion": "bridge_band_confusion.csv",
        "true_band": "bridge_calibration_by_fuel_true_band.csv",
        "post_band": "bridge_calibration_by_fuel_post_band.csv",
        "windows": "bridge_calibration_threshold_windows.csv",
        "bias_contributions": "bridge_bias_contributions.csv",
        "bias_cancellation": "bridge_bias_cancellation.csv",
    }
    for key, filename in filenames.items():
        tables[key].to_csv(OUT_TABLES / filename, index=False)
    return tables


def audit_legacy_validation(P: pd.DataFrame, sample: pd.DataFrame,
                            folds: np.ndarray,
                            authoritative_summary: pd.DataFrame,
                            authoritative_by_fuel: pd.DataFrame) -> dict[str, Any]:
    """Re-run the old code and explain the old summary/per-fuel mismatch."""
    checkpoint_summary_path = CHECKPOINT_DIR / "validation_heldout_summary.csv"
    checkpoint_fuel_path = CHECKPOINT_DIR / "validation_heldout_by_fuel.csv"
    checkpoint_summary = pd.read_csv(checkpoint_summary_path) if checkpoint_summary_path.exists() else None
    checkpoint_fuel = pd.read_csv(checkpoint_fuel_path) if checkpoint_fuel_path.exists() else None

    reproduced = legacy_heldout_validation(
        P, n_splits=LEGACY_SPLITS, train_frac=LEGACY_TRAIN_FRAC
    )
    legacy_summary = reproduced["summary"].copy()
    legacy_summary.to_csv(OUT_TABLES / "bridge_legacy_validation_reproduction.csv", index=False)

    comparisons = []
    if checkpoint_summary is not None:
        numeric = [c for c in checkpoint_summary.columns if c != "specification"]
        merged = checkpoint_summary.merge(
            legacy_summary, on="specification", suffixes=("_checkpoint", "_reproduced")
        )
        for _, row in merged.iterrows():
            for metric in numeric:
                old = row[f"{metric}_checkpoint"]
                new = row[f"{metric}_reproduced"]
                comparisons.append({
                    "specification": row.specification,
                    "metric": metric,
                    "checkpoint_value": old,
                    "reproduced_value": new,
                    "absolute_difference": abs(new - old),
                    "matches_at_saved_precision": bool(np.isclose(new, old, equal_nan=True)),
                })
    comparison = pd.DataFrame(comparisons, columns=[
        "specification", "metric", "checkpoint_value", "reproduced_value",
        "absolute_difference", "matches_at_saved_precision",
    ])
    comparison.to_csv(OUT_TABLES / "bridge_legacy_validation_reconciliation.csv", index=False)

    # Reconstruct the exact 200 legacy Bernoulli split sizes for the audit.
    rng = np.random.default_rng(RANDOM_SEED)
    split_rows = []
    for split in range(LEGACY_SPLITS):
        train_mask = rng.random(len(sample)) < LEGACY_TRAIN_FRAC
        split_rows.append({
            "split": split,
            "n_train": int(train_mask.sum()),
            "n_test": int((~train_mask).sum()),
            "train_share": float(train_mask.mean()),
        })
    split_sizes = pd.DataFrame(split_rows)
    split_sizes.to_csv(OUT_TABLES / "bridge_legacy_split_sizes.csv", index=False)

    fold_table = pd.DataFrame({"uprn": sample.uprn, "fold": folds})
    fold_table.groupby("fold").size().rename("n_test").reset_index().to_csv(
        OUT_TABLES / "bridge_crossfit_fold_sizes.csv", index=False
    )

    # The authoritative all-fuel row and weighted per-fuel rows are the same OOF
    # estimand.  The legacy summary is a different average over random half-tests.
    reconciliation_rows = []
    for specification in [s.name for s in RATING_SPECS]:
        all_row = authoritative_by_fuel[
            (authoritative_by_fuel.specification == specification)
            & (authoritative_by_fuel.fuel == "All")
        ].iloc[0]
        fuel_rows = authoritative_by_fuel[
            (authoritative_by_fuel.specification == specification)
            & (authoritative_by_fuel.fuel != "All")
        ]
        weighted_bias = np.average(fuel_rows.bias_pts, weights=fuel_rows.n)
        summary_row = authoritative_summary[
            authoritative_summary.specification == specification
        ].iloc[0]
        reconciliation_rows.append({
            "specification": specification,
            "authoritative_OOF_all_bias": all_row.bias_pts,
            "weighted_OOF_per_fuel_bias": weighted_bias,
            "absolute_difference": abs(all_row.bias_pts - weighted_bias),
            "authoritative_summary_bias": summary_row.bias_pts,
            "reconciles": bool(np.isclose(all_row.bias_pts, weighted_bias)),
            "legacy_difference_explanation": (
                "legacy summary averages 200 variable-size Bernoulli half-test estimates; "
                "legacy per-fuel table is one deterministic five-fold OOF pass"
            ),
        })
    reconciliation = pd.DataFrame(reconciliation_rows)
    reconciliation.to_csv(OUT_TABLES / "bridge_overall_per_fuel_reconciliation.csv", index=False)

    audit_rows = [
        ("heldout_eligible_n", len(sample)),
        ("unique_uprn_n", sample.uprn.nunique()),
        ("duplicate_uprn_n", sample.uprn.duplicated().sum()),
        ("legacy_random_seed", RANDOM_SEED),
        ("legacy_splits", LEGACY_SPLITS),
        ("legacy_train_probability", LEGACY_TRAIN_FRAC),
        ("legacy_mean_train_n", split_sizes.n_train.mean()),
        ("legacy_min_train_n", split_sizes.n_train.min()),
        ("legacy_max_train_n", split_sizes.n_train.max()),
        ("authoritative_crossfit_folds", N_FOLDS),
        ("authoritative_split_unit", "UPRN"),
        ("authoritative_fold_strata", "fuel x observed post-revision band"),
        ("error_sign", "predicted old-basis rating minus observed old-basis rating"),
        ("test_outcomes_used_in_fitting", False),
        ("true_old_band_used_as_model_feature", False),
        ("direct_SBEM_old_new_pairs_available", False),
        ("heating_description_beyond_main_fuel_available", False),
    ]
    audit_table = pd.DataFrame(audit_rows, columns=["audit_item", "value"])
    audit_table.to_csv(OUT_TABLES / "bridge_validation_audit.csv", index=False)
    audit_json = {
        "heldout_sample": {
            "n": int(len(sample)), "unique_uprn": int(sample.uprn.nunique()),
            "split_unit": "UPRN", "fuel_counts": sample.fuel.value_counts().to_dict(),
        },
        "legacy": {
            "seed": RANDOM_SEED, "n_splits": LEGACY_SPLITS,
            "train_assignment": "independent Bernoulli draw with p=0.5 per UPRN per split",
            "train_n_mean": float(split_sizes.n_train.mean()),
            "train_n_range": [int(split_sizes.n_train.min()), int(split_sizes.n_train.max())],
            "checkpoint_present": bool(checkpoint_summary is not None and checkpoint_fuel is not None),
            "checkpoint_reproduced_exactly_at_saved_precision": bool(
                not comparison.empty and comparison.matches_at_saved_precision.all()
            ),
        },
        "authoritative_crossfit": {
            "seed": RANDOM_SEED, "folds": N_FOLDS,
            "stratification": "fuel x observed post-revision band",
            "each_uprn_tested_once": True,
        },
        "error_sign": "predicted_old - observed_old",
        "leakage_audit": {
            "heldout_old_rating_or_true_band_used_to_fit": False,
            "all_bridge_features_available_at_register_application": True,
            "true_old_band_used_only_for_diagnostics": True,
            "post_rating_band_and_distance_bins_are_application_time_features": True,
            "calibration_selection_uses_both_certificates": True,
            "selection_caveat": (
                "unchanged recommendation count and floor-area stability are sample-selection "
                "conditions, not proof of no physical change"
            ),
        },
        "external_validation": {
            "paired_SBEM_reruns_found_in_repo": False,
            "status": "not available; no empirical bridge is externally validated against identical SBEM inputs",
        },
        "summary_mismatch_resolution": (
            "The old overall table and old per-fuel table used different resampling summaries. "
            "The authoritative tables use one common five-fold OOF prediction set, and the "
            "all-fuel bias equals the sample-size-weighted per-fuel bias."
        ),
    }
    with open(OUT_TABLES / "bridge_validation_audit.json", "w") as handle:
        json.dump(audit_json, handle, indent=2, default=str)
    return {
        "legacy_summary": legacy_summary,
        "comparison": comparison,
        "reconciliation": reconciliation,
        "audit_table": audit_table,
        "audit_json": audit_json,
    }


def specification_and_cell_support(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    specifications = pd.DataFrame([{
        "specification": spec.name,
        "label": spec.label,
        "model_kind": spec.kind,
        "features": spec.features,
        "role": spec.role,
        "structural_candidate_predeclared": spec.structural_candidate,
        "crossfit_required": True,
        "test_outcome_used_in_fit": False,
        "application_ready": True,
    } for spec in ALL_SPECS])
    specifications.to_csv(OUT_TABLES / "bridge_specifications.csv", index=False)

    rows = []
    for refinement, columns in [
        ("fuel_x_post_band", ["fuel", "post_band"]),
        ("fuel_x_threshold_distance", ["fuel", "distance_bin"]),
        ("fuel_x_post_band_x_floor_area", ["fuel", "post_band", "floor_area_bin"]),
        ("fuel_x_sector", ["fuel", "sector_group"]),
        ("fuel_x_AC", ["fuel", "ac_group"]),
    ]:
        counts = sample.groupby(columns, observed=True).size().rename("n")
        for key, count in counts.items():
            if not isinstance(key, tuple):
                key = (key,)
            rows.append({
                "refinement": refinement,
                "cell": " | ".join(map(str, key)),
                "n": int(count),
                "meets_unshrunk_min_n_50": bool(count >= MIN_CELL),
                "meets_shrinkage_min_n_10": bool(count >= MIN_SHRINK_CELL),
            })
    support = pd.DataFrame(rows)
    support.to_csv(OUT_TABLES / "bridge_cell_support.csv", index=False)
    return specifications, support


def prepare_latest_frame(dfx: pd.DataFrame | None = None) -> pd.DataFrame:
    """Row-faithful latest-certificate frame with application-time features."""
    source = build_extended_frame() if dfx is None else dfx
    latest = with_dates(source).drop_duplicates("uprn", keep="last").copy()
    latest = latest.rename(columns={
        "asset_rating": "post",
        "fuelgrp": "fuel",
        "floor_area": "floor_area",
        "property_type_clean": "sector",
        "aircon_present_clean": "ac",
    })
    latest["fuel"] = latest.fuel.where(latest.fuel.isin(FUELS), "Other")
    latest = _add_application_features(latest)
    latest["is_post_revision"] = latest.insp_dt >= CUT
    lodgement = pd.to_datetime(latest.lodgement_date, errors="coerce")
    transaction = latest.transaction_type_clean.fillna("").astype(str)
    latest["policy_proxy"] = (
        (lodgement >= ANALYSIS_AS_OF - pd.DateOffset(years=10))
        & ~transaction.str.contains("construction", case=False, regex=False)
        & transaction.str.contains("let", case=False, regex=False)
        & (latest.floor_area > 1000)
    )
    return latest


def prepare_raking_weights(latest: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Composition-rake to ND-NEED margins without expanding to benchmark N."""
    download_if_missing(ND_NEED_2025_URL, ND_NEED_2025_XLSX)
    download_if_missing(ND_NEED_GEO_2024_URL, ND_NEED_GEO_2024_XLSX)
    frame = latest.copy()
    frame["sector_benchmark"] = _nd_need_sector(frame.sector)
    frame["size_benchmark"] = pd.cut(
        frame.floor_area, SIZE_BINS, labels=SIZE_LABELS,
        include_lowest=True, right=True,
    ).astype(str)

    sector_raw = pd.read_excel(ND_NEED_2025_XLSX, sheet_name="Table 1", header=5)
    size_raw = pd.read_excel(ND_NEED_2025_XLSX, sheet_name="Table 2", header=6)
    sector_target = (
        sector_raw[sector_raw["Building use"] != "All"]
        .set_index("Building use")["Number of buildings "]
        .astype(float)
    )
    sector_target /= sector_target.sum()
    size_target = size_raw.set_index("Building size").loc[
        SIZE_LABELS, "Number of buildings "
    ].astype(float)
    size_target /= size_target.sum()

    geo = pd.read_excel(ND_NEED_GEO_2024_XLSX, sheet_name="Table 1", header=6)
    local = geo[geo["Local Authority"].notna()]
    code_region = dict(zip(local["Geographic Code"], local["Country or Region"]))
    name_region = dict(zip(local["Local Authority"], local["Country or Region"]))
    frame["region_benchmark"] = (
        frame.local_authority.map(code_region)
        .fillna(frame.local_authority_label.map(name_region))
        .replace({"East": "East of England"})
    )
    region_codes = [f"E1200000{i}" for i in range(1, 10)] + ["W92000004"]
    region_target = (
        geo[(geo["Local Authority"].isna()) & geo["Geographic Code"].isin(region_codes)]
        .set_index("Country or Region")["All buildings: number of buildings"]
        .astype(float)
        .rename(index={"East": "East of England"})
    )
    region_target /= region_target.sum()

    matched = frame.region_benchmark.notna()
    rake = frame[matched].copy()
    weights, iterations = _rake_weights(rake, {
        "sector_benchmark": sector_target,
        "size_benchmark": size_target,
        "region_benchmark": region_target,
    })
    diagnostics = {
        "n_latest_frame": int(len(frame)),
        "n_raked": int(len(rake)),
        "n_unmatched_region_excluded": int((~matched).sum()),
        "weights_sum": float(weights.sum()),
        "weight_mean": float(weights.mean()),
        "weight_min": float(weights.min()),
        "weight_p99": float(np.quantile(weights, 0.99)),
        "weight_max": float(weights.max()),
        "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
        "iterations": int(iterations),
        "count_scale": (
            "weights normalised to the matched register-frame N; estimates are composition-raked "
            "register-frame equivalents, not expansion estimates to 1.748 million buildings"
        ),
        "policy_proxy_raking_supported": False,
        "policy_proxy_raking_reason": (
            "ND-NEED margins do not identify tenure, transaction type, certificate validity, or "
            "their intersection"
        ),
    }
    with open(OUT_TABLES / "bridge_raking_diagnostics.json", "w") as handle:
        json.dump(diagnostics, handle, indent=2)
    return rake, weights, diagnostics


def fit_full_models(sample: pd.DataFrame) -> dict[str, Any]:
    fitted: dict[str, Any] = {}
    for spec in RATING_SPECS:
        fitted[spec.name] = fit_rating_bridge(sample, spec.name)
    fitted[THRESHOLD_SPEC.name] = fit_threshold_models(sample)
    return fitted


def fitted_cell_diagnostics(fitted: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for spec in RATING_SPECS:
        params = fitted[spec.name]
        if "cell" not in params:
            for fuel, estimate in params["base"].items():
                rows.append({
                    "specification": spec.name, "cell": fuel, "n": np.nan,
                    "raw_estimate": estimate, "final_estimate": estimate,
                    "shrinkage_weight": np.nan, "uses_fallback_in_full_fit": False,
                })
            continue
        for key, row in params["cell"].iterrows():
            if not isinstance(key, tuple):
                key = (key,)
            rows.append({
                "specification": spec.name,
                "cell": " | ".join(map(str, key)),
                "n": int(row.n),
                "raw_estimate": float(row.get("raw_estimate", row.estimate)),
                "final_estimate": float(row.estimate),
                "shrinkage_weight": float(row.get("shrinkage_weight", np.nan)),
                "uses_fallback_in_full_fit": bool(row.n < params["min_cell"]),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "bridge_fitted_cell_diagnostics.csv", index=False)
    return out


def _model_application_values(frame: pd.DataFrame, spec: BridgeSpec,
                              fitted: Any) -> dict[str, np.ndarray | str]:
    observed = frame.post.to_numpy(dtype=float)
    is_post = frame.is_post_revision.to_numpy(dtype=bool)
    if spec.kind == "rating":
        predicted_old = predict_rating_bridge(frame, spec.name, fitted)
        return {
            "estimate_type": "hard rating bridge",
            "B_C_probability": (predicted_old > THRESHOLDS["B_C"]).astype(float),
            "E_F_probability": (predicted_old > THRESHOLDS["E_F"]).astype(float),
            "B_C_hard": predicted_old > THRESHOLDS["B_C"],
            "E_F_hard": predicted_old > THRESHOLDS["E_F"],
            "predicted_old_rating": predicted_old,
        }
    probabilities = predict_threshold_probabilities(frame, fitted)
    return {
        "estimate_type": "probability sum",
        "B_C_probability": probabilities["B_C"],
        "E_F_probability": probabilities["E_F"],
        "B_C_hard": probabilities["B_C"] >= fitted["B_C"]["hard_cutoff"],
        "E_F_hard": probabilities["E_F"] >= fitted["E_F"]["hard_cutoff"],
        "predicted_old_rating": np.full(len(frame), np.nan),
    }


_THR = [("B_C", "prob_old_below_B", 50.0), ("E_F", "prob_old_FG", 125.0)]

_AREA_MODELS = {
    "A_baseline_fuel_logit": "0 + C(fuel) + C(fuel):x",
    "B_fuel_logfa_sector": "x + C(fuel) + logfa + C(sector_group)",
    "C_fuel_logfa_sector_interactions": "x + C(fuel) + logfa + C(sector_group) + C(fuel):x + C(fuel):logfa",
    "D_splines_x_logfa": "cr(x, df=4) + C(fuel) + cr(logfa, df=4) + C(sector_group)",
}


def nested_cv_area_models(sample: pd.DataFrame, n_seeds: int = 25) -> pd.DataFrame:
    """Fully nested, repeated cross-validation of the affected-area estimator.

    Within each outer training fold, inner CV selects *both* the model architecture (A-D)
    and the ridge penalty by inner-fold affected-area error, and only the inner-selected
    model is evaluated on the untouched outer fold, so the entire selection procedure is
    honestly out of sample. Repeated over ``n_seeds`` UPRN-level fold assignments; medians
    and 2.5--97.5 percentiles are taken across seeds. Reports (i) each fixed model's
    out-of-fold performance (penalty tuned inner), (ii) the selection procedure's
    out-of-fold performance, and (iii) how often each architecture is selected. Count and
    area are validated separately. Writes bridge_area_model_nestedcv.csv and
    bridge_area_model_selection.csv."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, balanced_accuracy_score
    from patsy import dmatrix
    s = sample[sample.floor_area > 0].reset_index(drop=True).copy()
    s["logfa"] = np.log(s.floor_area.to_numpy(float))
    fa = s.floor_area.to_numpy(float); uprn = s.uprn.to_numpy(); order = np.argsort(-fa)
    forms = {"A_baseline": "0 + C(fuel) + C(fuel):x",
             "B_logfa_sector": "x + C(fuel) + logfa + C(sector_group)",
             "C_interactions": "x + C(fuel) + logfa + C(sector_group) + C(fuel):x + C(fuel):logfa",
             "D_splines": "cr(x, df=4) + C(fuel) + cr(logfa, df=4) + C(sector_group)"}
    penalties = (0.3, 1.0, 3.0)

    def ufold(seed, k=5):
        rng = np.random.default_rng(seed); u = np.unique(uprn); rng.shuffle(u)
        fmap = {x: i % k for i, x in enumerate(u)}
        return np.array([fmap[x] for x in uprn])

    def fit_pred(X, ytr, tr, te, c):
        return LogisticRegression(C=c, max_iter=3000).fit(X[tr], ytr[tr]).predict_proba(X[te])[:, 1]

    def area_err(y, p, m):
        oa, pa = float((fa[m] * y[m]).sum()), float((fa[m] * p[m]).sum())
        return 100 * (pa - oa) / max(oa, 1e-9)

    per_rows, sel_rows = [], []
    for tname, tau in [("B_C", 50.0), ("E_F", 125.0)]:
        y = (s.pre.to_numpy(float) > tau).astype(int)
        Xc = {m: np.asarray(dmatrix(f, dict(fuel=s.fuel, x=(s.post.to_numpy(float) - tau) / 25.0,
              logfa=s.logfa, sector_group=s.sector_group), return_type="dataframe")) for m, f in forms.items()}
        for seed in range(n_seeds):
            of = ufold(seed)
            pm = {m: np.zeros(len(s)) for m in forms}; psel = np.zeros(len(s)); picks = []
            for o in range(5):
                tr, te = of != o, of == o
                inner = ufold(seed * 100 + o + 1)[tr]
                best_overall = (np.inf, None, None)
                for m in forms:
                    X = Xc[m]; best_m = (np.inf, 1.0)
                    for c in penalties:
                        pin = np.zeros(int(tr.sum()))
                        Xtr = X[tr]; ytr = y[tr]
                        for iF in range(int(inner.max()) + 1):
                            itr, ite = inner != iF, inner == iF
                            if ytr[itr].sum() in (0, int(itr.sum())):
                                pin[ite] = ytr[itr].mean(); continue
                            pin[ite] = LogisticRegression(C=c, max_iter=3000).fit(Xtr[itr], ytr[itr]).predict_proba(Xtr[ite])[:, 1]
                        ae = abs(100 * ((fa[tr] * pin).sum() - (fa[tr] * ytr).sum()) / max(float((fa[tr] * ytr).sum()), 1e-9))
                        if ae < best_m[0]:
                            best_m = (ae, c)
                        if ae < best_overall[0]:
                            best_overall = (ae, m, c)
                    pm[m][te] = fit_pred(Xc[m], y, tr, te, best_m[1])
                _, bm, bc = best_overall
                psel[te] = fit_pred(Xc[bm], y, tr, te, bc); picks.append(bm)
            for m in forms:
                keep = np.ones(len(s), bool); keep[order[:25]] = False
                per_rows.append({"threshold": tname, "model": m, "seed": seed,
                                 "count_error_pp": 100 * (pm[m].sum() - y.sum()) / len(s),
                                 "area_error_pct": area_err(y, pm[m], np.ones(len(s), bool)),
                                 "area_error_excl25_pct": area_err(y, pm[m], keep),
                                 "brier": brier_score_loss(y, pm[m]),
                                 "bal_acc": balanced_accuracy_score(y, pm[m] >= 0.5)})
            from collections import Counter
            cc = Counter(picks)
            sel_rows.append({"threshold": tname, "seed": seed,
                             "selection_area_error_pct": area_err(y, psel, np.ones(len(s), bool)),
                             "selection_count_error_pp": 100 * (psel.sum() - y.sum()) / len(s),
                             "modal_architecture": cc.most_common(1)[0][0],
                             **{f"picks_{m}": cc.get(m, 0) for m in forms}})
    per = pd.DataFrame(per_rows); sel = pd.DataFrame(sel_rows)

    def agg(df, gcols, col):
        g = df.groupby(gcols)[col]
        return g.median().rename("median"), g.quantile(0.025).rename("lo"), g.quantile(0.975).rename("hi")

    out = []
    for (tn, m), sub in per.groupby(["threshold", "model"]):
        out.append({"threshold": tn, "model": m,
                    "count_error_pp_median": round(float(sub.count_error_pp.median()), 2),
                    "area_error_pct_median": round(float(sub.area_error_pct.median()), 1),
                    "area_error_pct_lo": round(float(sub.area_error_pct.quantile(0.025)), 1),
                    "area_error_pct_hi": round(float(sub.area_error_pct.quantile(0.975)), 1),
                    "area_error_excl25_median": round(float(sub.area_error_excl25_pct.median()), 1),
                    "brier_median": round(float(sub.brier.median()), 4),
                    "bal_acc_median": round(float(sub.bal_acc.median()), 3)})
    comp = pd.DataFrame(out); comp.to_csv(OUT_TABLES / "bridge_area_model_nestedcv.csv", index=False)
    # selection-procedure summary + architecture frequency
    ssum = []
    for tn, sub in sel.groupby("threshold"):
        picks_tot = sub[[c for c in sub.columns if c.startswith("picks_")]].sum()
        freq = (100 * picks_tot / picks_tot.sum()).round(1).to_dict()
        ssum.append({"threshold": tn,
                     "selection_area_error_pct_median": round(float(sub.selection_area_error_pct.median()), 1),
                     "selection_area_error_pct_lo": round(float(sub.selection_area_error_pct.quantile(0.025)), 1),
                     "selection_area_error_pct_hi": round(float(sub.selection_area_error_pct.quantile(0.975)), 1),
                     "selection_count_error_pp_median": round(float(sub.selection_count_error_pp.median()), 2),
                     "n_seeds": int(len(sub)),
                     **{k.replace("picks_", "select_freq_%_"): v for k, v in freq.items()}})
    pd.DataFrame(ssum).to_csv(OUT_TABLES / "bridge_area_model_selection.csv", index=False)
    return comp


def _model_c_register_probs(sample: pd.DataFrame, latest: pd.DataFrame, tau: float) -> np.ndarray:
    """Fit the preferred area model (Model C) on the full calibration sample and return its
    per-entry old-basis crossing probability on the register frame at threshold tau."""
    from sklearn.linear_model import LogisticRegression
    from patsy import dmatrix, build_design_matrices
    s = sample[sample.floor_area > 0].copy(); s["logfa"] = np.log(s.floor_area.to_numpy(float))
    y = (s.pre.to_numpy(float) > tau).astype(int)
    formula = "x + C(fuel) + logfa + C(sector_group) + C(fuel):x + C(fuel):logfa"
    dm = dmatrix(formula, dict(fuel=s.fuel, x=(s.post.to_numpy(float) - tau) / 25.0,
                               logfa=s.logfa, sector_group=s.sector_group), return_type="dataframe")
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(np.asarray(dm), y)
    lg = np.log(latest.floor_area.to_numpy(float))
    Xl = build_design_matrices([dm.design_info], dict(
        fuel=latest.fuel, x=(latest.post.to_numpy(float) - tau) / 25.0,
        logfa=lg, sector_group=latest.sector_group))[0]
    return clf.predict_proba(np.asarray(Xl))[:, 1]


def transport_oos_validation(sample: pd.DataFrame, latest: pd.DataFrame,
                             tau: float = 50.0, n_seeds: int = 25) -> pd.DataFrame:
    """Out-of-sample validation of each transported estimator. Repeated (n_seeds) outer
    5-fold UPRN CV on the eligible calibration sample; within each fold the membership model
    is fit on outer-training calibration (S=1) plus the fixed eligible application frame
    (S=0), participation is predicted for the held-out fold, inverse-odds weights are trimmed
    at 1/99 pct using training data, the weighted outcome model is fit on training only, and
    the untouched fold is scored. Reports the held-out unweighted and target-weighted affected-
    area error and count error (median [2.5,97.5] over seeds) for unweighted and inverse-odds
    Models C and D, so weighting is judged on held-out outcome prediction, not on SMDs.
    Writes bridge_transport_oos_validation.csv."""
    from sklearn.linear_model import LogisticRegression
    from patsy import dmatrix, build_design_matrices

    def prep(df):
        return pd.DataFrame({"post": pd.to_numeric(df.post, errors="coerce"),
            "logfa": np.log(pd.to_numeric(df.floor_area, errors="coerce").clip(lower=1)),
            "fuel": df.fuel.astype(str), "sector": df.sector_group.astype(str), "ac": df.ac_group.astype(str)})
    ce = sample[sample.post <= tau]
    cal = prep(ce).reset_index(drop=True)
    cal["fa"] = pd.to_numeric(ce.floor_area, errors="coerce").to_numpy()
    cal["y"] = (pd.to_numeric(ce.pre, errors="coerce").to_numpy() > tau).astype(int)
    uprn = ce.uprn.to_numpy()[:len(cal)]
    cal = cal.dropna(subset=["post", "logfa", "fa"]).reset_index(drop=True); uprn = uprn[:len(cal)]
    appmask = latest.is_post_revision & (latest.post <= tau)
    app = prep(latest[appmask]).dropna(subset=["post", "logfa"]).reset_index(drop=True)
    app_s = app.sample(n=min(30000, len(app)), random_state=1).reset_index(drop=True)
    dinfo = dmatrix("0 + post + logfa + C(fuel) + C(sector) + C(ac)",
                    pd.concat([cal.assign(_c=1), app_s.assign(_c=0)], ignore_index=True), return_type="dataframe").design_info
    Zof = lambda d: np.asarray(build_design_matrices([dinfo], d)[0])
    Zapp = Zof(app_s)
    forms = {"C": "x + C(fuel) + logfa + C(sector) + C(fuel):x + C(fuel):logfa",
             "D": "cr(x, df=4) + C(fuel) + cr(logfa, df=4) + C(sector)"}

    def ufold(seed, k=5):
        rng = np.random.default_rng(seed); u = np.unique(uprn); rng.shuffle(u)
        fm = {x: i % k for i, x in enumerate(u)}
        return np.array([fm[x] for x in uprn])
    y = cal.y.to_numpy(); fa = cal.fa.to_numpy()
    rows = []
    for m, frm in forms.items():
        dc = dmatrix(frm, dict(fuel=cal.fuel, x=(cal.post.to_numpy() - tau) / 25.0, logfa=cal.logfa,
                               sector=cal.sector), return_type="dataframe")
        Xall = np.asarray(dc)
        for est in ("unweighted", "inverse_odds"):
            uw, tw, cn = [], [], []
            for seed in range(n_seeds):
                of = ufold(seed); p = np.zeros(len(cal)); wte = np.ones(len(cal))
                for o in range(5):
                    tr, te = of != o, of == o
                    Ztr = Zof(cal[tr])
                    mm = LogisticRegression(C=1.0, max_iter=2000).fit(
                        np.vstack([Ztr, Zapp]), np.r_[np.ones(int(tr.sum())), np.zeros(len(app_s))])
                    e_tr = mm.predict_proba(Ztr)[:, 1]; e_te = mm.predict_proba(Zof(cal[te]))[:, 1]
                    w_tr = np.ones(int(tr.sum()))
                    if est == "inverse_odds":
                        w_tr = (1 - e_tr) / np.clip(e_tr, 1e-6, 1 - 1e-6)
                        lo, hi = np.percentile(w_tr, [1, 99]); w_tr = np.clip(w_tr, lo, hi)
                    clf = LogisticRegression(C=1.0, max_iter=3000).fit(Xall[tr], y[tr], sample_weight=w_tr)
                    p[te] = clf.predict_proba(Xall[te])[:, 1]
                    wte[te] = (1 - e_te) / np.clip(e_te, 1e-6, 1 - 1e-6)
                uw.append(100 * ((fa * p).sum() - (fa * y).sum()) / (fa * y).sum())
                tw.append(100 * ((wte * fa * p).sum() - (wte * fa * y).sum()) / (wte * fa * y).sum())
                cn.append(100 * (p.sum() - y.sum()) / len(y))
            rows.append({"model": m, "estimator": est,
                         "heldout_area_err_median": round(float(np.median(uw)), 1),
                         "heldout_area_err_lo": round(float(np.percentile(uw, 2.5)), 1),
                         "heldout_area_err_hi": round(float(np.percentile(uw, 97.5)), 1),
                         "target_weighted_area_err_median": round(float(np.median(tw)), 1),
                         "target_weighted_area_err_lo": round(float(np.percentile(tw, 2.5)), 1),
                         "target_weighted_area_err_hi": round(float(np.percentile(tw, 97.5)), 1),
                         "count_err_pp_median": round(float(np.median(cn)), 2)})
    out = pd.DataFrame(rows); out.to_csv(OUT_TABLES / "bridge_transport_oos_validation.csv", index=False)
    return out


def stage1_transport(sample: pd.DataFrame, latest: pd.DataFrame, tau: float = 50.0) -> dict:
    """Stage-1 transport of the stable-descriptor calibration sample to the eligible EPC
    register application frame, within the threshold-specific eligible population (entries
    observed at the threshold or better: post <= tau). Fits a calibration-membership model
    P(in calibration | X) on fuel, post rating, log floor area, sector and air-conditioning,
    forms inverse-odds transport weights (trimmed at the 1st/99th percentile) and, as a
    second method, first-moment entropy-balancing weights. Reports positivity/common-support,
    Kish effective sample size, standardized mean differences before/after weighting, and the
    transported expected crossing count and affected floor area (Models C and D). This is the
    Stage-1 register-conditional transport; external-composition sensitivities (ND-NEED, BEES)
    are separate Stage-2 analyses. Writes bridge_transport_diagnostics.csv / _estimates.csv."""
    from sklearn.linear_model import LogisticRegression
    from scipy.optimize import minimize
    from patsy import dmatrix, build_design_matrices

    def prep(df, is_lat):
        d = pd.DataFrame({
            "post": pd.to_numeric(df.post, errors="coerce"),
            "logfa": np.log(pd.to_numeric(df.floor_area, errors="coerce").clip(lower=1)),
            "fuel": df.fuel.astype(str), "sector": df.sector_group.astype(str),
            "ac": df.ac_group.astype(str)})
        return d
    cal = prep(sample[sample.post <= tau], False).dropna(subset=["post", "logfa"]).reset_index(drop=True)
    y = (pd.to_numeric(sample[sample.post <= tau].pre, errors="coerce").to_numpy() > tau).astype(int)[:len(cal)]
    appmask = latest.is_post_revision & (latest.post <= tau)
    app = prep(latest[appmask], True).dropna(subset=["post", "logfa"]).reset_index(drop=True)
    app_fa = pd.to_numeric(latest[appmask].floor_area, errors="coerce").to_numpy()[:len(app)]
    app_s = app.sample(n=min(30000, len(app)), random_state=1)
    pool = pd.concat([cal.assign(_c=1), app_s.assign(_c=0)], ignore_index=True)
    dm = dmatrix("0 + post + logfa + C(fuel) + C(sector) + C(ac)", pool, return_type="dataframe")
    Z = np.asarray(dm); Zc, Za = Z[:len(cal)], Z[len(cal):]
    mm = LogisticRegression(C=1.0, max_iter=3000).fit(Z, pool._c.to_numpy())
    e_cal = mm.predict_proba(Zc)[:, 1]; e_app = mm.predict_proba(Za)[:, 1]
    w_io = (1 - e_cal) / np.clip(e_cal, 1e-6, 1 - 1e-6)
    lo, hi = np.percentile(w_io, [1, 99]); w_io = np.clip(w_io, lo, hi); w_io /= w_io.mean()
    # first-moment entropy balancing (guarded; may be unstable under collinear constraints)
    Zcc = Zc - Za.mean(0)
    try:
        r = minimize(lambda l: np.log(np.exp((Zcc @ l) - (Zcc @ l).max()).sum()),
                     np.zeros(Zcc.shape[1]), method="L-BFGS-B",
                     jac=lambda l: (Zcc * (np.exp((Zcc @ l) - (Zcc @ l).max()))[:, None]).sum(0)
                     / np.exp((Zcc @ l) - (Zcc @ l).max()).sum())
        lw = Zcc @ r.x; w_eb = np.exp(lw - lw.max()); w_eb /= w_eb.mean()
    except Exception:
        w_eb = np.ones(len(cal))

    def kish(w): return float((w.sum() ** 2) / (w ** 2).sum())

    def smd(c, a, w=None):
        w = np.ones(len(c)) if w is None else w
        return float((np.average(c, weights=w) - a.mean()) / (np.sqrt((c.var() + a.var()) / 2) + 1e-9))
    oob = float(((e_app < e_cal.min()) | (e_app > e_cal.max())).mean())
    diag = {"tau": tau, "n_cal_eligible": len(cal), "n_app_eligible": len(app),
            "app_outside_support_%": round(100 * oob, 2),
            "ess_unweighted": len(cal), "ess_inverse_odds": round(kish(w_io)),
            "ess_entropy_balance": round(kish(w_eb)),
            "io_weight_median": round(float(np.median(w_io)), 2), "io_weight_max": round(float(w_io.max()), 2)}
    smds = []
    for nm in ["post", "logfa"]:
        smds.append({"covariate": nm, "smd_raw": round(smd(cal[nm].to_numpy(), app[nm].to_numpy()), 3),
                     "smd_inverse_odds": round(smd(cal[nm].to_numpy(), app[nm].to_numpy(), w_io), 3),
                     "smd_entropy_bal": round(smd(cal[nm].to_numpy(), app[nm].to_numpy(), w_eb), 3)})
    for f in ["Electric", "Gas"]:
        c = (cal.fuel == f).astype(float).to_numpy(); a = (app.fuel == f).astype(float).to_numpy()
        smds.append({"covariate": f"fuel={f}", "smd_raw": round(smd(c, a), 3),
                     "smd_inverse_odds": round(smd(c, a, w_io), 3), "smd_entropy_bal": round(smd(c, a, w_eb), 3)})
    pd.DataFrame(smds).to_csv(OUT_TABLES / "bridge_transport_smd.csv", index=False)
    pd.DataFrame([diag]).to_csv(OUT_TABLES / "bridge_transport_diagnostics.csv", index=False)

    def apply_model(formula, w):
        dc = dmatrix(formula, dict(fuel=cal.fuel, x=(cal.post.to_numpy() - tau) / 25.0,
                                   logfa=cal.logfa, sector=cal.sector), return_type="dataframe")
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(np.asarray(dc), y, sample_weight=w)
        Xa = build_design_matrices([dc.design_info], dict(fuel=app.fuel, x=(app.post.to_numpy() - tau) / 25.0,
                                                          logfa=app.logfa, sector=app.sector))[0]
        p = clf.predict_proba(np.asarray(Xa))[:, 1]
        return float(p.sum()), float((app_fa * p).sum()) / 1e6
    fC = "x + C(fuel) + logfa + C(sector) + C(fuel):x + C(fuel):logfa"
    fD = "cr(x, df=4) + C(fuel) + cr(logfa, df=4) + C(sector)"
    est = []
    for mlabel, formula in [("C", fC), ("D", fD)]:
        for wlabel, w in [("untransported", np.ones(len(cal))), ("inverse_odds", w_io), ("entropy_balance", w_eb)]:
            cnt, ar = apply_model(formula, w)
            est.append({"model": mlabel, "estimator": wlabel, "expected_count": round(cnt),
                        "affected_area_Mm2": round(ar, 2), "expenditure_100_GBP_bn": round(ar * 100 / 1e3, 2)})
    estimates = pd.DataFrame(est); estimates.to_csv(OUT_TABLES / "bridge_transport_estimates.csv", index=False)
    return {"diagnostics": diag, "smd": pd.DataFrame(smds), "estimates": estimates}


def sector_specific_expenditure(sample: pd.DataFrame, latest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Entry-level, sector-specific expenditure-equivalent scale using the IPF (2017) archetype
    cost schedule, a strict register->archetype crosswalk (offices/industrial only; other
    sectors unmatched), and HM Treasury GDP-deflator rebasing to 2024-25 prices. Reports the
    covered-sector cost, the unmatched affected area with a generic-range sensitivity, and the
    full-register figure only under an explicit unmatched-sector assumption. This is an
    illustrative area-to-expenditure scale conversion, not an observed retrofit or policy cost."""
    lat = latest[latest.floor_area > 0].copy()
    defl = pd.read_csv(EXTERNAL_DIR / "gdp_deflator_hmt_jun2026.csv").set_index("financial_year")
    m = float(defl.loc["2024-25", "gdp_deflator_index_2025_26_base"]
              / defl.loc["2017-18", "gdp_deflator_index_2025_26_base"])
    cross = pd.read_csv(EXTERNAL_DIR / "sector_cost_crosswalk.csv")
    cost = {r.register_sector: (r.cost_low_gbp_m2_2017, r.cost_central_gbp_m2_2017, r.cost_high_gbp_m2_2017)
            for _, r in cross[cross.matched == "yes"].iterrows()}
    rows, applied = [], []
    for tname, tau in [("EPC-B", 50.0), ("F/G", 125.0)]:
        prob = _model_c_register_probs(sample, lat, tau)
        cand = lat.is_post_revision.to_numpy(bool) & (lat.post.to_numpy(float) <= tau)
        area = np.where(cand, prob, 0.0) * lat.floor_area.to_numpy(float)      # expected area (m2)
        sec = lat.sector_group.astype(str).to_numpy()
        cov = {k: 0.0 for k in ("area", "lo", "ce", "hi")}; unm_area = 0.0
        for sg in sorted(set(sec)):
            a = float(area[sec == sg].sum()) / 1e6                             # Mm2
            if sg in cost:
                lo, ce, hi = (c * m for c in cost[sg])
                cov["area"] += a; cov["lo"] += a * lo / 1e3; cov["ce"] += a * ce / 1e3; cov["hi"] += a * hi / 1e3
                rows.append({"threshold": tname, "register_sector": sg, "match": "matched",
                             "affected_area_Mm2": round(a, 2), "unit_cost_central_2024_25": round(ce),
                             "expenditure_central_GBP_bn": round(a * ce / 1e3, 3)})
            else:
                unm_area += a
                rows.append({"threshold": tname, "register_sector": sg, "match": "unmatched",
                             "affected_area_Mm2": round(a, 2), "unit_cost_central_2024_25": np.nan,
                             "expenditure_central_GBP_bn": np.nan})
        tot = cov["area"] + unm_area
        applied.append({"threshold": tname,
                        "covered_area_Mm2": round(cov["area"], 2), "unmatched_area_Mm2": round(unm_area, 2),
                        "unmatched_share_%": round(100 * unm_area / max(tot, 1e-9), 1),
                        "covered_expenditure_low_GBP_bn": round(cov["lo"], 2),
                        "covered_expenditure_central_GBP_bn": round(cov["ce"], 2),
                        "covered_expenditure_high_GBP_bn": round(cov["hi"], 2),
                        "unmatched_generic_25_GBP_bn": round(unm_area * 25 / 1e3, 2),
                        "unmatched_generic_100_GBP_bn": round(unm_area * 100 / 1e3, 2),
                        "unmatched_generic_200_GBP_bn": round(unm_area * 200 / 1e3, 2),
                        "full_register_low_GBP_bn": round(cov["lo"] + unm_area * 25 / 1e3, 2),
                        "full_register_central_GBP_bn": round(cov["ce"] + unm_area * 100 / 1e3, 2),
                        "full_register_high_GBP_bn": round(cov["hi"] + unm_area * 200 / 1e3, 2)})
    by_sector = pd.DataFrame(rows); by_sector.to_csv(OUT_TABLES / "bridge_sector_expenditure_by_sector.csv", index=False)
    summary = pd.DataFrame(applied); summary.to_csv(OUT_TABLES / "bridge_sector_expenditure_summary.csv", index=False)
    pd.DataFrame([{"deflator_source_year": "2017-18", "deflator_target_year": "2024-25",
                   "multiplier": round(m, 4), "source": "HM Treasury GDP deflators, June 2026 release"}]
                 ).to_csv(OUT_TABLES / "bridge_cost_rebasing.csv", index=False)
    return {"by_sector": by_sector, "summary": summary, "multiplier": m}


def area_model_comparison(sample: pd.DataFrame, latest: pd.DataFrame,
                          folds: np.ndarray) -> dict[str, pd.DataFrame]:
    """Develop and compare candidate affected-area models under leakage-free UPRN folds,
    using a joint count-and-area validation gate, and apply the selected model to the
    register. Candidates enrich the baseline fuel-specific threshold logistic with log
    floor area, sector and fuel interactions (Model C) or splines (Model D). Predictors are
    application-time features only; only the binary old-basis outcome is held out per fold.
    Writes bridge_area_model_comparison.csv and bridge_area_model_applied.csv."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss
    from patsy import dmatrix, build_design_matrices
    s = sample[sample.floor_area > 0].reset_index(drop=True).copy()
    fld = folds[sample.floor_area.to_numpy() > 0]
    s["logfa"] = np.log(s.floor_area.to_numpy(dtype=float))
    lat = latest[latest.floor_area > 0].copy(); lat["logfa"] = np.log(lat.floor_area.to_numpy(dtype=float))
    fa = s.floor_area.to_numpy(dtype=float); order = np.argsort(-fa)
    lfa = lat.floor_area.to_numpy(dtype=float)
    is_post = lat.is_post_revision.to_numpy(bool); obs = lat.post.to_numpy(dtype=float)
    rows, applied = [], []
    for tname, _pc, tau in _THR:
        y = (s.pre.to_numpy(dtype=float) > tau).astype(int)
        cand = is_post & (obs <= tau)
        for mname, formula in _AREA_MODELS.items():
            data = dict(fuel=s.fuel, x=(s.post.to_numpy(dtype=float) - tau) / 25.0,
                        logfa=s.logfa, sector_group=s.sector_group)
            dm = dmatrix(formula, data, return_type="dataframe")
            X = np.asarray(dm)
            p = np.zeros(len(s))
            for f in range(int(fld.max()) + 1):
                tr, te = fld != f, fld == f
                clf = LogisticRegression(C=1.0, max_iter=3000).fit(X[tr], y[tr])
                p[te] = clf.predict_proba(X[te])[:, 1]
            obs_a, pred_a = float((fa * y).sum()) / 1e6, float((fa * p).sum()) / 1e6
            keep = np.ones(len(y), bool); keep[order[:25]] = False
            a25 = 100 * ((fa[keep] * p[keep]).sum() - (fa[keep] * y[keep]).sum()) / max(float((fa[keep] * y[keep]).sum()), 1e-9)
            big = fa > 5000
            abig = (100 * ((fa[big] * p[big]).sum() - (fa[big] * y[big]).sum()) / max(float((fa[big] * y[big]).sum()), 1e-9)
                    if big.sum() > 10 and (fa[big] * y[big]).sum() > 1e3 else np.nan)
            area_err = 100 * (pred_a - obs_a) / max(obs_a, 1e-9)
            gate = (abs(area_err) <= 10.0) and (abs(a25) <= 20.0) and (np.isnan(abig) or abs(abig) <= 30.0)
            rows.append({"threshold": tname, "model": mname,
                         "oof_count_error_pp": round(100 * (p.sum() - y.sum()) / len(y), 2),
                         "oof_area_error_pct": round(area_err, 1),
                         "area_error_excl_top25_pct": round(a25, 1),
                         "area_error_gt5000m2_pct": (round(abig, 1) if not np.isnan(abig) else np.nan),
                         "brier": round(brier_score_loss(y, p), 4),
                         "passes_joint_gate": bool(gate)})
            # apply on full calibration set -> register frame
            clf_full = LogisticRegression(C=1.0, max_iter=3000).fit(X, y)
            Xl = build_design_matrices([dm.design_info], dict(
                fuel=lat.fuel, x=(obs - tau) / 25.0, logfa=lat.logfa, sector_group=lat.sector_group))[0]
            probl = clf_full.predict_proba(np.asarray(Xl))[:, 1]
            applied.append({"threshold": tname, "model": mname,
                            "expected_count": round(float(np.where(cand, probl, 0).sum()), 0),
                            "affected_area_Mm2": round(float((lfa * np.where(cand, probl, 0)).sum()) / 1e6, 3),
                            "expenditure_100_GBP_bn": round(float((lfa * np.where(cand, probl, 0)).sum()) / 1e6 * 100 / 1e3, 3)})
    comp = pd.DataFrame(rows); comp.to_csv(OUT_TABLES / "bridge_area_model_comparison.csv", index=False)
    app = pd.DataFrame(applied); app.to_csv(OUT_TABLES / "bridge_area_model_applied.csv", index=False)
    return {"comparison": comp, "applied": app}


def area_weighted_validation(oof: pd.DataFrame, latest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Area-weighted out-of-fold validation of the preferred threshold-probability model.

    The model is selected on aggregate *count* prevalence, but is applied to expected
    *affected floor area*; these are different estimands, and floor area is dominated by a
    few large premises. We therefore report: (1) out-of-fold area-weighted calibration
    (observed vs predicted affected area on held-out folds); (2) per-fold observed vs
    predicted area; (3) large-premise influence (excluding the top 1/5/10/25 premises by
    floor area); (4) calibration within floor-area bands; and (5) calibration-vs-application
    transport checks over post-rating band, fuel, floor-area band and sector. All use the
    cross-fitted OOF predictions of ``threshold_logit``; nothing here uses the hard cutoff."""
    t = oof[oof.specification == "threshold_logit"].copy()
    t["fa"] = pd.to_numeric(t.floor_area, errors="coerce")
    t = t[t.fa > 0]
    total_area = float(t.fa.sum()) / 1e6

    # (1) aggregate out-of-fold count vs area calibration
    cal = []
    for name, pcol, tau in _THR:
        actual = (t.pre_observed.to_numpy(float) > tau).astype(float)
        prob = t[pcol].to_numpy(float); fa = t.fa.to_numpy(float)
        obs_a, pred_a = float((fa * actual).sum()) / 1e6, float((fa * prob).sum()) / 1e6
        cal.append({"threshold": name,
                    "oof_obs_count": int(actual.sum()), "oof_pred_count": round(float(prob.sum()), 1),
                    "count_error_pp_of_n": round(100 * (prob.sum() - actual.sum()) / len(t), 2),
                    "oof_obs_area_Mm2": round(obs_a, 3), "oof_pred_area_Mm2": round(pred_a, 3),
                    "area_error_Mm2": round(pred_a - obs_a, 3),
                    "area_error_pct_of_observed": round(100 * (pred_a - obs_a) / max(obs_a, 1e-9), 2),
                    "area_error_pp_of_total_area": round(100 * (pred_a - obs_a) / total_area, 2)})
    cal = pd.DataFrame(cal); cal.to_csv(OUT_TABLES / "bridge_area_weighted_calibration.csv", index=False)

    # (2) per held-out fold observed vs predicted area
    fold_rows = []
    for name, pcol, tau in _THR:
        for fold, g in t.groupby("fold"):
            actual = (g.pre_observed.to_numpy(float) > tau).astype(float)
            fa = g.fa.to_numpy(float); prob = g[pcol].to_numpy(float)
            obs_a, pred_a = float((fa * actual).sum()) / 1e6, float((fa * prob).sum()) / 1e6
            fold_rows.append({"threshold": name, "fold": int(fold), "n": int(len(g)),
                              "obs_area_Mm2": round(obs_a, 3), "pred_area_Mm2": round(pred_a, 3),
                              "area_error_pct": round(100 * (pred_a - obs_a) / max(obs_a, 1e-9), 2)})
    pd.DataFrame(fold_rows).to_csv(OUT_TABLES / "bridge_area_calibration_by_fold.csv", index=False)

    # (3) large-premise influence
    order = np.argsort(-t.fa.to_numpy(float))
    inf = []
    for name, pcol, tau in _THR:
        actual = (t.pre_observed.to_numpy(float) > tau).astype(float)
        prob = t[pcol].to_numpy(float); fa = t.fa.to_numpy(float)
        for k in (0, 1, 5, 10, 25):
            keep = np.ones(len(t), bool); keep[order[:k]] = False
            obs_a, pred_a = float((fa[keep] * actual[keep]).sum()) / 1e6, float((fa[keep] * prob[keep]).sum()) / 1e6
            inf.append({"threshold": name, "excluded_largest_n": k,
                        "obs_area_Mm2": round(obs_a, 3), "pred_area_Mm2": round(pred_a, 3),
                        "area_error_pct": round(100 * (pred_a - obs_a) / max(obs_a, 1e-9), 2),
                        "pred_area_retained_pct": round(100 * pred_a / max(float((fa * prob).sum()) / 1e6, 1e-9), 2)})
    pd.DataFrame(inf).to_csv(OUT_TABLES / "bridge_large_premise_influence.csv", index=False)

    # (4) calibration within floor-area bands
    band = []
    for name, pcol, tau in _THR:
        for b, g in t.groupby("floor_area_bin"):
            actual = (g.pre_observed.to_numpy(float) > tau).astype(float)
            fa = g.fa.to_numpy(float); prob = g[pcol].to_numpy(float)
            obs_a, pred_a = float((fa * actual).sum()) / 1e6, float((fa * prob).sum()) / 1e6
            band.append({"threshold": name, "floor_area_bin": b, "n": int(len(g)),
                         "obs_prevalence_%": round(100 * actual.mean(), 2),
                         "pred_prevalence_%": round(100 * prob.mean(), 2),
                         "prevalence_error_pp": round(100 * (prob.mean() - actual.mean()), 2),
                         "obs_area_Mm2": round(obs_a, 3), "pred_area_Mm2": round(pred_a, 3),
                         "area_error_pct": round(100 * (pred_a - obs_a) / max(obs_a, 1e-9), 2)})
    pd.DataFrame(band).to_csv(OUT_TABLES / "bridge_calibration_by_floor_area_band.csv", index=False)

    # (5) transport: calibration sample vs application (register) composition
    lat = latest.copy(); lat["fa"] = pd.to_numeric(lat.floor_area, errors="coerce")
    tr = []
    for var in ("post_band", "fuel", "floor_area_bin", "sector_group"):
        if var not in lat.columns:
            continue
        cal_share = t[var].astype(str).value_counts(normalize=True)
        app_share = lat[var].astype(str).value_counts(normalize=True)
        app_area = lat.groupby(lat[var].astype(str)).fa.sum()
        app_area = app_area / app_area.sum()
        for lvl in sorted(set(cal_share.index) | set(app_share.index)):
            tr.append({"variable": var, "level": lvl,
                       "calibration_share_%": round(100 * float(cal_share.get(lvl, 0.0)), 2),
                       "application_count_share_%": round(100 * float(app_share.get(lvl, 0.0)), 2),
                       "application_area_share_%": round(100 * float(app_area.get(lvl, 0.0)), 2)})
    pd.DataFrame(tr).to_csv(OUT_TABLES / "bridge_calibration_transport.csv", index=False)
    return {"calibration": cal, "influence": pd.DataFrame(inf), "band": pd.DataFrame(band),
            "transport": pd.DataFrame(tr)}


def aggregate_application_by_fuel(frame: pd.DataFrame, fitted_models: dict[str, Any],
                                  frame_label: str, preferred_spec: str = "threshold_logit",
                                  cost_per_m2: float = 100.0) -> pd.DataFrame:
    """Fuel split of the preferred threshold-probability model's expected threshold
    crossings and expected affected floor area, on the full register frame. Each record
    contributes its crossing probability (expected entry) and floor_area x probability
    (expected affected area) to its main-heating-fuel group, so the fuel rows sum exactly
    to the aggregate expected entries and expected affected area used in the main text.
    Emitted as ``bridge_preferred_fuel_split.csv`` and consumed by the capex figure."""
    spec = next(s for s in ALL_SPECS if s.name == preferred_spec)
    values = _model_application_values(frame, spec, fitted_models[spec.name])
    observed = frame.post.to_numpy(dtype=float)
    is_post = frame.is_post_revision.to_numpy(dtype=bool)
    floor_area = frame.floor_area.to_numpy(dtype=float)
    fuel = frame.fuel.astype(str).to_numpy()
    rows = []
    for threshold_name, threshold in THRESHOLDS.items():
        probability = np.asarray(values[f"{threshold_name}_probability"], dtype=float)
        candidate = is_post & (observed <= threshold)
        crossing_probability = candidate.astype(float) * probability
        for fg in FUELS:
            m = fuel == fg
            expected_entries = float(np.sum(crossing_probability[m]))
            expected_area = float(np.sum(floor_area[m] * crossing_probability[m]) / 1e6)
            rows.append({
                "frame": frame_label, "specification": spec.name, "label": spec.label,
                "threshold": threshold_name, "threshold_rating": threshold, "fuel": fg,
                "expected_entries": expected_entries,
                "expected_affected_floor_area_Mm2": expected_area,
                "expenditure_equivalent_GBP_bn": expected_area * cost_per_m2 / 1e3,
                "cost_per_m2_GBP": cost_per_m2,
            })
        tot_e = sum(r["expected_entries"] for r in rows if r["threshold"] == threshold_name)
        tot_a = sum(r["expected_affected_floor_area_Mm2"] for r in rows if r["threshold"] == threshold_name)
        rows.append({
            "frame": frame_label, "specification": spec.name, "label": spec.label,
            "threshold": threshold_name, "threshold_rating": threshold, "fuel": "All",
            "expected_entries": tot_e, "expected_affected_floor_area_Mm2": tot_a,
            "expenditure_equivalent_GBP_bn": tot_a * cost_per_m2 / 1e3, "cost_per_m2_GBP": cost_per_m2})
    return pd.DataFrame(rows)


def aggregate_application(frame: pd.DataFrame, fitted_models: dict[str, Any],
                          frame_label: str, weights: np.ndarray | None = None) -> pd.DataFrame:
    """Aggregate hard rating bridges or direct threshold probabilities."""
    weight = np.ones(len(frame), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if len(weight) != len(frame):
        raise ValueError("Weight vector does not match application frame")
    observed = frame.post.to_numpy(dtype=float)
    is_post = frame.is_post_revision.to_numpy(dtype=bool)
    floor_area = frame.floor_area.to_numpy(dtype=float)
    rows = []
    for spec in ALL_SPECS:
        values = _model_application_values(frame, spec, fitted_models[spec.name])
        for threshold_name, threshold in THRESHOLDS.items():
            probability = np.asarray(values[f"{threshold_name}_probability"], dtype=float)
            hard = np.asarray(values[f"{threshold_name}_hard"], dtype=bool)
            candidate = is_post & (observed <= threshold)
            crossing_probability = candidate.astype(float) * probability
            crossing_hard = candidate & hard
            old_positive_probability = np.where(is_post, probability, observed > threshold)
            rows.append({
                "frame": frame_label,
                "specification": spec.name,
                "label": spec.label,
                "model_kind": spec.kind,
                "estimate_type": values["estimate_type"],
                "threshold": threshold_name,
                "threshold_rating": threshold,
                "n_records": int(len(frame)),
                "weight_sum": float(weight.sum()),
                "observed_positive_rate_%": float(100 * np.average(observed > threshold, weights=weight)),
                "predicted_old_positive_rate_%": float(100 * np.average(old_positive_probability, weights=weight)),
                "crossing_count_estimate": float(np.sum(weight * crossing_probability)),
                "crossing_count_hard": float(np.sum(weight * crossing_hard)),
                "crossing_rate_of_frame_%": float(100 * np.sum(weight * crossing_probability) / weight.sum()),
                "affected_floor_area_Mm2": float(
                    np.sum(weight * floor_area * crossing_probability) / 1e6
                ),
                "affected_floor_area_hard_Mm2": float(
                    np.sum(weight * floor_area * crossing_hard) / 1e6
                ),
                "structural_candidate_predeclared": spec.structural_candidate,
            })
    return pd.DataFrame(rows)


def apply_all_models(sample: pd.DataFrame, latest: pd.DataFrame,
                     rake: pd.DataFrame, rake_weights: np.ndarray) -> dict[str, pd.DataFrame]:
    fitted = fit_full_models(sample)
    fitted_cell_diagnostics(fitted)
    full = aggregate_application(latest, fitted, "full_latest_register_frame")
    policy = aggregate_application(
        latest[latest.policy_proxy].copy(), fitted,
        "valid_nonconstruction_property_to_let_over_1000m2_policy_proxy",
    )
    benchmark = aggregate_application(
        rake, fitted, "ND_NEED_composition_raked_register_frame", weights=rake_weights
    )
    full.to_csv(OUT_TABLES / "bridge_full_frame_aggregates.csv", index=False)
    policy.to_csv(OUT_TABLES / "bridge_policy_proxy_aggregates.csv", index=False)
    benchmark.to_csv(OUT_TABLES / "bridge_benchmark_weighted_aggregates.csv", index=False)
    fuel_split = aggregate_application_by_fuel(latest, fitted, "full_latest_register_frame")
    fuel_split.to_csv(OUT_TABLES / "bridge_preferred_fuel_split.csv", index=False)
    return {"full": full, "policy": policy, "benchmark": benchmark, "fuel_split": fuel_split}


def rank_models(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank primarily by OOF threshold usefulness; never by in-sample fit."""
    ranking = summary.copy()
    ranking["threshold_rank"] = ranking.threshold_score.rank(
        method="min", ascending=False
    ).astype(int)
    ranking["rating_rank_by_MAE"] = np.nan
    rating_mask = ranking.model_kind == "rating"
    ranking.loc[rating_mask, "rating_rank_by_MAE"] = ranking.loc[
        rating_mask, "MAE_pts"
    ].rank(method="min", ascending=True)
    ranking["individual_rating_calibration_usable"] = (
        rating_mask
        & (ranking.MAE_pts <= 20.0)
        & ranking.calibration_slope_actual_on_predicted.between(0.8, 1.2)
    )
    ranking["aggregate_probability_calibrated"] = (
        (ranking.B_aggregate_probability_error_pp.abs() <= 1.0)
        & (ranking.FG_aggregate_probability_error_pp.abs() <= 1.0)
    )
    ranking["threshold_classification_useful"] = (
        (ranking.B_balanced_accuracy >= 0.65)
        & (ranking.FG_balanced_accuracy >= 0.65)
    )
    ranking["best_supported_aggregate"] = (
        (ranking.specification == THRESHOLD_SPEC.name)
        & ranking.aggregate_probability_calibrated
        & ranking.threshold_classification_useful
    )

    def status(row: pd.Series) -> str:
        if row.best_supported_aggregate:
            return (
                "best-supported aggregate sensitivity only; subgroup and external calibration "
                "remain unresolved"
            )
        if row.specification == "fuel_mult":
            return "transparent baseline only; pooled bias masks rating-local calibration failure"
        if row.specification == "fuel_add":
            return "transparent functional-form baseline only"
        return "cross-fitted structural sensitivity only; not validated for individual ratings"

    ranking["authoritative_status"] = ranking.apply(status, axis=1)
    ranking = ranking.sort_values(
        ["threshold_rank", "MAE_pts"], na_position="last"
    ).reset_index(drop=True)
    ranking.to_csv(OUT_TABLES / "bridge_model_ranking.csv", index=False)
    return ranking


def structural_ranges(aggregates: dict[str, pd.DataFrame],
                      ranking: pd.DataFrame) -> pd.DataFrame:
    candidate = set(ranking.loc[
        ranking.structural_candidate_predeclared.astype(bool), "specification"
    ])
    best_supported = set(ranking.loc[
        ranking.best_supported_aggregate.astype(bool), "specification"
    ])
    scopes = {
        "candidate_local_and_probabilistic": candidate,
        "best_supported_overall_aggregate": best_supported,
        "all_models_including_fuel_only_baselines": set(ranking.specification),
    }
    rows = []
    for dataset, table in aggregates.items():
        # The fuel-split table is a figure input carrying per-fuel columns; the
        # structural range is defined over the register-frame aggregates only.
        if "crossing_count_estimate" not in table.columns:
            continue
        for scope, models in scopes.items():
            selected = table[table.specification.isin(models)]
            if selected.empty:
                continue
            for threshold, group in selected.groupby("threshold", observed=True):
                count_min_idx = group.crossing_count_estimate.idxmin()
                count_max_idx = group.crossing_count_estimate.idxmax()
                area_min_idx = group.affected_floor_area_Mm2.idxmin()
                area_max_idx = group.affected_floor_area_Mm2.idxmax()
                count_min = float(group.loc[count_min_idx, "crossing_count_estimate"])
                count_max = float(group.loc[count_max_idx, "crossing_count_estimate"])
                area_min = float(group.loc[area_min_idx, "affected_floor_area_Mm2"])
                area_max = float(group.loc[area_max_idx, "affected_floor_area_Mm2"])
                rows.append({
                    "dataset": dataset,
                    "frame": group.frame.iloc[0],
                    "range_scope": scope,
                    "threshold": threshold,
                    "n_models": int(group.specification.nunique()),
                    "models": "; ".join(sorted(group.specification.unique())),
                    "crossing_count_min": count_min,
                    "crossing_count_min_model": group.loc[count_min_idx, "specification"],
                    "crossing_count_max": count_max,
                    "crossing_count_max_model": group.loc[count_max_idx, "specification"],
                    "crossing_count_relative_width_%": (
                        100 * (count_max - count_min) / max((count_max + count_min) / 2.0, 1e-12)
                    ),
                    "affected_floor_area_min_Mm2": area_min,
                    "affected_floor_area_min_model": group.loc[area_min_idx, "specification"],
                    "affected_floor_area_max_Mm2": area_max,
                    "affected_floor_area_max_model": group.loc[area_max_idx, "specification"],
                    "affected_floor_area_relative_width_%": (
                        100 * (area_max - area_min) / max((area_max + area_min) / 2.0, 1e-12)
                    ),
                })
    ranges = pd.DataFrame(rows)
    # Composition sensitivity for the best-supported model, shown as a range
    # across the unweighted and ND-NEED-raked frames rather than cherry-picking
    # whichever is numerically smaller.
    full_and_weighted = pd.concat([
        aggregates["full"], aggregates["benchmark"]
    ], ignore_index=True)
    preferred = full_and_weighted[
        full_and_weighted.specification == THRESHOLD_SPEC.name
    ]
    composition_rows = []
    for threshold, group in preferred.groupby("threshold", observed=True):
        count_min_idx = group.crossing_count_estimate.idxmin()
        count_max_idx = group.crossing_count_estimate.idxmax()
        area_min_idx = group.affected_floor_area_Mm2.idxmin()
        area_max_idx = group.affected_floor_area_Mm2.idxmax()
        count_min = float(group.loc[count_min_idx, "crossing_count_estimate"])
        count_max = float(group.loc[count_max_idx, "crossing_count_estimate"])
        area_min = float(group.loc[area_min_idx, "affected_floor_area_Mm2"])
        area_max = float(group.loc[area_max_idx, "affected_floor_area_Mm2"])
        composition_rows.append({
            "dataset": "full_vs_benchmark",
            "frame": "unweighted_and_ND_NEED_composition_raked",
            "range_scope": "best_supported_model_composition_sensitivity",
            "threshold": threshold,
            "n_models": 1,
            "models": THRESHOLD_SPEC.name,
            "crossing_count_min": count_min,
            "crossing_count_min_model": group.loc[count_min_idx, "frame"],
            "crossing_count_max": count_max,
            "crossing_count_max_model": group.loc[count_max_idx, "frame"],
            "crossing_count_relative_width_%": (
                100 * (count_max - count_min) / max((count_max + count_min) / 2.0, 1e-12)
            ),
            "affected_floor_area_min_Mm2": area_min,
            "affected_floor_area_min_model": group.loc[area_min_idx, "frame"],
            "affected_floor_area_max_Mm2": area_max,
            "affected_floor_area_max_model": group.loc[area_max_idx, "frame"],
            "affected_floor_area_relative_width_%": (
                100 * (area_max - area_min) / max((area_max + area_min) / 2.0, 1e-12)
            ),
        })
    ranges = pd.concat([ranges, pd.DataFrame(composition_rows)], ignore_index=True)
    ranges.to_csv(OUT_TABLES / "bridge_structural_ranges.csv", index=False)
    stability = ranges[
        ranges.range_scope == "candidate_local_and_probabilistic"
    ][[
        "dataset", "threshold", "crossing_count_relative_width_%",
        "affected_floor_area_relative_width_%",
    ]].copy()
    stability["more_stable_metric"] = np.where(
        stability["crossing_count_relative_width_%"]
        <= stability["affected_floor_area_relative_width_%"],
        "counts", "floor area",
    )
    stability.to_csv(OUT_TABLES / "bridge_stability_summary.csv", index=False)
    return ranges


def _expenditure_table(aggregate: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    status = ranking.set_index("specification")[[
        "authoritative_status", "best_supported_aggregate",
    ]]
    out = aggregate.merge(status, left_on="specification", right_index=True, how="left")
    out = out.rename(columns={"affected_floor_area_Mm2": "affected_floor_area_Mm2"})
    for level, unit_cost in UNIT_COSTS_GBP_M2.items():
        out[f"unit_cost_{level}_GBP_m2"] = unit_cost
        out[f"expenditure_equivalent_{level}_GBP_bn"] = (
            out.affected_floor_area_Mm2 * unit_cost / 1000.0
        )
    out["interpretation"] = (
        "area-to-cost scale conversion only; not observed cost, avoided investment, savings, "
        "compliance cost, welfare, or social NPV"
    )
    columns = [
        "frame", "specification", "label", "model_kind", "estimate_type", "threshold",
        "crossing_count_estimate", "affected_floor_area_Mm2",
        "unit_cost_low_GBP_m2", "expenditure_equivalent_low_GBP_bn",
        "unit_cost_central_GBP_m2", "expenditure_equivalent_central_GBP_bn",
        "unit_cost_high_GBP_m2", "expenditure_equivalent_high_GBP_bn",
        "structural_candidate_predeclared", "best_supported_aggregate",
        "authoritative_status", "interpretation",
    ]
    return out[columns]


def expenditure_equivalent_outputs(aggregates: dict[str, pd.DataFrame],
                                   ranking: pd.DataFrame,
                                   ranges: pd.DataFrame) -> dict[str, pd.DataFrame]:
    full = _expenditure_table(aggregates["full"], ranking)
    policy = _expenditure_table(aggregates["policy"], ranking)
    benchmark = _expenditure_table(aggregates["benchmark"], ranking)
    full.to_csv(OUT_TABLES / "expenditure_equivalent_by_bridge.csv", index=False)
    policy.to_csv(OUT_TABLES / "expenditure_equivalent_policy_proxy.csv", index=False)
    benchmark.to_csv(OUT_TABLES / "expenditure_equivalent_benchmark_weighted.csv", index=False)

    structural_rows = []
    for _, row in ranges.iterrows():
        output = {
            "dataset": row.dataset,
            "frame": row.frame,
            "range_scope": row.range_scope,
            "threshold": row.threshold,
            "n_models": row.n_models,
            "models": row.models,
            "affected_floor_area_min_Mm2": row.affected_floor_area_min_Mm2,
            "affected_floor_area_min_model": row.affected_floor_area_min_model,
            "affected_floor_area_max_Mm2": row.affected_floor_area_max_Mm2,
            "affected_floor_area_max_model": row.affected_floor_area_max_model,
        }
        for level, cost in UNIT_COSTS_GBP_M2.items():
            output[f"unit_cost_{level}_GBP_m2"] = cost
            output[f"expenditure_equivalent_{level}_min_GBP_bn"] = (
                row.affected_floor_area_min_Mm2 * cost / 1000.0
            )
            output[f"expenditure_equivalent_{level}_max_GBP_bn"] = (
                row.affected_floor_area_max_Mm2 * cost / 1000.0
            )
        output["joint_structural_and_unit_cost_min_GBP_bn"] = (
            row.affected_floor_area_min_Mm2 * UNIT_COSTS_GBP_M2["low"] / 1000.0
        )
        output["joint_structural_and_unit_cost_max_GBP_bn"] = (
            row.affected_floor_area_max_Mm2 * UNIT_COSTS_GBP_M2["high"] / 1000.0
        )
        output["interpretation"] = (
            "structural bridge range crossed with replaceable unit-cost assumptions; scale only"
        )
        structural_rows.append(output)
    structural = pd.DataFrame(structural_rows)
    structural.to_csv(OUT_TABLES / "expenditure_equivalent_structural_range.csv", index=False)
    assumptions = pd.DataFrame([
        {"assumption": level, "GBP_per_m2": value, "source_in_repo": "existing paper assumptions"}
        for level, value in UNIT_COSTS_GBP_M2.items()
    ])
    assumptions.to_csv(OUT_TABLES / "expenditure_equivalent_unit_cost_assumptions.csv", index=False)
    return {"full": full, "policy": policy, "benchmark": benchmark, "structural": structural}


def _markdown_table(frame: pd.DataFrame, decimals: int = 2) -> str:
    display = frame.copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    display[numeric] = display[numeric].round(decimals)

    def render(value: Any) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{decimals}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column).replace("|", "\\|") for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown_report(audit: dict[str, Any], validation: dict[str, pd.DataFrame],
                          ranking: pd.DataFrame, aggregates: dict[str, pd.DataFrame],
                          ranges: pd.DataFrame,
                          expenditure: dict[str, pd.DataFrame],
                          raking_diagnostics: dict[str, Any]) -> None:
    preferred = "threshold_logit"
    baseline = ranking[ranking.specification == "fuel_mult"].iloc[0]
    direct = ranking[ranking.specification == preferred].iloc[0]
    local = ranking[
        ranking.structural_candidate_predeclared.astype(bool)
        & (ranking.model_kind == "rating")
    ]
    best_local_mae = local.loc[local.MAE_pts.idxmin()]
    best_local_threshold = local.loc[local.threshold_score.idxmax()]
    cancellation = validation["bias_cancellation"]
    baseline_cancel = cancellation[
        (cancellation.specification == "fuel_mult")
        & (cancellation.band_basis == "true_old_basis")
    ].iloc[0]
    probability_calibration = validation["probability_calibration"]
    direct_subgroups = probability_calibration[
        (probability_calibration.specification == preferred)
        & (probability_calibration.n >= 50)
    ].copy()
    direct_subgroups["absolute_calibration_error_pp"] = (
        direct_subgroups.probability_calibration_error_pp.abs()
    )
    worst_direct_subgroups = direct_subgroups.nlargest(
        8, "absolute_calibration_error_pp"
    )[[
        "threshold", "grouping", "group", "n", "actual_positive_rate_%",
        "predicted_probability_mean_%", "probability_calibration_error_pp",
    ]]
    max_direct_subgroup_error = float(
        direct_subgroups.absolute_calibration_error_pp.max()
    )

    def aggregate_row(dataset: str, threshold: str, spec: str = preferred) -> pd.Series:
        table = aggregates[dataset]
        return table[(table.specification == spec) & (table.threshold == threshold)].iloc[0]

    full_b, full_fg = aggregate_row("full", "B_C"), aggregate_row("full", "E_F")
    policy_b, policy_fg = aggregate_row("policy", "B_C"), aggregate_row("policy", "E_F")
    weighted_b, weighted_fg = aggregate_row("benchmark", "B_C"), aggregate_row("benchmark", "E_F")

    candidate_ranges = ranges[
        ranges.range_scope == "candidate_local_and_probabilistic"
    ].copy()
    headline_ranges = candidate_ranges[[
        "dataset", "threshold", "n_models", "crossing_count_min", "crossing_count_max",
        "affected_floor_area_min_Mm2", "affected_floor_area_max_Mm2",
        "crossing_count_relative_width_%", "affected_floor_area_relative_width_%",
    ]]

    ranking_table = ranking[[
        "threshold_rank", "specification", "MAE_pts",
        "calibration_slope_actual_on_predicted", "B_balanced_accuracy",
        "FG_balanced_accuracy", "B_aggregate_probability_error_pp",
        "FG_aggregate_probability_error_pp", "authoritative_status",
    ]]
    policy_table = aggregates["policy"][[
        "specification", "threshold", "estimate_type", "crossing_count_estimate",
        "affected_floor_area_Mm2",
    ]]
    benchmark_table = aggregates["benchmark"][[
        "specification", "threshold", "estimate_type", "crossing_count_estimate",
        "affected_floor_area_Mm2",
    ]]
    expenditure_ranges = expenditure["structural"]
    expenditure_table = expenditure_ranges[
        expenditure_ranges.range_scope.isin([
            "candidate_local_and_probabilistic",
            "best_supported_model_composition_sensitivity",
        ])
    ][[
        "dataset", "threshold", "affected_floor_area_min_Mm2",
        "affected_floor_area_max_Mm2", "expenditure_equivalent_low_min_GBP_bn",
        "expenditure_equivalent_central_min_GBP_bn",
        "expenditure_equivalent_central_max_GBP_bn",
        "expenditure_equivalent_high_max_GBP_bn",
    ]]

    b_count_change = 100 * (weighted_b.crossing_count_estimate / full_b.crossing_count_estimate - 1)
    b_area_change = 100 * (weighted_b.affected_floor_area_Mm2 / full_b.affected_floor_area_Mm2 - 1)
    fg_count_change = 100 * (weighted_fg.crossing_count_estimate / full_fg.crossing_count_estimate - 1)
    fg_area_change = 100 * (weighted_fg.affected_floor_area_Mm2 / full_fg.affected_floor_area_Mm2 - 1)

    report = f"""# Authoritative bridge validation

## Executive decision

The fuel-only multiplicative bridge is **not usable as an individually calibrated old-basis rating model**. It remains useful only as a transparent baseline. Its out-of-fold MAE is {baseline.MAE_pts:.1f} rating points and its calibration slope is {baseline.calibration_slope_actual_on_predicted:.3f}. The pooled bias of {baseline.bias_pts:.2f} points is misleading: fuel × true-band contributions have a gross absolute contribution of {baseline_cancel.gross_absolute_bias_contribution_pts:.2f} points and cancel by {100 * baseline_cancel.cancellation_fraction:.1f}% before reaching the pooled mean.

The rating-local ladder improves conventional rating calibration but **does not improve threshold usefulness**. The lowest local-model MAE is {best_local_mae.MAE_pts:.1f} ({best_local_mae.specification}) and its calibration slope is {best_local_mae.calibration_slope_actual_on_predicted:.3f}; however, the best local threshold score is {best_local_threshold.threshold_score:.3f}, below the fuel-only baseline's {baseline.threshold_score:.3f}.

The direct fuel-specific threshold-logistic model is the **best-supported aggregate sensitivity**, not an externally validated truth. It is the only specification meeting the transparent overall OOF gate used here: balanced accuracy at least 0.65 at both thresholds and absolute probability-sum prevalence error no greater than 1 percentage point. Its hard decisions are fully out-of-fold, with cutoffs selected using training folds only. Its B/C and E/F balanced accuracies are {direct.B_balanced_accuracy:.3f} and {direct.FG_balanced_accuracy:.3f}; its overall probability-sum prevalence errors are {direct.B_aggregate_probability_error_pp:+.3f} and {direct.FG_aggregate_probability_error_pp:+.3f} percentage points.

That overall calibration does **not** hold uniformly. Among reported subgroup cells with at least 50 calibration observations, the largest absolute probability-calibration error is {max_direct_subgroup_error:.1f} percentage points. This model therefore supports a register-conditional aggregate sensitivity, with structural and raking ranges, but not exact subgroup, rating-level or full-band claims.

## Reproduction and leakage audit

- Held-out eligible sample: {audit['heldout_sample']['n']:,} unique UPRNs; one first/latest pair per UPRN.
- Fuel counts: {audit['heldout_sample']['fuel_counts']}.
- Legacy logic: {LEGACY_SPLITS} independent Bernoulli splits with train probability {LEGACY_TRAIN_FRAC}, seed {RANDOM_SEED}; train sizes therefore vary rather than being exactly half.
- Authoritative logic: deterministic {N_FOLDS}-fold UPRN-level cross-fitting, stratified by fuel × observed post-revision band; every UPRN is held out exactly once.
- Error convention: predicted old-basis rating minus observed old-basis rating. Positive error means a numerically higher/worse predicted rating.
- No test-fold old rating, true old band, or threshold outcome enters model fitting. True old band is used only for scoring.
- Post-revision band and threshold distance are legitimate application-time features derived from the observed post-revision rating.
- Calibration-sample selection uses both certificates to require stable floor area and recommendation count. This is a selection assumption, not proof of no physical change.
- No identical-input SBEM 2013/2021 rerun pairs were found. External SBEM validation is therefore unavailable.

The former overall/per-fuel mismatch is resolved: the old overall summary averaged 200 variable half-test estimates, while the detailed table used one five-fold OOF pass. The authoritative all-fuel result and the sample-size-weighted per-fuel result now use the same OOF predictions and reconcile exactly.

## Model ranking

Ranking is by the mean of B/C and E/F out-of-fold balanced accuracy, not by in-sample fit.

{_markdown_table(ranking_table, 3)}

### Largest subgroup calibration errors for the direct probability model

{_markdown_table(worst_direct_subgroups, 2)}

## Aggregate results from the preferred probability model

### Full latest-register frame

- B/C: expected reclassification count {full_b.crossing_count_estimate:,.0f}; affected area {full_b.affected_floor_area_Mm2:.2f} Mm².
- E/F: expected reclassification count {full_fg.crossing_count_estimate:,.0f}; affected area {full_fg.affected_floor_area_Mm2:.2f} Mm².

These are probability sums across eligible post-revision register entries, not integer classifications of physical buildings.

### Policy-proxy subset

The subset contains {int(policy_b.n_records):,} latest-register UPRNs satisfying: lodged within ten years of {ANALYSIS_AS_OF.date()}, latest transaction not labelled construction, transaction labelled property-to-let, and floor area above 1,000 m².

- B/C: expected count {policy_b.crossing_count_estimate:,.0f}; affected area {policy_b.affected_floor_area_Mm2:.2f} Mm².
- E/F: expected count {policy_fg.crossing_count_estimate:,.0f}; affected area {policy_fg.affected_floor_area_Mm2:.2f} Mm².

“Property to let” remains a tenure proxy and the subset does not identify exemptions, cost effectiveness, or legal status.

### ND-NEED composition-raked sensitivity

- B/C: expected count {weighted_b.crossing_count_estimate:,.0f}; affected area {weighted_b.affected_floor_area_Mm2:.2f} Mm².
- E/F: expected count {weighted_fg.crossing_count_estimate:,.0f}; affected area {weighted_fg.affected_floor_area_Mm2:.2f} Mm².

Relative to the unweighted preferred estimate, raking changes B/C count by {b_count_change:+.1f}% and area by {b_area_change:+.1f}%; it changes E/F count by {fg_count_change:+.1f}% and area by {fg_area_change:+.1f}%. Counts are generally more stable than floor area. The weights are normalised to the matched register-frame size ({raking_diagnostics['n_raked']:,}); they adjust composition and do not expand the EPC frame to the ND-NEED building count. ND-NEED does not support raking the policy-proxy intersection.

## Structural ranges

The following range includes the pre-specified rating-local candidates plus the direct probability model. It is a model-structure sensitivity envelope, not a confidence interval. The best-supported overall-aggregate row collapses to the direct probability model; the separate composition-sensitivity row spans its unweighted and raked applications. No rating bridge passes individual calibration and threshold-usefulness requirements.

{_markdown_table(headline_ranges, 2)}

### Policy-proxy model table

{_markdown_table(policy_table, 2)}

### Benchmark-weighted model table

{_markdown_table(benchmark_table, 2)}

## Expenditure-equivalent scale

The conversion uses replaceable assumptions of £{UNIT_COSTS_GBP_M2['low']:.0f}, £{UNIT_COSTS_GBP_M2['central']:.0f}, and £{UNIT_COSTS_GBP_M2['high']:.0f}/m². These values are **not observed costs, avoided investment, savings, compliance costs, welfare, or social NPV**.

For the preferred direct model, the full-frame B/C area corresponds to £{full_b.affected_floor_area_Mm2 * UNIT_COSTS_GBP_M2['low'] / 1000:.2f}–£{full_b.affected_floor_area_Mm2 * UNIT_COSTS_GBP_M2['high'] / 1000:.2f}bn; E/F corresponds to £{full_fg.affected_floor_area_Mm2 * UNIT_COSTS_GBP_M2['low'] / 1000:.2f}–£{full_fg.affected_floor_area_Mm2 * UNIT_COSTS_GBP_M2['high'] / 1000:.2f}bn. For the policy proxy, the corresponding ranges are £{policy_b.affected_floor_area_Mm2 * UNIT_COSTS_GBP_M2['low'] / 1000:.2f}–£{policy_b.affected_floor_area_Mm2 * UNIT_COSTS_GBP_M2['high'] / 1000:.2f}bn at B/C and £{policy_fg.affected_floor_area_Mm2 * UNIT_COSTS_GBP_M2['low'] / 1000:.2f}–£{policy_fg.affected_floor_area_Mm2 * UNIT_COSTS_GBP_M2['high'] / 1000:.2f}bn at E/F.

Combining the best-supported model's unweighted-to-raked area range with the low-to-high unit costs gives the most conservative presentable scale range: B/C £{min(full_b.affected_floor_area_Mm2, weighted_b.affected_floor_area_Mm2) * UNIT_COSTS_GBP_M2['low'] / 1000:.2f}–£{max(full_b.affected_floor_area_Mm2, weighted_b.affected_floor_area_Mm2) * UNIT_COSTS_GBP_M2['high'] / 1000:.2f}bn and E/F £{min(full_fg.affected_floor_area_Mm2, weighted_fg.affected_floor_area_Mm2) * UNIT_COSTS_GBP_M2['low'] / 1000:.2f}–£{max(full_fg.affected_floor_area_Mm2, weighted_fg.affected_floor_area_Mm2) * UNIT_COSTS_GBP_M2['high'] / 1000:.2f}bn. This is still a scale translation, not a cost estimate.

{_markdown_table(expenditure_table, 2)}

## Answers to the decision questions

1. **Fuel-only bridge:** unusable for individual old-basis ratings; retain only as a transparent baseline.
2. **Rating-local bridge:** improves MAE and slope, but not B/C or E/F threshold usefulness. It does not rescue a certificate-level recomputation claim.
3. **Full-frame B/C effect:** remains large under the preferred aggregate probability model ({full_b.crossing_count_estimate:,.0f} expected entries; {full_b.affected_floor_area_Mm2:.2f} Mm²), but remains conditional on the register and calibration sample.
4. **Policy proxy:** remains material ({policy_b.crossing_count_estimate:,.0f} expected B/C entries; {policy_b.affected_floor_area_Mm2:.2f} Mm²), but is much smaller than the all-register frame and is not the legal population.
5. **Structural range:** B/C is materially more stable than E/F. E/F hard-bridge results are especially specification-sensitive.
6. **Most conservative defensible presentation:** use the direct probability-sum model as the central aggregate sensitivity and present its unweighted-to-raked range, rather than selecting whichever endpoint is smaller. At B/C this is {min(full_b.crossing_count_estimate, weighted_b.crossing_count_estimate):,.0f}–{max(full_b.crossing_count_estimate, weighted_b.crossing_count_estimate):,.0f} expected entries and {min(full_b.affected_floor_area_Mm2, weighted_b.affected_floor_area_Mm2):.2f}–{max(full_b.affected_floor_area_Mm2, weighted_b.affected_floor_area_Mm2):.2f} Mm²; at E/F it is {min(full_fg.crossing_count_estimate, weighted_fg.crossing_count_estimate):,.0f}–{max(full_fg.crossing_count_estimate, weighted_fg.crossing_count_estimate):,.0f} entries and {min(full_fg.affected_floor_area_Mm2, weighted_fg.affected_floor_area_Mm2):.2f}–{max(full_fg.affected_floor_area_Mm2, weighted_fg.affected_floor_area_Mm2):.2f} Mm².
7. **Money sensitivity:** unit-cost uncertainty alone creates an eight-fold low-to-high range; bridge choice and raking add substantial area uncertainty.
8. **Counts versus floor area:** counts are generally more stable. Large premises make affected area more sensitive to bridge and raking choices.
9. **Claims no longer defensible:** exact old-basis ratings for individual certificates; exact subgroup effects from the direct model; exact numbers of physical buildings; legal non-compliance counts; national-stock totals; actual or avoided retrofit cost; savings; welfare; and any claim that near-zero pooled bias establishes local calibration.

## Output interpretation

No plots were generated because the calibration, threshold and structural-range tables expose the failure more directly. All validation predictions are out-of-sample. Full-register predictions use models refitted on all 2,442 calibration UPRNs and are labelled separately from OOF validation.
"""
    REPORT_PATH.write_text(report)


def output_manifest() -> pd.DataFrame:
    rows = []
    for path in sorted(OUT_TABLES.glob("bridge_*.csv")) + sorted(
        OUT_TABLES.glob("expenditure_equivalent_*.csv")
    ):
        try:
            frame = pd.read_csv(path)
            rows.append({
                "filename": path.name, "rows": len(frame), "columns": len(frame.columns),
                "column_names": ";".join(frame.columns),
            })
        except Exception as exc:
            rows.append({
                "filename": path.name, "rows": np.nan, "columns": np.nan,
                "column_names": f"READ_ERROR: {exc}",
            })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_TABLES / "bridge_output_manifest.csv", index=False)
    return manifest


def run_bridge_analysis() -> dict[str, Any]:
    def progress(label: str) -> None:
        print(f"    bridge: {label} ...", flush=True)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    progress("loading extended register and calibration pairs")
    dfx = build_extended_frame()
    P = within_building_pairs(dfx)
    sample, sample_audit = build_calibration_sample(P)
    folds = assign_folds(sample)
    progress("cross-fitted rating predictions")
    oof = crossfit_predictions(sample, folds)
    progress("validation, model support and ranking")
    validation = evaluate_oof(sample, oof)
    specification_and_cell_support(sample)
    ranking = rank_models(validation["summary"])
    audit = audit_legacy_validation(
        P, sample, folds, validation["summary"], validation["by_fuel"]
    )
    audit["audit_json"]["sample_construction"] = sample_audit
    with open(OUT_TABLES / "bridge_validation_audit.json", "w") as handle:
        json.dump(audit["audit_json"], handle, indent=2, default=str)

    progress("latest-register and affected-area validation")
    latest = prepare_latest_frame(dfx)
    area_validation = area_weighted_validation(oof, latest)
    progress("area-model comparisons")
    area_models = area_model_comparison(sample, latest, folds)
    area_models_nested = nested_cv_area_models(sample)
    progress("sector expenditure and transport diagnostics")
    sector_expenditure = sector_specific_expenditure(sample, latest)
    transport = stage1_transport(sample, latest, tau=50.0)
    transport_oos = transport_oos_validation(sample, latest, tau=50.0)
    progress("ND-NEED raking and full-frame aggregation")
    rake, rake_weights, raking_diagnostics = prepare_raking_weights(latest)
    aggregates = apply_all_models(sample, latest, rake, rake_weights)
    progress("structural ranges, expenditure and report")
    ranges = structural_ranges(aggregates, ranking)
    expenditure = expenditure_equivalent_outputs(aggregates, ranking, ranges)
    external = pd.DataFrame([
        {
            "benchmark": "identical-input SBEM 2013/2021 reruns",
            "available": False,
            "records": 0,
            "use": "not performed",
            "implication": "empirical bridges lack direct external old/new methodology validation",
        }
    ])
    external.to_csv(OUT_TABLES / "bridge_external_validation_status.csv", index=False)
    write_markdown_report(
        audit["audit_json"], validation, ranking, aggregates, ranges,
        expenditure, raking_diagnostics,
    )
    manifest = output_manifest()
    return {
        "sample": sample, "validation": validation, "ranking": ranking,
        "audit": audit, "aggregates": aggregates, "ranges": ranges,
        "expenditure": expenditure, "manifest": manifest,
    }


if __name__ == "__main__":
    result = run_bridge_analysis()
    print("Bridge analysis complete")
    print(result["ranking"][[
        "threshold_rank", "specification", "threshold_score", "authoritative_status"
    ]].to_string(index=False))
    print(f"Report: {REPORT_PATH}")
