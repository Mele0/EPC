"""MAE-rescue analysis for empirical old-basis EPC rating bridges.

This module is intentionally isolated from :mod:`bridge_analysis`.  It writes
only ``mae_rescue_*`` outputs and ``bridge_MAE_rescue.md``; the authoritative
``bridge_*`` output family is read-only.

The objective gates are fixed in code before model execution:

* rating MAE <= 18 points and calibration slope >= 0.8;
* B/C and E/F balanced accuracy >= 0.70;
* aggregate probability error within +/-1 percentage point;
* maximum absolute subgroup probability error <= 10 percentage points for
  reported cells with n >= 50;
* at least 75% of the baseline calibration sample and 90% formal register
  application coverage for an unrestricted individual-rating claim.

Run with::

    python -m epc_artefact.mae_rescue
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any, Callable

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer, StandardScaler

from .bridge_analysis import (
    FUELS,
    THRESHOLDS,
    _fit_binary_logit,
    _training_balanced_cutoff,
    build_calibration_sample,
    prepare_latest_frame,
)
from .config import CUT, OUT_TABLES, RANDOM_SEED, UNIFIED_PARQUET, ensure_dirs
from .data import broad_band


REPORT_PATH = OUT_TABLES / "bridge_MAE_rescue.md"
REC_LONG_PARQUET = UNIFIED_PARQUET.with_name("recommendations_long_clean.parquet")
RAW_CERT_DIR = Path(os.environ.get(
    "EPC_RAW_DATA_DIR", str(UNIFIED_PARQUET.parent / "raw"))).expanduser().resolve()

RATING_MAE_GATE = 18.0
RATING_SLOPE_GATE = 0.80
THRESHOLD_BALANCED_ACCURACY_GATE = 0.70
AGGREGATE_PROBABILITY_ERROR_GATE_PP = 1.0
SUBGROUP_CALIBRATION_GATE_PP = 10.0
MIN_SAMPLE_RETENTION = 0.75
MIN_APPLICATION_COVERAGE = 0.90
MIN_SUBGROUP_N = 50
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3


def validate_mae_rescue_inputs() -> None:
    """Fail early unless the Supplementary 1.18 inputs are fully provisioned."""
    absent = [path for path in (UNIFIED_PARQUET, REC_LONG_PARQUET) if not path.is_file()]
    if absent:
        raise FileNotFoundError(
            "The Supplementary 1.18 MAE-rescue stage requires both the wide unified register and "
            "the recommendation-text parquet. Missing: "
            + ", ".join(str(path) for path in absent))

    register_required = {
        "uprn", "inspection_year", "invalid_asset_rating", "extreme_asset_rating",
        "missing_asset_rating", "target_emissions", "other_fuel_desc",
        "special_energy_uses", "renewable_sources", *PAIR_BASE_FIELDS,
        *SAME_METHOD_FIELDS,
    }
    register_columns = set(pq.ParquetFile(UNIFIED_PARQUET).schema_arrow.names)
    missing_register = sorted(register_required - register_columns)
    recommendation_columns = set(pq.ParquetFile(REC_LONG_PARQUET).schema_arrow.names)
    missing_recommendations = sorted(
        {"certificate_number", "recommendation"} - recommendation_columns)
    if missing_register or missing_recommendations:
        details = []
        if missing_register:
            details.append(f"wide register missing {missing_register}")
        if missing_recommendations:
            details.append(f"recommendation parquet missing {missing_recommendations}")
        raise ValueError("MAE-rescue input schema is incomplete: " + "; ".join(details))


RECOMMENDATION_CODE_COLUMNS = [
    f"recommendation_code_count_{code}"
    for code in [
        "EPC-C1", "EPC-C2", "EPC-C3", "EPC-E1", "EPC-E2", "EPC-E3",
        "EPC-E4", "EPC-E5", "EPC-E6", "EPC-E7", "EPC-E8", "EPC-F1",
        "EPC-F2", "EPC-F3", "EPC-F4", "EPC-F5", "EPC-F6", "EPC-H1",
        "EPC-H2", "EPC-H3", "EPC-H4", "EPC-H5", "EPC-H6", "EPC-H7",
        "EPC-H8", "EPC-L1", "EPC-L2", "EPC-L3", "EPC-L4", "EPC-L5",
        "EPC-L6", "EPC-L7", "EPC-R1", "EPC-R2", "EPC-R3", "EPC-R4",
        "EPC-R5", "EPC-V1", "EPC-W1", "EPC-W2", "EPC-W3", "EPC-W4",
        "USER",
    ]
]

PAIR_BASE_FIELDS = [
    "certificate_number", "asset_rating", "inspection_date", "lodgement_date",
    "main_heating_fuel_clean", "main_heating_fuel_raw", "floor_area",
    "n_recommendations", "unique_recommendation_codes", "property_type_clean",
    "aircon_present_clean", "building_environment_clean", "building_level",
    "new_build_benchmark", "existing_stock_benchmark", "standard_emissions",
    "target_emissions", "typical_emissions", "building_emissions",
    "primary_energy_value", "aircon_kw_rating", "estimated_aircon_kw_rating",
    "ac_inspection_commissioned", "transaction_type_clean", "uprn_source",
    "days_between_inspection_and_lodgement", "n_short_payback",
    "n_medium_payback", "n_long_payback", "text_length_total", "text_length_mean",
] + RECOMMENDATION_CODE_COLUMNS

SAME_METHOD_FIELDS = [
    "certificate_number", "asset_rating", "inspection_date", "lodgement_date",
    "main_heating_fuel_clean", "floor_area", "n_recommendations",
    "unique_recommendation_codes", "property_type_clean", "aircon_present_clean",
    "building_environment_clean", "building_level", "new_build_benchmark",
    "existing_stock_benchmark",
]

TEXT_KEYWORDS = {
    "heat_pump": "heat pump",
    "direct_electric": "direct electric|electric resistance|panel heater|storage heater",
    "boiler": "boiler",
    "lighting": "light|lamp",
    "insulation": "insulat",
    "glazing": "glaz|window",
    "solar": "solar|photovoltaic|pv ",
    "controls": "control|thermostat|bms",
    "ventilation": "ventilat|air handling",
    "cooling": "air condition|cooling|chiller",
    "hot_water": "hot water|dhw",
}


@dataclass(frozen=True)
class SampleDefinition:
    name: str
    pair_source: str
    description: str
    mask: Callable[[pd.DataFrame], pd.Series]
    selection_uses_old_outcome: bool = False
    application_scope: str = "all post-revision register entries"


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _valid_register_sql() -> str:
    return f"""
        SELECT *
        FROM read_parquet('{UNIFIED_PARQUET}')
        WHERE uprn IS NOT NULL
          AND inspection_date IS NOT NULL
          AND inspection_year BETWEEN 2012 AND 2025
          AND NOT invalid_asset_rating
          AND NOT extreme_asset_rating
          AND NOT missing_asset_rating
          AND asset_rating > 0
          AND floor_area > 0
    """


def _pair_projection(alias: str, suffix: str, fields: list[str]) -> str:
    return ",\n".join(
        f"{alias}.{_q(field)} AS {_q(field + '_' + suffix)}" for field in fields
    )


def extract_cross_method_pairs(mode: str) -> pd.DataFrame:
    """Extract either first/latest or nearest pre/post pairs from the raw register."""
    fields = PAIR_BASE_FIELDS
    projection_pre = _pair_projection("pre", "pre", fields)
    projection_post = _pair_projection("post", "post", fields)
    valid = _valid_register_sql()
    if mode == "first_latest":
        sql = f"""
        WITH valid AS ({valid}), ordered AS (
            SELECT *,
                   row_number() OVER (
                       PARTITION BY uprn ORDER BY inspection_date, lodgement_date, certificate_number
                   ) AS rn_first,
                   row_number() OVER (
                       PARTITION BY uprn ORDER BY inspection_date DESC, lodgement_date DESC, certificate_number DESC
                   ) AS rn_last
            FROM valid
        ), pre AS (SELECT * FROM ordered WHERE rn_first=1),
           post AS (SELECT * FROM ordered WHERE rn_last=1)
        SELECT pre.uprn, {projection_pre}, {projection_post}
        FROM pre JOIN post USING (uprn)
        WHERE pre.inspection_date < TIMESTAMP '2022-06-15'
          AND post.inspection_date >= TIMESTAMP '2022-06-15'
        """
    elif mode == "nearest":
        sql = f"""
        WITH valid AS ({valid}), pre_ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY uprn ORDER BY inspection_date DESC, lodgement_date DESC, certificate_number DESC
            ) rn
            FROM valid WHERE inspection_date < TIMESTAMP '2022-06-15'
        ), post_ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY uprn ORDER BY inspection_date, lodgement_date, certificate_number
            ) rn
            FROM valid WHERE inspection_date >= TIMESTAMP '2022-06-15'
        ), pre AS (SELECT * FROM pre_ranked WHERE rn=1),
           post AS (SELECT * FROM post_ranked WHERE rn=1)
        SELECT pre.uprn, {projection_pre}, {projection_post}
        FROM pre JOIN post USING (uprn)
        """
    else:
        raise ValueError(mode)
    frame = duckdb.connect().execute(sql).fetchdf()
    return prepare_pair_features(add_recommendation_text_features(frame))


def add_recommendation_text_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add transparent keyword counts from the registered recommendation text."""
    selected = frame[["certificate_number_post"]].drop_duplicates().rename(
        columns={"certificate_number_post": "certificate_number"}
    )
    con = duckdb.connect()
    con.register("selected_certificates", selected)
    expressions = []
    for label, pattern in TEXT_KEYWORDS.items():
        escaped = pattern.replace("'", "''")
        expressions.append(
            f"sum(CASE WHEN regexp_matches(lower(coalesce(recommendation,'')), '{escaped}') "
            f"THEN 1 ELSE 0 END) AS rec_text_{label}"
        )
    query = f"""
        SELECT r.certificate_number, {', '.join(expressions)}
        FROM read_parquet('{REC_LONG_PARQUET}') r
        JOIN selected_certificates s USING (certificate_number)
        GROUP BY r.certificate_number
    """
    features = con.execute(query).fetchdf()
    out = frame.merge(
        features, left_on="certificate_number_post", right_on="certificate_number",
        how="left",
    ).drop(columns=["certificate_number"])
    keyword_columns = [f"rec_text_{label}" for label in TEXT_KEYWORDS]
    out[keyword_columns] = out[keyword_columns].fillna(0)
    return out


def _fuel_group(values: pd.Series) -> np.ndarray:
    text = values.fillna("").astype(str).str.lower()
    return np.where(text.str.contains("electr"), "Electric",
                    np.where(text.str.contains("gas"), "Gas", "Other"))


def prepare_pair_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["pre"] = pd.to_numeric(out.asset_rating_pre, errors="coerce")
    out["post"] = pd.to_numeric(out.asset_rating_post, errors="coerce")
    out["fuel"] = _fuel_group(out.main_heating_fuel_clean_post)
    out["fuel_pre"] = _fuel_group(out.main_heating_fuel_clean_pre)
    out["pre_band"] = broad_band(out.pre.to_numpy())
    out["post_band"] = broad_band(out.post.to_numpy())
    out["gap_years"] = (
        pd.to_datetime(out.inspection_date_post) - pd.to_datetime(out.inspection_date_pre)
    ).dt.days / 365.25
    out["floor_area"] = pd.to_numeric(out.floor_area_post, errors="coerce")
    out["log_floor_area"] = np.log1p(out.floor_area.clip(lower=0))
    out["sector"] = out.property_type_clean_post.fillna("Missing")
    out["ac"] = out.aircon_present_clean_post.fillna("Missing")
    out["environment"] = out.building_environment_clean_post.fillna("Missing")
    out["main_fuel_raw"] = out.main_heating_fuel_raw_post.fillna("Missing")
    out["transaction"] = out.transaction_type_clean_post.fillna("Missing")
    out["uprn_source_feature"] = out.uprn_source_post.fillna("Missing")
    out["floor_stable_2pct"] = (
        (out.floor_area_post - out.floor_area_pre).abs()
        / out.floor_area_pre.replace(0, np.nan)
    ) < 0.02
    out["fuel_stable"] = out.fuel == out.fuel_pre
    out["recommendation_count_stable"] = out.n_recommendations_post == out.n_recommendations_pre
    out["recommendation_codes_stable"] = (
        out.unique_recommendation_codes_post.fillna("")
        == out.unique_recommendation_codes_pre.fillna("")
    )
    out["sector_stable"] = (
        out.property_type_clean_post.fillna("") == out.property_type_clean_pre.fillna("")
    )
    out["ac_stable"] = (
        out.aircon_present_clean_post.fillna("") == out.aircon_present_clean_pre.fillna("")
    )
    out["environment_stable"] = (
        out.building_environment_clean_post.fillna("")
        == out.building_environment_clean_pre.fillna("")
    )
    out["building_level_stable"] = (
        out.building_level_post.fillna(-999999) == out.building_level_pre.fillna(-999999)
    )
    out["benchmark_stable"] = (
        (out.new_build_benchmark_post.fillna(-999999)
         == out.new_build_benchmark_pre.fillna(-999999))
        & (out.existing_stock_benchmark_post.fillna(-999999)
           == out.existing_stock_benchmark_pre.fillna(-999999))
    )
    return out[(out.pre > 0) & (out.post > 0) & (out.gap_years > 0)].reset_index(drop=True)


def extract_same_method_pairs() -> pd.DataFrame:
    fields = SAME_METHOD_FIELDS
    lagged = ",\n".join(
        f"lag({_q(field)}) OVER w AS {_q(field + '_earlier')}" for field in fields
    )
    later = ",\n".join(f"{_q(field)} AS {_q(field + '_later')}" for field in fields)
    sql = f"""
    WITH valid AS ({_valid_register_sql()}), lagged AS (
        SELECT uprn, {later}, {lagged}
        FROM valid
        WINDOW w AS (
            PARTITION BY uprn ORDER BY inspection_date, lodgement_date, certificate_number
        )
    )
    SELECT * FROM lagged
    WHERE inspection_date_earlier IS NOT NULL
      AND inspection_date_later > inspection_date_earlier
      AND ((inspection_date_earlier < TIMESTAMP '2022-06-15'
            AND inspection_date_later < TIMESTAMP '2022-06-15')
        OR (inspection_date_earlier >= TIMESTAMP '2022-06-15'
            AND inspection_date_later >= TIMESTAMP '2022-06-15'))
    """
    out = duckdb.connect().execute(sql).fetchdf()
    out["earlier"] = out.asset_rating_earlier.astype(float)
    out["later"] = out.asset_rating_later.astype(float)
    out["gap_years"] = (
        pd.to_datetime(out.inspection_date_later) - pd.to_datetime(out.inspection_date_earlier)
    ).dt.days / 365.25
    out["period"] = np.where(
        pd.to_datetime(out.inspection_date_later) < CUT, "pre_pre", "post_post"
    )
    out["fuel"] = _fuel_group(out.main_heating_fuel_clean_later)
    out["earlier_band"] = broad_band(out.earlier.to_numpy())
    out["floor_stable_2pct"] = (
        (out.floor_area_later - out.floor_area_earlier).abs()
        / out.floor_area_earlier.replace(0, np.nan)
    ) < 0.02
    out["fuel_stable"] = (
        out.main_heating_fuel_clean_later == out.main_heating_fuel_clean_earlier
    )
    out["recommendation_count_stable"] = (
        out.n_recommendations_later == out.n_recommendations_earlier
    )
    out["recommendation_codes_stable"] = (
        out.unique_recommendation_codes_later.fillna("")
        == out.unique_recommendation_codes_earlier.fillna("")
    )
    for name, column in [
        ("sector_stable", "property_type_clean"),
        ("ac_stable", "aircon_present_clean"),
        ("environment_stable", "building_environment_clean"),
    ]:
        out[name] = (
            out[f"{column}_later"].fillna("") == out[f"{column}_earlier"].fillna("")
        )
    out["building_level_stable"] = (
        out.building_level_later.fillna(-999999)
        == out.building_level_earlier.fillna(-999999)
    )
    out["benchmark_stable"] = (
        (out.new_build_benchmark_later.fillna(-999999)
         == out.new_build_benchmark_earlier.fillna(-999999))
        & (out.existing_stock_benchmark_later.fillna(-999999)
           == out.existing_stock_benchmark_earlier.fillna(-999999))
    )
    return out.reset_index(drop=True)


def assign_group_folds(groups: pd.Series, n_folds: int = N_OUTER_FOLDS,
                       seed: int = RANDOM_SEED) -> np.ndarray:
    unique = pd.Series(groups.astype(str).unique()).sort_values().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    mapping = {group: i % n_folds for i, group in enumerate(unique)}
    return groups.astype(str).map(mapping).to_numpy(dtype=int)


def rating_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    slope = intercept = np.nan
    if len(actual) >= 2 and np.unique(predicted).size > 1:
        slope, intercept = np.polyfit(predicted, actual, 1)
    return {
        "n": int(len(actual)),
        "MAE_pts": float(np.mean(np.abs(error))),
        "median_absolute_error_pts": float(np.median(np.abs(error))),
        "bias_pts": float(np.mean(error)),
        "median_error_pts": float(np.median(error)),
        "error_p10_pts": float(np.quantile(error, 0.10)),
        "error_p50_pts": float(np.quantile(error, 0.50)),
        "error_p90_pts": float(np.quantile(error, 0.90)),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "band_accuracy": float(np.mean(broad_band(actual) == broad_band(predicted))),
    }


def threshold_metrics(actual_rating: np.ndarray, probability: np.ndarray,
                      threshold: float, hard: np.ndarray | None = None) -> dict[str, float | int]:
    actual_rating = np.asarray(actual_rating, dtype=float)
    # The positive classes requested for the paper are B-or-better at B/C and
    # F/G at E/F.  Higher numeric ratings are worse, so the inequalities differ.
    actual = actual_rating <= threshold if threshold == THRESHOLDS["B_C"] else actual_rating > threshold
    probability = np.asarray(probability, dtype=float)
    predicted = probability >= 0.5 if hard is None else np.asarray(hard, dtype=bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "sensitivity": float(sensitivity), "specificity": float(specificity),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "PPV": float(ppv), "NPV": float(npv),
        "agreement": float(np.mean(actual == predicted)),
        "aggregate_probability_error_pp": float(
            100 * (probability.mean() - actual.mean())
        ),
        "brier_score": float(np.mean((probability - actual.astype(float)) ** 2)),
    }


def _make_preprocessor(numeric: list[str], categorical: list[str],
                       spline: bool = False, ordinal: bool = False) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []
    numeric = list(dict.fromkeys(numeric))
    categorical = list(dict.fromkeys(categorical))
    if spline and "post" in numeric:
        transformers.append((
            "post_spline",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("spline", SplineTransformer(n_knots=5, degree=2, include_bias=False)),
                ("scale", StandardScaler()),
            ]),
            ["post"],
        ))
        numeric = [column for column in numeric if column != "post"]
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        encoder: Any = (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                           encoded_missing_value=-1)
            if ordinal else
            OneHotEncoder(handle_unknown="ignore", min_frequency=5)
        )
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", encoder),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.2)


class BridgeRegressor(BaseEstimator, RegressorMixin):
    """Wrap an estimator so inner CV scores the final old-rating prediction."""

    def __init__(self, base_estimator: Any, mode: str = "direct"):
        self.base_estimator = base_estimator
        self.mode = mode

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.estimator_ = clone(self.base_estimator)
        post = np.asarray(X["post"], dtype=float)
        y = np.asarray(y, dtype=float)
        if self.mode == "additive":
            target = y - post
        elif self.mode == "log_ratio":
            target = np.log(np.clip(y, 1e-6, None) / np.clip(post, 1e-6, None))
        else:
            target = y
        self.estimator_.fit(X, target)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(self.estimator_.predict(X), dtype=float)
        post = np.asarray(X["post"], dtype=float)
        if self.mode == "additive":
            return post + raw
        if self.mode == "log_ratio":
            return post * np.exp(np.clip(raw, -3, 3))
        return raw


BASIC_NUMERIC = ["post"]
BASIC_CATEGORICAL = ["fuel"]
DESCRIPTOR_NUMERIC = [
    "post", "log_floor_area", "aircon_kw_rating_post",
    "estimated_aircon_kw_rating_post", "ac_inspection_commissioned_post",
    "building_level_post", "new_build_benchmark_post", "existing_stock_benchmark_post",
    "post_inspection_year",
]
DESCRIPTOR_CATEGORICAL = [
    "fuel", "main_fuel_raw", "sector", "ac", "environment", "transaction",
    "uprn_source_feature", "post_band",
]
COMPONENT_NUMERIC = [
    "post", "building_emissions_post", "standard_emissions_post",
    "target_emissions_post", "typical_emissions_post", "primary_energy_value_post",
]
RECOMMENDATION_NUMERIC = [
    "post", "n_recommendations_post", "n_short_payback_post", "n_medium_payback_post",
    "n_long_payback_post", "text_length_total_post", "text_length_mean_post",
] + [f"{column}_post" for column in RECOMMENDATION_CODE_COLUMNS] + [
    f"rec_text_{label}" for label in TEXT_KEYWORDS
]
RICH_NUMERIC = list(dict.fromkeys(
    DESCRIPTOR_NUMERIC + COMPONENT_NUMERIC + RECOMMENDATION_NUMERIC
    + ["days_between_inspection_and_lodgement_post"]
))
RICH_CATEGORICAL = DESCRIPTOR_CATEGORICAL


def _add_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["post_inspection_year"] = pd.to_datetime(out.inspection_date_post).dt.year
    return out


def nested_regression_oof(frame: pd.DataFrame, numeric: list[str], categorical: list[str],
                          model: str, mode: str = "direct") -> tuple[np.ndarray, list[dict[str, Any]]]:
    data = _add_model_features(frame)
    folds = assign_group_folds(data.uprn)
    predictions = np.empty(len(data), dtype=float)
    selected_parameters: list[dict[str, Any]] = []

    if model in {"ridge", "spline_ridge"}:
        preprocessor = _make_preprocessor(numeric, categorical, spline=model == "spline_ridge")
        base = Pipeline([("prep", preprocessor), ("model", Ridge())])
        estimator = BridgeRegressor(base, mode=mode)
        grid = {"base_estimator__model__alpha": [0.1, 1.0, 10.0, 100.0]}
    elif model == "hist_gradient_boosting":
        preprocessor = _make_preprocessor(numeric, categorical, ordinal=True)
        base = Pipeline([
            ("prep", preprocessor),
            ("model", HistGradientBoostingRegressor(
                loss="absolute_error", random_state=RANDOM_SEED, max_iter=300,
            )),
        ])
        estimator = BridgeRegressor(base, mode=mode)
        grid = {
            "base_estimator__model__max_leaf_nodes": [7, 15],
            "base_estimator__model__min_samples_leaf": [20, 50],
            "base_estimator__model__l2_regularization": [1.0, 10.0],
        }
    elif model == "random_forest":
        preprocessor = _make_preprocessor(numeric, categorical, ordinal=True)
        base = Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=250, random_state=RANDOM_SEED, n_jobs=-1,
            )),
        ])
        estimator = BridgeRegressor(base, mode=mode)
        grid = {
            "base_estimator__model__max_depth": [5, None],
            "base_estimator__model__min_samples_leaf": [5, 20],
        }
    else:
        raise ValueError(model)

    for fold in range(N_OUTER_FOLDS):
        train = data[folds != fold]
        test = data[folds == fold]
        inner = KFold(n_splits=N_INNER_FOLDS, shuffle=True,
                      random_state=RANDOM_SEED + fold)
        search = GridSearchCV(
            estimator, grid, scoring="neg_mean_absolute_error", cv=inner,
            n_jobs=-1, refit=True,
        )
        search.fit(train, train.pre.to_numpy(dtype=float))
        predictions[folds == fold] = search.predict(test)
        selected_parameters.append({"fold": fold, **search.best_params_})
    return predictions, selected_parameters


def fuel_baseline_oof(frame: pd.DataFrame, mode: str) -> np.ndarray:
    folds = assign_group_folds(frame.uprn)
    predictions = np.empty(len(frame), dtype=float)
    for fold in range(N_OUTER_FOLDS):
        train, test = frame[folds != fold], frame[folds == fold]
        if mode == "multiplicative":
            parameter = (train.pre / train.post).groupby(train.fuel).median()
            predictions[folds == fold] = test.post * test.fuel.map(parameter)
        elif mode == "additive":
            parameter = (train.pre - train.post).groupby(train.fuel).mean()
            predictions[folds == fold] = test.post + test.fuel.map(parameter)
        elif mode == "log_ratio":
            parameter = np.log(train.pre / train.post).groupby(train.fuel).median()
            predictions[folds == fold] = test.post * np.exp(test.fuel.map(parameter))
        else:
            raise ValueError(mode)
    return predictions


def isotonic_oof(frame: pd.DataFrame) -> np.ndarray:
    folds = assign_group_folds(frame.uprn)
    predictions = np.empty(len(frame), dtype=float)
    for fold in range(N_OUTER_FOLDS):
        train, test = frame[folds != fold], frame[folds == fold]
        for fuel in FUELS:
            tr = train[train.fuel == fuel]
            mask = (folds == fold) & (frame.fuel.to_numpy() == fuel)
            model = IsotonicRegression(out_of_bounds="clip", y_min=0)
            model.fit(tr.post, tr.pre)
            predictions[mask] = model.predict(frame.loc[mask, "post"])
    return predictions


def component_bridge_oof(frame: pd.DataFrame) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Predict old BER and SER ratios separately, then reconstruct AR=50*BER/SER."""
    data = _add_model_features(frame)
    folds = assign_group_folds(data.uprn)
    predictions = np.empty(len(data), dtype=float)
    selections: list[dict[str, Any]] = []
    numeric = RICH_NUMERIC
    categorical = RICH_CATEGORICAL

    def estimator(alpha: float) -> Pipeline:
        return Pipeline([
            ("prep", _make_preprocessor(numeric, categorical, spline=True)),
            ("model", Ridge(alpha=alpha)),
        ])

    for fold in range(N_OUTER_FOLDS):
        train = data[folds != fold].reset_index(drop=True)
        test = data[folds == fold]
        inner = KFold(n_splits=N_INNER_FOLDS, shuffle=True,
                      random_state=RANDOM_SEED + fold)
        scores: dict[float, list[float]] = {alpha: [] for alpha in [0.1, 1.0, 10.0, 100.0]}
        for train_index, validation_index in inner.split(train):
            inner_train = train.iloc[train_index]
            validation = train.iloc[validation_index]
            for alpha in scores:
                ber_model, ser_model = estimator(alpha), estimator(alpha)
                ber_target = np.log(
                    inner_train.building_emissions_pre.clip(lower=1e-6)
                    / inner_train.building_emissions_post.clip(lower=1e-6)
                )
                ser_target = np.log(
                    inner_train.standard_emissions_pre.clip(lower=1e-6)
                    / inner_train.standard_emissions_post.clip(lower=1e-6)
                )
                ber_model.fit(inner_train, ber_target)
                ser_model.fit(inner_train, ser_target)
                ber = validation.building_emissions_post * np.exp(
                    np.clip(ber_model.predict(validation), -3, 3)
                )
                ser = validation.standard_emissions_post * np.exp(
                    np.clip(ser_model.predict(validation), -3, 3)
                )
                reconstructed = 50 * ber / np.clip(ser, 1e-6, None)
                scores[alpha].append(float(np.mean(np.abs(reconstructed - validation.pre))))
        best_alpha = min(scores, key=lambda alpha: np.mean(scores[alpha]))
        ber_model, ser_model = estimator(best_alpha), estimator(best_alpha)
        ber_model.fit(
            train,
            np.log(train.building_emissions_pre.clip(lower=1e-6)
                   / train.building_emissions_post.clip(lower=1e-6)),
        )
        ser_model.fit(
            train,
            np.log(train.standard_emissions_pre.clip(lower=1e-6)
                   / train.standard_emissions_post.clip(lower=1e-6)),
        )
        ber = test.building_emissions_post * np.exp(
            np.clip(ber_model.predict(test), -3, 3)
        )
        ser = test.standard_emissions_post * np.exp(
            np.clip(ser_model.predict(test), -3, 3)
        )
        predictions[folds == fold] = 50 * ber / np.clip(ser, 1e-6, None)
        selections.append({"fold": fold, "alpha": best_alpha})
    return predictions, selections


def residual_distribution_oof(frame: pd.DataFrame) -> tuple[
        np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    """Nested OOF point model plus empirical training-residual probabilities."""
    data = _add_model_features(frame)
    folds = assign_group_folds(data.uprn)
    predictions = np.empty(len(data), dtype=float)
    probabilities = {name: np.empty(len(data), dtype=float) for name in THRESHOLDS}
    selections: list[dict[str, Any]] = []
    base_pipeline = Pipeline([
        ("prep", _make_preprocessor(RICH_NUMERIC, RICH_CATEGORICAL, spline=True)),
        ("model", Ridge()),
    ])
    estimator = BridgeRegressor(base_pipeline, mode="direct")
    grid = {"base_estimator__model__alpha": [0.1, 1.0, 10.0, 100.0]}
    for fold in range(N_OUTER_FOLDS):
        train = data[folds != fold].reset_index(drop=True)
        test = data[folds == fold]
        inner = KFold(n_splits=N_INNER_FOLDS, shuffle=True,
                      random_state=RANDOM_SEED + fold)
        search = GridSearchCV(
            estimator, grid, scoring="neg_mean_absolute_error", cv=inner,
            n_jobs=-1, refit=True,
        )
        search.fit(train, train.pre)
        test_prediction = search.predict(test)
        predictions[folds == fold] = test_prediction

        # Residuals are themselves generated out-of-fold entirely inside the
        # outer training set; the outer test fold never enters this calibration.
        inner_oof = cross_val_predict(
            clone(search.best_estimator_), train, train.pre,
            cv=inner, n_jobs=-1, method="predict",
        )
        residual_frame = pd.DataFrame({
            "fuel": train.fuel.to_numpy(),
            "post_band": train.post_band.to_numpy(),
            "residual": train.pre.to_numpy(dtype=float) - inner_oof,
        })
        for threshold_name, threshold in THRESHOLDS.items():
            output = []
            for point, fuel, band in zip(test_prediction, test.fuel, test.post_band):
                pool = residual_frame[
                    (residual_frame.fuel == fuel) & (residual_frame.post_band == band)
                ].residual
                if len(pool) < 30:
                    pool = residual_frame[residual_frame.fuel == fuel].residual
                if len(pool) < 30:
                    pool = residual_frame.residual
                simulated = point + pool.to_numpy()
                output.append(float(np.mean(
                    simulated <= threshold
                    if threshold_name == "B_C" else simulated > threshold
                )))
            probabilities[threshold_name][folds == fold] = output
        selections.append({"fold": fold, **search.best_params_})
    return predictions, probabilities, selections


def simple_threshold_logit_oof(frame: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    folds = assign_group_folds(frame.uprn)
    output: dict[str, dict[str, np.ndarray]] = {}
    for threshold_name, threshold in THRESHOLDS.items():
        probability = np.empty(len(frame), dtype=float)
        hard = np.empty(len(frame), dtype=bool)
        for fold in range(N_OUTER_FOLDS):
            train, test = frame[folds != fold], frame[folds == fold]
            coefficients: dict[str, np.ndarray] = {}
            train_probability = np.empty(len(train), dtype=float)
            test_probability = np.empty(len(test), dtype=float)
            for fuel in FUELS:
                train_mask = train.fuel.to_numpy() == fuel
                test_mask = test.fuel.to_numpy() == fuel
                x = (train.loc[train_mask, "post"].to_numpy() - threshold) / 25.0
                old_rating = train.loc[train_mask, "pre"].to_numpy()
                y = (
                    old_rating <= threshold if threshold_name == "B_C"
                    else old_rating > threshold
                ).astype(float)
                coefficients[fuel] = _fit_binary_logit(x, y)
                beta = coefficients[fuel]
                train_probability[train_mask] = 1 / (
                    1 + np.exp(-(beta[0] + beta[1] * x))
                )
                xt = (test.loc[test_mask, "post"].to_numpy() - threshold) / 25.0
                test_probability[test_mask] = 1 / (
                    1 + np.exp(-(beta[0] + beta[1] * xt))
                )
            train_actual = (
                train.pre.to_numpy() <= threshold if threshold_name == "B_C"
                else train.pre.to_numpy() > threshold
            )
            cutoff = _training_balanced_cutoff(train_actual, train_probability)
            probability[folds == fold] = test_probability
            hard[folds == fold] = test_probability >= cutoff
        output[threshold_name] = {"probability": probability, "hard": hard}
    return output


def calibrated_threshold_spline_oof(frame: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    data = _add_model_features(frame)
    folds = assign_group_folds(data.uprn)
    output: dict[str, dict[str, np.ndarray]] = {}
    for threshold_name, threshold in THRESHOLDS.items():
        probability = np.empty(len(data), dtype=float)
        hard = np.empty(len(data), dtype=bool)
        for fold in range(N_OUTER_FOLDS):
            train = data[folds != fold].reset_index(drop=True)
            test = data[folds == fold]
            y = (
                train.pre.to_numpy() <= threshold if threshold_name == "B_C"
                else train.pre.to_numpy() > threshold
            ).astype(int)
            pipeline = Pipeline([
                ("prep", _make_preprocessor(RICH_NUMERIC, RICH_CATEGORICAL, spline=True)),
                ("model", LogisticRegression(
                    max_iter=3000, class_weight="balanced", solver="liblinear",
                    random_state=RANDOM_SEED,
                )),
            ])
            inner = KFold(n_splits=N_INNER_FOLDS, shuffle=True,
                          random_state=RANDOM_SEED + fold)
            search = GridSearchCV(
                pipeline, {"model__C": [0.1, 1.0, 10.0]},
                scoring="balanced_accuracy", cv=inner, n_jobs=-1, refit=True,
            )
            search.fit(train, y)
            calibrated = CalibratedClassifierCV(
                estimator=clone(search.best_estimator_), method="isotonic",
                cv=N_INNER_FOLDS, ensemble=False,
            )
            calibrated.fit(train, y)
            p = calibrated.predict_proba(test)[:, 1]
            probability[folds == fold] = p
            hard[folds == fold] = p >= 0.5
        output[threshold_name] = {"probability": probability, "hard": hard}
    return output


def ordinal_band_oof(frame: pd.DataFrame) -> tuple[
        np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    data = _add_model_features(frame)
    folds = assign_group_folds(data.uprn)
    prediction = np.empty(len(data), dtype=float)
    probabilities = {name: np.empty(len(data), dtype=float) for name in THRESHOLDS}
    selections: list[dict[str, Any]] = []
    labels = ["A/B", "C", "D", "E", "F/G"]
    for fold in range(N_OUTER_FOLDS):
        train = data[folds != fold].reset_index(drop=True)
        test = data[folds == fold]
        y = pd.Categorical(train.pre_band, categories=labels, ordered=True).codes
        pipeline = Pipeline([
            ("prep", _make_preprocessor(RICH_NUMERIC, RICH_CATEGORICAL, ordinal=True)),
            ("model", HistGradientBoostingClassifier(
                loss="log_loss", random_state=RANDOM_SEED, max_iter=300,
            )),
        ])
        inner = KFold(n_splits=N_INNER_FOLDS, shuffle=True,
                      random_state=RANDOM_SEED + fold)
        search = GridSearchCV(
            pipeline,
            {"model__max_leaf_nodes": [7, 15], "model__min_samples_leaf": [20, 50],
             "model__l2_regularization": [1.0, 10.0]},
            scoring="balanced_accuracy", cv=inner, n_jobs=-1, refit=True,
        )
        search.fit(train, y)
        class_probability = search.predict_proba(test)
        classes = search.best_estimator_.named_steps["model"].classes_.astype(int)
        expanded = np.zeros((len(test), len(labels)))
        expanded[:, classes] = class_probability
        band_medians = train.groupby("pre_band", observed=True).pre.median()
        fallback = train.pre.median()
        midpoint = np.array([band_medians.get(label, fallback) for label in labels])
        prediction[folds == fold] = expanded @ midpoint
        probabilities["B_C"][folds == fold] = expanded[:, 0]
        probabilities["E_F"][folds == fold] = expanded[:, 4]
        selections.append({"fold": fold, **search.best_params_})
    return prediction, probabilities, selections


def authoritative_enriched_sample(
        first_latest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Enrich the exact 2,442-row authoritative sample with post-only fields.

    The authoritative sample is the source of truth for membership and the
    old/post ratings.  The raw extraction contributes only certificate fields
    that existed at the time of the post-revision certificate.  This avoids a
    subtle tie-ordering discrepancy between DuckDB and pandas while ensuring
    that none of the new predictors uses the old outcome or a future record.
    """
    authoritative, audit = build_calibration_sample()
    raw = first_latest.drop_duplicates("uprn", keep="last").set_index("uprn")
    missing = set(authoritative.uprn.astype(str)) - set(raw.index.astype(str))
    if missing:
        raise AssertionError(f"Raw enrichment misses {len(missing)} authoritative UPRNs")
    out = raw.loc[authoritative.uprn.astype(str)].reset_index()
    aligned = authoritative.set_index("uprn").loc[out.uprn]
    out["pre"] = aligned.pre.to_numpy(dtype=float)
    out["post"] = aligned.post.to_numpy(dtype=float)
    out["fuel"] = aligned.fuel.to_numpy()
    out["pre_band"] = broad_band(out.pre.to_numpy())
    out["post_band"] = broad_band(out.post.to_numpy())
    out["gap_years"] = aligned.gap.to_numpy(dtype=float)
    out["authoritative_member"] = True
    if len(out) != audit["n_heldout_eligible"] or out.uprn.nunique() != len(out):
        raise AssertionError("Authoritative enriched sample membership is not one row per UPRN")
    return out.reset_index(drop=True), audit


def _hard_probabilities(prediction: np.ndarray) -> dict[str, np.ndarray]:
    rating = np.asarray(prediction, dtype=float)
    return {
        "B_C": (rating <= THRESHOLDS["B_C"]).astype(float),
        "E_F": (rating > THRESHOLDS["E_F"]).astype(float),
    }


def _evaluate_prediction(
        model_name: str,
        frame: pd.DataFrame,
        prediction: np.ndarray | None,
        probabilities: dict[str, np.ndarray],
        hard: dict[str, np.ndarray] | None = None,
        model_type: str = "rating",
        feature_set: str = "",
        diagnostics: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rating = rating_metrics(frame.pre, prediction) if prediction is not None else {
        "n": int(len(frame)), "MAE_pts": np.nan,
        "median_absolute_error_pts": np.nan, "bias_pts": np.nan,
        "median_error_pts": np.nan, "error_p10_pts": np.nan,
        "error_p50_pts": np.nan, "error_p90_pts": np.nan,
        "calibration_slope": np.nan, "calibration_intercept": np.nan,
        "band_accuracy": np.nan,
    }
    threshold_rows: list[dict[str, Any]] = []
    threshold_results: dict[str, dict[str, Any]] = {}
    for threshold_name, threshold in THRESHOLDS.items():
        result = threshold_metrics(
            frame.pre.to_numpy(), probabilities[threshold_name], threshold,
            None if hard is None else hard[threshold_name],
        )
        threshold_results[threshold_name] = result
        threshold_rows.append({
            "model": model_name,
            "model_type": model_type,
            "threshold": threshold_name,
            "threshold_rating": threshold,
            "n": len(frame),
            "positive_class": "B_or_better" if threshold_name == "B_C" else "F_or_G",
            "observed_positive_rate": float(
                (frame.pre <= threshold).mean() if threshold_name == "B_C"
                else (frame.pre > threshold).mean()
            ),
            "mean_predicted_probability": float(np.mean(probabilities[threshold_name])),
            **result,
        })
    row = {
        "model": model_name,
        "model_type": model_type,
        "feature_set": feature_set,
        **rating,
        "B_C_balanced_accuracy": threshold_results["B_C"]["balanced_accuracy"],
        "E_F_balanced_accuracy": threshold_results["E_F"]["balanced_accuracy"],
        "B_C_probability_error_pp": threshold_results["B_C"]["aggregate_probability_error_pp"],
        "E_F_probability_error_pp": threshold_results["E_F"]["aggregate_probability_error_pp"],
        "nested_validation": True,
        "all_predictions_out_of_fold": True,
        "parameter_selections_json": json.dumps(diagnostics, default=str),
    }
    row["rating_gate_pass"] = bool(
        pd.notna(row["MAE_pts"])
        and row["MAE_pts"] <= RATING_MAE_GATE
        and row["calibration_slope"] >= RATING_SLOPE_GATE
    )
    row["B_C_gate_pass"] = bool(
        row["B_C_balanced_accuracy"] >= THRESHOLD_BALANCED_ACCURACY_GATE
        and abs(row["B_C_probability_error_pp"]) <= AGGREGATE_PROBABILITY_ERROR_GATE_PP
    )
    row["E_F_gate_pass"] = bool(
        row["E_F_balanced_accuracy"] >= THRESHOLD_BALANCED_ACCURACY_GATE
        and abs(row["E_F_probability_error_pp"]) <= AGGREGATE_PROBABILITY_ERROR_GATE_PP
    )
    return row, threshold_rows


def run_model_ladder(
        frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    """Run the predeclared model ladder and retain OOF predictions in memory."""
    rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    prediction_store: dict[str, dict[str, Any]] = {}

    def add_rating(name: str, prediction: np.ndarray, features: str,
                   diagnostics: Any = None,
                   probabilities: dict[str, np.ndarray] | None = None) -> None:
        probs = _hard_probabilities(prediction) if probabilities is None else probabilities
        row, thresholds = _evaluate_prediction(
            name, frame, prediction, probs, feature_set=features,
            diagnostics=diagnostics,
        )
        rows.append(row)
        threshold_rows.extend(thresholds)
        prediction_store[name] = {"rating": prediction, "probabilities": probs}

    for mode, name in [
        ("multiplicative", "fuel_only_multiplicative"),
        ("additive", "fuel_only_additive"),
        ("log_ratio", "fuel_only_log_ratio"),
    ]:
        prediction = fuel_baseline_oof(frame, mode)
        add_rating(name, prediction, "post rating + cleaned main-fuel group")

    prediction = isotonic_oof(frame)
    add_rating("fuel_isotonic", prediction, "post rating, isotonic within fuel")

    regression_specs = [
        ("ridge_basic", BASIC_NUMERIC, BASIC_CATEGORICAL, "ridge", "direct"),
        ("ridge_post_descriptors", DESCRIPTOR_NUMERIC, DESCRIPTOR_CATEGORICAL, "ridge", "direct"),
        ("ridge_components", COMPONENT_NUMERIC, DESCRIPTOR_CATEGORICAL, "ridge", "direct"),
        ("ridge_recommendations", RECOMMENDATION_NUMERIC, DESCRIPTOR_CATEGORICAL, "ridge", "direct"),
        ("ridge_rich", RICH_NUMERIC, RICH_CATEGORICAL, "ridge", "direct"),
        ("additive_ridge_rich", RICH_NUMERIC, RICH_CATEGORICAL, "ridge", "additive"),
        ("log_ratio_ridge_rich", RICH_NUMERIC, RICH_CATEGORICAL, "ridge", "log_ratio"),
        ("spline_ridge_rich", RICH_NUMERIC, RICH_CATEGORICAL, "spline_ridge", "direct"),
    ]
    for name, numeric, categorical, estimator_name, mode in regression_specs:
        prediction, selection = nested_regression_oof(
            frame, numeric, categorical, estimator_name, mode=mode,
        )
        add_rating(
            name, prediction,
            f"{mode}; numeric={','.join(numeric)}; categorical={','.join(categorical)}",
            selection,
        )

    component_prediction, selection = component_bridge_oof(frame)
    add_rating(
        "BER_SER_component_bridge", component_prediction,
        "separate log bridges for building- and standard-emissions rates; reconstructed AR",
        selection,
    )

    prediction, probabilities, selection = residual_distribution_oof(frame)
    add_rating(
        "spline_ridge_residual_distribution", prediction,
        "rich spline ridge plus nested empirical fuel x post-band residual distribution",
        selection, probabilities,
    )

    for name, estimator_name in [
        ("hist_gradient_boosting_rich", "hist_gradient_boosting"),
        ("random_forest_rich", "random_forest"),
    ]:
        prediction, selection = nested_regression_oof(
            frame, RICH_NUMERIC, RICH_CATEGORICAL, estimator_name,
        )
        add_rating(name, prediction, "rich post-certificate features", selection)

    prediction, probabilities, selection = ordinal_band_oof(frame)
    add_rating(
        "ordinal_band_gradient_boosting", prediction,
        "rich features; multiclass broad-band probabilities", selection, probabilities,
    )

    threshold_models = [
        ("direct_threshold_fuel_logit", simple_threshold_logit_oof(frame),
         "post-rating distance and fuel"),
        ("direct_threshold_spline_calibrated", calibrated_threshold_spline_oof(frame),
         "rich post features; nested spline logistic and isotonic probability calibration"),
    ]
    for name, outputs, features in threshold_models:
        probabilities = {threshold: values["probability"] for threshold, values in outputs.items()}
        hard = {threshold: values["hard"] for threshold, values in outputs.items()}
        row, thresholds = _evaluate_prediction(
            name, frame, None, probabilities, hard=hard, model_type="threshold",
            feature_set=features,
        )
        rows.append(row)
        threshold_rows.extend(thresholds)
        prediction_store[name] = {"rating": None, "probabilities": probabilities, "hard": hard}

    model_table = pd.DataFrame(rows)
    threshold_table = pd.DataFrame(threshold_rows)
    subgroup_table = build_subgroup_calibration(frame, prediction_store)
    max_subgroup = (
        subgroup_table[subgroup_table.n >= MIN_SUBGROUP_N]
        .groupby(["model", "threshold"], observed=True).calibration_error_pp
        .apply(lambda x: x.abs().max()).unstack()
    )
    for threshold in THRESHOLDS:
        mapping = max_subgroup.get(threshold, pd.Series(dtype=float))
        model_table[f"{threshold}_max_abs_subgroup_error_pp"] = model_table.model.map(mapping)
        model_table[f"{threshold}_subgroup_gate_pass"] = (
            model_table[f"{threshold}_max_abs_subgroup_error_pp"] <= SUBGROUP_CALIBRATION_GATE_PP
        )
    model_table["individual_rating_claim_supported"] = (
        model_table.rating_gate_pass
        & model_table.B_C_subgroup_gate_pass.fillna(False)
        & model_table.E_F_subgroup_gate_pass.fillna(False)
    )
    model_table["individual_threshold_claim_supported"] = (
        model_table.B_C_gate_pass & model_table.E_F_gate_pass
        & model_table.B_C_subgroup_gate_pass.fillna(False)
        & model_table.E_F_subgroup_gate_pass.fillna(False)
    )
    policy_mask = (
        frame.transaction.astype(str).str.contains("property to let", case=False, na=False)
        & ~frame.transaction.astype(str).str.contains("construction", case=False, na=False)
        & (frame.floor_area_post > 1000)
    ).to_numpy()
    model_table["policy_proxy_like_n"] = int(policy_mask.sum())
    for index, model_row in model_table.iterrows():
        output = prediction_store[model_row.model]
        if policy_mask.sum() >= 20 and output["rating"] is not None:
            policy_rating = rating_metrics(
                frame.loc[policy_mask, "pre"], np.asarray(output["rating"])[policy_mask],
            )
            model_table.loc[index, "policy_proxy_like_MAE_pts"] = policy_rating["MAE_pts"]
            model_table.loc[index, "policy_proxy_like_calibration_slope"] = policy_rating["calibration_slope"]
        for threshold_name, threshold in THRESHOLDS.items():
            if policy_mask.sum() < 20:
                continue
            result = threshold_metrics(
                frame.loc[policy_mask, "pre"],
                np.asarray(output["probabilities"][threshold_name])[policy_mask],
                threshold,
                None if "hard" not in output else np.asarray(output["hard"][threshold_name])[policy_mask],
            )
            model_table.loc[index, f"policy_proxy_like_{threshold_name}_balanced_accuracy"] = result["balanced_accuracy"]
            model_table.loc[index, f"policy_proxy_like_{threshold_name}_probability_error_pp"] = result["aggregate_probability_error_pp"]
    model_table["policy_proxy_rating_gate_pass"] = (
        (model_table.policy_proxy_like_MAE_pts <= RATING_MAE_GATE)
        & (model_table.policy_proxy_like_calibration_slope >= RATING_SLOPE_GATE)
    )
    for threshold_name in THRESHOLDS:
        model_table[f"{threshold_name}_gate_including_subgroups"] = (
            model_table[f"{threshold_name}_gate_pass"]
            & model_table[f"{threshold_name}_subgroup_gate_pass"].fillna(False)
        )
        model_table[f"policy_proxy_{threshold_name}_gate_pass"] = (
            (model_table[f"policy_proxy_like_{threshold_name}_balanced_accuracy"]
             >= THRESHOLD_BALANCED_ACCURACY_GATE)
            & (model_table[f"policy_proxy_like_{threshold_name}_probability_error_pp"].abs()
               <= AGGREGATE_PROBABILITY_ERROR_GATE_PP)
        )
    model_table["individual_rating_claim_supported"] &= (
        model_table.policy_proxy_rating_gate_pass.fillna(False)
    )
    model_table["individual_threshold_claim_supported"] &= (
        model_table.policy_proxy_B_C_gate_pass.fillna(False)
        & model_table.policy_proxy_E_F_gate_pass.fillna(False)
    )
    model_table["rank_score"] = (
        model_table.MAE_pts.fillna(1000)
        + 2 * model_table.B_C_probability_error_pp.abs()
        + 2 * model_table.E_F_probability_error_pp.abs()
        + 10 * (1 - model_table.B_C_balanced_accuracy)
        + 10 * (1 - model_table.E_F_balanced_accuracy)
    )
    model_table["rank"] = model_table.rank_score.rank(method="min").astype(int)
    return model_table.sort_values("rank"), threshold_table, subgroup_table, prediction_store


def build_subgroup_calibration(
        frame: pd.DataFrame, prediction_store: dict[str, dict[str, Any]]) -> pd.DataFrame:
    data = frame.copy()
    data["floor_area_bin"] = pd.cut(
        data.floor_area_post,
        [0, 100, 500, 1000, 5000, np.inf],
        labels=["0-100", "101-500", "501-1000", "1001-5000", ">5000"],
        include_lowest=True,
    ).astype(str)
    groupings = {
        "fuel": data.fuel.fillna("Missing"),
        "post_band": data.post_band.astype(str),
        "sector_use": data.sector.fillna("Missing"),
        "floor_area_bin": data.floor_area_bin.fillna("Missing"),
        "AC_status": data.ac.fillna("Missing"),
        "policy_proxy_like": np.where(
            data.transaction.astype(str).str.contains("property to let", case=False, na=False)
            & ~data.transaction.astype(str).str.contains("construction", case=False, na=False)
            & (data.floor_area_post > 1000),
            "property_to_let_over_1000m2", "other_calibration_entries",
        ),
    }
    rows: list[dict[str, Any]] = []
    for model_name, outputs in prediction_store.items():
        for threshold_name, threshold in THRESHOLDS.items():
            probability = np.asarray(outputs["probabilities"][threshold_name], dtype=float)
            observed = (
                (data.pre.to_numpy() <= threshold).astype(float)
                if threshold_name == "B_C"
                else (data.pre.to_numpy() > threshold).astype(float)
            )
            for grouping_name, values in groupings.items():
                for value in pd.Series(values).drop_duplicates():
                    mask = pd.Series(values).to_numpy() == value
                    if not mask.any():
                        continue
                    rows.append({
                        "diagnostic_type": "threshold_probability",
                        "model": model_name,
                        "threshold": threshold_name,
                        "grouping": grouping_name,
                        "group": value,
                        "n": int(mask.sum()),
                        "observed_rate": float(observed[mask].mean()),
                        "mean_predicted_probability": float(probability[mask].mean()),
                        "calibration_error_pp": float(100 * (probability[mask].mean() - observed[mask].mean())),
                        "reported_for_gate": bool(mask.sum() >= MIN_SUBGROUP_N),
                    })
        if outputs["rating"] is not None:
            prediction = np.asarray(outputs["rating"], dtype=float)
            rating_groups: list[tuple[str, pd.Series]] = [
                ("fuel", data.fuel.fillna("Missing")),
                ("true_old_band", data.pre_band.astype(str)),
                ("post_observed_band", data.post_band.astype(str)),
                ("fuel_x_true_old_band", data.fuel.astype(str) + "|" + data.pre_band.astype(str)),
                ("fuel_x_post_observed_band", data.fuel.astype(str) + "|" + data.post_band.astype(str)),
            ]
            for threshold_name, threshold in THRESHOLDS.items():
                for width in [10, 20, 30]:
                    included = (data.post - threshold).abs() <= width
                    rating_groups.append((
                        f"post_within_{width}_points_of_{threshold_name}",
                        pd.Series(np.where(included, "inside", "outside"), index=data.index),
                    ))
            for grouping_name, values in rating_groups:
                for value in pd.Series(values).drop_duplicates():
                    mask = pd.Series(values).to_numpy() == value
                    metrics = rating_metrics(data.loc[mask, "pre"], prediction[mask])
                    rows.append({
                        "diagnostic_type": "rating_error",
                        "model": model_name, "threshold": "",
                        "grouping": grouping_name, "group": value,
                        "n": int(mask.sum()), **metrics,
                        "observed_rate": np.nan, "mean_predicted_probability": np.nan,
                        "calibration_error_pp": np.nan,
                        "reported_for_gate": bool(mask.sum() >= MIN_SUBGROUP_N),
                    })
    return pd.DataFrame(rows)


def same_method_basic_oof(frame: pd.DataFrame) -> np.ndarray:
    """Nested-OOF basic prediction of a later same-method rating."""
    data = frame.copy()
    data["log_floor_area"] = np.log1p(data.floor_area_earlier.clip(lower=0))
    data["sector"] = data.property_type_clean_earlier.fillna("Missing")
    data["ac"] = data.aircon_present_clean_earlier.fillna("Missing")
    data["environment"] = data.building_environment_clean_earlier.fillna("Missing")
    data["fuel_earlier"] = _fuel_group(data.main_heating_fuel_clean_earlier)
    numeric = ["earlier", "gap_years", "log_floor_area", "n_recommendations_earlier"]
    categorical = ["fuel_earlier", "sector", "ac", "environment", "earlier_band"]
    folds = assign_group_folds(data.uprn)
    prediction = np.empty(len(data), dtype=float)
    pipeline = Pipeline([
        ("prep", _make_preprocessor(numeric, categorical, spline=True)),
        ("model", Ridge()),
    ])
    for fold in range(N_OUTER_FOLDS):
        train = data[folds != fold]
        test = data[folds == fold]
        unique_groups = train.uprn.nunique()
        n_inner = min(N_INNER_FOLDS, unique_groups)
        if n_inner < 2:
            prediction[folds == fold] = train.later.median()
            continue
        inner = GroupKFold(n_splits=n_inner)
        search = GridSearchCV(
            pipeline, {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
            scoring="neg_mean_absolute_error", cv=inner, n_jobs=-1, refit=True,
        )
        search.fit(train, train.later, groups=train.uprn)
        prediction[folds == fold] = search.predict(test)
    return prediction


def _same_method_rows(
        frame: pd.DataFrame, prediction: np.ndarray, model: str,
        period: str, gap_label: str, stability: str,
) -> list[dict[str, Any]]:
    work = frame.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    grouping_specs = [("overall", pd.Series("All", index=work.index)),
                      ("fuel", work.fuel), ("earlier_band", work.earlier_band)]
    for grouping, groups in grouping_specs:
        for group in pd.Series(groups).drop_duplicates():
            mask = pd.Series(groups).to_numpy() == group
            actual = work.loc[mask, "later"].to_numpy(dtype=float)
            predicted = np.asarray(prediction)[mask]
            metrics = rating_metrics(actual, predicted)
            row: dict[str, Any] = {
                "period": period, "maximum_gap": gap_label,
                "stability_definition": stability, "model": model,
                "grouping": grouping, "group": group,
                "error_sign_definition": "predicted_later_minus_observed_later",
                **metrics,
            }
            for threshold_name, threshold in THRESHOLDS.items():
                agreement = np.mean((actual > threshold) == (predicted > threshold))
                balanced = balanced_accuracy_score(actual > threshold, predicted > threshold)
                row[f"{threshold_name}_agreement"] = float(agreement)
                row[f"{threshold_name}_balanced_accuracy"] = float(balanced)
            rows.append(row)
    return rows


def run_same_method_noise(pairs: pd.DataFrame) -> pd.DataFrame:
    """Estimate stable-reassessment variation without crossing the 2022 change."""
    stability_definitions = {
        "fuel_floor": pairs.fuel_stable & pairs.floor_stable_2pct,
        "fuel_floor_recommendation_count": (
            pairs.fuel_stable & pairs.floor_stable_2pct
            & pairs.recommendation_count_stable
        ),
        "strict_observed_descriptors": (
            pairs.fuel_stable & pairs.floor_stable_2pct
            & pairs.recommendation_count_stable & pairs.sector_stable
            & pairs.ac_stable
        ),
        "very_strict_observed_descriptors": (
            pairs.fuel_stable & pairs.floor_stable_2pct
            & pairs.recommendation_count_stable & pairs.recommendation_codes_stable
            & pairs.sector_stable & pairs.ac_stable & pairs.environment_stable
            & pairs.building_level_stable
        ),
    }
    gaps = [("<=6_months", 0.5), ("<=12_months", 1.0),
            ("<=24_months", 2.0), ("all_gaps", np.inf)]
    rows: list[dict[str, Any]] = []
    for period in ["pre_pre", "post_post"]:
        for stability, stable_mask in stability_definitions.items():
            for gap_label, maximum in gaps:
                subset = pairs[
                    (pairs.period == period) & stable_mask & (pairs.gap_years <= maximum)
                ].copy()
                if len(subset) < 50:
                    continue
                identity = subset.earlier.to_numpy(dtype=float)
                rows.extend(_same_method_rows(
                    subset, identity, "identity_no_change", period, gap_label, stability,
                ))
                if len(subset) >= 150 and subset.uprn.nunique() >= 100:
                    basic = same_method_basic_oof(subset)
                    rows.extend(_same_method_rows(
                        subset, basic, "nested_ridge_basic_descriptors",
                        period, gap_label, stability,
                    ))
    return pd.DataFrame(rows)


def _composition(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in FUELS:
        result[f"fuel_{value.lower()}_pct"] = float(100 * (frame.fuel == value).mean())
    for value in ["A/B", "C", "D", "E", "F/G"]:
        result[f"post_band_{value.replace('/', '_')}_pct"] = float(
            100 * (frame.post_band.astype(str) == value).mean()
        )
    return result


def _latest_scope_coverage(latest: pd.DataFrame, scope: str) -> tuple[float, float]:
    if scope == "electric_gas":
        mask = latest.fuel.isin(["Electric", "Gas"])
    elif scope == "near_B_C_30":
        mask = (latest.post - 50).abs() <= 30
    elif scope == "near_E_F_30":
        mask = (latest.post - 125).abs() <= 30
    else:
        mask = pd.Series(True, index=latest.index)
    full = float(100 * mask.mean())
    policy = latest.policy_proxy
    policy_pct = float(100 * mask[policy].mean()) if policy.any() else np.nan
    return full, policy_pct


def sample_definitions() -> list[SampleDefinition]:
    current = lambda d: pd.Series(True, index=d.index)
    stable_baseline = lambda d: d.floor_stable_2pct & d.recommendation_count_stable
    return [
        SampleDefinition(
            "current_authoritative", "authoritative", "Exact committed 2,442-UPRN sample", current,
        ),
        SampleDefinition(
            "strict_all_observed_descriptors", "authoritative",
            "Current sample plus unchanged fuel, recommendation codes, sector/use, AC, environment and building level",
            lambda d: d.fuel_stable & d.floor_stable_2pct & d.recommendation_count_stable
            & d.recommendation_codes_stable & d.sector_stable & d.ac_stable
            & d.environment_stable & d.building_level_stable,
        ),
        SampleDefinition(
            "stable_fuel_floor_only", "first_latest",
            "First/latest pair; unchanged fuel and floor area; recommendation count allowed to change",
            lambda d: d.fuel_stable & d.floor_stable_2pct,
        ),
        SampleDefinition(
            "relaxed_recommendation_count", "first_latest",
            "First/latest pair; floor area stable; fuel may change and recommendation count may change",
            lambda d: d.floor_stable_2pct,
        ),
        SampleDefinition(
            "nearest_pre_post_baseline", "nearest",
            "Nearest certificates either side of change; stable floor and recommendation count",
            stable_baseline,
        ),
        SampleDefinition(
            "maximum_gap_6_months", "nearest",
            "Nearest pair, baseline stability, <=6-month gap",
            lambda d: stable_baseline(d) & (d.gap_years <= 0.5),
        ),
        SampleDefinition(
            "maximum_gap_12_months", "nearest",
            "Nearest pair, baseline stability, <=12-month gap",
            lambda d: stable_baseline(d) & (d.gap_years <= 1.0),
        ),
        SampleDefinition(
            "maximum_gap_24_months", "nearest",
            "Nearest pair, baseline stability, <=24-month gap",
            lambda d: stable_baseline(d) & (d.gap_years <= 2.0),
        ),
        SampleDefinition(
            "post_within_30_of_B_C", "authoritative",
            "Current sample restricted to post ratings within 30 points of B/C",
            lambda d: (d.post - 50).abs() <= 30,
            application_scope="near_B_C_30",
        ),
        SampleDefinition(
            "post_within_30_of_E_F", "authoritative",
            "Current sample restricted to post ratings within 30 points of E/F",
            lambda d: (d.post - 125).abs() <= 30,
            application_scope="near_E_F_30",
        ),
        SampleDefinition(
            "electric_and_gas_only", "authoritative",
            "Current sample restricted to electric and gas fuel groups",
            lambda d: d.fuel.isin(["Electric", "Gas"]),
            application_scope="electric_gas",
        ),
        SampleDefinition(
            "exclude_abs_change_over_75_diagnostic", "authoritative",
            "Outcome-selected diagnostic excluding absolute old/post movements >75 points",
            lambda d: (d.pre - d.post).abs() <= 75,
            selection_uses_old_outcome=True,
        ),
    ]


def run_sample_ladder(
        authoritative: pd.DataFrame, first_latest: pd.DataFrame,
        nearest: pd.DataFrame, latest: pd.DataFrame,
) -> pd.DataFrame:
    sources = {"authoritative": authoritative, "first_latest": first_latest,
               "nearest": nearest}
    baseline_n = len(authoritative)
    rows: list[dict[str, Any]] = []
    for definition in sample_definitions():
        source = sources[definition.pair_source]
        subset = source[definition.mask(source)].reset_index(drop=True)
        coverage_full, coverage_policy = _latest_scope_coverage(
            latest, definition.application_scope,
        )
        base = {
            "sample": definition.name,
            "description": definition.description,
            "pair_source": definition.pair_source,
            "selection_uses_old_outcome": definition.selection_uses_old_outcome,
            "n": len(subset), "unique_uprns": subset.uprn.nunique(),
            "retention_vs_authoritative_pct": float(100 * len(subset) / baseline_n),
            "formal_full_frame_scope_pct": coverage_full,
            "formal_policy_proxy_scope_pct": coverage_policy,
            **(_composition(subset) if len(subset) else {}),
        }
        if len(subset) < 150 or subset.uprn.nunique() < 100:
            rows.append({**base, "model": "not_estimated",
                         "reason": "insufficient support for five-fold nested validation"})
            continue
        for model_name in ["fuel_only_multiplicative", "spline_ridge_rich"]:
            if model_name == "fuel_only_multiplicative":
                prediction = fuel_baseline_oof(subset, "multiplicative")
            else:
                prediction, _ = nested_regression_oof(
                    subset, RICH_NUMERIC, RICH_CATEGORICAL, "spline_ridge",
                )
            metrics = rating_metrics(subset.pre, prediction)
            threshold_result: dict[str, Any] = {}
            for threshold_name, threshold in THRESHOLDS.items():
                value = threshold_metrics(
                    subset.pre, _hard_probabilities(prediction)[threshold_name], threshold,
                )
                threshold_result[f"{threshold_name}_balanced_accuracy"] = value["balanced_accuracy"]
                threshold_result[f"{threshold_name}_aggregate_probability_error_pp"] = value["aggregate_probability_error_pp"]
            rows.append({**base, "model": model_name, "reason": "", **metrics,
                         **threshold_result})
    return pd.DataFrame(rows)


def build_data_inventory() -> pd.DataFrame:
    """Inventory locally available predictors and relevant absent source formats."""
    con = duckdb.connect()
    fields = {
        "asset_rating": ("Outcome/current rating", True, False, "Core target and predictor"),
        "building_emissions": ("BER proxy", True, False, "Numerator in rating formula"),
        "standard_emissions": ("SER proxy", True, False, "Denominator in rating formula"),
        "target_emissions": ("Target emissions", True, False, "Calculation component"),
        "typical_emissions": ("Typical emissions", True, False, "Calculation component"),
        "primary_energy_value": ("Primary energy", True, False, "May capture system/load differences"),
        "main_heating_fuel_raw": ("Detailed recorded fuel text", True, False, "More granular than cleaned fuel"),
        "main_heating_fuel_clean": ("Cleaned fuel", True, False, "Baseline bridge stratifier"),
        "building_environment_clean": ("Heating/ventilation/AC environment", True, False, "Partial system-type proxy"),
        "aircon_present_clean": ("AC presence", True, False, "Cooling/load proxy"),
        "aircon_kw_rating": ("Recorded AC capacity", True, False, "Cooling intensity; very sparse"),
        "estimated_aircon_kw_rating": ("Estimated AC capacity", True, False, "Cooling intensity; sparse"),
        "property_type_clean": ("Sector/use", True, False, "Activity/load proxy"),
        "building_level": ("Building level", True, False, "Form/use proxy"),
        "new_build_benchmark": ("New-build benchmark", True, False, "Reference-building proxy"),
        "existing_stock_benchmark": ("Existing-stock benchmark", True, False, "Reference-building proxy"),
        "transaction_type_clean": ("Transaction", True, False, "Context/tenure proxy"),
        "uprn_source": ("UPRN source", True, False, "Record-quality proxy"),
        "days_between_inspection_and_lodgement": ("Lodgement lag", True, False, "Process/metadata proxy"),
        "n_recommendations": ("Recommendation count", True, False, "Recorded improvement proxy"),
        "unique_recommendation_codes": ("Recommendation codes", True, False, "System/fabric recommendation proxy"),
        "other_fuel_desc": ("Other fuel description", True, False, "Potential fuel detail but empty"),
        "special_energy_uses": ("Special energy uses", True, False, "Potential load detail but empty"),
        "renewable_sources": ("Renewables", True, False, "Potential generation detail but empty"),
    }
    expressions = ["count(*) AS n"] + [
        f"sum(CASE WHEN {_q(field)} IS NULL OR trim(cast({_q(field)} AS varchar))='' THEN 1 ELSE 0 END) AS {_q(field)}"
        for field in fields
    ]
    counts = con.execute(
        f"SELECT {', '.join(expressions)} FROM read_parquet('{UNIFIED_PARQUET}')"
    ).fetchone()
    n = counts[0]
    missing = dict(zip(fields, counts[1:]))
    rows: list[dict[str, Any]] = []
    for field, (description, application, leakage, relevance) in fields.items():
        rows.append({
            "file_or_table": str(UNIFIED_PARQUET), "field_name": field,
            "description": description, "availability": "present",
            "n_rows": n, "missingness_pct": 100 * missing[field] / n,
            "available_at_full_register_application_time": application,
            "usable_without_outcome_leakage": not leakage,
            "expected_relevance": relevance,
            "notes": "Post-certificate value used; old/pre value is never a predictor",
        })
    formula_mae, formula_median = con.execute(f"""
        SELECT avg(abs(asset_rating - 50 * building_emissions / nullif(standard_emissions, 0))),
               median(abs(asset_rating - 50 * building_emissions / nullif(standard_emissions, 0)))
        FROM read_parquet('{UNIFIED_PARQUET}')
        WHERE asset_rating > 0 AND building_emissions IS NOT NULL
          AND standard_emissions > 0
    """).fetchone()
    rows.append({
        "file_or_table": str(UNIFIED_PARQUET), "field_name": "asset_rating_formula_audit",
        "description": "AR versus 50 x building-emissions / standard-emissions",
        "availability": "derived", "n_rows": n, "missingness_pct": np.nan,
        "available_at_full_register_application_time": True,
        "usable_without_outcome_leakage": True,
        "expected_relevance": "Shows components are nearly algebraic restatements, not independent physical detail",
        "notes": f"Register-wide absolute discrepancy: mean {formula_mae:.3f} points; median {formula_median:.3f} points",
    })

    recommendation_n, recommendation_missing = con.execute(f"""
        SELECT count(*), sum(CASE WHEN recommendation IS NULL OR trim(recommendation)='' THEN 1 ELSE 0 END)
        FROM read_parquet('{REC_LONG_PARQUET}')
    """).fetchone()
    rows.append({
        "file_or_table": str(REC_LONG_PARQUET), "field_name": "recommendation",
        "description": "Full recommendation text/details", "availability": "present",
        "n_rows": recommendation_n,
        "missingness_pct": 100 * recommendation_missing / recommendation_n,
        "available_at_full_register_application_time": True,
        "usable_without_outcome_leakage": True,
        "expected_relevance": "Text proxies for HVAC, lighting, controls, fabric and renewables",
        "notes": "Only text attached to the post certificate is used",
    })
    raw_certificates = sorted(RAW_CERT_DIR.glob("certificates-*.csv"))
    raw_recommendations = sorted(RAW_CERT_DIR.glob("recommendations-*.csv"))
    for files, label, description in [
        (raw_certificates, "annual certificate CSVs", "Raw downloaded EPC register certificates"),
        (raw_recommendations, "annual recommendation CSVs", "Raw downloaded recommendation records"),
    ]:
        rows.append({
            "file_or_table": str(RAW_CERT_DIR / label), "field_name": "multiple",
            "description": description, "availability": "present" if files else "absent",
            "n_rows": np.nan, "missingness_pct": np.nan,
            "available_at_full_register_application_time": True,
            "usable_without_outcome_leakage": True,
            "expected_relevance": "Source-level audit and fields retained in clean parquet",
            "notes": f"{len(files)} files: {files[0].name if files else ''} to {files[-1].name if files else ''}",
        })

    searched_extensions = [".xml", ".nct", ".inp", ".json", ".html", ".pdf", ".zip", ".7z"]
    absent_types = [
        ("raw EPC XML", ["*.xml"]),
        ("BRUKL reports", ["*BRUKL*.pdf", "*BRUKL*.html"]),
        ("SBEM/NCM project inputs", ["*.nct", "*.inp"]),
        ("assessor software exports", ["*SBEM*.xml", "*NCM*.xml", "*SBEM*.json"]),
    ]
    search_roots = [RAW_CERT_DIR] if RAW_CERT_DIR.is_dir() else []
    for label, patterns in absent_types:
        matches: list[Path] = []
        for root in search_roots:
            for pattern in patterns:
                matches.extend(
                    path for path in root.rglob(pattern)
                    if not any(part in {".venv", ".git", "node_modules", "site-packages"}
                               for part in path.parts)
                )
        if label == "raw EPC XML":
            verified: list[Path] = []
            for path in matches:
                name_match = any(token in path.name.lower() for token in ["epc", "certificate"])
                try:
                    content = path.read_bytes()[:65536].lower()
                except OSError:
                    content = b""
                content_match = any(
                    token in content
                    for token in [b"asset-rating", b"property-type", b"non-domestic epc"]
                )
                if name_match or content_match:
                    verified.append(path)
            matches = verified
        rows.append({
            "file_or_table": "; ".join(str(root) for root in search_roots),
            "field_name": label, "description": label,
            "availability": "present" if matches else "absent",
            "n_rows": len(matches), "missingness_pct": 100.0 if not matches else np.nan,
            "available_at_full_register_application_time": False,
            "usable_without_outcome_leakage": True,
            "expected_relevance": "Potentially decisive: same-input reruns or richer physical-system inputs",
            "notes": (
                "Matches: " + "; ".join(str(path) for path in matches[:5])
                if matches else
                f"No relevant files found in the configured raw-data directory for {', '.join(searched_extensions)}"
            ),
        })
    rows.append({
        "file_or_table": str(UNIFIED_PARQUET), "field_name": "software_version_or_calculation_tool",
        "description": "Assessor software/version metadata", "availability": "absent",
        "n_rows": n, "missingness_pct": 100.0,
        "available_at_full_register_application_time": False,
        "usable_without_outcome_leakage": True,
        "expected_relevance": "Could isolate implementation/version effects",
        "notes": "No corresponding field in the 146-column unified certificate table",
    })
    rows.append({
        "file_or_table": str(UNIFIED_PARQUET), "field_name": "heat_pump_vs_resistance_system",
        "description": "Explicit heating-system technology", "availability": "absent",
        "n_rows": n, "missingness_pct": 100.0,
        "available_at_full_register_application_time": False,
        "usable_without_outcome_leakage": True,
        "expected_relevance": "High for electricity-rating response",
        "notes": "Recommendation keywords are weak proxies; no installed-system field distinguishes heat pumps from resistance heating",
    })
    return pd.DataFrame(rows)


def sbem_feasibility_sources() -> pd.DataFrame:
    """Official sources checked for a lawful, valid same-input rerun pilot."""
    rows = [
        {
            "source": "UK-NCM current iSBEM England downloads", "authority": "UK-NCM/HSE",
            "url": "https://www.uk-ncm.org.uk/page.jsp?id=35", "searched_for": "Current Part L 2021 engine and files",
            "finding": "iSBEM v6.1.e implements the England 2021 NCM effective 15 June 2022; installer, databases and manuals are offered after licence acceptance.",
            "same_input_pair_usable_now": False,
            "blocker": "Current engine alone cannot produce the old-method side; licence must be accepted by the user and software is Windows/Access based.",
        },
        {
            "source": "UK-NCM previous versions", "authority": "UK-NCM/HSE",
            "url": "https://www.uk-ncm.org.uk/page.jsp?id=21", "searched_for": "Pre-2022 iSBEM v5.6b engine",
            "finding": "The site lists v5.6b and v6.1 versions, but says previous versions may be downloaded/used only in exceptional circumstances for projects created in the older version or after a PC update.",
            "same_input_pair_usable_now": False,
            "blocker": "A new research pilot is not within the stated exceptional-use cases without written HSE/BRE permission.",
        },
        {
            "source": "UK-NCM software licence/disclaimer", "authority": "UK-NCM/HSE",
            "url": "https://www.uk-ncm.org.uk/disclaimer.jsp", "searched_for": "Legal reuse terms",
            "finding": "Use requires acceptance of a non-transferable, non-exclusive royalty-free licence; reverse engineering and redistribution are restricted.",
            "same_input_pair_usable_now": False,
            "blocker": "Licence terms were not accepted and installers were not redistributed.",
        },
        {
            "source": "iSBEM v6.1.e installation instructions", "authority": "UK-NCM/HSE",
            "url": "https://www.uk-ncm.org.uk/filelibrary/Installation_instructions_for_iSBEM_v6.1.e.pdf",
            "searched_for": "Execution requirements",
            "finding": "The official front end requires Microsoft Access or Access Runtime and local Windows installation.",
            "same_input_pair_usable_now": False,
            "blocker": f"Host is {platform.system()} {platform.machine()}; no Windows/Access runtime or configured VM is available.",
        },
        {
            "source": "iSBEM v5.6b installation instructions", "authority": "UK-NCM/HSE",
            "url": "https://www.uk-ncm.org.uk/filelibrary/Installation_instructions_for_iSBEM_v5.6.b.pdf",
            "searched_for": "Old-engine execution requirements",
            "finding": "The old front end likewise requires Microsoft Access/Windows.",
            "same_input_pair_usable_now": False,
            "blocker": "No licensed old engine plus Windows/Access environment is available.",
        },
        {
            "source": "UK-NCM iSBEM FAQ", "authority": "UK-NCM/HSE",
            "url": "https://www.uk-ncm.org.uk/faq.jsp?id=9", "searched_for": "Project format and cross-version conversion",
            "finding": "Project inputs are .nct files. Conversion to a newer version can require user action and completion of new fields.",
            "same_input_pair_usable_now": False,
            "blocker": "No legally reusable public .nct test case was located; conversion/adaptation would have to be documented as an input change.",
        },
        {
            "source": "Approved NCM software list", "authority": "UK Government",
            "url": "https://assets.publishing.service.gov.uk/media/68f8b1daec6267c615ed8f64/2025-10-15_NCM_approved_software_programs_for_buildings_other_than_dwellings.pdf",
            "searched_for": "Official engine status",
            "finding": "iSBEM 6.1 variants are approved for Part L 2021 calculations.",
            "same_input_pair_usable_now": False,
            "blocker": "Approval establishes provenance, not a paired public test dataset.",
        },
        {
            "source": "National Calculation Methodology modelling guide", "authority": "UK Government",
            "url": "https://www.gov.uk/government/publications/national-calculation-methodology-modelling-guide-for-buildings-other-than-dwellings-in-england-and-wales",
            "searched_for": "Official methodology and reusable model inputs",
            "finding": "Official modelling guidance is public, but no same-building old/new executable input pair was provided with the publication.",
            "same_input_pair_usable_now": False,
            "blocker": "A modelling guide cannot substitute for identical complete SBEM inputs run in both engines.",
        },
        {
            "source": "UK cost-optimal report", "authority": "UK Government",
            "url": "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/770783/2nd_UK_Cost_Optimal_Report.pdf",
            "searched_for": "Reference-building/test-case inputs",
            "finding": "The report describes seven SBEM reference buildings, but no reusable .nct input package was located.",
            "same_input_pair_usable_now": False,
            "blocker": "Published summary outputs are not identical executable inputs.",
        },
        {
            "source": "Configured raw-data directory search", "authority": "Local filesystem",
            "url": "", "searched_for": "*.nct, *.inp, raw XML, BRUKL, SBEM/NCM exports and archives",
            "finding": "No valid SBEM/NCM input file or old/new paired project was found in the configured raw-data directory.",
            "same_input_pair_usable_now": False,
            "blocker": "No same-input building model exists locally.",
        },
    ]
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _markdown_table(frame: pd.DataFrame, digits: int = 2) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].round(digits)
    def clean(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")
    headers = [clean(column) for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend(
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def build_summary(
        model_table: pd.DataFrame, sample_table: pd.DataFrame,
        same_method: pd.DataFrame, feasibility: pd.DataFrame,
) -> pd.DataFrame:
    rating_models = model_table[model_table.MAE_pts.notna()].copy()
    best = rating_models.loc[rating_models.MAE_pts.idxmin()]
    baseline = model_table.set_index("model").loc["fuel_only_multiplicative"]
    reduction = 100 * (baseline.MAE_pts - best.MAE_pts) / baseline.MAE_pts
    rating_pass = model_table[model_table.individual_rating_claim_supported]
    threshold_pass = model_table[model_table.individual_threshold_claim_supported]
    strict_noise = same_method[
        (same_method.stability_definition == "strict_observed_descriptors")
        & (same_method.model == "identity_no_change")
        & (same_method.grouping == "overall")
        & (same_method.maximum_gap == "<=6_months")
    ]
    noise_values = ", ".join(
        f"{row.period}: {row.MAE_pts:.1f}" for row in strict_noise.itertuples()
    )
    defensible_samples = sample_table[
        ~sample_table.selection_uses_old_outcome.fillna(False)
        & sample_table.MAE_pts.notna()
    ]
    best_sample = defensible_samples.loc[defensible_samples.MAE_pts.idxmin()]
    rows = [
        {
            "question": "Can the baseline MAE be materially reduced?",
            "answer": "Yes" if best.MAE_pts <= RATING_MAE_GATE or reduction >= 25 else "No",
            "evidence": f"Baseline OOF MAE {baseline.MAE_pts:.2f}; best {best.model} {best.MAE_pts:.2f}; relative reduction {reduction:.1f}%.",
        },
        {
            "question": "Good enough for individual old-basis ratings?",
            "answer": "Yes" if len(rating_pass) else "No",
            "evidence": (
                ", ".join(rating_pass.model) if len(rating_pass)
                else f"No model jointly passed MAE<={RATING_MAE_GATE}, slope>={RATING_SLOPE_GATE} and subgroup gates."
            ),
        },
        {
            "question": "Good enough for individual B/C and E/F reclassification?",
            "answer": "Yes" if len(threshold_pass) else "No",
            "evidence": (
                ", ".join(threshold_pass.model) if len(threshold_pass)
                else "No model jointly passed both threshold discrimination, aggregate-probability, subgroup-calibration and policy-proxy stability gates."
            ),
        },
        {
            "question": "Is remaining error irreducible reassessment noise?",
            "answer": "Substantial component; not point-identified",
            "evidence": f"Strict stable same-method identity MAE within six months: {noise_values}. Physical/input changes cannot be ruled out completely.",
        },
        {
            "question": "Would direct same-input SBEM/NCM reruns help and are they feasible now?",
            "answer": "They would help; not feasible from the available public/local files and current licensed environment",
            "evidence": f"{int((~feasibility.same_input_pair_usable_now).sum())} searched source entries yielded no executable lawful same-input old/new pair.",
        },
        {
            "question": "Which defensible sample/model combination has lowest MAE?",
            "answer": f"{best_sample['sample']} / {best_sample['model']}",
            "evidence": f"n={int(best_sample.n):,}; OOF MAE {best_sample.MAE_pts:.2f}; retention {best_sample.retention_vs_authoritative_pct:.1f}%.",
        },
        {
            "question": "Strongest defensible paper framing",
            "answer": "Aggregate, cross-fitted threshold probabilities with structural sensitivity",
            "evidence": "Do not claim exact reconstructed ratings or individual reclassification; retain register-conditional aggregate probability framing.",
        },
        {
            "question": "Claims impossible without full SBEM inputs",
            "answer": "Exact old-method rating and legal/policy status for an individual register entry",
            "evidence": "The public register omits complete geometry, zoning, HVAC efficiencies and load schedules needed for identical-engine reruns.",
        },
    ]
    return pd.DataFrame(rows)


def write_report(
        audit: dict[str, Any], inventory: pd.DataFrame, same_method: pd.DataFrame,
        sample_table: pd.DataFrame, model_table: pd.DataFrame,
        threshold_table: pd.DataFrame, subgroup_table: pd.DataFrame,
        feasibility: pd.DataFrame, summary: pd.DataFrame,
        bridge_hashes_unchanged: bool,
) -> None:
    baseline = model_table.set_index("model").loc["fuel_only_multiplicative"]
    rating_models = model_table[model_table.MAE_pts.notna()]
    best = rating_models.loc[rating_models.MAE_pts.idxmin()]
    relative_reduction = 100 * (baseline.MAE_pts - best.MAE_pts) / baseline.MAE_pts
    short_noise = same_method[
        (same_method.stability_definition == "strict_observed_descriptors")
        & (same_method.model == "identity_no_change")
        & (same_method.grouping == "overall")
        & (same_method.maximum_gap.isin(["<=6_months", "<=12_months", "<=24_months"]))
    ][["period", "maximum_gap", "n", "MAE_pts", "median_absolute_error_pts",
       "bias_pts", "calibration_slope", "B_C_agreement", "E_F_agreement"]]
    ridge_six_month = same_method[
        (same_method.stability_definition == "strict_observed_descriptors")
        & (same_method.model == "nested_ridge_basic_descriptors")
        & (same_method.grouping == "overall")
        & (same_method.maximum_gap == "<=6_months")
    ].set_index("period")
    ranking_columns = [
        "rank", "model", "MAE_pts", "calibration_slope",
        "B_C_balanced_accuracy", "E_F_balanced_accuracy",
        "B_C_probability_error_pp", "E_F_probability_error_pp",
        "B_C_max_abs_subgroup_error_pp", "E_F_max_abs_subgroup_error_pp",
        "individual_rating_claim_supported", "individual_threshold_claim_supported",
    ]
    ranking = model_table[ranking_columns].sort_values("rank")
    policy_columns = [
        "model", "policy_proxy_like_n", "policy_proxy_like_MAE_pts",
        "policy_proxy_like_calibration_slope",
        "policy_proxy_like_B_C_balanced_accuracy", "policy_proxy_like_E_F_balanced_accuracy",
        "policy_proxy_like_B_C_probability_error_pp", "policy_proxy_like_E_F_probability_error_pp",
    ]
    policy = model_table[policy_columns].sort_values("policy_proxy_like_MAE_pts", na_position="last")
    sample_view = sample_table[
        [column for column in [
            "sample", "model", "n", "retention_vs_authoritative_pct", "MAE_pts",
            "calibration_slope", "B_C_balanced_accuracy", "E_F_balanced_accuracy",
            "formal_full_frame_scope_pct", "formal_policy_proxy_scope_pct",
            "selection_uses_old_outcome", "reason",
        ] if column in sample_table]
    ]
    inventory_view = inventory[
        ["field_name", "availability", "missingness_pct",
         "available_at_full_register_application_time", "usable_without_outcome_leakage",
         "expected_relevance"]
    ]
    formula_note = inventory.loc[
        inventory.field_name == "asset_rating_formula_audit", "notes"
    ].iloc[0]
    file_inventory = inventory[inventory.field_name.isin([
        "raw EPC XML", "BRUKL reports", "SBEM/NCM project inputs",
        "assessor software exports", "software_version_or_calculation_tool",
        "heat_pump_vs_resistance_system",
    ])][["field_name", "availability", "n_rows", "expected_relevance", "notes"]]
    strict_sample = sample_table[
        (sample_table["sample"] == "strict_all_observed_descriptors")
        & (sample_table.model == "spline_ridge_rich")
    ].iloc[0]
    threshold_best = threshold_table.sort_values(
        ["threshold", "balanced_accuracy"], ascending=[True, False]
    ).groupby("threshold", observed=True).head(5)[
        ["threshold", "model", "balanced_accuracy", "sensitivity", "specificity",
         "PPV", "NPV", "aggregate_probability_error_pp", "brier_score"]
    ]
    direct_fuel_threshold = threshold_table[
        threshold_table.model == "direct_threshold_fuel_logit"
    ][["threshold", "balanced_accuracy", "PPV", "NPV",
       "aggregate_probability_error_pp", "brier_score"]]
    direct_fuel_model = model_table.set_index("model").loc["direct_threshold_fuel_logit"]
    gate_text = (
        f"MAE <= {RATING_MAE_GATE:g}; rating calibration slope >= {RATING_SLOPE_GATE:.2f}; "
        f"B/C and E/F balanced accuracy >= {THRESHOLD_BALANCED_ACCURACY_GATE:.2f}; "
        f"aggregate probability error within +/-{AGGREGATE_PROBABILITY_ERROR_GATE_PP:g} pp; "
        f"maximum reported subgroup error <= {SUBGROUP_CALIBRATION_GATE_PP:g} pp; "
        f"sample retention >= {100*MIN_SAMPLE_RETENTION:.0f}% and formal application coverage >= {100*MIN_APPLICATION_COVERAGE:.0f}%."
    )
    conclusion = (
        "The MAE limitation is not rescued for individual-rating claims."
        if not model_table.individual_rating_claim_supported.any()
        else "At least one model clears the predeclared individual-rating gates."
    )
    threshold_conclusion = (
        "No candidate clears the joint individual B/C and E/F classification-and-calibration gates."
        if not model_table.individual_threshold_claim_supported.any()
        else "At least one candidate clears the joint individual threshold gates."
    )
    report = f"""# Bridge MAE rescue analysis

## Executive decision

{conclusion} The committed fuel-only multiplicative baseline has an out-of-fold MAE of **{baseline.MAE_pts:.2f} rating points**. The best point model is **{best.model}** at **{best.MAE_pts:.2f} points**, a **{relative_reduction:.1f}%** reduction. Its calibration slope is **{best.calibration_slope:.3f}**. {threshold_conclusion}

Same-method repeats demonstrate that this is not only a bridge-specification problem. Even when fuel, floor area, recommendation count, sector/use and AC status are unchanged, short-gap reassessments have material rating dispersion. These samples are a practical reassessment-noise benchmark, not a proof of zero physical change; unrecorded inputs, assessor choices and genuine minor changes remain inseparable in the public register.

The strongest defensible paper is therefore still an **aggregate, register-conditional threshold-probability analysis with model and sample sensitivity**, not a certificate-level reconstruction of exact old-basis ratings or individual reclassification.

## Scope and audit trail

- Starting authoritative commit: `209f746`.
- Exact authoritative cross-method sample: **{audit['n_heldout_eligible']:,} unique UPRNs**, one first/latest pair per UPRN.
- Random seed: **{RANDOM_SEED}**; outer folds: **{N_OUTER_FOLDS}**; nested tuning/calibration folds: **{N_INNER_FOLDS}**.
- All validation predictions are out of sample. Outer assignment is deterministic at UPRN level. Same-method inner validation also groups repeated UPRNs.
- Prediction error is `predicted old-basis rating - observed old-basis rating`; positive values mean a numerically worse predicted rating.
- Predictors use the post-revision certificate and fields available with it. Old rating, old components and later certificates are outcomes only. The explicitly outcome-selected movement-trim sample is labelled diagnostic and cannot justify a main claim.
- Existing `bridge_*` files were hashed before and after this module: **{'unchanged' if bridge_hashes_unchanged else 'CHANGED — audit failure'}**.

## Predeclared gates

{gate_text}

These are joint gates. A low MAE alone does not establish useful individual threshold classification, and good discrimination alone does not establish calibrated aggregate probabilities.

## Data inventory and richer inputs

The processed register contains emissions components, primary energy, detailed fuel text, use/sector, AC and environment descriptors, recommendation codes, and full recommendation text. However, it does not contain the complete SBEM building model. In particular, there is no explicit installed-system variable that reliably separates heat pumps from resistance electric heating, no assessor software/version field, and no geometry/zoning/load-schedule input.

The asset rating is algebraically almost determined by recorded building and standard emissions rates, so using those post fields adds little independent information; predicting their *old-method* values remains the hard task. Formula audit: **{formula_note}**. Full inventory: `outputs/tables/mae_rescue_data_inventory.csv`.

{_markdown_table(inventory_view.head(28))}

The targeted local file/archive search found:

{_markdown_table(file_inventory)}

## Same-method reassessment-noise benchmark

The identity model predicts the later same-method rating with the earlier one. The descriptor ridge is nested and out of sample. The short-gap strict results below are the most relevant empirical noise benchmark.

For this table only, error is `predicted later rating - observed later rating`.

{_markdown_table(short_noise)}

The basic descriptor ridge reduces six-month MAE only to **{ridge_six_month.loc['pre_pre', 'MAE_pts']:.2f}** points in pre/pre repeats and **{ridge_six_month.loc['post_post', 'MAE_pts']:.2f}** in post/post repeats. It removes the mean time-trend bias and restores the calibration slope, but does not eliminate certificate-level dispersion.

This variation places a non-trivial lower-bound-style benchmark under the cross-method problem. It is not a formal irreducible-error variance decomposition: the public register cannot verify that all physical and operational inputs are identical. Nevertheless, an empirical bridge trained on the same public fields cannot be expected to recover information that already changes within same-method stable-descriptor reassessments.

## Model ladder

{_markdown_table(ranking)}

Transparent fuel-only models are retained as baselines. Post-rating splines, register descriptors, recommendation text/codes, BER/SER component construction, isotonic calibration and diagnostic tree ensembles test whether observable nonlinearity or extra public fields close the gap. Hyperparameters are chosen only inside outer training folds.

### Threshold usefulness

{_markdown_table(threshold_best)}

Hard thresholding of a point prediction is distinguished from a calibrated probability model. Aggregate probability error can be small while individual classifications or subgroups remain poor, and vice versa.

Positive classes are **old-basis B-or-better** at B/C and **old-basis F/G** at E/F.

The direct fuel-specific logistic model is the clearest example:

{_markdown_table(direct_fuel_threshold)}

It clears the headline balanced-accuracy and aggregate-probability gates for both thresholds in the full calibration sample. It does **not** rescue individual claims: B/C has a maximum reported subgroup calibration error of **{direct_fuel_model.B_C_max_abs_subgroup_error_pp:.1f} pp**; the F/G positive predictive value is only **{100*direct_fuel_threshold.loc[direct_fuel_threshold.threshold == 'E_F', 'PPV'].iloc[0]:.1f}%**; and in the policy-proxy-like subset balanced accuracy falls to **{direct_fuel_model.policy_proxy_like_B_C_balanced_accuracy:.3f}** at B/C and **{direct_fuel_model.policy_proxy_like_E_F_balanced_accuracy:.3f}** at E/F. This model remains useful for aggregate probability estimation, not for declaring a particular UPRN reclassified.

### Policy-proxy-like calibration subset

This validation subset uses post certificates with a property-to-let transaction label, floor area above 1,000 m2 and no construction label. It is only a transaction/size proxy, not verified private-rental tenure or legal applicability.

{_markdown_table(policy)}

## Calibration-sample ladder

{_markdown_table(sample_view)}

The ladder was predeclared before comparing results. The strict observed-descriptor sample reaches **{strict_sample.MAE_pts:.2f}** MAE with slope **{strict_sample.calibration_slope:.3f}**, but retains only **{strict_sample.retention_vs_authoritative_pct:.1f}%** of the authoritative calibration frame (n={int(strict_sample.n):,}) and still misses the B/C discrimination gate. This is evidence that tighter input stability helps, not a general-register rescue. The outcome-trimmed diagnostic is not eligible for substantive selection. Gap, stability and threshold-local restrictions are reported with their retained n and their formal register-frame scope; a restriction that improves MAE by discarding relevant cases does not establish a general bridge.

## Same-input SBEM/NCM rerun feasibility

No pilot table is reported because no valid identical-input pair could be executed. Calling two different old/new project files “paired” would be invalid. Official sources establish that the current and previous engines exist, but the old-version page restricts use to exceptional legacy-project circumstances, the front ends require Windows/Microsoft Access, and no legally reusable public `.nct` test case was located.

{_markdown_table(feasibility[['source', 'finding', 'same_input_pair_usable_now', 'blocker']])}

A valid next step would require (1) written HSE/BRE permission for v5.6b research use, (2) a legally shareable complete `.nct` input, and (3) a Windows/Access environment. The identical input should be opened/run under both v5.6b and v6.1, with every conversion or newly required field logged. A representative rerun panel would directly identify methodology effects and is the clearest route beyond public-register reassessment noise.

## Answers to the eight questions

{_markdown_table(summary)}

## Claims after this analysis

Defensible:

- The fuel-only bridge is a transparent baseline, not an individual-rating reconstruction.
- Cross-fitted aggregate threshold probabilities can be reported, conditional on the EPC register, with bridge/sample structural ranges and subgroup calibration disclosed.
- Same-method stable-descriptor repeats document a substantial register reassessment-noise benchmark.

Not defensible from the public register alone:

- An exact pre-2022-method rating for a particular post-2022 certificate.
- A claim that a particular UPRN crossed B/C or E/F solely because of the accounting revision unless a validated classifier clears the relevant individual gate.
- A claim that the empirical bridge reruns SBEM/NCM or holds all physical inputs fixed.
- A legal-compliance claim, a physical-building census claim, or an exact retrofit cost attached to an individual inferred crossing.

## Output map

- `mae_rescue_data_inventory.csv`: local data/files/fields and leakage/applicability audit.
- `mae_rescue_same_method_noise.csv`: same-method identity and descriptor-model noise by period, gap, fuel and band.
- `mae_rescue_sample_ladder.csv`: systematic cross-method sample restrictions and retained scope.
- `mae_rescue_model_ladder.csv`: nested-OOF point/model ranking and gates.
- `mae_rescue_threshold_performance.csv`: full B/C and E/F confusion/performance metrics.
- `mae_rescue_subgroup_calibration.csv`: threshold calibration and rating-error diagnostics by fuel, bands, sector, size, AC, threshold distance and policy proxy.
- `mae_rescue_sbem_feasibility_sources.csv`: official/local same-input rerun search.
- `mae_rescue_summary.csv`: question-level decisions and evidence.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    validate_mae_rescue_inputs()
    ensure_dirs()
    protected_before = {
        path: _sha256(path)
        for path in OUT_TABLES.glob("bridge_*") if path.is_file()
    }
    protected_before.update({
        path: _sha256(path)
        for path in OUT_TABLES.glob("expenditure_equivalent_*") if path.is_file()
    })

    inventory = build_data_inventory()
    feasibility = sbem_feasibility_sources()
    first_latest = extract_cross_method_pairs("first_latest")
    nearest = extract_cross_method_pairs("nearest")
    authoritative, audit = authoritative_enriched_sample(first_latest)
    same_method_pairs = extract_same_method_pairs()
    same_method = run_same_method_noise(same_method_pairs)
    latest = prepare_latest_frame()
    sample_table = run_sample_ladder(authoritative, first_latest, nearest, latest)
    model_table, threshold_table, subgroup_table, _ = run_model_ladder(authoritative)
    summary = build_summary(model_table, sample_table, same_method, feasibility)

    outputs = {
        "mae_rescue_data_inventory.csv": inventory,
        "mae_rescue_same_method_noise.csv": same_method,
        "mae_rescue_sample_ladder.csv": sample_table,
        "mae_rescue_model_ladder.csv": model_table,
        "mae_rescue_threshold_performance.csv": threshold_table,
        "mae_rescue_subgroup_calibration.csv": subgroup_table,
        "mae_rescue_sbem_feasibility_sources.csv": feasibility,
        "mae_rescue_summary.csv": summary,
    }
    for filename, table in outputs.items():
        table.to_csv(OUT_TABLES / filename, index=False)

    protected_after = {path: _sha256(path) for path in protected_before}
    unchanged = protected_before == protected_after
    if not unchanged:
        changed = [str(path) for path in protected_before if protected_before[path] != protected_after[path]]
        raise AssertionError(f"Protected authoritative outputs changed: {changed}")
    write_report(
        audit, inventory, same_method, sample_table, model_table,
        threshold_table, subgroup_table, feasibility, summary, unchanged,
    )
    print(json.dumps({
        "report": str(REPORT_PATH),
        "tables": {name: len(table) for name, table in outputs.items()},
        "authoritative_n": audit["n_heldout_eligible"],
        "protected_outputs_unchanged": unchanged,
    }, indent=2))


if __name__ == "__main__":
    main()
