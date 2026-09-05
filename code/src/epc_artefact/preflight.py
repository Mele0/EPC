"""Fail-fast validation for the immutable manuscript input bundle.

The publication entry point calls this module before any long-running analysis.
It prevents a missing or partial Data Asset from triggering mutable web fallbacks,
and it verifies that the release inputs match the checksums and structural
contracts used for the manuscript.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import DATA_DIR, FRANCE_INPUT_PARQUET, INPUT_EXTERNAL_DIR, UNIFIED_PARQUET
from .data import ANALYSIS_COLUMNS, EXTENDED_COLUMNS, EXTENDED_FRAME, verify_checksum


CHECKSUM_MANIFEST = DATA_DIR / "input_checksums.sha256"
MANUSCRIPT_EW_CERTIFICATES = 1_245_956
MANUSCRIPT_EW_UPRNS = 842_673

CORE_EXTERNAL_FILES = (
    "gdp_deflator_hmt_jun2026.csv",
    "ipf_cost_schedule.csv",
    "sector_cost_crosswalk.csv",
    "nd_need_2025_supporting_tables.xlsx",
    "nd_need_2024_geographical_annex.xlsx",
    "owid_carbon_intensity_electricity.csv",
)
MAP_EXTERNAL_FILES = (
    "ne_50m_admin_0_countries.geojson",
    "ne_50m_admin_0_map_subunits.geojson",
)
METERED_DERIVED_FILES = (
    "metered_nondomestic_electricity_EW.csv",
    "metered_nondomestic_gas_EW.csv",
)
MANIFEST_CORE_REQUIRED_FILES = (
    "nd_artefact_extended.parquet",
    *(f"external/{name}" for name in CORE_EXTERNAL_FILES),
    *(f"external/{name}" for name in METERED_DERIVED_FILES),
)
MANIFEST_FRANCE_REQUIRED_FILES = (
    "france/dpe_window_2025H2_2026H1.parquet",
)
MANIFEST_MAE_REQUIRED_FILES = (
    "non_domestic_epc_unified_certificate_level.parquet",
    "recommendations_long_clean.parquet",
)


class InputValidationError(RuntimeError):
    """Raised before analysis when the release Data Asset is incomplete or changed."""


def _manifest_entries(path: Path) -> list[tuple[str, Path]]:
    if not path.is_file():
        raise InputValidationError(
            f"Missing input checksum manifest: {path}. Build the Data Asset from "
            "the repository data/ layout documented in docs/data_manifest.md.")
    entries: list[tuple[str, Path]] = []
    for line_number, raw in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            digest, relative_text = line.split(maxsplit=1)
        except ValueError as error:
            raise InputValidationError(
                f"Malformed checksum manifest line {line_number}: {raw!r}") from error
        relative = Path(relative_text.lstrip("*"))
        if relative.is_absolute() or ".." in relative.parts:
            raise InputValidationError(
                f"Unsafe checksum-manifest path on line {line_number}: {relative}")
        entries.append((digest, DATA_DIR / relative))
    if not entries:
        raise InputValidationError(f"Input checksum manifest is empty: {path}")
    return entries


def _validate_manifest(errors: list[str], *, include_map: bool,
                       include_france: bool, include_mae_rescue: bool) -> None:
    try:
        entries = _manifest_entries(CHECKSUM_MANIFEST)
    except (InputValidationError, OSError) as error:
        errors.append(str(error))
        return
    for digest, path in entries:
        try:
            verify_checksum(path, digest)
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
            errors.append(str(error))
    recorded = {str(path.relative_to(DATA_DIR)) for _, path in entries}
    required = set(MANIFEST_CORE_REQUIRED_FILES)
    if include_map:
        required.update(f"external/{name}" for name in MAP_EXTERNAL_FILES)
    if include_france:
        required.update(MANIFEST_FRANCE_REQUIRED_FILES)
    if include_mae_rescue:
        required.update(MANIFEST_MAE_REQUIRED_FILES)
    missing = sorted(required - recorded)
    if missing:
        errors.append(f"Input checksum manifest omits required files: {missing}")


def _validate_ew_release(errors: list[str]) -> None:
    if not EXTENDED_FRAME.is_file():
        if UNIFIED_PARQUET.is_file():
            errors.append(
                f"The raw register exists at {UNIFIED_PARQUET}, but the publication "
                f"run requires the frozen extended frame at {EXTENDED_FRAME}.")
        else:
            errors.append(f"Missing frozen England/Wales frame: {EXTENDED_FRAME}")
        return
    try:
        parquet = pq.ParquetFile(EXTENDED_FRAME)
        columns = set(parquet.schema_arrow.names)
        missing = sorted((set(ANALYSIS_COLUMNS) | EXTENDED_COLUMNS) - columns)
        if missing:
            errors.append(f"{EXTENDED_FRAME} is missing required columns: {missing}")
        if parquet.metadata.num_rows != MANUSCRIPT_EW_CERTIFICATES:
            errors.append(
                f"{EXTENDED_FRAME} has {parquet.metadata.num_rows:,} rows; expected "
                f"{MANUSCRIPT_EW_CERTIFICATES:,}.")
        uprns = pq.read_table(EXTENDED_FRAME, columns=["uprn"])["uprn"]
        unique_uprns = int(pc.count_distinct(uprns).as_py())
        if unique_uprns != MANUSCRIPT_EW_UPRNS:
            errors.append(
                f"{EXTENDED_FRAME} has {unique_uprns:,} unique UPRNs; expected "
                f"{MANUSCRIPT_EW_UPRNS:,}.")
    except Exception as error:  # pyarrow emits several format/schema error types
        errors.append(f"Cannot validate {EXTENDED_FRAME}: {error}")


def _require_external(errors: list[str], names: tuple[str, ...]) -> None:
    for name in names:
        path = INPUT_EXTERNAL_DIR / name
        if not path.is_file():
            errors.append(f"Missing frozen external input: {path}")


def _validate_metered_inputs(errors: list[str]) -> None:
    derived = [INPUT_EXTERNAL_DIR / name for name in METERED_DERIVED_FILES]
    if not all(path.is_file() for path in derived):
        errors.append(
            "Missing frozen DESNZ metered-energy pair: "
            f"{', '.join(str(path) for path in derived)}.")


def _validate_france_release(errors: list[str]) -> None:
    from .france import FIELDS, MANUSCRIPT_WINDOW_RECORDS

    if not FRANCE_INPUT_PARQUET.is_file():
        errors.append(
            f"Missing frozen France manuscript snapshot: {FRANCE_INPUT_PARQUET}. "
            "The live ADEME dataset is mutable and is not accepted for a publication run.")
        return
    try:
        verify_checksum(FRANCE_INPUT_PARQUET, require_checksum=True)
        parquet = pq.ParquetFile(FRANCE_INPUT_PARQUET)
        missing = sorted(set(FIELDS) - set(parquet.schema_arrow.names))
        if missing:
            errors.append(f"{FRANCE_INPUT_PARQUET} is missing required columns: {missing}")
        if parquet.metadata.num_rows != MANUSCRIPT_WINDOW_RECORDS:
            errors.append(
                f"{FRANCE_INPUT_PARQUET} has {parquet.metadata.num_rows:,} rows; expected "
                f"{MANUSCRIPT_WINDOW_RECORDS:,}.")
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        errors.append(str(error))


def validate_release_inputs(*, include_france: bool = True,
                            include_map: bool = True,
                            include_mae_rescue: bool = False) -> None:
    """Validate every immutable input needed by the requested publication stages."""
    errors: list[str] = []
    _validate_manifest(
        errors, include_map=include_map, include_france=include_france,
        include_mae_rescue=include_mae_rescue)
    _validate_ew_release(errors)
    _require_external(errors, CORE_EXTERNAL_FILES)
    _validate_metered_inputs(errors)
    if include_map:
        _require_external(errors, MAP_EXTERNAL_FILES)
    if include_france:
        _validate_france_release(errors)
    if include_mae_rescue:
        recommendations = DATA_DIR / "recommendations_long_clean.parquet"
        if not UNIFIED_PARQUET.is_file():
            errors.append(f"MAE-rescue input is missing: {UNIFIED_PARQUET}")
        if not recommendations.is_file():
            errors.append(f"MAE-rescue input is missing: {recommendations}")
        if UNIFIED_PARQUET.is_file() and recommendations.is_file():
            try:
                from .mae_rescue import validate_mae_rescue_inputs
                validate_mae_rescue_inputs()
            except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
                errors.append(str(error))
    if errors:
        detail = "\n  - ".join(errors)
        raise InputValidationError(f"Release input validation failed:\n  - {detail}")
