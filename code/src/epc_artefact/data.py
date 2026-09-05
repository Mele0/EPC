"""Data layer: load and clean the non-domestic EPC register, and read the DESNZ
metered-energy series used for ground-truth validation.

Shared helpers (``broad_band``, ``with_dates``) live here because they are used by
both the analysis and figure modules.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import ANALYSIS_FRAME, DATA_DIR, UNIFIED_PARQUET, EW_CODE


EXTENDED_FRAME = DATA_DIR / "nd_artefact_extended.parquet"
EXTENDED_CACHE = ANALYSIS_FRAME.with_name("nd_artefact_extended.parquet")
ANALYSIS_COLUMNS = [
    "certificate_number", "uprn", "asset_rating", "asset_rating_band_clean",
    "property_type_clean", "inspection_date", "inspection_year",
    "main_heating_fuel_clean", "aircon_present_clean", "floor_area",
    "standard_emissions", "building_emissions", "typical_emissions",
    "primary_energy_value", "transaction_type_clean", "n_recommendations",
    "invalid_asset_rating", "extreme_asset_rating", "missing_asset_rating",
]
EXTENDED_COLUMNS = {
    *ANALYSIS_COLUMNS,
    "local_authority", "local_authority_label", "country", "lodgement_date",
    "payback_triple", "rec_codes",
}


class DownloadChecksumError(RuntimeError):
    """Raised when a cached or newly downloaded input fails SHA-256 verification."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of ``path`` without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_sidecar(path: Path) -> Path:
    """Return the standard checksum sidecar path for a downloaded input."""
    path = Path(path)
    return path.with_name(path.name + ".sha256")


def source_checksum_marker(path: Path) -> Path:
    """Return the provenance marker paired with a derived cache."""
    path = Path(path)
    return path.with_name(path.name + ".source.sha256")


def _normalise_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    digest = value.strip().split()[0].lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"Invalid SHA-256 digest: {value!r}")
    return digest


def _read_checksum_sidecar(path: Path) -> str | None:
    sidecar = checksum_sidecar(path)
    if not sidecar.exists():
        return None
    return _normalise_sha256(sidecar.read_text(encoding="ascii"))


def _fsync_directory(path: Path) -> None:
    """Persist a completed rename on filesystems that support directory fsync."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_checksum_sidecar(path: Path, digest: str | None = None) -> Path:
    """Atomically write a sha256sum-compatible checksum sidecar for ``path``."""
    path = Path(path)
    digest = _normalise_sha256(digest) if digest is not None else sha256_file(path)
    sidecar = checksum_sidecar(path)
    _atomic_write_text(sidecar, f"{digest}  {path.name}\n")
    return sidecar


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a derived parquet cache without exposing a partial final file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        write_checksum_sidecar(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_source_marker(marker: Path, source: Path, digest: str) -> None:
    _atomic_write_text(marker, f"{_normalise_sha256(digest)}  {source.name}\n")


def _cache_matches_source(cache: Path, marker: Path, source: Path) -> bool:
    """Return true only for a complete cache derived from the current immutable input."""
    if not cache.is_file() or not marker.is_file():
        return False
    try:
        recorded_source = _normalise_sha256(marker.read_text(encoding="ascii"))
        if recorded_source != sha256_file(source):
            return False
        verify_checksum(cache, require_checksum=True)
        return True
    except (OSError, ValueError, DownloadChecksumError):
        return False


def verify_checksum(path: Path, expected_sha256: str | None = None,
                    *, require_checksum: bool = False) -> str:
    """Verify ``path`` against an explicit digest or its ``.sha256`` sidecar.

    The explicit digest takes precedence.  The actual digest is returned so callers
    can record it after a trusted, first-time import.  Missing checksum metadata is
    only accepted when ``require_checksum`` is false.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = (_normalise_sha256(expected_sha256)
                if expected_sha256 is not None else _read_checksum_sidecar(path))
    if expected is None and require_checksum:
        raise DownloadChecksumError(
            f"No SHA-256 checksum supplied for {path}; provide expected_sha256 or "
            f"the sidecar {checksum_sidecar(path)}")
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise DownloadChecksumError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def verify_or_record_checksum(path: Path, expected_sha256: str | None = None) -> str:
    """Verify existing checksum metadata, or record a sidecar for a trusted input."""
    path = Path(path)
    expected = expected_sha256 if expected_sha256 is not None else _read_checksum_sidecar(path)
    actual = verify_checksum(path, expected)
    if expected is None:
        write_checksum_sidecar(path, actual)
    return actual


def fuel_group(series: pd.Series) -> np.ndarray:
    """Group the cleaned main-heating-fuel field into Electric / Gas / Other."""
    f = series.astype(str).str.lower()
    return np.where(f.str.contains("electr"), "Electric",
                    np.where(f.str.contains("gas"), "Gas", "Other"))


def broad_band(ar: np.ndarray) -> np.ndarray:
    """Broad EPC class from asset rating (A/B<=50, C<=75, D<=100, E<=125, F/G>125)."""
    return np.select([ar <= 50, ar <= 75, ar <= 100, ar <= 125], ["A/B", "C", "D", "E"], "F/G")


def with_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse inspection dates, drop unusable rows, and sort chronologically."""
    df = df.copy()
    df["insp_dt"] = pd.to_datetime(df["inspection_date"], errors="coerce")
    return df.dropna(subset=["insp_dt", "asset_rating", "uprn"]).sort_values("insp_dt")


def build_analysis_frame(force: bool = False) -> pd.DataFrame:
    """Load the unified non-domestic register, clean it, tag fuel; cache to parquet."""
    sources = (EXTENDED_FRAME, UNIFIED_PARQUET)
    source = next((candidate for candidate in sources if candidate.exists()), None)
    if source is None:
        raise FileNotFoundError(
            f"Missing input register. Expected {UNIFIED_PARQUET} or the release-ready "
            f"extended frame at {EXTENDED_FRAME}. See README for data provenance or "
            "set EPC_DATA_DIR to the directory that contains it."
        )
    if not force and _cache_matches_source(
            ANALYSIS_FRAME, source_checksum_marker(ANALYSIS_FRAME), source):
        return pd.read_parquet(ANALYSIS_FRAME)
    available = set(pq.ParquetFile(source).schema_arrow.names)
    missing = sorted(set(ANALYSIS_COLUMNS) - available)
    if missing:
        raise ValueError(f"Input register {source} is missing required columns: {missing}")
    df = pq.read_table(source, columns=ANALYSIS_COLUMNS).to_pandas()
    for c in ("invalid_asset_rating", "extreme_asset_rating", "missing_asset_rating"):
        df[c] = df[c].astype("boolean").fillna(False)
    keep = (~df.invalid_asset_rating) & (~df.extreme_asset_rating) & (~df.missing_asset_rating)
    df = df[keep & df.asset_rating.notna() & (df.floor_area > 0)]
    df = df[(df.inspection_year >= 2012) & (df.inspection_year <= 2025)].copy()
    df["fuelgrp"] = fuel_group(df["main_heating_fuel_clean"])
    source_digest = sha256_file(source)
    _atomic_write_parquet(df, ANALYSIS_FRAME)
    _write_source_marker(source_checksum_marker(ANALYSIS_FRAME), source, source_digest)
    return df


def build_extended_frame(force: bool = False) -> pd.DataFrame:
    """Analysis frame plus the extra register fields used by the robustness
    checks: country (from the local-authority ONS code), lodgement date (EPC
    validity runs 10 years from lodgement), the short/medium/long payback triple
    and the pipe-joined set of recommendation codes (for the stricter
    no-recorded-works calibration). Cached alongside the analysis frame."""
    if EXTENDED_FRAME.exists() and (not force or not UNIFIED_PARQUET.exists()):
        cached = pd.read_parquet(EXTENDED_FRAME)
        missing = sorted(EXTENDED_COLUMNS - set(cached.columns))
        if not missing:
            return cached
        raise ValueError(
            f"Extended release frame {EXTENDED_FRAME} is missing required columns: {missing}")
    if UNIFIED_PARQUET.exists() and not force and _cache_matches_source(
            EXTENDED_CACHE, source_checksum_marker(EXTENDED_CACHE), UNIFIED_PARQUET):
        cached = pd.read_parquet(EXTENDED_CACHE)
        missing = sorted(EXTENDED_COLUMNS - set(cached.columns))
        if not missing:
            return cached
    df = build_analysis_frame()
    if not UNIFIED_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing both a complete extended release frame at {EXTENDED_FRAME} and the raw "
            f"input register at {UNIFIED_PARQUET} needed to build one."
        )
    extra_cols = ["certificate_number", "local_authority", "local_authority_label",
                  "lodgement_date",
                  "n_short_payback", "n_medium_payback", "n_long_payback",
                  "unique_recommendation_codes"]
    ex = pq.read_table(UNIFIED_PARQUET, columns=extra_cols).to_pandas().drop_duplicates("certificate_number")
    la = ex.local_authority.astype(str)
    ex["country"] = np.where(la.str.startswith("W"), "Wales",
                             np.where(la.str.startswith("E"), "England", "Other"))
    ex["payback_triple"] = (ex.n_short_payback.astype(str) + "_"
                            + ex.n_medium_payback.astype(str) + "_"
                            + ex.n_long_payback.astype(str))
    ex["rec_codes"] = ex.unique_recommendation_codes.astype(str)
    out = df.merge(ex[["certificate_number", "local_authority", "local_authority_label",
                       "country", "lodgement_date",
                       "payback_triple", "rec_codes"]], on="certificate_number", how="left")
    source_digest = sha256_file(UNIFIED_PARQUET)
    _atomic_write_parquet(out, EXTENDED_CACHE)
    _write_source_marker(source_checksum_marker(EXTENDED_CACHE), UNIFIED_PARQUET, source_digest)
    return out


def download_if_missing(url: str, path: Path, expected_sha256: str | None = None,
                        *, timeout: int = 180) -> Path:
    """Atomically download an immutable input and verify its SHA-256 checksum.

    ``expected_sha256`` may be supplied from a release manifest.  When it is omitted,
    the function verifies an existing ``<filename>.sha256`` sidecar.  For a trusted
    first download (or a pre-provisioned file without a sidecar), it records such a
    sidecar so later runs fail clearly if the cache changes.  A failed or interrupted
    transfer never replaces the last complete file.
    """
    path = Path(path)
    if path.exists():
        verify_or_record_checksum(path, expected_sha256)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    expected = _normalise_sha256(expected_sha256)
    if expected is None:
        expected = _read_checksum_sidecar(path)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "epc-research-artefact"})
        with os.fdopen(descriptor, "wb") as target, \
                urllib.request.urlopen(request, timeout=timeout) as response:
            digest = hashlib.sha256()
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
                digest.update(block)
            target.flush()
            os.fsync(target.fileno())
        actual = digest.hexdigest()
        if expected is not None and actual != expected:
            raise DownloadChecksumError(
                f"SHA-256 mismatch downloading {url}: expected {expected}, got {actual}")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        write_checksum_sidecar(path, actual)
        return path
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def desnz_nd_per_meter(xlsx: Path, kind: str) -> pd.DataFrame:
    """England & Wales non-domestic mean consumption per meter, by year, from a DESNZ workbook."""
    xl = pd.ExcelFile(xlsx)
    raw = pd.read_excel(xl, sheet_name="2024", header=None, nrows=8)
    hrow = raw.index[raw.apply(lambda r: r.astype(str).str.contains("Code", case=False).any(), axis=1)][0]
    hdr = pd.read_excel(xl, sheet_name="2024", header=None, skiprows=hrow, nrows=1).iloc[0].astype(str).tolist()
    col = next(i for i, h in enumerate(hdr)
               if all(t in h.lower().replace("\n", " ") for t in ("non", "mean", "consumption")))
    rows = []
    for y in range(2012, 2025):
        try:
            d = pd.read_excel(xl, sheet_name=str(y), skiprows=hrow)
            d.columns = [f"c{i}" for i in range(len(d.columns))]
            ew = d[d.c0.astype(str).str.strip() == EW_CODE]
            if len(ew):
                rows.append([y, float(ew.iloc[0][f"c{col}"])])
        except Exception:
            pass
    return pd.DataFrame(rows, columns=["year", f"nd_{kind}_mean_kWh_per_meter"]).dropna().sort_values("year")
