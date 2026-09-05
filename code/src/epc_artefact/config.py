"""Paths and constants for the EPC carbon-accounting artefact analysis.

The three writable domains are deliberately separate:

* immutable inputs: ``EPC_DATA_DIR`` (``<repo>/data`` locally, ``/data`` in a
  Code Ocean Reproducible Run);
* generated results: ``EPC_OUTPUT_DIR`` (``<repo>/outputs`` locally,
  ``/results`` in Code Ocean); and
* disposable caches: ``EPC_SCRATCH_DIR`` (``<repo>/.cache`` locally,
  ``/scratch`` in Code Ocean).

The ``code/run`` driver sets the Code Ocean paths explicitly.  Keeping caches
out of the data directory means a Capsule never needs to write to a mounted
Data Asset.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CODE_ROOT.parent
DATA_DIR = Path(os.environ.get("EPC_DATA_DIR", REPO_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("EPC_OUTPUT_DIR", REPO_ROOT / "outputs"))
SCRATCH_DIR = Path(os.environ.get("EPC_SCRATCH_DIR", REPO_ROOT / ".cache"))
INPUT_EXTERNAL_DIR = DATA_DIR / "external"
EXTERNAL_DIR = SCRATCH_DIR / "external"
OUT_TABLES = OUTPUT_DIR / "tables"
# Doubly robust estimator results reported in Table 2 Panel B and Supplementary S29-S37.
# See doubly_robust_estimates/README.md.
DR_ESTIMATES_DIR = CODE_ROOT / "doubly_robust_estimates"
OUT_FIGURES = OUTPUT_DIR / "figures"

# Input: cleaned non-domestic EPC register at certificate level (see README for provenance).
UNIFIED_PARQUET = DATA_DIR / "non_domestic_epc_unified_certificate_level.parquet"
ANALYSIS_FRAME = SCRATCH_DIR / "nd_artefact_analysis.parquet"  # cached analytic subset

# DESNZ subnational consumption statistics (downloaded automatically if absent).
DESNZ_ELEC_URL = "https://assets.publishing.service.gov.uk/media/69426e905431f4f94d7f0b5e/Subnational_electricity_consumption_statistics_2005-2024.xlsx"
DESNZ_GAS_URL = "https://assets.publishing.service.gov.uk/media/69f9bf208f3213f7384ecae6/Subnational_gas_consumption_statistics_2005-2024.xlsx"
DESNZ_ELEC_XLSX = EXTERNAL_DIR / "subnat_elec_2005_2024.xlsx"
DESNZ_GAS_XLSX = EXTERNAL_DIR / "subnat_gas_2005_2024.xlsx"

# Independent administrative benchmark for register-frame representativeness.
# The main 2025 workbook provides building-use and size margins; the latest
# published geographical annex supplies region margins (stock at March 2024).
ND_NEED_2025_URL = (
    "https://assets.publishing.service.gov.uk/media/68a5c772a6acbbc7fb96a3ce/"
    "Non-domestic_National_Energy_Efficiency_Data-Framework_2025_supporting_data_tables.xlsx"
)
ND_NEED_GEO_2024_URL = (
    "https://assets.publishing.service.gov.uk/media/6762b3a3ff2c870561bde777/"
    "ND-NEED_2024_Geographical_Annex_Data_Tables_-_final.xlsx"
)
ND_NEED_2025_XLSX = EXTERNAL_DIR / "nd_need_2025_supporting_tables.xlsx"
ND_NEED_GEO_2024_XLSX = EXTERNAL_DIR / "nd_need_2024_geographical_annex.xlsx"

# International structural-exposure typology (Figure 3 / map).
OWID_CARBON_INTENSITY_URL = "https://ourworldindata.org/grapher/carbon-intensity-electricity.csv"
OWID_CARBON_INTENSITY_CSV = EXTERNAL_DIR / "owid_carbon_intensity_electricity.csv"
NE_COUNTRIES_GEOJSON_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
                            "master/geojson/ne_50m_admin_0_countries.geojson")
NE_COUNTRIES_GEOJSON = EXTERNAL_DIR / "ne_50m_admin_0_countries.geojson"
NE_SUBUNITS_GEOJSON_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
                           "master/geojson/ne_50m_admin_0_map_subunits.geojson")
NE_SUBUNITS_GEOJSON = EXTERNAL_DIR / "ne_50m_admin_0_map_subunits.geojson"
TYPOLOGY_START_YEAR = 2015
TYPOLOGY_END_YEAR = 2023

# France DPE 2026 coefficient reform (factor-revision replication, Supplementary).
DPE_DATASET = "dpe03existant"        # ADEME data-fair dataset id (existing-dwelling DPE)
DPE_API_BASE = f"https://data.ademe.fr/data-fair/api/v1/datasets/{DPE_DATASET}/lines"
FRANCE_INPUT_PARQUET = DATA_DIR / "france" / "dpe_window_2025H2_2026H1.parquet"
FRANCE_CACHE_PARQUET = SCRATCH_DIR / "france" / "dpe_window_2025H2_2026H1.parquet"
# Prefer the immutable manuscript snapshot.  Download to scratch only when a
# frozen snapshot has not been supplied.
FRANCE_WINDOW_PARQUET = (
    FRANCE_INPUT_PARQUET if FRANCE_INPUT_PARQUET.exists() else FRANCE_CACHE_PARQUET
)
FRANCE_DATA_DIR = FRANCE_WINDOW_PARQUET.parent
FRANCE_WINDOW = ("2025-07-01", "2026-06-30")
DPE_COEF_OLD = 2.3
DPE_COEF_NEW = 1.9

# Analysis constants
CUT = pd.Timestamp("2022-06-15")     # Part L 2021 / SAP 10 effective date
ELEC_FACTOR_OLD = 0.519              # SAP 2012 electricity carbon factor (kgCO2/kWh)
ELEC_FACTOR_NEW = 0.136              # SAP 10.1 electricity carbon factor (kgCO2/kWh)
BROAD = ["A/B", "C", "D", "E", "F/G"]   # broad EPC classes
RANDOM_SEED = 42
EW_CODE = "K04000001"                # England & Wales aggregate row in DESNZ tables

_EXTERNALS_SEEDED = False


def _atomic_copy(source: Path, target: Path) -> None:
    """Replace a scratch copy atomically so immutable inputs always win."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".part", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def ensure_dirs() -> None:
    global _EXTERNALS_SEEDED
    for p in (OUT_TABLES, OUT_FIGURES, EXTERNAL_DIR, SCRATCH_DIR / "france"):
        p.mkdir(parents=True, exist_ok=True)

    # The analysis modules use one external-data work directory for both frozen
    # snapshots and downloaded fallbacks.  Seed writable scratch from every
    # regular, non-hidden file in the immutable input mount without modifying it.
    # This includes checksum sidecars when the release Data Asset provides them.
    if INPUT_EXTERNAL_DIR.is_dir() and not _EXTERNALS_SEEDED:
        for source in INPUT_EXTERNAL_DIR.iterdir():
            if source.is_file() and not source.name.startswith("."):
                target = EXTERNAL_DIR / source.name
                _atomic_copy(source, target)
        _EXTERNALS_SEEDED = True
