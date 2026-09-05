from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from epc_artefact import analysis, data, dr_estimator, france, preflight  # noqa: E402


class DownloadHardeningTests(unittest.TestCase):
    def test_atomic_download_records_and_enforces_checksum(self) -> None:
        payload = b"immutable input\n"
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "source.bin"
            with mock.patch.object(
                    data.urllib.request, "urlopen", return_value=io.BytesIO(payload)):
                result = data.download_if_missing(
                    "https://example.invalid/source.bin", destination,
                    expected_sha256=expected)

            self.assertEqual(result, destination)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertTrue(data.checksum_sidecar(destination).is_file())
            self.assertEqual(data.verify_checksum(destination), expected)

            destination.write_bytes(b"tampered")
            with self.assertRaises(data.DownloadChecksumError):
                data.download_if_missing("https://example.invalid/source.bin", destination)

    def test_checksum_mismatch_leaves_no_partial_or_final_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "source.bin"
            with mock.patch.object(
                    data.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong")):
                with self.assertRaises(data.DownloadChecksumError):
                    data.download_if_missing(
                        "https://example.invalid/source.bin", destination,
                        expected_sha256=hashlib.sha256(b"right").hexdigest())
            self.assertFalse(destination.exists())
            self.assertFalse(data.checksum_sidecar(destination).exists())
            self.assertEqual(list(Path(directory).glob("*.part")), [])


class FrameFallbackTests(unittest.TestCase):
    @staticmethod
    def _extended_frame() -> pd.DataFrame:
        row = {column: None for column in data.EXTENDED_COLUMNS}
        row.update({
            "certificate_number": "cert-1", "uprn": "uprn-1",
            "asset_rating": 60.0, "asset_rating_band_clean": "C",
            "property_type_clean": "Office", "inspection_date": "2024-01-01",
            "inspection_year": 2024, "main_heating_fuel_clean": "Grid electricity",
            "aircon_present_clean": "No", "floor_area": 100.0,
            "standard_emissions": 10.0, "building_emissions": 12.0,
            "typical_emissions": 11.0, "primary_energy_value": 120.0,
            "transaction_type_clean": "Mandatory issue", "n_recommendations": 2,
            "invalid_asset_rating": False, "extreme_asset_rating": False,
            "missing_asset_rating": False, "local_authority": "E09000001",
            "local_authority_label": "City of London", "country": "England",
            "lodgement_date": "2024-01-02", "payback_triple": "1_1_0",
            "rec_codes": "EPC-E1|EPC-L1",
        })
        return pd.DataFrame([row])

    def test_analysis_and_extended_loaders_accept_canonical_extended_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extended = root / "data" / "nd_artefact_extended.parquet"
            extended.parent.mkdir()
            self._extended_frame().to_parquet(extended, index=False)
            analysis_cache = root / "scratch" / "nd_artefact_analysis.parquet"
            extended_cache = root / "scratch" / "nd_artefact_extended.parquet"
            missing_unified = root / "data" / "missing-unified.parquet"
            with mock.patch.multiple(
                    data, ANALYSIS_FRAME=analysis_cache, UNIFIED_PARQUET=missing_unified,
                    EXTENDED_FRAME=extended, EXTENDED_CACHE=extended_cache):
                analysis = data.build_analysis_frame(force=True)
                supplied = data.build_extended_frame(force=True)
                analysis_cache.write_bytes(b"stale partial cache")
                rebuilt = data.build_analysis_frame()

            self.assertEqual(len(analysis), 1)
            self.assertEqual(analysis.iloc[0].fuelgrp, "Electric")
            self.assertTrue(analysis_cache.is_file())
            self.assertTrue(data.checksum_sidecar(analysis_cache).is_file())
            self.assertTrue(data.source_checksum_marker(analysis_cache).is_file())
            self.assertEqual(len(rebuilt), 1)
            self.assertEqual(supplied.iloc[0].country, "England")
            self.assertFalse(extended_cache.exists())


class ReleasePreflightTests(unittest.TestCase):
    def test_complete_non_france_bundle_passes_and_missing_france_fails_early(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root / "external"
            external.mkdir(parents=True)
            extended = root / "nd_artefact_extended.parquet"
            FrameFallbackTests._extended_frame().to_parquet(extended, index=False)
            required = (preflight.CORE_EXTERNAL_FILES + preflight.MAP_EXTERNAL_FILES
                        + preflight.METERED_DERIVED_FILES)
            for name in required:
                (external / name).write_text("frozen input\n", encoding="utf-8")
            manifest = root / "input_checksums.sha256"
            inputs = [extended, *(external / name for name in required)]
            manifest.write_text("".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                f"{path.relative_to(root)}\n"
                for path in inputs
            ), encoding="ascii")

            with mock.patch.multiple(
                    preflight, DATA_DIR=root, INPUT_EXTERNAL_DIR=external,
                    EXTENDED_FRAME=extended, UNIFIED_PARQUET=root / "missing.parquet",
                    FRANCE_INPUT_PARQUET=root / "france" / "missing.parquet",
                    CHECKSUM_MANIFEST=manifest, MANUSCRIPT_EW_CERTIFICATES=1,
                    MANUSCRIPT_EW_UPRNS=1):
                preflight.validate_release_inputs(include_france=False)
                with self.assertRaisesRegex(
                        preflight.InputValidationError, "Missing frozen France"):
                    preflight.validate_release_inputs(include_france=True)

    def test_mae_inputs_must_be_checksum_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "input_checksums.sha256"
            manifest.write_text(
                "0" * 64 + "  nd_artefact_extended.parquet\n", encoding="ascii")
            errors: list[str] = []
            with mock.patch.multiple(
                    preflight, DATA_DIR=root, CHECKSUM_MANIFEST=manifest,
                    MANIFEST_CORE_REQUIRED_FILES=("nd_artefact_extended.parquet",)):
                preflight._validate_manifest(
                    errors, include_map=False, include_france=False,
                    include_mae_rescue=True)
            self.assertTrue(any(
                "non_domestic_epc_unified_certificate_level.parquet" in error
                and "recommendations_long_clean.parquet" in error
                for error in errors
            ))


class MeteredSnapshotTests(unittest.TestCase):
    def test_frozen_derived_series_prevent_live_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root / "external"
            tables = root / "tables"
            external.mkdir()
            tables.mkdir()
            pd.DataFrame({
                "year": [2012, 2013],
                "nd_electricity_mean_kWh_per_meter": [100.0, 90.0],
            }).to_csv(external / "metered_nondomestic_electricity_EW.csv", index=False)
            pd.DataFrame({
                "year": [2012, 2013],
                "nd_gas_mean_kWh_per_meter": [200.0, 180.0],
            }).to_csv(external / "metered_nondomestic_gas_EW.csv", index=False)
            pd.DataFrame({
                "inspection_year": [2012, 2013], "BER": [10.0, 9.0],
                "AR": [100.0, 90.0], "pe": [200.0, 180.0],
            }).to_csv(tables / "register_year_trend.csv", index=False)
            with mock.patch.multiple(analysis, EXTERNAL_DIR=external, OUT_TABLES=tables), \
                    mock.patch.object(
                        analysis, "download_if_missing",
                        side_effect=AssertionError("network fallback must not run")):
                result = analysis.metered_validation()
            self.assertEqual(result.metered_idx.round(0).tolist(), [100.0, 90.0])


class FranceHardeningTests(unittest.TestCase):
    @staticmethod
    def _api_rows(n: int) -> list[dict[str, str]]:
        return [
            {field: f"{field}-{index}" for field in france.FIELDS}
            for index in range(n)
        ]

    def test_limited_download_uses_distinct_atomic_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            canonical = Path(directory) / "window.parquet"
            canonical.write_bytes(b"full-release-sentinel")

            def fake_fetch(url: str, retries: int = 5) -> dict:
                del retries
                if "size=0" in url:
                    return {"total": 3}
                return {"results": self._api_rows(3), "next": None}

            with mock.patch.object(france, "ensure_dirs"), \
                    mock.patch.object(france, "_fetch", side_effect=fake_fetch):
                sample = france.download_window(canonical, max_records=2)

            self.assertEqual(canonical.read_bytes(), b"full-release-sentinel")
            self.assertEqual(sample, canonical.with_name("window.sample-2.parquet"))
            self.assertEqual(pq.ParquetFile(sample).metadata.num_rows, 2)
            self.assertTrue(data.checksum_sidecar(sample).is_file())

    def test_full_download_rejects_changed_source_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "window.parquet"

            def fake_fetch(url: str, retries: int = 5) -> dict:
                del url, retries
                return {"total": 3}

            with mock.patch.object(france, "ensure_dirs"), \
                    mock.patch.object(france, "_fetch", side_effect=fake_fetch):
                with self.assertRaises(france.FranceValidationError):
                    france.download_window(destination, expected_records=2)
            self.assertFalse(destination.exists())

    def test_full_sanity_failure_is_fatal(self) -> None:
        with self.assertRaisesRegex(france.FranceValidationError, "record_count"):
            france.validate_sanity_checks({"record_count": False, "partition": True})

    def test_manuscript_headline_and_heating_contract_is_exact(self) -> None:
        summary = {**france.MANUSCRIPT_SUMMARY,
                   "match_all": france.MANUSCRIPT_REPRODUCTION_PCT}
        heating = pd.DataFrame([
            {"heat_group": group, "n": values[0], "passoires": values[1],
             "exits": values[2]}
            for group, values in france.MANUSCRIPT_HEATING.items()
        ])
        checks = france.manuscript_release_checks(
            france.MANUSCRIPT_WINDOW_RECORDS, summary, heating)
        self.assertTrue(all(checks.values()))

        drifted = dict(summary)
        drifted["exits"] += 1
        failed = france.manuscript_release_checks(
            france.MANUSCRIPT_WINDOW_RECORDS, drifted, heating)
        self.assertFalse(failed["headline_exits_matches_manuscript"])
        with self.assertRaisesRegex(france.FranceValidationError, "headline_exits"):
            france.validate_sanity_checks(failed)


class DRHardeningTests(unittest.TestCase):
    @staticmethod
    def _comparison(recomputed_count: float, recomputed_area: float,
                    *, threshold: str = "EPC-B", model: str = "C") -> pd.DataFrame:
        return pd.DataFrame([
            {"threshold": threshold, "model": model, "quantity": "count",
             "reported": 94_911.0, "recomputed": recomputed_count,
             "percent_difference": 100 * (recomputed_count - 94_911.0) / 94_911.0},
            {"threshold": threshold, "model": model, "quantity": "area_Mm2",
             "reported": 57.24, "recomputed": recomputed_area,
             "percent_difference": 100 * (recomputed_area - 57.24) / 57.24},
        ])

    def test_known_minor_dr_deltas_pass(self) -> None:
        dr_estimator.validate_reported_agreement(self._comparison(94_690.0, 57.18))

    def test_material_dr_drift_fails(self) -> None:
        with self.assertRaises(dr_estimator.DRValidationError):
            dr_estimator.validate_reported_agreement(self._comparison(93_000.0, 55.0))

    def test_secondary_gate_accepts_expected_cross_fit_sensitivity(self) -> None:
        comparison = self._comparison(
            94_911.0 * 1.0499, 57.24 * 0.9501, threshold="F/G", model="D")
        dr_estimator.validate_reported_agreement(comparison)

    def test_secondary_gate_rejects_drift_above_five_percent(self) -> None:
        comparison = self._comparison(
            94_911.0 * 1.0501, 57.24, threshold="F/G", model="D")
        with self.assertRaisesRegex(dr_estimator.DRValidationError, "limit 5.00%"):
            dr_estimator.validate_reported_agreement(comparison)


class OptionalMaeStageTests(unittest.TestCase):
    @staticmethod
    def _runtime_with_stubs(extra: dict[str, object] | None = None):
        stubs = {
            "epc_artefact.analysis": types.SimpleNamespace(run_analysis=None),
            "epc_artefact.figures": types.SimpleNamespace(make_figures=None),
            "epc_artefact.markov": types.SimpleNamespace(run_markov=None),
            "epc_artefact.robustness": types.SimpleNamespace(run_robustness=None),
            "epc_artefact.validation": types.SimpleNamespace(
                heldout_validation=None, latest_frame_characteristics=None,
                policy_proxy_subsets=None, representativeness_assessment=None),
        }
        if extra:
            stubs.update(extra)
        sys.modules.pop("epc_artefact.run", None)

        return stubs

    def test_default_mae_stage_runs(self) -> None:
        validate = mock.Mock()
        execute = mock.Mock()
        stubs = self._runtime_with_stubs({
            "epc_artefact.mae_rescue": types.SimpleNamespace(
                main=execute, validate_mae_rescue_inputs=validate),
        })
        with mock.patch.dict(sys.modules, stubs):
            runtime = importlib.import_module("epc_artefact.run")
            ran = runtime.run_optional_mae_rescue()
        self.assertTrue(ran)
        validate.assert_called_once_with()
        execute.assert_called_once_with()

    def test_mae_stage_can_be_explicitly_skipped(self) -> None:
        stubs = self._runtime_with_stubs()
        output = io.StringIO()
        with mock.patch.dict(sys.modules, stubs), contextlib.redirect_stdout(output):
            runtime = importlib.import_module("epc_artefact.run")
            ran = runtime.run_optional_mae_rescue(False)
        self.assertFalse(ran)
        self.assertIn("MAE-rescue analysis skipped", output.getvalue())


if __name__ == "__main__":
    unittest.main()
