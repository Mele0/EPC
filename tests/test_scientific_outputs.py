"""Focused contracts for auditable bootstrap and manuscript table outputs."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from epc_artefact import analysis, paper_tables


class FixedFactorBootstrapTests(unittest.TestCase):
    def test_every_bootstrap_replicate_is_written_with_provenance(self) -> None:
        frame = pd.DataFrame({
            "uprn": ["u1", "u2", "u3"],
            "inspection_date": ["2023-01-01"] * 3,
            "asset_rating": [40.0, 100.0, 120.0],
            "fuelgrp": ["Electric", "Gas", "Other"],
        })
        pairs = pd.DataFrame({
            "straddle": [True] * 6,
            "drec": [0] * 6,
            "dfa": [0.0] * 6,
            "ar_f": [80.0, 90.0, 120.0, 130.0, 150.0, 160.0],
            "ar_l": [40.0, 50.0, 100.0, 100.0, 100.0, 100.0],
            "fuel": ["Electric", "Electric", "Gas", "Gas", "Other", "Other"],
            "dt_f": pd.to_datetime(["2021-01-01"] * 6),
            "dt_l": pd.to_datetime(["2023-01-01"] * 6),
        })
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(analysis, "OUT_TABLES", Path(directory)):
            result = analysis.fixed_factor_recompute(frame, pairs, n_boot=7)
            draws = pd.read_csv(Path(directory) / "fixed_factor_bootstrap_draws.csv")

        self.assertEqual(len(draws), 7)
        self.assertEqual(draws.replicate.tolist(), list(range(7)))
        self.assertEqual(result["bootstrap"]["replicates"], 7)
        self.assertEqual(result["bootstrap"]["seed"], analysis.RANDOM_SEED)
        self.assertEqual(result["bootstrap"]["rng"], "numpy.random.default_rng")


class PaperTableTests(unittest.TestCase):
    def test_france_display_shares_close_and_labels_match_manuscript(self) -> None:
        heat = pd.DataFrame({
            "heat_group": ["Electricity", "Gas", "Wood/biomass", "Heat network",
                           "Oil/other-fossil"],
            "n": [1_122_161, 1_065_224, 89_460, 334_035, 124_107],
            "passoires": [1, 1, 1, 1, 1],
            "exits": [1, 1, 1, 1, 1],
            "exit_rate_passoires_%": [1.0] * 5,
            "share_of_exits_%": [20.0] * 5,
        })
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(paper_tables, "PAPER_DIR", Path(directory)), \
                mock.patch.object(paper_tables, "_load", return_value=heat):
            table = paper_tables.table3_france()

        self.assertEqual(table["Flow share (%)"].tolist(), [41.0, 39.0, 3.3, 12.2, 4.5])
        self.assertEqual(float(table["Flow share (%)"].sum()), 100.0)
        self.assertIn("Wood or biomass", table["Heating energy"].tolist())
        self.assertIn("Oil/other", table["Heating energy"].tolist())


if __name__ == "__main__":
    unittest.main()
