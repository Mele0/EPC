"""Fast data-contract checks for the manuscript-synchronized figures."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from epc_artefact.figures import (  # noqa: E402
    _asset_rating_change,
    _cohort_blocks,
    _share_label,
)
from epc_artefact.map_figure import MAP_VMAX, _country_name_column  # noqa: E402


class FigureDataContractTests(unittest.TestCase):
    def test_figure2_cohort_is_sequential_and_conserves_counts(self) -> None:
        blocks = _cohort_blocks(
            {
                "n_buildings": 842_673,
                "n_reassessed_buildings": 169_888,
                "n_straddle_2022": 106_228,
                "straddle_improved": 85_554,
            },
            {"improved_majority_formula_%": 100 * 49_878 / 85_554},
        )
        self.assertEqual([row["total"] for row in blocks],
                         [842_673, 169_888, 106_228, 85_554])
        self.assertEqual([row["selected"][1] for row in blocks],
                         [169_888, 106_228, 85_554, 49_878])
        for row in blocks:
            self.assertEqual(row["selected"][1] + row["remainder"][1], row["total"])

    def test_share_labels_match_manuscript_precision(self) -> None:
        self.assertEqual(_share_label(4.0), "4")
        self.assertEqual(_share_label(22.3), "22.3")

    def test_figure3b_retains_negative_asset_rating_change(self) -> None:
        source = pd.DataFrame({"mean_dAR": [-34.2, -18.4, -9.1]})
        self.assertListEqual(_asset_rating_change(source).tolist(), [-34.2, -18.4, -9.1])

    def test_figure4_uses_manuscript_scale(self) -> None:
        self.assertEqual(MAP_VMAX, 70)
        self.assertEqual(_country_name_column(pd.DataFrame({"NAME": ["France"]})), "NAME")


if __name__ == "__main__":
    unittest.main()
