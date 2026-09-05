"""Fast checks for the portable filesystem and Capsule contract."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_package_declares_supported_python(self) -> None:
        with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)["project"]
        self.assertEqual(project["requires-python"], ">=3.11")
        locked = [
            line.strip()
            for line in (REPO_ROOT / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(project["dependencies"] + project["optional-dependencies"]["map"], locked)

    def test_environment_lock_matches_runtime_lock(self) -> None:
        self.assertEqual(
            (REPO_ROOT / "requirements.txt").read_bytes(),
            (REPO_ROOT / "environment" / "requirements.txt").read_bytes(),
        )
        self.assertEqual(
            (REPO_ROOT / "requirements-lock.txt").read_bytes(),
            (REPO_ROOT / "environment" / "requirements-lock.txt").read_bytes(),
        )
        post_install = (REPO_ROOT / "environment" / "postInstall").read_text()
        self.assertIn("/environment/requirements-lock.txt", post_install)

    def test_config_honours_separate_runtime_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            data_dir = base / "data"
            output_dir = base / "results"
            scratch_dir = base / "scratch"
            (data_dir / "external").mkdir(parents=True)
            for name in (
                "gdp_deflator_hmt_jun2026.csv",
                "ipf_cost_schedule.csv",
                "sector_cost_crosswalk.csv",
            ):
                (data_dir / "external" / name).write_text("test\n", encoding="utf-8")

            code = """
import json
from epc_artefact.config import (
    ANALYSIS_FRAME, DATA_DIR, EXTERNAL_DIR, OUTPUT_DIR, SCRATCH_DIR, ensure_dirs,
)
ensure_dirs()
print(json.dumps({
    "data": str(DATA_DIR), "output": str(OUTPUT_DIR),
    "scratch": str(SCRATCH_DIR), "analysis": str(ANALYSIS_FRAME),
    "external_files": sorted(path.name for path in EXTERNAL_DIR.iterdir()),
}))
"""
            environment = os.environ.copy()
            environment.update(
                {
                    "EPC_DATA_DIR": str(data_dir),
                    "EPC_OUTPUT_DIR": str(output_dir),
                    "EPC_SCRATCH_DIR": str(scratch_dir),
                    "PYTHONPATH": str(REPO_ROOT / "code" / "src"),
                }
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            paths = json.loads(completed.stdout)
            self.assertEqual(paths["data"], str(data_dir))
            self.assertEqual(paths["output"], str(output_dir))
            self.assertEqual(paths["scratch"], str(scratch_dir))
            self.assertTrue(paths["analysis"].startswith(str(scratch_dir)))
            self.assertEqual(
                paths["external_files"],
                [
                    "gdp_deflator_hmt_jun2026.csv",
                    "ipf_cost_schedule.csv",
                    "sector_cost_crosswalk.csv",
                ],
            )

    def test_capsule_driver_is_headless_and_executable(self) -> None:
        driver = REPO_ROOT / "run"
        capsule_driver = REPO_ROOT / "code" / "run"
        self.assertTrue(os.access(driver, os.X_OK), f"{driver} is not executable")
        self.assertTrue(
            os.access(capsule_driver, os.X_OK), f"{capsule_driver} is not executable"
        )
        environment = os.environ.copy()
        environment["EPC_PYTHON"] = sys.executable
        completed = subprocess.run(
            [str(driver), "--help"], check=True, capture_output=True, text=True,
            env=environment,
        )
        self.assertIn("--skip-france", completed.stdout)
        self.assertIn("--skip-map", completed.stdout)
        self.assertIn("--skip-mae-rescue", completed.stdout)

    def test_copy_from_git_code_folder_is_self_contained(self) -> None:
        code_root = REPO_ROOT / "code"
        self.assertTrue((code_root / "scripts" / "reproduce_all.py").is_file())
        self.assertTrue((code_root / "src" / "epc_artefact" / "config.py").is_file())
        self.assertTrue((code_root / "doubly_robust_estimates" /
                         "bridge_stage1_frozen.csv").is_file())
        driver = (code_root / "run").read_text(encoding="utf-8")
        self.assertIn('${code_dir}/src', driver)
        self.assertIn('${code_dir}/scripts/reproduce_all.py', driver)
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied_code = Path(temporary_directory) / "code"
            shutil.copytree(code_root, copied_code)
            environment = os.environ.copy()
            environment["EPC_PYTHON"] = sys.executable
            completed = subprocess.run(
                [str(copied_code / "run"), "--help"], check=True,
                capture_output=True, text=True, env=environment,
            )
            self.assertIn("--skip-france", completed.stdout)
            environment["PYTHONPATH"] = str(copied_code / "src")
            completed = subprocess.run(
                [sys.executable, "-c",
                 "from epc_artefact.config import DR_ESTIMATES_DIR; "
                 "assert (DR_ESTIMATES_DIR / 'bridge_stage1_frozen.csv').is_file()"],
                check=True, capture_output=True, text=True, env=environment,
            )
            self.assertEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
