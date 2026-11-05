"""The same evidence rules apply to generation and supplied-model evaluation."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from opennpu_quant._io import atomic_json, read_json, sha256
from opennpu_quant.public_run import evaluate_config
from opennpu_quant.report import report
from opennpu_quant.run_records import current_records, record_directory
from opennpu_quant.run_storage import archive_evaluation, finish_status, new_status
from tests.test_public_run import evaluation_fixture


class RunRecordsTests(unittest.TestCase):
    def test_both_modes_require_the_status_attestation(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, config, catalog = evaluation_fixture(Path(temp))
            evaluate_config(config, catalog, Path(temp), paths)
            for mode, name in (
                ("generate", "reliability"),
                ("evaluate", config["name"]),
            ):
                with self.subTest(mode=mode):
                    root = paths.runs / name
                    status = read_json(root / ".state/status.json")
                    self.assertEqual(2, status["schema_version"])
                    self.assertTrue(status["config_identity"])
                    conditions = status["models"]["toy"]["conditions"]
                    self.assertEqual({"fp32", "pot_rtn"}, set(conditions))
                    self.assertTrue(
                        all("record" not in row for row in conditions.values())
                    )
                    records, _ = current_records(root)
                    self.assertTrue(all(row["complete"] for row in records))
                    directory = root / ".state/toy/pot_rtn"
                    result = read_json(directory / "result.json")
                    result["metrics"]["top1"]["value"] = 0.25
                    atomic_json(directory / "result.json", result)
                    # Updating the file and proof still cannot replace the result
                    # attested by the completed condition in status.json.
                    proof = read_json(directory / "complete.json")
                    proof["result_sha256"] = sha256(directory / "result.json")
                    atomic_json(directory / "complete.json", proof)
                    rows = report(root)["rows"]
                    self.assertEqual("unverified", rows[1]["state"])
                    self.assertIsNone(rows[1]["value_pct"])
                    self.assertTrue(rows[0]["complete"])
                    (root / ".state/status.json").unlink()
                    rows = report(root)["rows"]
                    self.assertTrue(all(row["state"] == "unverified" for row in rows))
                    self.assertTrue(all(row["value_pct"] is None for row in rows))

    def test_interrupted_or_running_conditions_are_never_completed(self):
        for state in (
            "interrupted",
            "running",
            "failed",
            "pending",
            "reused",
            "completed",
        ):
            with self.subTest(state=state):
                status = new_status({"toy": {"fp32": {}}}, {})
                status["models"]["toy"]["conditions"]["fp32"]["state"] = state
                finish_status(status)
                self.assertEqual(
                    "completed" if state == "reused" else state, status["state"]
                )

    def test_old_archive_is_not_silently_approved_for_reuse(self):
        from opennpu_quant.run_storage import restore_evaluation

        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp)
            archive = state / ".evaluations/old"
            condition = archive / "toy/fp32"
            atomic_json(condition / "complete.json", dict(identity="matching"))
            atomic_json(archive / "status.json", dict(schema_version=1))
            self.assertFalse(restore_evaluation(state, "toy", "fp32", "matching"))
            self.assertFalse((state / "toy").exists())

    def test_report_folders_are_not_runs_and_snapshots_are_not_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths, config, catalog = evaluation_fixture(root)
            evaluate_config(config, catalog, root, paths)
            run = paths.runs / config["name"]
            published = root / "published"
            report(run, output=published, formats="csv,json")

            def files(folder):
                return {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}

            before = files(published)
            # A report folder holds settings.json, but it is not a run.
            self.assertIsNone(record_directory(published))
            with self.assertRaisesRegex(ValueError, "Not a run directory"):
                report(published)
            self.assertEqual(before, files(published))
            # A copied state folder is a snapshot: it is reported, and a report
            # of another run never overwrites it.
            snapshot = root / "snapshot"
            shutil.copytree(run / ".state", snapshot)
            self.assertEqual(snapshot, record_directory(snapshot))
            self.assertEqual(2, len(report(snapshot)["rows"]))
            kept = files(snapshot)
            with self.assertRaisesRegex(ValueError, "another run folder"):
                report(run, output=snapshot)
            self.assertEqual(kept, files(snapshot))

    def test_schema_one_archives_remain_reportable(self):
        from tests.test_result_reliability import fixture

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, config = fixture(root)
            state = root / "run/.state"
            (state / "toy/fp32").mkdir(parents=True)
            atomic_json(state / "effective_config.json", config)
            conditions = {c: dict(state="pending") for c in config["conditions"]}
            atomic_json(
                state / "status.json",
                dict(
                    state="completed",
                    attempt_id="a",
                    models=dict(toy=dict(state="completed", conditions=conditions)),
                ),
            )
            atomic_json(
                state / "settings.json",
                dict(schema_version=1, mode="generate", config=config, models={}),
            )
            self.assertEqual(2, len(report(root / "run")["rows"]))
            archive_evaluation(state, config)
            archive = next((state / ".evaluations").iterdir())
            self.assertEqual(archive, record_directory(archive))
            self.assertEqual(2, len(report(archive)["rows"]))

    def test_reporting_does_not_import_a_run_driver(self):
        process = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "import sys,json; import opennpu_quant.report; "
                "print(json.dumps([m for m in sys.modules if m in "
                "('opennpu_quant.runner','opennpu_quant.public_run','torch')]))",
            ],
            env=os.environ.copy(),
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual([], json.loads(process.stdout))


if __name__ == "__main__":
    unittest.main()
