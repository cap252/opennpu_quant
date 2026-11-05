"""One writer for every report folder: destination checks, settings and diagnostics."""

import contextlib
import copy
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from opennpu_quant._io import read_json
from opennpu_quant._locking import RunLock, check_report_destination
from opennpu_quant.cli import main
from opennpu_quant.public_run import evaluate_config
from opennpu_quant.report import report
from tests.test_public_run import evaluation_fixture


def snapshot(folder):
    return {
        str(path.relative_to(folder)): path.read_bytes()
        for path in folder.rglob("*")
        if path.is_file()
    }


class ReportOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        with contextlib.redirect_stdout(io.StringIO()):
            paths, config, catalog = evaluation_fixture(cls.root)
            evaluate_config(config, catalog, cls.root, paths)
            other = copy.deepcopy(config)
            other["name"] = "other"
            evaluate_config(other, catalog, cls.root, paths)
        cls.run_folder = paths.runs / config["name"]
        cls.other = paths.runs / "other"
        # The fixture also generates models using the same public state format.
        cls.standalone = next(
            p.parent.parent
            for p in paths.runs.glob("*/.state/settings.json")
            if read_json(p)["mode"] == "generate"
        )

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_reports_never_write_into_run_state_or_another_run(self):
        before = snapshot(self.other)
        for source, destination, message in (
            (
                self.run_folder,
                self.run_folder / ".state",
                r"inside a run's \.state folder",
            ),
            (self.run_folder, self.other / ".state/tables", r"inside a run's \.state"),
            (self.run_folder, self.other, "another run folder"),
            (self.run_folder, self.standalone, "another run folder"),
            (self.standalone, self.run_folder, "another run folder"),
        ):
            with (
                self.subTest(source=source.name, destination=str(destination)),
                self.assertRaisesRegex(ValueError, message),
            ):
                report(source, output=destination)
        self.assertEqual(before, snapshot(self.other))
        self.assertFalse((self.other / ".state/tables").exists())
        # A refused destination releases the source lock.
        with RunLock(self.run_folder / ".lock"):
            pass

    def test_run_folder_and_separate_folders_are_accepted(self):
        alias = self.root / "run-alias"
        alias.symlink_to(self.run_folder, target_is_directory=True)
        for destination in (
            self.run_folder,
            alias,
            self.root / "tables",
            self.run_folder / "tables",
        ):
            with self.subTest(destination=str(destination)):
                check_report_destination(self.run_folder, destination)
                self.assertEqual(
                    2, len(report(self.run_folder, output=destination)["rows"])
                )
                self.assertTrue((destination / "results.csv").is_file())

    def test_results_command_checks_its_output(self):
        export = self.root / "export"
        report(self.run_folder, output=export, formats="csv,json")
        stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as exited,
        ):
            main(["results", str(export / "records.json"), "--output", str(self.other)])
        self.assertEqual(2, exited.exception.code)
        self.assertIn("another run folder", stderr.getvalue())

    def test_filters_do_not_change_environment_diagnostics(self):
        tables = self.root / "filtered"
        value = report(
            self.run_folder,
            output=tables,
            conditions=["pot_rtn"],
            formats="csv,json",
            table_profile="opennpu_raw",
        )
        whole = {
            row["condition"]: row
            for row in read_json(self.run_folder / "settings.json")[
                "environment_diagnostics"
            ]
        }
        expected = {
            key: item
            for key, item in whole["pot_rtn"].items()
            if key not in ("model", "condition")
        }
        self.assertEqual(
            [whole["pot_rtn"]],
            read_json(tables / "settings.json")["environment_diagnostics"],
        )
        self.assertEqual(
            [expected],
            [
                r["environment_diagnostic"]
                for r in read_json(tables / "summary.json")["rows"]
            ],
        )
        metadata = read_json(tables / "raw_data.metadata.json")["records"]
        self.assertEqual([expected], [r["environment_diagnostic"] for r in metadata])
        # Recovery still needs the FP32 row in the selection.
        self.assertIsNone(value["rows"][0]["recovery_pct"])

    def test_public_reports_use_the_shared_writer(self):
        module = importlib.import_module("opennpu_quant.report")
        original = module._write_report
        calls = []

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        run_settings = (self.run_folder / "settings.json").read_bytes()
        with patch.object(module, "_write_report", side_effect=spy):
            report(self.run_folder, conditions=["fp32"])
            report(self.run_folder, output=self.root / "spy", conditions=["fp32"])
        # The run folder's settings.json keeps describing the whole run.
        self.assertEqual(run_settings, (self.run_folder / "settings.json").read_bytes())
        self.assertEqual([False, True], [c["settings"] is not None for c in calls])
        self.assertTrue(all(len(c["baselines"]) == 2 for c in calls))
        destination = self.root / "interrupted"
        with (
            patch.object(module, "_write_report", side_effect=OSError("interrupted")),
            self.assertRaisesRegex(OSError, "interrupted"),
        ):
            report(self.run_folder, output=destination)
        self.assertFalse((destination / "settings.json").exists())
        for directory in (self.run_folder, destination):
            with RunLock(directory / ".lock"):
                pass


def _create_run_after_report_check(
    root, paths, config, catalog, checked, finished, channel
):
    try:
        if not checked.wait(30):
            raise RuntimeError("Destination check did not reach the barrier")
        evaluate_config(config, catalog, root, paths)
        target = paths.runs / config["name"]
        result = ("ok", snapshot(target))
    except BaseException as error:
        result = ("error", type(error).__name__, str(error))
    finally:
        finished.set()
    # The parent waits for the event before reading. Do not block that event
    # on a snapshot larger than the multiprocessing pipe buffer.
    channel.send(result)
    channel.close()


class ReportDestinationRaceTests(unittest.TestCase):
    def test_new_run_after_check_is_protected_under_the_destination_lock(self):
        import multiprocessing
        import opennpu_quant._locking as locks

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths, config, catalog = evaluation_fixture(root)
            evaluate_config(config, catalog, root, paths)
            source = paths.runs / config["name"]
            other = copy.deepcopy(config)
            other.update(name="created-after-check")
            other["evaluation"]["limit"] = 1
            target = paths.runs / other["name"]
            ctx = multiprocessing.get_context("spawn")
            checked, finished = ctx.Event(), ctx.Event()
            parent, child = ctx.Pipe()
            worker = ctx.Process(
                target=_create_run_after_report_check,
                args=(root, paths, other, catalog, checked, finished, child),
            )
            original = locks.check_report_destination
            checks = []

            def check_then_pause(src, dst):
                original(src, dst)
                checks.append(str(dst))
                checked.set()
                if not finished.wait(30):
                    raise RuntimeError("Concurrent run did not complete")

            worker.start()
            child.close()
            try:
                with patch.object(
                    locks, "check_report_destination", side_effect=check_then_pause
                ):
                    with self.assertRaisesRegex(ValueError, "another run folder"):
                        report(source, output=target)
                self.assertTrue(parent.poll(30))
                result = parent.recv()
                worker.join(30)
                self.assertEqual(0, worker.exitcode)
                self.assertEqual("ok", result[0], result)
                self.assertEqual(result[1], snapshot(target))
                self.assertEqual([str(target)], checks)
            finally:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(10)
                parent.close()


if __name__ == "__main__":
    unittest.main()
