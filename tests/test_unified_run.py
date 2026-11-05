import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from opennpu_quant._io import read_json, sha256
from opennpu_quant.runner import run
from opennpu_quant.report import report
from tests.test_result_reliability import fixture


@contextlib.contextmanager
def _read_only_tree(root):
    paths = [root, *root.rglob("*")]
    modes = {path: path.stat().st_mode & 0o777 for path in paths}
    try:
        for path in paths:
            path.chmod(0o555 if path.is_dir() else 0o444)
        yield
    finally:
        for path, mode in modes.items():
            path.chmod(mode)


def _read_only_report_worker(pipe, source, destination):
    from opennpu_quant.report import _write_report

    def publish(*args, **kwargs):
        pipe.send(("locked",))
        if not pipe.poll(60) or pipe.recv() != "release":
            raise RuntimeError("Report barrier timed out")
        return _write_report(*args, **kwargs)

    try:
        with patch("opennpu_quant.report._write_report", side_effect=publish):
            result = report(source, output=destination)
        pipe.send(("done", len(result["rows"])))
    except BaseException as error:
        pipe.send(("error", type(error).__name__, str(error)))
    finally:
        pipe.close()


class ReadOnlyReportTests(unittest.TestCase):
    def test_all_report_commands_read_only_inputs_with_or_without_lock(self):
        import csv
        import subprocess
        import sys
        from opennpu_quant._io import atomic_json
        from tests.test_results import record
        from tests.test_review_fixes import ReportPolicyTests

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for lock_exists in (False, True):
                for kind in ("report", "results"):
                    with self.subTest(command=kind, lock_exists=lock_exists):
                        case = root / (kind + str(lock_exists))
                        source = case / "run"
                        source.mkdir(parents=True)
                        if kind == "report":
                            ReportPolicyTests().fixture(source)
                            command = ["report", str(source)]
                        else:
                            atomic_json(
                                source / "records.json",
                                dict(
                                    schema_version=1,
                                    records=[record(), record("q", 0.78)],
                                ),
                            )
                            command = ["results", str(source / "records.json")]
                        base = [
                            sys.executable,
                            "-B",
                            "-m",
                            "opennpu_quant",
                            *command,
                        ]
                        expected = case / "expected"
                        result = subprocess.run(
                            [*base, "--output", str(expected)],
                            cwd=root,
                            capture_output=True,
                            text=True,
                        )
                        self.assertEqual(0, result.returncode, result.stderr)
                        lock = source / ".lock"
                        if lock_exists:
                            lock.touch()
                        else:
                            lock.unlink(missing_ok=True)
                        before = {
                            str(p.relative_to(source)): p.read_bytes()
                            for p in source.rglob("*")
                            if p.is_file()
                        }
                        destination = case / "tables"
                        with _read_only_tree(source):
                            result = subprocess.run(
                                [*base, "--output", str(destination)],
                                cwd=root,
                                capture_output=True,
                                text=True,
                            )
                        self.assertEqual(0, result.returncode, result.stderr)
                        self.assertNotIn("Traceback", result.stderr)
                        actual = (destination / "results.csv").read_bytes()
                        self.assertEqual(
                            (expected / "results.csv").read_bytes(), actual
                        )
                        self.assertEqual(
                            2, len(list(csv.DictReader(io.StringIO(actual.decode()))))
                        )
                        self.assertEqual(
                            before,
                            {
                                str(p.relative_to(source)): p.read_bytes()
                                for p in source.rglob("*")
                                if p.is_file()
                            },
                        )

    def test_read_only_source_is_locked_until_report_publication(self):
        import multiprocessing as mp
        from opennpu_quant._locking import run_directory_lock
        from tests.test_review_fixes import ReportPolicyTests

        context = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "run"
            source.mkdir()
            ReportPolicyTests().fixture(source)
            with _read_only_tree(source):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_read_only_report_worker,
                    args=(child, source, root / "tables"),
                )
                process.start()
                child.close()
                try:
                    self.assertTrue(parent.poll(60), "Report did not acquire its locks")
                    self.assertEqual(("locked",), parent.recv())
                    for fixed in (False, True):
                        with self.assertRaisesRegex(
                            RuntimeError, "Another process owns"
                        ):
                            with run_directory_lock(source, fixed=fixed):
                                self.fail(
                                    "Writer entered while the input was being read"
                                )
                    with self.assertRaisesRegex(RuntimeError, "Another process owns"):
                        report(source, output=root / "other")
                    self.assertFalse((root / "other").exists())
                    self.assertFalse((source / ".lock").exists())
                    parent.send("release")
                    self.assertTrue(parent.poll(60))
                    self.assertEqual(("done", 2), parent.recv())
                    process.join(10)
                    self.assertEqual(0, process.exitcode)
                finally:
                    if process.is_alive():
                        process.terminate()
                    process.join(10)
                    parent.close()
            with run_directory_lock(source, fixed=False):
                pass
            self.assertTrue((root / "tables/results.csv").is_file())

    def test_read_only_existing_lock_is_respected_and_errors_release_locks(self):
        from opennpu_quant._locking import RunLock, run_directory_lock
        from tests.test_review_fixes import ReportPolicyTests

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "run"
            source.mkdir()
            ReportPolicyTests().fixture(source)
            with RunLock(source / ".lock"), _read_only_tree(source):
                with self.assertRaisesRegex(RuntimeError, "Another process owns"):
                    report(source, output=root / "tables")
                self.assertFalse((root / "tables").exists())
            with run_directory_lock(source, fixed=False):
                pass
            with _read_only_tree(source):
                with patch(
                    "opennpu_quant.report._write_report",
                    side_effect=OSError("interrupted"),
                ):
                    with self.assertRaisesRegex(OSError, "interrupted"):
                        report(source, output=root / "tables")
                self.assertEqual(2, len(report(source, output=root / "tables")["rows"]))
            with run_directory_lock(source, fixed=False):
                pass


def no_generation():
    stack = contextlib.ExitStack()
    for name in ("calibrate", "quantize", "apply_adaround"):
        stack.enter_context(
            patch(
                "opennpu_quant.runner." + name,
                side_effect=AssertionError("unexpected " + name),
            )
        )
    return stack


class GenerationReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths, self.config = fixture(self.root)
        self.config["output"]["save_qdq_models"] = True
        self.config["conditions"] = {"pot_rtn": self.config["conditions"]["pot_rtn"]}
        self.config["evaluation"]["limit"] = 1
        self.out = self.paths.runs / self.config["name"]

    def execute(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run(self.config, self.paths, **kwargs, formats="csv,json,md")

    def test_each_recipe_is_resolved_once_per_run(self):
        from opennpu_quant.models.spec import ModelSpec

        # A second resolution repeats the recipe checks and any replacement warning.
        with patch.object(ModelSpec, "for_run", wraps=ModelSpec.for_run) as resolve:
            self.execute()
        self.assertEqual(1, resolve.call_count)

    def test_standalone_report_holds_both_locks_and_releases_them_on_error(self):
        from opennpu_quant._locking import RunLock

        self.config["evaluation"]["limit"] = 2
        self.execute()
        destination = self.root / "tables"
        with RunLock(self.out / ".lock"):
            with self.assertRaisesRegex(RuntimeError, "Another process owns"):
                report(self.out, output=destination)
        self.assertFalse(destination.exists())
        with RunLock(destination / ".lock"):
            with self.assertRaisesRegex(RuntimeError, "Another process owns"):
                report(self.out, output=destination)
            # Failure to acquire the output lock must release the input lock.
            with RunLock(self.out / ".lock"):
                pass

        def fail_write(*args, **kwargs):
            for directory in (self.out, destination):
                with self.assertRaisesRegex(RuntimeError, "Another process owns"):
                    with RunLock(directory / ".lock"):
                        pass
            raise OSError("report publication interrupted")

        with patch("opennpu_quant.report._write_report", side_effect=fail_write):
            with self.assertRaisesRegex(OSError, "report publication interrupted"):
                report(self.out, output=destination)
        for directory in (self.out, destination):
            with RunLock(directory / ".lock"):
                pass
        alias = self.root / "run-alias"
        alias.symlink_to(self.out, target_is_directory=True)
        self.assertTrue(report(self.out, output=alias)["rows"][0]["complete"])
        self.assertTrue(report(self.out, output=destination)["rows"][0]["complete"])
        self.assertTrue((destination / "results.csv").is_file())

    def test_evaluation_change_reuses_model_and_preserves_previous_attempt(self):
        self.execute()
        old = self.out / ".state/toy/pot_rtn/result.json"
        payload = old.read_bytes()
        model_hash = sha256(old.with_name("model.onnx"))
        self.config["evaluation"]["limit"] = 2
        with no_generation():
            result = self.execute()
        self.assertTrue(result["rows"][0]["complete"])
        self.assertEqual("reused", read_json(old)["model_generation"]["state"])
        self.assertEqual(model_hash, sha256(old.with_name("model.onnx")))
        archives = list(
            (self.out / ".state/.evaluations").glob("*/toy/pot_rtn/result.json")
        )
        self.assertEqual(1, len(archives))
        self.assertEqual(payload, archives[0].read_bytes())
        self.assertEqual(1, report(archives[0].parents[2])["rows"][0]["samples"])
        with (
            no_generation(),
            patch(
                "opennpu_quant.runner.evaluate", side_effect=AssertionError("inference")
            ),
        ):
            self.execute()

    def test_condition_subset_and_return_restore_verified_evaluations(self):
        import copy
        from opennpu_quant._io import atomic_json

        policy = self.config["conditions"]["pot_rtn"]
        all_conditions = {"fp32": {}, "pot_rtn": policy, "rtn_copy": policy}
        self.config["evaluation"]["limit"] = 2
        self.config["conditions"] = copy.deepcopy(all_conditions)
        self.execute()
        original = {
            name: (self.out / ".state/toy" / name / "result.json").read_bytes()
            for name in all_conditions
        }
        with (
            no_generation(),
            patch(
                "opennpu_quant.runner.evaluate", side_effect=AssertionError("inference")
            ),
        ):
            for names in (("fp32", "pot_rtn"), tuple(all_conditions)):
                self.config["conditions"] = {n: all_conditions[n] for n in names}
                result = self.execute()
                self.assertEqual(len(names), len(result["rows"]))
                self.assertTrue(all(row["complete"] for row in result["rows"]))
                status = read_json(self.out / ".state/status.json")
                for name in names:
                    self.assertEqual(
                        "reused", status["models"]["toy"]["conditions"][name]["state"]
                    )
                    self.assertEqual(
                        original[name],
                        (self.out / ".state/toy" / name / "result.json").read_bytes(),
                    )
        # Report rebinding is explicit and bound to the active configuration.
        status["models"]["toy"]["conditions"]["rtn_copy"]["reused_config_identity"] = (
            "changed"
        )
        atomic_json(self.out / ".state/status.json", status)
        # This result's original full config matches again. Change report policy to
        # exercise the rebinding check independently of measurement identities.
        config = read_json(self.out / ".state/settings.json")["config"]
        config["report"]["target_mean_recovery"] = 90
        settings = read_json(self.out / ".state/settings.json")
        settings["config"] = config
        atomic_json(self.out / ".state/settings.json", settings)
        self.assertTrue(all(not row["complete"] for row in report(self.out)["rows"]))

    def test_changed_scope_evaluates_and_restores_exact_prior_scope(self):
        self.execute()
        before = (self.out / ".state/toy/pot_rtn/result.json").read_bytes()
        self.config["evaluation"]["limit"] = 2
        with no_generation():
            self.execute()
        self.config["evaluation"]["limit"] = 1
        with (
            no_generation(),
            patch(
                "opennpu_quant.runner.evaluate", side_effect=AssertionError("inference")
            ),
        ):
            result = self.execute()
        self.assertEqual(1, result["rows"][0]["samples"])
        self.assertEqual("reused", result["rows"][0]["state"])
        self.assertFalse(result["rows"][0]["complete"])
        self.assertEqual(
            before, (self.out / ".state/toy/pot_rtn/result.json").read_bytes()
        )

    def test_corrupt_archive_is_not_restored(self):
        self.execute()
        self.config["evaluation"]["limit"] = 2
        self.execute()
        archived = next(
            (self.out / ".state/.evaluations").glob("*/toy/pot_rtn/result.json")
        )
        archived.write_text("{}")
        self.config["evaluation"]["limit"] = 1
        with patch(
            "opennpu_quant.runner.evaluate", side_effect=RuntimeError("must evaluate")
        ):
            with self.assertRaisesRegex(RuntimeError, "must evaluate"):
                self.execute()
        self.assertFalse(read_json(self.out / "summary.json")["rows"][0]["complete"])

    def test_saved_model_reuse_does_not_require_saving_another_copy(self):
        self.execute()
        self.config["output"]["save_qdq_models"] = False
        self.config["evaluation"]["limit"] = 2
        with no_generation():
            self.execute()
        result = read_json(self.out / ".state/toy/pot_rtn/result.json")
        self.assertEqual("reused", result["model_generation"]["state"])
        self.assertTrue(result["model_generation"]["saved"])
        self.assertFalse((self.out / ".state/toy/pot_rtn/model.onnx").exists())
        self.assertEqual(
            1, len(list((self.out / ".state/.generated").glob("*/*/complete.json")))
        )

    def test_saved_generation_survives_failed_evaluation(self):
        with patch(
            "opennpu_quant.runner.evaluate",
            side_effect=RuntimeError("evaluation failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                self.execute()
        self.assertEqual(
            1, len(list((self.out / ".state/.generated").glob("*/*/complete.json")))
        )
        self.assertFalse((self.out / ".state/toy/pot_rtn/complete.json").exists())
        with no_generation():
            result = self.execute()
        self.assertEqual("completed", result["rows"][0]["state"])
        self.assertEqual(
            "reused",
            read_json(self.out / ".state/toy/pot_rtn/result.json")["model_generation"][
                "state"
            ],
        )

    def test_unsaved_model_is_regenerated_for_a_new_evaluation(self):
        self.config["output"]["save_qdq_models"] = False
        self.execute()
        old = read_json(self.out / ".state/toy/pot_rtn/result.json")
        self.assertEqual(
            "model_not_saved_regeneration_required", old["model_generation"]["reuse"]
        )
        self.assertFalse((self.out / ".state/.generated").exists())
        self.config["evaluation"]["limit"] = 2
        with patch(
            "opennpu_quant.runner.quantize",
            side_effect=RuntimeError("regeneration required"),
        ):
            with self.assertRaisesRegex(RuntimeError, "regeneration required"):
                self.execute()
        row = read_json(self.out / "summary.json")["rows"][0]
        self.assertEqual("failed", row["state"])
        self.assertFalse(row["complete"])

    def test_tampered_model_and_missing_completion_are_not_reused(self):
        for corrupt in (True, False):
            with self.subTest(corrupt=corrupt):
                self.execute(force=True)
                cache = next((self.out / ".state/.generated").glob("*/*"))
                if corrupt:
                    with (cache / "model.onnx").open("ab") as stream:
                        stream.write(b"changed")
                else:
                    (cache / "complete.json").unlink()
                self.config["evaluation"]["limit"] = (
                    2 if self.config["evaluation"]["limit"] == 1 else 1
                )
                with patch(
                    "opennpu_quant.runner.quantize", side_effect=RuntimeError("rebuild")
                ):
                    with self.assertRaisesRegex(
                        ValueError if corrupt else RuntimeError,
                        "Generated model artifact changed" if corrupt else "rebuild",
                    ):
                        self.execute()
                self.assertFalse(
                    read_json(self.out / "summary.json")["rows"][0]["complete"]
                )

    def test_force_and_recalibrate_do_not_reuse_saved_model(self):
        for flag in ("force", "recalibrate"):
            with self.subTest(flag=flag):
                self.execute(force=True)
                with patch(
                    "opennpu_quant.runner.quantize",
                    side_effect=RuntimeError("regenerated"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "regenerated"):
                        self.execute(**{flag: True})

    def test_failed_recovery_refreshes_the_success_report(self):
        self.execute()
        (self.out / ".state/toy/pot_rtn/complete.json").write_text("{")
        with patch(
            "opennpu_quant.runner.evaluate",
            side_effect=RuntimeError("evaluation failed"),
        ) as evaluate:
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                self.execute()
        self.assertEqual(1, evaluate.call_count)
        row = read_json(self.out / "summary.json")["rows"][0]
        self.assertEqual("failed", row["state"])
        self.assertFalse(row["complete"])

    def test_external_weights_change_invalidates_generated_model(self):
        self.paths, self.config = fixture(self.root / "external", external=True)
        self.config["output"]["save_qdq_models"] = True
        self.config["conditions"] = {"pot_rtn": self.config["conditions"]["pot_rtn"]}
        self.config["evaluation"]["limit"] = 1
        self.out = self.paths.runs / self.config["name"]
        self.execute()
        weights = self.paths.models / "toy/weights.bin"
        data = bytearray(weights.read_bytes())
        data[0] ^= 1
        weights.write_bytes(data)
        self.config["evaluation"]["limit"] = 2
        with patch(
            "opennpu_quant.runner.quantize", side_effect=RuntimeError("rebuild")
        ):
            with self.assertRaisesRegex(RuntimeError, "rebuild"):
                self.execute()
        self.assertFalse(read_json(self.out / "summary.json")["rows"][0]["complete"])

    def test_short_adaround_saved_before_evaluation_and_reused(self):
        policy = dict(self.config["conditions"]["pot_rtn"], adaround=True)
        self.config["conditions"] = {"pot_adaround": policy}
        self.config["adaround"]["steps"] = 2
        self.execute()
        record = read_json(self.out / ".state/toy/pot_adaround/result.json")
        self.assertIsNotNone(record["adaround"])
        self.config["evaluation"]["limit"] = 2
        with no_generation():
            self.execute()
        self.assertEqual(
            "reused",
            read_json(self.out / ".state/toy/pot_adaround/result.json")[
                "model_generation"
            ]["state"],
        )

    def test_interrupted_evaluation_archive_resumes_without_losing_results(self):
        from opennpu_quant._io import atomic_json
        from opennpu_quant.run_storage import finish_archive

        self.execute()
        payload = (self.out / ".state/toy/pot_rtn/result.json").read_bytes()
        archive = self.out / ".state/.evaluations/interrupted"
        archive.mkdir(parents=True)
        (self.out / ".state/toy").replace(archive / "toy")
        atomic_json(
            self.out / ".state/.evaluation-transition.json",
            dict(
                destination=".evaluations/interrupted",
                names=["toy", "settings.json", "status.json"],
            ),
        )
        finish_archive(self.out / ".state")
        self.assertEqual(payload, (archive / "toy/pot_rtn/result.json").read_bytes())
        self.assertEqual(1, report(archive)["rows"][0]["samples"])
        self.assertFalse((self.out / ".state/.evaluation-transition.json").exists())

    def test_generation_device_depends_on_calibration_and_adaround(self):
        from opennpu_quant.runner import _ModelRun
        from opennpu_quant.configuration import runtime, scheme_for
        from opennpu_quant.ort.config import OrtConfig

        def key(device, adaround=False):
            job = _ModelRun(
                "toy",
                self.config,
                self.paths,
                self.out,
                runtime(self.config),
                "code",
                {},
            )
            job.ort = device
            policy = dict(self.config["conditions"]["pot_rtn"], adaround=adaround)
            return job._generation_identity(
                policy, scheme_for(self.config, "toy", policy["scheme"])
            )

        cpu, gpu = OrtConfig.cpu(), OrtConfig.cuda()
        with (
            patch(
                "opennpu_quant.runner.environment",
                side_effect=lambda ort: ort.to_dict(),
            ),
            patch("torch.cuda.device", side_effect=lambda _: contextlib.nullcontext()),
            patch(
                "opennpu_quant.ort.gpu_environment.training_environment",
                side_effect=lambda torch, device: {"device": device},
            ),
        ):
            self.config["calibration"]["device"] = "cpu"
            self.assertEqual(key(cpu), key(gpu))
            self.assertNotEqual(key(cpu, True), key(gpu, True))
            self.config["calibration"]["device"] = "run"
            self.assertNotEqual(key(cpu), key(gpu))


class ReviewRecoveryTests(unittest.TestCase):
    def test_archive_retries_after_request_restoration_and_partial_report_copy(self):
        import opennpu_quant.run_storage as storage

        for interruption in ("request", "copy"):
            with (
                self.subTest(interruption=interruption),
                tempfile.TemporaryDirectory() as temp,
            ):
                paths, config = fixture(Path(temp))
                run(config, paths, recalibrate=True, formats="csv,json,md")
                config["evaluation"]["limit"] = 1
                injected = []
                unlink, copyfile = Path.unlink, storage.shutil.copyfile

                def stop_unlink(path, *args, **kwargs):
                    if path.name == ".evaluation-transition.json":
                        injected.append(True)
                        raise KeyboardInterrupt("request restoration")
                    return unlink(path, *args, **kwargs)

                def stop_copy(src, dst, *args, **kwargs):
                    if Path(src).name == "settings.json":
                        Path(dst).write_bytes(Path(src).read_bytes()[:5])
                        injected.append(True)
                        raise OSError("partial report")
                    return copyfile(src, dst, *args, **kwargs)

                target = (
                    patch.object(Path, "unlink", stop_unlink)
                    if interruption == "request"
                    else patch.object(storage.shutil, "copyfile", stop_copy)
                )
                with target, self.assertRaises((KeyboardInterrupt, OSError)):
                    run(config, paths, formats="csv,json,md")
                self.assertTrue(injected)
                result = run(config, paths, formats="csv,json,md")
                self.assertEqual(1, result["rows"][0]["samples"])
                self.assertEqual("completed", result["rows"][0]["state"])
                self.assertFalse(result["rows"][0]["full_evaluation"])
                self.assertFalse(
                    (
                        paths.runs
                        / config["name"]
                        / ".state/.evaluation-transition.json"
                    ).exists()
                )

    def test_user_model_named_prepared_can_resume(self):
        import yaml

        with tempfile.TemporaryDirectory() as temp:
            paths, config = fixture(Path(temp))
            (paths.models / "toy").rename(paths.models / "prepared")
            path = paths.models / "prepared/recipe.yaml"
            recipe = yaml.safe_load(path.read_text())
            recipe["name"] = "prepared"
            path.write_text(yaml.safe_dump(recipe))
            config["models"] = ["prepared"]
            config["percentiles"]["prepared"] = config["percentiles"].pop("toy")
            config["conditions"] = {"fp32": {}}
            self.assertTrue(
                run(config, paths, formats="csv,json,md")["rows"][0]["complete"]
            )
            self.assertTrue(
                run(config, paths, formats="csv,json,md")["rows"][0]["complete"]
            )
