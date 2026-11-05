"""One rule for the state an exception leaves a condition in, on every run path."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from opennpu_quant._io import read_json
from opennpu_quant.cli import THREAD_VARIABLES, main
from opennpu_quant.exceptions import (
    EXPECTED_ERRORS,
    ExecutionError,
    InvalidOutputError,
    OutputMismatchError,
    condition_state,
    describe,
    error_record,
    expected,
)
from opennpu_quant.ort.profile import ATOL, RTOL, output_difference
from opennpu_quant.public_run import evaluate_config
from opennpu_quant.run_records import current_records
from opennpu_quant.runner import run
from tests.test_public_run import evaluation_fixture
from tests.test_result_reliability import fixture

# Exception, state it leaves, and whether it is reported as a message.
CASES = (
    (ValueError("bad input"), "failed", True),
    (RuntimeError("ONNX Runtime failure"), "failed", True),
    (OSError("disk error"), "failed", True),
    (ImportError("optional package"), "failed", True),
    (OutputMismatchError("rerun differs"), "failed", True),
    (InvalidOutputError("Nonfinite prediction"), "failed", True),
    (AssertionError("check failed"), "failed", False),
    (TypeError("programming error"), "failed", False),
    (KeyError("missing key"), "failed", False),
    (MemoryError("out of memory"), "failed", False),
    (KeyboardInterrupt(), "interrupted", False),
    (SystemExit(3), "interrupted", False),
)


@contextlib.contextmanager
def first_evaluation_raises(error):
    """Raise ``error`` from the first evaluation; later ones run normally."""
    from opennpu_quant.evaluation import loop

    original = loop.evaluate
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(None)
        if len(calls) == 1:
            raise error
        return original(*args, **kwargs)

    with patch.object(loop, "evaluate", side_effect=evaluate):
        yield calls


class ConditionStateTests(unittest.TestCase):
    def test_every_exception_kind_leaves_one_state(self):
        for error, state, reported in CASES:
            with self.subTest(error=type(error).__name__):
                self.assertEqual(state, condition_state(error))
                self.assertEqual(state, condition_state(error, preparing=True))
                self.assertEqual(reported, expected(error))

    def test_missing_file_while_preparing_stays_pending(self):
        missing = FileNotFoundError("manifest.json")
        self.assertEqual("pending", condition_state(missing, preparing=True))
        self.assertEqual("failed", condition_state(missing))

    def test_execution_errors_keep_their_builtin_types(self):
        self.assertTrue(issubclass(OutputMismatchError, RuntimeError))
        self.assertTrue(issubclass(InvalidOutputError, ValueError))
        for kind in (ExecutionError, OutputMismatchError, InvalidOutputError):
            self.assertTrue(issubclass(kind, EXPECTED_ERRORS))

    def test_description_omits_an_empty_message(self):
        self.assertEqual("ValueError: bad", describe(ValueError("bad")))
        self.assertEqual("KeyboardInterrupt", describe(KeyboardInterrupt()))
        # Error records kept in status.json read the same way.
        for error in (ValueError("bad"), KeyboardInterrupt()):
            self.assertEqual(describe(error), describe(error_record(error)))

    def test_command_line_reports_expected_errors_with_exit_code_2(self):
        for error, _, reported in CASES:
            stderr = io.StringIO()
            with (
                self.subTest(error=type(error).__name__),
                patch("opennpu_quant.cli.dispatch", side_effect=error),
                contextlib.redirect_stderr(stderr),
            ):
                if reported:
                    with self.assertRaises(SystemExit) as exited:
                        main(["report", "unused"])
                    self.assertEqual(2, exited.exception.code)
                    self.assertIn(describe(error), stderr.getvalue())
                else:
                    with self.assertRaises(type(error)):
                        main(["report", "unused"])


class EvaluationLoopTests(unittest.TestCase):
    def test_nonfinite_predictions_are_invalid_outputs(self):
        from onnx import helper, numpy_helper
        from opennpu_quant import evaluate
        from tests.library_fixtures import evaluator, samples, toy_model

        model = toy_model()
        model.graph.initializer.append(
            numpy_helper.from_array(np.zeros(1, np.float32), "zero")
        )
        model.graph.node.append(helper.make_node("Div", ["logits", "zero"], ["scaled"]))
        model.graph.output[0].name = "scaled"
        with self.assertRaisesRegex(
            InvalidOutputError, "Nonfinite prediction"
        ) as raised:
            evaluate(model, samples(), evaluator=evaluator())
        # Callers that caught the former ValueError still do.
        self.assertIsInstance(raised.exception, ValueError)


class OutputDifferenceTests(unittest.TestCase):
    def test_agrees_with_assert_allclose(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            reference = rng.normal(size=6).astype(np.float32)
            noise = rng.normal(scale=10 ** rng.uniform(-7, -3), size=6)
            output = (reference + noise).astype(np.float32)
            if rng.random() < 0.1:
                output[0] = np.nan
            try:
                np.testing.assert_allclose(output, reference, atol=ATOL, rtol=RTOL)
                matches = True
            except AssertionError:
                matches = False
            self.assertEqual(matches, output_difference(output, reference) is None)

    def test_names_the_reason(self):
        for output, reference, reason in (
            (np.zeros((1, 3)), np.zeros((1, 4)), "shape (1, 3) instead of (1, 4)"),
            (
                np.array([np.inf, 1.0]),
                np.array([0.0, 1.0]),
                "1 of 2 values differ, 1 of them non-finite",
            ),
            (
                np.array([1.001, 1.0]),
                np.array([1.0, 1.0]),
                "1 of 2 values differ, max absolute difference 0.001, "
                "max relative difference 0.001",
            ),
            (np.array([True, False]), np.array([True, False]), None),
        ):
            with self.subTest(reason=reason):
                self.assertEqual(reason, output_difference(output, reference))


class EvaluationPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.paths, cls.config, cls.catalog = evaluation_fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def settings(self, name):
        settings = copy.deepcopy(self.config)
        settings["name"] = name
        return settings

    def test_any_error_fails_only_its_condition(self):
        for index, (error, state, reported) in enumerate(CASES):
            if state != "failed":
                continue
            name = f"failed-{index}"
            logs = (
                self.assertNoLogs("opennpu_quant.public_run", "ERROR")
                if reported
                else self.assertLogs("opennpu_quant.public_run", "ERROR")
            )
            with self.subTest(error=type(error).__name__):
                with first_evaluation_raises(error), logs as captured:
                    value = evaluate_config(
                        self.settings(name), self.catalog, self.root, self.paths
                    )
                self.assertEqual(
                    dict(state="failed", completed=1, pending=0, failed=1),
                    value["execution"],
                )
                rows = read_json(self.paths.runs / name / ".state/status.json")
                rows = rows["models"]["toy"]["conditions"]
                self.assertEqual("failed", rows["fp32"]["state"])
                self.assertEqual(
                    dict(type=type(error).__name__, message=str(error)),
                    rows["fp32"]["error"],
                )
                self.assertEqual("completed", rows["pot_rtn"]["state"])
                if not reported:
                    # Unexpected errors keep their traceback in the log.
                    self.assertIn("toy/fp32", captured.output[0])
                    self.assertIs(error, captured.records[0].exc_info[1])

    def test_stop_request_interrupts_the_run(self):
        for index, (error, state, _) in enumerate(CASES):
            if state != "interrupted":
                continue
            name = f"stopped-{index}"
            with self.subTest(error=type(error).__name__):
                with first_evaluation_raises(error), self.assertRaises(type(error)):
                    evaluate_config(
                        self.settings(name), self.catalog, self.root, self.paths
                    )
                status = read_json(self.paths.runs / name / ".state/status.json")
                self.assertEqual("interrupted", status["state"])
                rows = status["models"]["toy"]["conditions"]
                self.assertEqual("interrupted", rows["fp32"]["state"])
                self.assertEqual("pending", rows["pot_rtn"]["state"])
                reasons = {
                    r["condition"]: r.get("failure_reason")
                    for r in current_records(self.paths.runs / name)[0]
                }
                self.assertEqual(describe(error), reasons["fp32"])

    def test_report_abort_preserves_completed_conditions_and_resumes_without_retrying_writer(
        self,
    ):
        from opennpu_quant import public_run

        original = public_run.report_from_settings_locked
        for phase in ("initial", "running", "completed", "reused", "final"):
            for error in (
                KeyboardInterrupt("report stopped"),
                OSError("report disk error"),
            ):
                name = f"report-{phase}-{type(error).__name__}"
                config = self.settings(name)
                config["conditions"] = ["fp32"]
                folder = self.paths.runs / name
                if phase == "reused":
                    evaluate_config(config, self.catalog, self.root, self.paths)
                calls = []
                failures = []

                def fail_report(*args, **kwargs):
                    status = read_json(folder / ".state/status.json")
                    condition = status["models"]["toy"]["conditions"]["fp32"]["state"]
                    calls.append(condition)
                    if (
                        (phase == "initial" and len(calls) == 1)
                        or (phase == "final" and status["finished"] is not None)
                        or (
                            phase in ("running", "completed", "reused")
                            and condition == phase
                        )
                    ):
                        failures.append(None)
                        raise error
                    return original(*args, **kwargs)

                with self.subTest(phase=phase, error=type(error).__name__):
                    with (
                        patch.object(
                            public_run,
                            "report_from_settings_locked",
                            side_effect=fail_report,
                        ),
                        self.assertRaises(type(error)) as raised,
                    ):
                        evaluate_config(config, self.catalog, self.root, self.paths)
                    self.assertIs(error, raised.exception)
                    self.assertEqual(1, len(failures))
                    status = read_json(folder / ".state/status.json")
                    self.assertEqual(
                        "interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "failed",
                        status["state"],
                    )
                    self.assertIsNotNone(status["finished"])
                    self.assertEqual(error_record(error), status["error"])
                    model = status["models"]["toy"]
                    current = model["conditions"]["fp32"]
                    completed = phase in ("completed", "reused", "final")
                    if completed:
                        self.assertEqual("completed", model["state"])
                        self.assertEqual(
                            "reused" if phase == "reused" else "completed",
                            current["state"],
                        )
                        proof = folder / ".state/toy/fp32/complete.json"
                        before = proof.read_bytes()
                    else:
                        self.assertEqual(
                            "pending" if phase == "initial" else status["state"],
                            model["state"],
                        )
                    with patch.object(
                        public_run,
                        "_evaluate_prepared",
                        wraps=public_run._evaluate_prepared,
                    ) as evaluated:
                        result = evaluate_config(
                            config, self.catalog, self.root, self.paths
                        )
                    self.assertEqual(1, result["execution"]["completed"])
                    self.assertEqual(0 if completed else 1, evaluated.call_count)
                    if completed:
                        self.assertEqual(before, proof.read_bytes())

    def test_preparation_interrupt_and_report_failure_keep_original_error(self):
        from opennpu_quant import public_run

        config = self.settings("prepare-interrupt")
        config["conditions"] = ["fp32"]
        folder = self.paths.runs / config["name"]
        original = public_run.report_from_settings_locked
        calls = []
        error = KeyboardInterrupt("preparation stopped")

        def report_or_fail(*args, **kwargs):
            calls.append(None)
            if len(calls) > 1:
                raise OSError("cannot save interruption report")
            return original(*args, **kwargs)

        with (
            patch.object(public_run, "_prepare_evaluation", side_effect=error),
            patch.object(
                public_run, "report_from_settings_locked", side_effect=report_or_fail
            ),
            self.assertLogs("opennpu_quant.public_run", "ERROR"),
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            evaluate_config(config, self.catalog, self.root, self.paths)
        self.assertIs(error, raised.exception)
        self.assertEqual(2, len(calls))
        status = read_json(folder / ".state/status.json")
        self.assertEqual("interrupted", status["state"])
        self.assertIsNotNone(status["finished"])
        self.assertEqual(error_record(error), status["error"])
        self.assertEqual("interrupted", status["models"]["toy"]["state"])

    def test_dry_run_interrupt_creates_no_output(self):
        config = self.settings("dry-interrupt")
        with (
            patch(
                "opennpu_quant.public_run._prepare_evaluation",
                side_effect=KeyboardInterrupt(),
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            evaluate_config(config, self.catalog, self.root, self.paths, dry_run=True)
        self.assertFalse((self.paths.runs / config["name"]).exists())

    def test_dry_run_reports_unexpected_preparation_errors(self):
        with (
            patch(
                "opennpu_quant.data.samples.preprocessing_implementation",
                side_effect=TypeError("programming error"),
            ),
            self.assertLogs("opennpu_quant.public_run", "ERROR"),
        ):
            value = evaluate_config(
                self.settings("dry"), self.catalog, self.root, self.paths, dry_run=True
            )
        self.assertEqual(dict(ready=0, pending=0, failed=2), value["validation"])
        self.assertEqual(
            ["TypeError: programming error"] * 2,
            [row["reason"] for row in value["preparation"]],
        )

    @patch.dict(os.environ, {key: "1" for key in THREAD_VARIABLES})
    def test_command_line_finishes_the_run_after_an_unexpected_error(self):
        config = self.root / "unexpected.yaml"
        config.write_text(yaml.safe_dump(self.settings("command")))
        stdout = io.StringIO()
        with (
            first_evaluation_raises(TypeError("programming error")),
            contextlib.redirect_stdout(stdout),
            self.assertLogs("opennpu_quant.public_run", "ERROR"),
            self.assertRaises(SystemExit) as exited,
        ):
            main(
                [
                    "run",
                    "-c",
                    str(config),
                    "--home",
                    str(self.root),
                    "--local-config",
                    str(self.root / "local.yaml"),
                    "--json",
                ]
            )
        self.assertEqual(1, exited.exception.code)
        rows = json.loads(stdout.getvalue())["rows"]
        self.assertEqual(["failed", "completed"], [row["state"] for row in rows])

    @patch.dict(os.environ, {key: "1" for key in THREAD_VARIABLES})
    def test_command_line_counts_the_conditions_of_a_pending_run(self):
        config = self.root / "pending.yaml"
        config.write_text(yaml.safe_dump(self.settings("pending")))
        stdout = io.StringIO()
        with (
            patch(
                "opennpu_quant.public_run._assets",
                side_effect=FileNotFoundError("network.onnx"),
            ),
            contextlib.redirect_stdout(stdout),
            self.assertRaises(SystemExit) as exited,
        ):
            main(
                [
                    "run",
                    "-c",
                    str(config),
                    "--home",
                    str(self.root),
                    "--local-config",
                    str(self.root / "local.yaml"),
                ]
            )
        self.assertEqual(1, exited.exception.code)
        self.assertIn("Conditions: completed=0, failed=0, pending=2", stdout.getvalue())
        status = read_json(self.paths.runs / "pending/.state/status.json")
        self.assertEqual("pending", status["models"]["toy"]["state"])


class GenerationPathTests(unittest.TestCase):
    def test_generation_stops_and_records_the_same_states(self):
        for error, state in (
            (OutputMismatchError("rerun differs"), "failed"),
            (TypeError("programming error"), "failed"),
            (KeyboardInterrupt(), "interrupted"),
            (SystemExit(3), "interrupted"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                tempfile.TemporaryDirectory() as temp,
            ):
                paths, config = fixture(Path(temp))
                config["conditions"] = {"fp32": {}}
                with (
                    patch("opennpu_quant.runner.evaluate", side_effect=error),
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaises(type(error)),
                ):
                    run(config, paths, formats="csv")
                status = read_json(paths.runs / config["name"] / ".state/status.json")
                self.assertEqual(state, status["state"])
                self.assertEqual(type(error).__name__, status["error"]["type"])
                self.assertEqual(
                    state, status["models"]["toy"]["conditions"]["fp32"]["state"]
                )


if __name__ == "__main__":
    unittest.main()
