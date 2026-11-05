"""Recover internal evaluation evidence once, without retrying failed inference."""

import contextlib
import json
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from opennpu_quant import runner
from opennpu_quant._io import atomic_json, read_json, sha256
from opennpu_quant.evaluation import loop
from opennpu_quant.public_run import evaluate_config
from opennpu_quant.report import report
from tests.test_public_run import evaluation_fixture
from tests.test_result_reliability import fixture


def recovery_fixture(root, mode, *, adaround=False):
    if mode == "generate":
        paths, config = fixture(root)
        config["output"]["save_qdq_models"] = True
        if adaround:
            config["conditions"]["pot_rtn"]["adaround"] = True
            config["adaround"]["steps"] = 2
        execute = lambda: runner.run(config, paths)
    else:
        paths, config, catalog = evaluation_fixture(root)
        execute = lambda: evaluate_config(config, catalog, root, paths)
    config["evaluation"]["limit"] = 1
    execute()
    out = paths.runs / config["name"]
    return paths, config, out, execute


def forbid_generation():
    stack = contextlib.ExitStack()
    for name in ("calibrate", "quantize", "apply_adaround"):
        stack.enter_context(
            patch.object(runner, name, side_effect=AssertionError(name))
        )
    return stack


def failure_probe(mode):
    """Subprocess watchdog also catches a future accidental infinite retry loop."""
    with tempfile.TemporaryDirectory() as tmp:
        _, _, out, execute = recovery_fixture(Path(tmp), mode)
        directory = out / ".state/toy/pot_rtn"
        (directory / "complete.json").write_text("{broken")
        target = (
            "opennpu_quant.runner.evaluate"
            if mode == "generate"
            else "opennpu_quant.evaluation.loop.evaluate"
        )
        with (
            forbid_generation(),
            patch(target, side_effect=RuntimeError("persistent failure")) as run,
        ):
            try:
                execute()
            except RuntimeError as error:
                if mode != "generate" or str(error) != "persistent failure":
                    raise
            calls = run.call_count
        row = report(out)["rows"][1]
        archives = list((out / ".state/.evaluations").glob("*/toy/pot_rtn"))
        # A later user invocation may try again. Within this invocation there
        # must be just one failed evaluation, with no stale completion proof.
        return dict(
            calls=calls,
            state=row["state"],
            value=row["value_pct"],
            archives=len(archives),
            complete=(directory / "complete.json").exists(),
            other=report(out)["rows"][0]["state"],
        )


def cli_recovery_probe(mode, fail):
    import cv2
    import yaml
    from opennpu_quant.cli import main, THREAD_VARIABLES

    for name in THREAD_VARIABLES:
        os.environ.setdefault(name, "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cv2.setNumThreads(1)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _, config, out, _ = recovery_fixture(root, mode)
        config_path = root / "recovery.yaml"
        config_path.write_text(yaml.safe_dump(config))
        directory = out / ".state/toy/pot_rtn"
        (directory / "complete.json").write_text("{broken")
        target = (
            "opennpu_quant.runner.evaluate"
            if mode == "generate"
            else "opennpu_quant.evaluation.loop.evaluate"
        )
        original = runner.evaluate if mode == "generate" else loop.evaluate
        stdout, stderr = io.StringIO(), io.StringIO()
        options = (
            dict(side_effect=RuntimeError("persistent failure"))
            if fail
            else dict(wraps=original)
        )
        code = 0
        with (
            forbid_generation(),
            patch(target, **options) as evaluation,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            try:
                main(
                    [
                        "run",
                        "-c",
                        str(config_path),
                        "--home",
                        str(root),
                        "--local-config",
                        str(root / "local.yaml"),
                        "--json",
                    ]
                )
            except SystemExit as error:
                code = error.code
        value = json.loads(stdout.getvalue()) if stdout.getvalue() else None
        return dict(
            code=code,
            calls=evaluation.call_count,
            json=value is not None,
            warning="Preserved" in stderr.getvalue(),
            failed=value.get("execution", {}).get("failed") if value else None,
        )


class EvaluationRecoveryTests(unittest.TestCase):
    def test_corrupt_internal_records_recover_once_and_then_reuse(self):
        for mode in ("generate", "evaluate"):
            for damage in (
                "metric",
                "result_json",
                "complete_json",
                "missing_result",
                "missing_complete",
            ):
                with (
                    self.subTest(mode=mode, damage=damage),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    _, _, out, execute = recovery_fixture(Path(tmp), mode)
                    directory = out / ".state/toy/pot_rtn"
                    result_path = directory / "result.json"
                    proof_path = directory / "complete.json"
                    expected = read_json(result_path)["metrics"]
                    if damage == "metric":
                        value = read_json(result_path)
                        value["metrics"]["top1"]["value"] = 0.12345
                        atomic_json(result_path, value)
                    elif damage == "result_json":
                        result_path.write_text("{broken")
                    elif damage == "complete_json":
                        proof_path.write_text("{broken")
                    elif damage == "missing_result":
                        result_path.unlink()
                    else:
                        proof_path.unlink()
                    preserved = {
                        p.name: p.read_bytes()
                        for p in directory.iterdir()
                        if p.is_file()
                    }
                    self.assertEqual("unverified", report(out)["rows"][1]["state"])
                    target = (
                        "opennpu_quant.runner.evaluate"
                        if mode == "generate"
                        else "opennpu_quant.evaluation.loop.evaluate"
                    )
                    original = runner.evaluate if mode == "generate" else loop.evaluate
                    with (
                        forbid_generation(),
                        patch(target, wraps=original) as evaluation,
                    ):
                        value = execute()
                    self.assertEqual(1, evaluation.call_count)
                    self.assertEqual("reused", value["rows"][0]["state"])
                    row = value["rows"][1]
                    self.assertEqual("completed", row["state"])
                    self.assertEqual(
                        (1, 2, False),
                        (
                            row["samples"],
                            row["expected_samples"],
                            row["full_evaluation"],
                        ),
                    )
                    self.assertEqual(expected, read_json(result_path)["metrics"])
                    archives = list((out / ".state/.evaluations").glob("*/toy/pot_rtn"))
                    self.assertEqual(1, len(archives))
                    self.assertEqual(
                        preserved,
                        {
                            p.name: p.read_bytes()
                            for p in archives[0].iterdir()
                            if p.is_file()
                        },
                    )
                    with (
                        forbid_generation(),
                        patch(target, side_effect=AssertionError("already completed")),
                    ):
                        reused = execute()
                    self.assertEqual(
                        ["reused", "reused"], [r["state"] for r in reused["rows"]]
                    )

    def test_saved_adaround_model_is_not_retrained_during_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, out, execute = recovery_fixture(Path(tmp), "generate", adaround=True)
            model = out / "models/toy/pot_rtn/network.onnx"
            before = sha256(model)
            (out / ".state/toy/pot_rtn/complete.json").write_text("{")
            with (
                forbid_generation(),
                patch.object(runner, "evaluate", wraps=runner.evaluate) as evaluate,
            ):
                result = execute()
            self.assertEqual(1, evaluate.call_count)
            self.assertEqual("completed", result["rows"][1]["state"])
            self.assertEqual(before, sha256(model))

    def test_persistent_inference_failure_is_not_retried(self):
        for mode in ("generate", "evaluate"):
            with self.subTest(mode=mode):
                command = [
                    sys.executable,
                    "-B",
                    "-c",
                    "import json,sys; from tests.test_evaluation_recovery import failure_probe; print(json.dumps(failure_probe(sys.argv[1])))",
                    mode,
                ]
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    env=os.environ.copy(),
                    cwd=Path(__file__).resolve().parents[1],
                    timeout=30,
                )
                self.assertEqual(0, process.returncode, process.stderr)
                result = json.loads(process.stdout.splitlines()[-1])
                self.assertEqual(
                    dict(
                        calls=1,
                        state="failed",
                        value=None,
                        archives=1,
                        complete=False,
                        other="reused",
                    ),
                    result,
                )

    def test_cli_recovery_keeps_json_separate_and_failure_codes(self):
        for mode in ("generate", "evaluate"):
            for fail in (False, True):
                with self.subTest(mode=mode, fail=fail):
                    command = [
                        sys.executable,
                        "-B",
                        "-c",
                        "import json,sys; from tests.test_evaluation_recovery import cli_recovery_probe; print(json.dumps(cli_recovery_probe(sys.argv[1], sys.argv[2] == 'yes')))",
                        mode,
                        "yes" if fail else "no",
                    ]
                    process = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        env=os.environ.copy(),
                        cwd=Path(__file__).resolve().parents[1],
                        timeout=30,
                    )
                    self.assertEqual(0, process.returncode, process.stderr)
                    value = json.loads(process.stdout.splitlines()[-1])
                    expected_code = (2 if mode == "generate" else 1) if fail else 0
                    self.assertEqual(expected_code, value["code"], value)
                    self.assertEqual(1, value["calls"], value)
                    self.assertTrue(value["warning"], value)
                    self.assertEqual(
                        not (fail and mode == "generate"), value["json"], value
                    )
                    if mode == "evaluate":
                        self.assertEqual(int(fail), value["failed"], value)

    def test_unavailable_or_changed_identity_does_not_start_evaluation(self):
        for mode in ("generate", "evaluate"):
            for damage in ("both_json", "input", "stored_identity"):
                with (
                    self.subTest(mode=mode, damage=damage),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    paths, _, out, execute = recovery_fixture(Path(tmp), mode)
                    directory = out / ".state/toy/pot_rtn"
                    (directory / "complete.json").write_text("{broken")
                    if damage == "both_json":
                        (directory / "result.json").write_text("{broken")
                    elif damage == "input":
                        manifest = paths.data / "imagenet/manifest.json"
                        info = read_json(manifest)
                        info["validation"][0]["target"] = 0
                        atomic_json(manifest, info)
                    else:
                        value = read_json(directory / "result.json")
                        value["evidence"]["identity"] = "another-execution"
                        atomic_json(directory / "result.json", value)
                    target = (
                        "opennpu_quant.runner.evaluate"
                        if mode == "generate"
                        else "opennpu_quant.evaluation.loop.evaluate"
                    )
                    with (
                        forbid_generation(),
                        patch(
                            target, side_effect=AssertionError("must not evaluate")
                        ) as evaluate,
                    ):
                        if mode == "generate":
                            with self.assertRaisesRegex(ValueError, "identity"):
                                execute()
                        else:
                            value = execute()
                            self.assertGreater(value["execution"]["failed"], 0)
                    self.assertEqual(0, evaluate.call_count)
                    self.assertFalse(
                        list((out / ".state/.evaluations").glob("*/toy/pot_rtn"))
                    )

    def test_damaged_saved_model_is_not_an_evaluation_record_repair(self):
        for proof_damage in (False, True):
            with (
                self.subTest(proof_damage=proof_damage),
                tempfile.TemporaryDirectory() as tmp,
            ):
                _, _, out, execute = recovery_fixture(Path(tmp), "generate")
                directory = out / ".state/toy/pot_rtn"
                if proof_damage:
                    (directory / "complete.json").write_text("{broken")
                model = out / "models/toy/pot_rtn/network.onnx"
                with model.open("ab") as stream:
                    stream.write(b"changed")
                with (
                    forbid_generation(),
                    patch.object(
                        runner, "evaluate", side_effect=AssertionError("bad model")
                    ) as evaluate,
                ):
                    with self.assertRaisesRegex(ValueError, "model artifact changed"):
                        execute()
                self.assertEqual(0, evaluate.call_count)
                self.assertIsNone(report(out)["rows"][1]["value_pct"])
