import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from opennpu_quant._io import read_json, sha256
from opennpu_quant.configuration import validate
from opennpu_quant.evaluation.capture import verify_capture
from opennpu_quant.report import report
from opennpu_quant.runner import run
from tests.test_result_reliability import fixture


class RunCaptureTests(unittest.TestCase):
    def test_raw_capture_matches_inputs_outputs_and_predictions_and_detects_tampering(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            paths, cfg = fixture(Path(temp))
            cfg["conditions"] = {"fp32": {}}
            cfg["output"]["raw_samples"] = 1
            result = run(cfg, paths, formats="csv,json,md")
            directory = paths.runs / cfg["name"] / ".state/toy/fp32"
            raw = read_json(directory / "raw.json")
            self.assertEqual(1, raw["summary"]["samples"])
            self.assertTrue(verify_capture(directory))
            row = raw["samples"][0]
            with np.load(directory / row["tensors"], allow_pickle=False) as arrays:
                inputs = arrays[row["mapping"]["inputs"]["images"]]
                outputs = arrays[row["mapping"]["outputs"]["logits"]]
                self.assertEqual((1, 3, 4, 4), inputs.shape)
                self.assertEqual((1, 6), outputs.shape)
                self.assertTrue(np.isfinite(outputs).all())
            self.assertTrue((directory / row["prediction"]).is_file())
            with patch(
                "opennpu_quant.runner.evaluate", side_effect=AssertionError("rerun")
            ):
                run(cfg, paths, formats="csv,json,md")
            target = directory / row["tensors"]
            target.write_bytes(target.read_bytes() + b"tampered")
            self.assertFalse(verify_capture(directory))
            self.assertFalse(report(directory.parents[1])["rows"][0]["complete"])
            self.assertEqual(2, result["rows"][0]["samples"])

    def test_capture_budget_ids_and_previous_evidence_survive_force(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, cfg = fixture(Path(temp))
            cfg["conditions"] = {"fp32": {}}
            cfg["output"].update(raw_sample_ids=["3"], raw_max_bytes=100000)
            run(cfg, paths, formats="csv,json,md")
            directory = paths.runs / cfg["name"] / ".state/toy/fp32"
            first = read_json(directory / "raw.json")
            self.assertEqual(["3"], [r["id"] for r in first["samples"]])
            hashes = {name: sha256(directory / name) for name in first["files"]}
            run(cfg, paths, force=True, formats="csv,json,md")
            self.assertEqual(
                hashes, {name: sha256(directory / name) for name in hashes}
            )
            cfg["output"]["raw_max_bytes"] = 1
            result = run(cfg, paths, formats="csv,json,md")
            raw = read_json(directory / "raw.json")
            self.assertEqual(0, raw["summary"]["samples"])
            self.assertEqual(["3"], raw["summary"]["budget_skipped"])
            self.assertTrue(result["rows"][0]["complete"])

    def test_capture_is_optional_and_invalid_options_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, cfg = fixture(Path(temp))
            cfg["conditions"] = {"fp32": {}}
            run(cfg, paths, formats="csv,json,md")
            directory = paths.runs / cfg["name"] / ".state/toy/fp32"
            self.assertFalse((directory / "raw.json").exists())
            for options in (
                {"raw_samples": True},
                {"raw_samples": -1},
                {"raw_samples": 1.5},
                {"raw_sample_ids": ["x", "x"]},
                {"raw_max_bytes": 0},
                {"raw_samples": 2, "raw_sample_ids": ["x"]},
            ):
                bad = copy.deepcopy(cfg)
                bad["output"].update(options)
                with self.subTest(options=options), self.assertRaises(ValueError):
                    validate(bad)

    def test_failed_evaluation_never_publishes_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, cfg = fixture(Path(temp))
            cfg["conditions"] = {"fp32": {}}
            cfg["output"]["raw_samples"] = 1
            from opennpu_quant.models.adapter import ModelEvaluator

            original = ModelEvaluator.update

            def interrupted(evaluator, outputs, sample):
                if str(sample.sample_id) == "3":
                    raise RuntimeError("interrupted evaluation")
                return original(evaluator, outputs, sample)

            with (
                patch.object(ModelEvaluator, "update", interrupted),
                self.assertRaisesRegex(RuntimeError, "interrupted"),
            ):
                run(cfg, paths, formats="csv,json,md")
            directory = paths.runs / cfg["name"] / ".state/toy/fp32"
            self.assertFalse((directory / "complete.json").exists())
            self.assertFalse((directory / "raw.json").exists())
            self.assertTrue(list(directory.glob(".raw-pending-*/*.npz")))

    def test_capture_preserves_raw_values_before_postprocessing_mutation(self):
        from types import SimpleNamespace
        from opennpu_quant.evaluation.capture import RawCapture
        from opennpu_quant._feeds import feed_digest

        class MutableEvaluator:
            def update(self, outputs, sample):
                outputs["logits"][:] = 9
                self.prediction = {"top1": 1}

        with tempfile.TemporaryDirectory() as temp:
            feeds = {"input": np.array([[1, 2]], np.float32)}
            values = {"logits": np.array([[3, 4]], np.float32)}
            expected = feed_digest(values)
            capture = RawCapture(MutableEvaluator(), temp, {"raw_samples": 1})
            capture.update(values, SimpleNamespace(sample_id="first", feeds=feeds))
            capture.finish()
            capture.publish()
            raw = read_json(Path(temp) / "raw.json")
            row = raw["samples"][0]
            self.assertEqual(expected, row["output_digest"])
            with np.load(Path(temp) / row["tensors"], allow_pickle=False) as stored:
                np.testing.assert_array_equal(
                    [[3, 4]], stored[row["mapping"]["outputs"]["logits"]]
                )
                np.testing.assert_array_equal(
                    [[1, 2]], stored[row["mapping"]["inputs"]["input"]]
                )
            self.assertEqual({"top1": 1}, read_json(Path(temp) / row["prediction"]))


class MetadataCaptureReportTests(unittest.TestCase):
    def test_report_with_raw_capture_needs_no_numerical_imports(self):
        import subprocess, sys, json

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths, config = fixture(root)
            config["conditions"] = {"fp32": {}}
            config["output"]["raw_samples"] = 1
            run(config, paths, formats="csv,json,md")
            code = """import importlib.abc,json,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'numpy','onnx','onnxruntime','torch'}:
            raise ImportError('numerical import: '+fullname)
sys.meta_path.insert(0,Block())
from opennpu_quant.report import report
print(json.dumps(report(sys.argv[1],output=sys.argv[2])))
"""
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    code,
                    str(paths.runs / config["name"]),
                    str(root / "report"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertTrue(json.loads(result.stdout)["rows"][0]["complete"])
