"""Numerical environment identity, FP32-only input needs and GPU evidence scope."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from opennpu_quant._io import atomic_json, read_json
from opennpu_quant.ort.cpu_environment import cpu_environment
from opennpu_quant.ort.config import OrtConfig
from opennpu_quant.ort.session import environment
from opennpu_quant.ort.profile import summarize_profile, assert_cuda_observed
from opennpu_quant.ort.diagnostics import compare
from opennpu_quant.runner import run
from tests.test_result_reliability import fixture


class RuntimeBoundaryTests(unittest.TestCase):
    def test_effective_numpy_features_and_flags_change_environment(self):
        before = environment(OrtConfig.cpu())
        enabled = before["cpu_numerics"]["features"] or {}
        candidates = [x for x in ("AVX2", "FMA3", "AVX512F") if enabled.get(x)]
        if not candidates:
            self.assertIsInstance(before["cpu_numerics"]["implementation"], dict)
            return
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json; from opennpu_quant.ort.session import environment; "
                "from opennpu_quant.ort.config import OrtConfig; "
                "print(json.dumps(environment(OrtConfig.cpu())))",
            ],
            env=dict(os.environ, NPY_DISABLE_CPU_FEATURES=",".join(candidates)),
            capture_output=True,
            text=True,
            check=True,
        )
        after = json.loads(child.stdout)
        self.assertNotEqual(before, after)
        for name in candidates:
            self.assertFalse(after["cpu_numerics"]["features"][name])

    def test_fp32_only_without_calibration_and_environment_reuse_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, config = fixture(Path(temp))
            config["conditions"] = {"fp32": {}}
            manifest = paths.data / "imagenet/manifest.json"
            value = read_json(manifest)
            value["calibration"] = []
            atomic_json(manifest, value)
            with patch(
                "opennpu_quant.runner.calibrate",
                side_effect=AssertionError("calibration called"),
            ):
                run(config, paths)
                run(config, paths)
            status = read_json(paths.runs / config["name"] / "status.json")
            self.assertEqual(
                "reused", status["models"]["toy"]["conditions"]["fp32"]["state"]
            )
            with patch.dict(os.environ, {"NPY_DISABLE_CPU_FEATURES": "AVX2"}):
                with self.assertRaisesRegex(ValueError, "execution identity changed"):
                    run(config, paths)
            config["name"] = "mixed"
            config["conditions"]["pot_rtn"] = {
                "scheme": "asym_pc_pot",
                "adaround": False,
            }
            # Keep a valid quantization policy from the preset.
            from opennpu_quant.configuration import load_config

            config["conditions"]["pot_rtn"] = load_config()["conditions"]["pot_rtn"]
            with self.assertRaisesRegex(ValueError, "Calibration sample count"):
                run(config, paths)

    def test_environment_missing_is_pending_and_driver_changes_warn(self):
        base = dict(device="cpu", cpu_numerics=cpu_environment())
        self.assertEqual("pass", compare(base, base)["state"])
        self.assertEqual(
            "pending", compare({"device": "cpu"}, {"device": "cpu"})["state"]
        )
        other = copy.deepcopy(base)
        other["cpu_numerics"]["features"] = {"changed": True}
        self.assertEqual("warning", compare(base, other)["state"])
        base.update(
            device="cuda:0",
            kernel_driver="driver one",
            driver_libraries={"libraries": [dict(name="libcuda.so.1", sha256="a")]},
            loaded_libraries={
                "libraries": [
                    dict(name=n, sha256="x")
                    for n in (
                        "libcudart.so",
                        "libcublas.so",
                        "libcudnn.so",
                        "libonnxruntime_providers_cuda.so",
                    )
                ]
            },
        )
        self.assertEqual("pass", compare(base, base)["state"])
        for field in ("kernel_driver", "driver_libraries"):
            other = copy.deepcopy(base)
            if field == "kernel_driver":
                other[field] = "driver two"
            else:
                other[field]["libraries"][0]["sha256"] = "b"
            self.assertEqual("warning", compare(base, other)["state"])
            other.pop(field)
            self.assertEqual("pending", compare(base, other)["state"])

    def test_cuda_observation_does_not_attest_full_placement(self):
        events = [
            dict(cat="Node", args=dict(provider=p, op_name="Conv"))
            for p in ["CUDAExecutionProvider"] + ["CPUExecutionProvider"] * 99
        ]
        result = summarize_profile(events)
        assert_cuda_observed(["CUDAExecutionProvider"], result)
        self.assertEqual(
            "cpu_compute_observed", result["sampled_main_compute_placement"]
        )
        self.assertEqual(99, result["cpu_main_compute"][0]["calls"])
        self.assertEqual("not_verified", result["full_network_placement"])


class EvaluationCoverageTests(unittest.TestCase):
    def test_manifest_coverage_changes_block_full_result_reuse(self):
        from opennpu_quant.data.samples import Dataset

        for change in ("expected", "length", "both"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                paths, config = fixture(Path(temp))
                config["conditions"] = {"fp32": {}}
                config["evaluation"]["limit"] = None if change == "expected" else 2
                manifest = paths.data / "imagenet/manifest.json"
                calibration = Dataset(manifest).identity("calibration", 2)
                first = run(config, paths)
                self.assertTrue(first["rows"][0]["full_evaluation"])
                output = paths.runs / config["name"]
                result = output / "toy/fp32/result.json"
                original = result.read_bytes()
                data = read_json(manifest)
                if change != "length":
                    data["expected_samples"] = 3
                if change != "expected":
                    data["validation"].append(dict(id=4, path="0.png", target=5))
                atomic_json(manifest, data)
                self.assertEqual(
                    calibration, Dataset(manifest).identity("calibration", 2)
                )
                with self.assertRaisesRegex(
                    ValueError, "execution identity changed.*--force"
                ):
                    run(config, paths)
                self.assertEqual(original, result.read_bytes())
                self.assertFalse(
                    read_json(output / "summary.json")["rows"][0]["complete"]
                )
                if change == "expected":
                    with self.assertRaisesRegex(
                        ValueError, "Missing evaluation samples"
                    ):
                        run(config, paths, force=True)
                elif change == "both":
                    forced = run(config, paths, force=True)
                    row = forced["rows"][0]
                    self.assertEqual(
                        (2, 3, False, False),
                        (
                            row["samples"],
                            row["expected_samples"],
                            row["full_evaluation"],
                            row["complete"],
                        ),
                    )


class RuntimePlacementReuseTests(unittest.TestCase):
    def test_config_identity_ignores_only_gpu_placement_and_arena(self):
        from opennpu_quant._identity import run_config_identity
        from opennpu_quant.configuration import load_config

        config = load_config()
        config["runtime"]["device"] = "cuda:0"
        config["calibration"]["device"] = "cuda:1"
        other = copy.deepcopy(config)
        other["runtime"]["device"] = "cuda:3"
        other["calibration"]["device"] = "cuda:2"
        other["runtime"]["cuda"]["gpu_mem_limit"] = 1024
        self.assertEqual(run_config_identity(config), run_config_identity(other))
        for path, value in (("window_bytes", 1024), ("window_samples", 2)):
            changed = copy.deepcopy(other)
            changed["adaround"][path] = value
            self.assertNotEqual(
                run_config_identity(config), run_config_identity(changed)
            )
        other["runtime"]["device"] = "cpu"
        self.assertNotEqual(run_config_identity(config), run_config_identity(other))

    def test_arena_change_does_not_invalidate_actual_completed_result(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, config = fixture(Path(temp))
            config["conditions"] = {"fp32": {}}
            run(config, paths)
            config["runtime"]["cuda"]["gpu_mem_limit"] = 1024
            with patch(
                "opennpu_quant.runner._ModelRun._evaluate",
                side_effect=AssertionError("unexpected reevaluation"),
            ):
                summary = run(config, paths)
            self.assertTrue(summary["rows"][0]["complete"])
            status = read_json(paths.runs / config["name"] / "status.json")
            self.assertEqual(
                "reused", status["models"]["toy"]["conditions"]["fp32"]["state"]
            )
