"""GPU numerical identity without requiring CUDA in CPU CI."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from opennpu_quant.ort.config import OrtConfig
from opennpu_quant.ort.session import environment
from opennpu_quant.ort.gpu_environment import training_environment
from opennpu_quant.quant.storage import statistics_key

MODULE = "opennpu_quant.ort.gpu_environment"


class GPUEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.device = dict(
            name="GPU model",
            capability=[8, 9],
            multiprocessors=142,
            driver_api=12080,
            ordinal=0,
            uuid="GPU-a",
        )
        self.devices = patch(MODULE + ".device_record", return_value=self.device)
        self.libraries = patch(
            MODULE + ".library_identity",
            return_value={"cudnn": "a", "cublas": "b", "provider": "c"},
        )
        self.devices.start()
        self.libs = self.libraries.start()
        self.addCleanup(self.devices.stop)
        self.addCleanup(self.libraries.stop)

    def test_hardware_library_flags_change_cache_keys(self):
        config = OrtConfig.cuda()
        before = statistics_key("model", "inputs", 2048, config)
        for field, value in (
            ("name", "other"),
            ("capability", [6, 1]),
            ("multiprocessors", 28),
            ("driver_api", 12090),
        ):
            with self.subTest(field=field):
                device = {**self.device, field: value}
                with patch(MODULE + ".device_record", return_value=device):
                    self.assertNotEqual(
                        before, statistics_key("model", "inputs", 2048, config)
                    )
        with patch(MODULE + ".library_identity", return_value={"cudnn": "changed"}):
            self.assertNotEqual(before, statistics_key("model", "inputs", 2048, config))
        with patch.dict("os.environ", {"NVIDIA_TF32_OVERRIDE": "0"}):
            self.assertNotEqual(before, statistics_key("model", "inputs", 2048, config))

    def test_uuid_ordinal_memory_do_not_change_numerical_identity(self):
        before = environment(OrtConfig.cuda())
        with patch(
            MODULE + ".device_record",
            return_value={**self.device, "uuid": "GPU-b", "ordinal": 2},
        ) as query:
            after = environment(OrtConfig.cuda(2, memory_limit=1024))
            query.assert_called_once_with(2)
        self.assertEqual(before, after)
        config = OrtConfig.cuda()
        opts = [dict(x) for x in config.provider_options]
        opts[0]["cudnn_conv_algo_search"] = "DEFAULT"
        changed = replace(
            config, provider_options=tuple(tuple(x.items()) for x in opts)
        )
        self.assertNotEqual(before, environment(changed))

    def test_cpu_does_not_query_cuda(self):
        with patch(
            MODULE + ".gpu_environment", side_effect=AssertionError("CUDA on CPU")
        ):
            self.assertNotIn("cuda_numerics", environment(OrtConfig.cpu()))
            self.assertIsNone(training_environment(None, "cpu"))

    def test_training_uses_actual_torch_device_without_ort_cuda_provider(self):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(current_device=lambda: 3),
            __version__="x",
            version=SimpleNamespace(cuda="12.4"),
            backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90100)),
        )
        with patch(
            MODULE + ".gpu_environment", return_value={"device": {}, "flags": {}}
        ) as get:
            value = training_environment(torch, "cuda")
            get.assert_called_once_with(3, include_provider=False)
            self.assertEqual(value["torch"]["cudnn"], 90100)
            self.assertEqual(value["flags"]["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")

    def test_imported_statistics_reject_changed_gpu_and_missing_evidence(self):
        from opennpu_quant.quant.calibration import (
            validate_statistics,
            CalibrationStats,
        )
        from opennpu_quant.quant.config import CalibrationConfig

        # Environment is checked before the expensive graph layout.
        config = OrtConfig.cuda()
        stats = Mock(spec=CalibrationStats)
        stats.model_identity = "model"
        stats.histograms = SimpleNamespace(bins=2048)
        stats.histogram_bins = 2048
        stats.environment = environment(config)
        with patch(
            "opennpu_quant.quant.calibration.model_identity", return_value="model"
        ):
            for old in (
                dict(stats.environment, cuda_numerics={}),
                {k: v for k, v in stats.environment.items() if k != "cuda_numerics"},
            ):
                stats.environment = deepcopy(old)
                with self.assertRaisesRegex(ValueError, "ORT environment mismatch"):
                    validate_statistics(
                        None, stats, config=CalibrationConfig(), ort=config
                    )

    def test_checkpoint_identity_includes_training_numerics(self):
        # Exercise the actual checkpoint gate before training on CPU. Changing
        # only the independently identified training runtime must reject reuse.
        import tempfile
        from pathlib import Path
        from opennpu_quant import OrtConfig, QuantizationConfig, quantize
        from opennpu_quant.quant.adaround import apply_adaround
        from opennpu_quant.quant.config import AdaroundConfig
        from tests.library_fixtures import toy_model, feeds

        fp32 = toy_model()
        config = OrtConfig.cpu()
        result = quantize(fp32, feeds, config=QuantizationConfig(), ort=config)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            with patch(
                MODULE + ".training_environment", return_value={"device": "first"}
            ):
                apply_adaround(
                    fp32,
                    result,
                    feeds,
                    config=AdaroundConfig(
                        steps=1, batch_size=1, window_samples=1, window_steps=1
                    ),
                    ort=config,
                    checkpoint_dir=checkpoint,
                )
            with patch(
                MODULE + ".training_environment", return_value={"device": "second"}
            ):
                with self.assertRaisesRegex(ValueError, "Checkpoint.*changed"):
                    apply_adaround(
                        fp32,
                        result,
                        feeds,
                        config=AdaroundConfig(
                            steps=1, batch_size=1, window_samples=1, window_steps=1
                        ),
                        ort=config,
                        checkpoint_dir=checkpoint,
                    )


if __name__ == "__main__":
    unittest.main()


class DeviceRecordingTests(unittest.TestCase):
    def test_record_hides_uuid_without_losing_distinct_device_evidence(self):
        from opennpu_quant.ort.gpu_environment import recorded_device, device_identifier

        uuid = "GPU-12345678-1234-1234-1234-123456789abc"
        first = recorded_device({"uuid": uuid, "name": "GPU", "ordinal": 2})
        self.assertNotIn("uuid", first)
        self.assertEqual(64, len(first["uuid_sha256"]))
        self.assertEqual(device_identifier(uuid[4:]), first["uuid_sha256"])
        self.assertEqual(first, recorded_device(first))
        self.assertNotEqual(first["uuid_sha256"], device_identifier(uuid + "0"))
        self.assertIsNone(recorded_device({"uuid": ""})["uuid_sha256"])
        self.assertEqual("GPU", first["name"])

    def test_kernel_driver_reads_text_and_tolerates_missing_or_unreadable(self):
        import tempfile
        from pathlib import Path
        from opennpu_quant.ort.diagnostics import kernel_driver

        text = "NVRM version: NVIDIA UNIX x86_64 Kernel Module  595.99.02"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "version"
            path.write_text(text + "\n")
            self.assertEqual(text, kernel_driver(path))
            self.assertIsNone(kernel_driver(Path(temp) / "missing"))
            with patch.object(Path, "read_text", side_effect=PermissionError):
                self.assertIsNone(kernel_driver(path))

    def test_cuda_evaluation_records_driver_fields_for_diagnostics(self):
        import numpy as np
        from opennpu_quant.evaluation.loop import _evaluate_stream
        from opennpu_quant.ort.cpu_environment import cpu_environment
        from opennpu_quant.ort.diagnostics import compare

        class Evaluator:
            def reset(self):
                pass

            def update(self, values, sample):
                pass

            def finalize(self):
                return {"accuracy": 1.0}

        session = SimpleNamespace(get_outputs=lambda: [SimpleNamespace(name="y")])
        samples = [SimpleNamespace(sample_id=str(i), feeds={}) for i in range(2)]
        loaded = {
            "libraries": [
                dict(name=n, sha256="x")
                for n in (
                    "libcudart.so.12",
                    "libcublas.so.12",
                    "libcudnn.so.9",
                    "libonnxruntime_providers_cuda.so",
                )
            ]
        }
        drivers = dict(
            available=True,
            libraries=[dict(name="libcuda.so.1", sha256="d", state="loaded")],
            not_loaded=[],
        )
        recorded = {}
        for ort in (OrtConfig.cpu(), OrtConfig.cuda(0)):
            measured = dict(
                providers=list(ort.providers), cpu_numerics=cpu_environment()
            )
            with (
                patch(
                    "opennpu_quant.evaluation.loop.environment", return_value=measured
                ),
                patch(
                    "opennpu_quant.ort.cuda_libs.loaded_libraries", return_value=loaded
                ),
                patch(
                    "opennpu_quant.ort.cuda_libs.loaded_driver_libraries",
                    return_value=drivers,
                ),
                patch(
                    "opennpu_quant.ort.diagnostics.kernel_driver",
                    return_value="driver 595.99.02",
                ),
            ):
                result = _evaluate_stream(
                    session,
                    samples,
                    evaluator=Evaluator(),
                    ort=ort,
                    fetch=lambda index, sample: {"y": np.zeros(1, np.float32)},
                )
            recorded[ort.providers[0]] = result.runtime
        cuda = recorded["CUDAExecutionProvider"]
        self.assertEqual("driver 595.99.02", cuda["kernel_driver"])
        self.assertEqual(drivers, cuda["driver_libraries"])
        self.assertEqual("pass", compare(cuda, deepcopy(cuda))["state"])
        # Records without the two fields cannot pass the GPU comparison.
        old = deepcopy(cuda)
        old.pop("kernel_driver")
        old.pop("driver_libraries")
        self.assertEqual(
            ["kernel_driver", "driver_libraries"],
            compare(old, deepcopy(old))["unverified_fields"],
        )
        cpu = recorded["CPUExecutionProvider"]
        self.assertNotIn("kernel_driver", cpu)
        self.assertNotIn("driver_libraries", cpu)

    def test_capture_does_not_collect_raw_gpu_uuids(self):
        import json
        from opennpu_quant.ort.diagnostics import capture
        from opennpu_quant.ort.gpu_environment import recorded_visibility

        uuid = "GPU-12345678-1234-1234-1234-123456789abc"
        with (
            patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": uuid + ",1"}),
            patch(
                "opennpu_quant.ort.diagnostics.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=0, stdout="GPU name, driver\n", stderr=""
                ),
            ) as query,
        ):
            recorded = capture()
        self.assertNotIn("uuid", query.call_args.args[0][1])
        self.assertNotIn(uuid, json.dumps(recorded))
        self.assertEqual(
            recorded_visibility(uuid) + ",1", recorded["cuda_visible_devices"]
        )
        self.assertIsNone(recorded_visibility(None))
        self.assertEqual("2,1", recorded_visibility("2,1"))
