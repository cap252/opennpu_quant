"""Native ORT failures retain their cause and reach run's failure handling."""

import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as n

from opennpu_quant.ort.config import OrtConfig
from opennpu_quant.ort.errors import runtime_errors
from opennpu_quant.ort.fetch import run_tensors
from opennpu_quant.ort.session import create_session


def runtime_error_model():
    model = h.make_model(
        h.make_graph(
            [
                h.make_node("Flatten", ["images"], ["flat"]),
                h.make_node("Shape", ["flat"], ["shape"]),
                h.make_node("Add", ["shape", "offset"], ["target"]),
                h.make_node("Reshape", ["flat", "target"], ["logits"]),
            ],
            "runtime-error",
            [h.make_tensor_value_info("images", 1, [1, 3, 4, 4])],
            [h.make_tensor_value_info("logits", 1, [1, None])],
            [n.from_array(np.array([0, 1], dtype=np.int64), "offset")],
        ),
        opset_imports=[h.make_opsetid("", 18)],
        ir_version=10,
    )
    onnx.checker.check_model(model, full_check=True)
    return model


class OrtErrorTests(unittest.TestCase):
    def test_real_runtime_and_input_errors_preserve_native_cause(self):
        from onnxruntime.capi import onnxruntime_pybind11_state as native

        session = create_session(runtime_error_model(), OrtConfig.cpu())
        for dtype, error in (
            (np.float32, native.RuntimeException),
            (np.float64, native.InvalidArgument),
        ):
            with (
                self.subTest(dtype=dtype),
                self.assertRaisesRegex(
                    RuntimeError, "ONNX Runtime " + error.__name__
                ) as caught,
            ):
                run_tensors(
                    session, ["logits"], {"images": np.ones((1, 3, 4, 4), dtype=dtype)}
                )
            self.assertIsInstance(caught.exception.__cause__, error)

    def test_real_session_creation_error_is_translated(self):
        from onnxruntime.capi import onnxruntime_pybind11_state as native

        model = runtime_error_model()
        model.graph.node[0].op_type = "MissingOperator"
        with self.assertRaisesRegex(
            RuntimeError, "ONNX Runtime InvalidGraph"
        ) as caught:
            create_session(model.SerializeToString(), OrtConfig.cpu())
        self.assertIsInstance(caught.exception.__cause__, native.InvalidGraph)

    def test_known_ort_errors_and_unrelated_errors_are_distinct(self):
        from onnxruntime.capi import onnxruntime_pybind11_state as native

        names = {
            name
            for name, value in vars(native).items()
            if isinstance(value, type) and issubclass(value, Exception)
        }
        # The verified release's classes, plus any a newer runtime registers.
        self.assertLessEqual(
            {
                "Fail",
                "InvalidArgument",
                "NoSuchFile",
                "NoModel",
                "EngineError",
                "RuntimeException",
                "InvalidProtobuf",
                "ModelLoaded",
                "NotImplemented",
                "InvalidGraph",
                "EPFail",
            },
            names,
        )
        for name in sorted(names):
            error = getattr(native, name)("backend failure")
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(RuntimeError, "ONNX Runtime " + name) as caught,
            ):
                with runtime_errors():
                    raise error
            self.assertIs(caught.exception.__cause__, error)
        # Same class name, but not raised by the native runtime.
        Fail = type("Fail", (Exception,), {})
        for error in (
            TypeError("bug"),
            AssertionError("bug"),
            RuntimeError("python"),
            Fail("python"),
            KeyboardInterrupt(),
            SystemExit(7),
        ):
            with (
                self.subTest(type=type(error)),
                self.assertRaises(type(error)) as caught,
            ):
                with runtime_errors():
                    raise error
            self.assertIs(caught.exception, error)
