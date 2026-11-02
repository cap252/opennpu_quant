"""Replay repeats inference through run_tensors, like the other session callers."""

import unittest

import numpy as np
from onnx import TensorProto, helper

from opennpu_quant.ort.config import OrtConfig
from opennpu_quant.ort.fetch import run_tensors
from opennpu_quant.ort.session import create_session

FEED = {"x": np.array([[-1.5, 2.0, -3.25, 4.0]], np.float32)}


def session():
    """y = Relu(x), z = Neg(x), and x itself as a passthrough graph output."""
    values = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 4]) for n in "yzx"
    ]
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"]), helper.make_node("Neg", ["x"], ["z"])],
        "repeat",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        values,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    return create_session(model, OrtConfig.cpu())


class InputContractTests(unittest.TestCase):
    def test_overridable_initializer_is_forwarded(self):
        from onnx import numpy_helper

        graph = helper.make_graph(
            [helper.make_node("Add", ["x", "w"], ["y"])],
            "override",
            [
                helper.make_tensor_value_info(n, TensorProto.FLOAT, [1])
                for n in ("x", "w")
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
            [numpy_helper.from_array(np.array([1.0], np.float32), "w")],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
        )
        s = create_session(model, OrtConfig.cpu())
        feed = {"x": np.array([1.0], np.float32), "w": np.array([5.0], np.float32)}
        np.testing.assert_array_equal(
            s.run(["y"], feed)[0], run_tensors(s, ["y"], feed)[0]
        )
        self.assertEqual([6.0], run_tensors(s, ["y"], feed)[0].tolist())
        self.assertEqual([2.0], run_tensors(s, ["y"], {"x": feed["x"]})[0].tolist())

    def test_passthrough_checks_dtype_and_shape(self):
        for value in (np.ones((1, 4), np.float64), np.ones((2, 4), np.float32)):
            with self.subTest(shape=value.shape, dtype=str(value.dtype)):
                with self.assertRaisesRegex(ValueError, "Input (dtype|shape)"):
                    run_tensors(session(), ["x"], {"x": value})
        np.testing.assert_array_equal(FEED["x"], run_tensors(session(), ["x"], FEED)[0])


if __name__ == "__main__":
    unittest.main()
