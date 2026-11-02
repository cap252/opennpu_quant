"""Small input assets shared by public evaluation tests."""

import numpy as np
import onnx


def lfw_assets(root):
    source = root / "assets"
    source.mkdir()
    h = onnx.helper
    model = h.make_model(
        h.make_graph(
            [
                h.make_node("GlobalAveragePool", ["images"], ["avg"]),
                h.make_node("Flatten", ["avg"], ["flat"], axis=1),
                h.make_node("MatMul", ["flat", "weight"], ["logits"]),
            ],
            "fixture",
            [h.make_tensor_value_info("images", 1, [1, 3, 4, 4])],
            [h.make_tensor_value_info("logits", 1, [1, 6])],
            [
                onnx.numpy_helper.from_array(
                    np.arange(18, dtype=np.float32).reshape(3, 6), "weight"
                )
            ],
        ),
        opset_imports=[h.make_opsetid("", 18)],
        ir_version=10,
    )
    onnx.save(model, source / "model.onnx")
    return source
