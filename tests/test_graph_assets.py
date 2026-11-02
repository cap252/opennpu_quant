"""Public model I/O preserves external-weight and sparse-tensor protections."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx

from opennpu_quant import OrtConfig
from opennpu_quant._io import sha256
from opennpu_quant.graph.model import (
    load_model,
    external_data_files,
    model_identity,
    bundle_identity,
    save_model,
)
from opennpu_quant.ort.session import create_session
from opennpu_quant.public_run import evaluate_config
from tests.test_public_run import evaluation_fixture


def sparse_model(root, *, external=True, component="values", container="initializer"):
    h = onnx.helper
    values = onnx.numpy_helper.from_array(np.array([2.0, 3.0], np.float32), "weight")
    indices = onnx.numpy_helper.from_array(np.array([0, 3], np.int64), "indices")
    sparse = h.make_sparse_tensor(values, indices, [2, 2])
    if external:
        tensor = getattr(sparse, component)
        (root / "weights.bin").write_bytes(tensor.raw_data)
        onnx.external_data_helper.set_external_data(tensor, "weights.bin")
        tensor.ClearField("raw_data")
    graph = h.make_graph(
        [h.make_node("MatMul", ["images", "weight"], ["scores"])],
        "sparse",
        [h.make_tensor_value_info("images", 1, [1, 2])],
        [h.make_tensor_value_info("scores", 1, [1, 2])],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=10)
    if container == "initializer":
        model.graph.sparse_initializer.append(sparse)
    elif container == "constant":
        model.graph.node.insert(
            0, h.make_node("Constant", [], ["weight"], sparse_value=sparse)
        )
    elif container == "subgraph":
        branch = copy.deepcopy(model.graph)
        del branch.input[:]
        branch.sparse_initializer.append(sparse)
        del model.graph.node[:]
        model.graph.initializer.append(
            onnx.numpy_helper.from_array(np.array(True), "condition")
        )
        model.graph.node.append(
            h.make_node(
                "If", ["condition"], ["scores"], then_branch=branch, else_branch=branch
            )
        )
    elif container in ("function", "list_attribute"):
        attribute = (
            {"sparse_value": sparse}
            if container == "function"
            else {"sparse_values": [sparse]}
        )
        node = h.make_node("Constant", [], ["weight"], **attribute)
        model.functions.append(
            h.make_function(
                "test", "SparseWeight", [], ["weight"], [node], [h.make_opsetid("", 18)]
            )
        )
        model.opset_import.append(h.make_opsetid("test", 1))
        model.graph.node.insert(
            0, h.make_node("SparseWeight", [], ["weight"], domain="test")
        )
    else:
        raise ValueError("Unknown sparse fixture container")
    path = root / "model.onnx"
    path.write_bytes(model.SerializeToString())
    return path


class GraphAssetsTests(unittest.TestCase):
    def test_sparse_external_data_rejected_by_public_load_identity_and_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = sparse_model(root)
            for source in (path, onnx.load(path, load_external_data=False)):
                for operation in (load_model, model_identity):
                    with self.assertRaisesRegex(ValueError, "External sparse ONNX"):
                        operation(source)
                with self.assertRaisesRegex(ValueError, "External sparse ONNX"):
                    save_model(source, root / "copy.onnx")
            with self.assertRaisesRegex(ValueError, "External sparse ONNX"):
                bundle_identity(path)
            self.assertFalse((root / "copy.onnx").exists())

    def test_nested_external_sparse_values_and_indices_are_rejected(self):
        for container in (
            "initializer",
            "constant",
            "subgraph",
            "function",
            "list_attribute",
        ):
            for component in ("values", "indices"):
                with (
                    self.subTest(container=container, component=component),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    path = sparse_model(
                        Path(tmp), component=component, container=container
                    )
                    for operation in (external_data_files, load_model):
                        with self.assertRaisesRegex(ValueError, "External sparse ONNX"):
                            operation(path)
                    with self.assertRaisesRegex(ValueError, "External sparse ONNX"):
                        load_model(onnx.load(path, load_external_data=False))

    def test_embedded_sparse_weights_load_and_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = sparse_model(Path(tmp), external=False)
            model = load_model(path)
            self.assertTrue(model.graph.sparse_initializer[0].values.raw_data)
            session = create_session(model, OrtConfig.cpu())
            result = session.run(None, {"images": np.ones((1, 2), np.float32)})
            np.testing.assert_array_equal([[2.0, 3.0]], result[0])

    def test_external_weights_are_required_before_evaluation_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, cfg, catalog = evaluation_fixture(root)
            cfg["conditions"] = ["fp32"]
            desc = catalog["models"]["toy"]["variants"]["fp32"]["network"]
            path = root / desc["path"]
            model = onnx.load(path)
            onnx.save_model(
                model,
                str(path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="weights.bin",
                size_threshold=0,
            )
            weights = path.parent / "weights.bin"
            desc.update(
                sha256=sha256(path), external_data={"weights.bin": sha256(weights)}
            )
            missing_pin = copy.deepcopy(catalog)
            missing_pin["models"]["toy"]["variants"]["fp32"]["network"].pop(
                "external_data"
            )
            self.assertEqual(
                1,
                evaluate_config(cfg, missing_pin, root, paths, dry_run=True)[
                    "validation"
                ]["failed"],
            )
            self.assertEqual(
                1, evaluate_config(cfg, catalog, root, paths)["execution"]["completed"]
            )
            original = weights.read_bytes()
            weights.write_bytes(original + b"changed")
            result = evaluate_config(cfg, catalog, root, paths)
            self.assertEqual(1, result["execution"]["failed"])
            self.assertFalse(result["rows"][0]["complete"])
            weights.write_bytes(original)
            self.assertEqual(
                1, evaluate_config(cfg, catalog, root, paths)["execution"]["completed"]
            )
            weights.unlink()
            self.assertNotEqual(
                1, evaluate_config(cfg, catalog, root, paths)["execution"]["completed"]
            )

    def test_external_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = sparse_model(root, external=False)
            model = onnx.load(path)
            tensor = onnx.numpy_helper.from_array(np.ones((2, 2), np.float32), "weight")
            del model.graph.sparse_initializer[:]
            onnx.external_data_helper.set_external_data(tensor, "../weights.bin")
            tensor.ClearField("raw_data")
            model.graph.initializer.append(tensor)
            path.write_bytes(model.SerializeToString())
            with self.assertRaisesRegex(
                ValueError, "External data must stay within the model bundle"
            ):
                load_model(path)
