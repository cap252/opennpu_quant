import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper

from opennpu_quant._io import sha256
from opennpu_quant.models.spec import ModelSpec
from opennpu_quant.quant.scheme import Scheme
from tools.make_fixed_spatial_bundle import POLICIES, create, fixed_model


def model():
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["images"], ["output"])],
            "dynamic",
            [helper.make_tensor_value_info("images", 1, [1, 3, "height", "width"])],
            [helper.make_tensor_value_info("output", 1, [1, 3, "height", "width"])],
            [numpy_helper.from_array(np.array([0.125], np.float32), "weight")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )


class FixedSpatialBundleTests(unittest.TestCase):
    def test_only_input_dimensions_change_and_source_is_preserved(self):
        value = model()
        before = value.SerializeToString()
        fixed = fixed_model(value, 512)
        self.assertEqual(before, value.SerializeToString())
        self.assertEqual(
            [1, 3, 512, 512],
            [dim.dim_value for dim in fixed.graph.input[0].type.tensor_type.shape.dim],
        )
        fixed.graph.input[0].CopyFrom(value.graph.input[0])
        self.assertEqual(before, fixed.SerializeToString())

    def test_invalid_input_contracts_are_rejected(self):
        for shape in ([2, 3, "h", "w"], [1, 4, "h", "w"], [1, 3, 320, 320]):
            with self.subTest(shape=shape):
                value = model()
                value.graph.input[0].CopyFrom(
                    helper.make_tensor_value_info("images", 1, shape)
                )
                with self.assertRaises(ValueError):
                    fixed_model(value, 512)
        for size in (True, 0, -1, 512.5):
            with self.assertRaises(ValueError):
                fixed_model(model(), size)

    def test_bundle_hashes_and_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "network.onnx"
            onnx.save(model(), source)
            original = source.read_bytes()
            variants = {"fp32": dict(scheme=None)}
            variants.update(
                {f"q{i}": dict(scheme=Scheme().to_dict()) for i in range(6)}
            )
            for row in variants.values():
                row["network"] = dict(path=source.name, sha256=sha256(source))
            catalog = dict(schema_version=1, models={})
            for name in [*POLICIES, "resnet18"]:
                recipe = ModelSpec.load(name).recipe
                recipe["graphs"] = dict(network="network.onnx")
                catalog["models"][name] = dict(
                    recipe=recipe,
                    variants=variants,
                    defaults={key: key for key in variants},
                )
            # An extra search candidate must survive conversion even when unused.
            for row in catalog["models"].values():
                row["variants"] = dict(
                    row["variants"], extra=dict(row["variants"]["q0"])
                )
            path = root / "catalog.json"
            path.write_text(json.dumps(catalog))
            output = root / "fixed"
            config = create(path, output)
            self.assertTrue(config.is_file())
            saved = json.loads((output / "catalog.json").read_text())
            self.assertEqual(6, len(saved["models"]))
            for name, row in saved["models"].items():
                self.assertEqual(
                    ModelSpec.load(name).recipe["preprocess"],
                    row["recipe"]["preprocess"],
                )
                self.assertEqual(8, len(row["variants"]))
                self.assertEqual(catalog["models"][name]["defaults"], row["defaults"])
                for variant in row["variants"].values():
                    desc = variant["network"]
                    self.assertEqual(desc["sha256"], sha256(output / desc["path"]))
            self.assertEqual(original, source.read_bytes())
            preserved = saved["models"]["resnet18"]["variants"]["extra"]["network"]
            self.assertEqual(sha256(source), preserved["sha256"])
            self.assertEqual(original, (output / preserved["path"]).read_bytes())
            with self.assertRaisesRegex(ValueError, "already exists"):
                create(path, output)
            # A failure partway through never publishes an incomplete bundle.
            final = list(catalog["models"].values())[-1]
            final["variants"] = json.loads(json.dumps(final["variants"]))
            final["variants"]["q5"]["network"]["sha256"] = "0" * 64
            path.write_text(json.dumps(catalog))
            failed = root / "failed"
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                create(path, failed)
            self.assertFalse(failed.exists())
            self.assertFalse(list(root.glob(".failed-*")))


if __name__ == "__main__":
    unittest.main()
