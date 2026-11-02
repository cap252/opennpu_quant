"""Structural parity is independent of framework output tolerances."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from opennpu_quant.export.finalize import finalize, outputs
from opennpu_quant.models.spec import ModelSpec, model_names


def graph():
    nodes = [
        h.make_node("Conv", ["images", "weight"], ["head"], name="stem"),
        h.make_node("LeakyRelu", ["head"], ["a"], alpha=0.1, name="activation"),
        h.make_node("Neg", ["head"], ["b"], name="negative"),
        h.make_node("Sub", ["a", "b"], ["output"], name="merge"),
        h.make_node("Shape", ["head"], ["size"], name="shape"),
    ]
    g = h.make_graph(
        nodes,
        "test",
        [h.make_tensor_value_info("images", 1, [1, 1, "h", "w"])],
        [
            h.make_tensor_value_info("output", 1, [1, 1, "h", "w"]),
            h.make_tensor_value_info("size", 7, [4]),
        ],
        [nh.from_array(np.ones((1, 1, 1, 1), np.float32), "weight")],
    )
    return h.make_model(g, ir_version=8, opset_imports=[h.make_opsetid("", 17)])


class ExportContractTests(unittest.TestCase):
    def test_numerical_diagnostic_continues_to_graph_checks_and_saves_model(self):
        model = graph()
        recipe = dict(
            name="contract_fixture",
            task="detection",
            export=dict(family="fixture"),
            graphs=dict(network="network.onnx"),
        )
        feed = {"images": np.ones((1, 1, 2, 3), np.float32)}
        expected = outputs(model, [feed])
        expected[0]["output"] += np.float32(0.5)
        for mode in ("strict", "functional"):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                raw = root / "raw.onnx"
                onnx.save(model, raw)
                result = finalize(raw, recipe, [feed], expected, root, validation=mode)
                self.assertTrue((root / "network.onnx").is_file())
                self.assertEqual("passed", result["state"])
                self.assertEqual("differences", result["framework_diagnostics_state"])
                self.assertFalse(result["framework_parity_passed"])
                self.assertFalse(result["framework_raw_numerical_passed"])
                self.assertEqual(
                    0.5, result["checks"][0]["framework_raw"]["errors"]["output"]
                )
                self.assertEqual(0.0, result["checks"][0]["raw_final"]["output"])
                self.assertEqual(0.0, result["checks"][0]["partition"]["output"])

    def test_missing_fixtures_cannot_be_reported_as_success(self):
        for feeds, references in (([], []), ([{}], []), ([{}], [{}, {}])):
            with self.subTest(feeds=feeds, references=references):
                with self.assertRaisesRegex(ValueError, "one framework reference"):
                    finalize(None, {}, feeds, references, None)

    def test_raw_final_numerical_failure_is_still_fatal(self):
        model = graph()
        recipe = dict(
            name="contract_fixture",
            task="detection",
            export=dict(family="fixture"),
            graphs=dict(network="network.onnx"),
        )
        feed = {"images": np.ones((1, 1, 2, 3), np.float32)}
        expected = outputs(model, [feed])
        changed = copy.deepcopy(expected)
        changed[0]["output"] += np.float32(0.5)
        for mode in ("strict", "functional"):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                raw = root / "raw.onnx"
                onnx.save(model, raw)
                with (
                    patch(
                        "opennpu_quant.export.finalize.outputs",
                        autospec=True,
                        side_effect=[expected, changed],
                    ),
                    self.assertRaises(AssertionError),
                ):
                    finalize(raw, recipe, [feed], expected, root, validation=mode)
                self.assertFalse((root / "network.onnx").exists())

    def test_historical_contract_is_not_claimed_by_public_export(self):
        model = graph()
        recipe = dict(
            name="contract_fixture",
            task="detection",
            export=dict(family="fixture"),
            graphs=dict(network="network.onnx"),
        )
        feed = {"images": np.ones((1, 1, 2, 3), np.float32)}
        expected = outputs(model, [feed])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw.onnx"
            onnx.save(model, raw)
            for mode in ("strict", "functional"):
                value = finalize(raw, recipe, [feed], expected, root, validation=mode)
                self.assertEqual(value["release_structure"]["state"], "not_checked")


class PackagedExportSettingsTests(unittest.TestCase):
    def test_reviewed_export_settings_cover_all_supported_models(self):
        expected = json.loads(
            (Path(__file__).parent / "fixtures/export_settings.json").read_text()
        )
        self.assertEqual(set(model_names()), set(expected))
        for name, settings in expected.items():
            self.assertEqual(settings, ModelSpec.load(name).recipe["export"])
