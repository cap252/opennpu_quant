"""CNN30 support boundaries do not block custom recipes or metadata reporting."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from opennpu_quant.models.spec import ModelSpec, _RETIRED_MODELS
from opennpu_quant.evaluation_config import load
from opennpu_quant.public_run import evaluate_config
from opennpu_quant.report import report
from opennpu_quant._io import atomic_json
from tests.test_public_run import evaluation_fixture


class Cnn30SupportTests(unittest.TestCase):
    def test_removed_models_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths, cfg, catalog = evaluation_fixture(root)
            for model in sorted(_RETIRED_MODELS):
                with self.subTest(model=model):
                    with self.assertRaisesRegex(
                        ValueError, "Unsupported model in the CNN30 release"
                    ):
                        ModelSpec.load(model)
                    changed = copy.deepcopy(catalog)
                    row = changed["models"].pop("toy")
                    row["recipe"]["name"] = model
                    changed["models"][model] = row
                    atomic_json(root / "catalog.json", changed)
                    with self.assertRaisesRegex(
                        ValueError, "Unsupported model in the CNN30 release"
                    ):
                        load(root / "evaluate.yaml")
                    self.assertFalse((paths.runs / cfg["name"]).exists())

    def test_custom_results_report_without_model_support(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths, cfg, catalog = evaluation_fixture(root)
            value = evaluate_config(cfg, catalog, root, paths)
            self.assertEqual(2, value["execution"]["completed"])
            with patch(
                "opennpu_quant.models.spec.require_supported",
                side_effect=AssertionError("report attempted model loading"),
            ):
                self.assertEqual(2, len(report(paths.runs / cfg["name"])["rows"]))
