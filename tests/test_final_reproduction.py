import hashlib
from tests.library_fixtures import small_config

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from opennpu_quant.models.evaluation_profiles import apply_profile
from opennpu_quant.models.spec import ModelSpec


class FinalReproductionTests(unittest.TestCase):
    def test_research_topk_ties_preserve_public_default(self):
        from opennpu_quant.evaluation.metrics import TopKClassification
        from opennpu_quant.models.adapter import ModelEvaluator
        from opennpu_quant.models.contracts import EvaluationSample

        logits = {"logits": np.ones((1, 6), np.float32)}
        sample = EvaluationSample("0", {}, 5)
        default = TopKClassification()
        default.update(logits, sample)
        self.assertEqual({"top1": 0.0, "top5": 0.0}, default.finalize())
        recipe = apply_profile(ModelSpec.load("resnet18").recipe, "numpy_topk_v1")
        bundle = SimpleNamespace(spec=ModelSpec(recipe), finish=lambda out, meta: out)
        evaluator = ModelEvaluator(bundle, None)
        evaluator.update(logits, sample)
        self.assertEqual({"top1": 1.0, "top5": 1.0}, evaluator.finalize())
        self.assertEqual([5, 4, 3, 2, 1], evaluator.prediction["top5"])

    def test_training_input_guard_rejects_drift_before_calibration(self):
        from opennpu_quant._feeds import feed_digest
        from opennpu_quant.configuration import validate
        from opennpu_quant.runner import _ModelRun
        from opennpu_quant import CalibrationConfig

        feed = {"images": np.zeros((1, 3, 2, 2), np.float32)}
        digest = hashlib.sha256(feed_digest(feed).encode()).hexdigest()
        config = small_config()
        name = config["models"][0]
        config["calibration"]["expected_samples_identity"] = {name: "bad"}
        with self.assertRaisesRegex(ValueError, "SHA256"):
            validate(config)
        config["calibration"]["input_cache"] = {"enabled": False}
        config["calibration"]["expected_samples_identity"] = {name: "0" * 64}
        validate(config)
        fake = SimpleNamespace(
            statistics=None, config=config, name=name, factory=lambda: iter([feed])
        )
        with patch("opennpu_quant.runner.calibrate") as calibrate:
            with self.assertRaisesRegex(ValueError, "Calibration inputs differ"):
                _ModelRun._collect_statistics(fake, CalibrationConfig())
            calibrate.assert_not_called()
        config["calibration"]["expected_samples_identity"][name] = digest
        config["calibration"]["device"] = "cpu"
        with patch(
            "opennpu_quant.runner.runtime", side_effect=RuntimeError("guard accepted")
        ):
            with self.assertRaisesRegex(RuntimeError, "guard accepted"):
                _ModelRun._collect_statistics(fake, CalibrationConfig())
