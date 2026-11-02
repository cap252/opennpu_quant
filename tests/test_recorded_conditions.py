"""Condition-specific recorded percentiles and the generated model reuse boundary."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from opennpu_quant._io import read_json
from opennpu_quant.configuration import (
    load_config,
    scheme_for,
    scheme_for_condition,
    validate,
    _scheme_key,
)
from opennpu_quant.quant.scheme import Scheme
from opennpu_quant.runner import _ModelRun, run
from tests.test_result_reliability import fixture


class RecordedConditionTests(unittest.TestCase):
    def policies(self, config, model):
        fields = config["schemes"]["pot"]
        return {
            (model, _scheme_key(fields), False): {99.999},
            (model, _scheme_key(fields), True): {99.99},
        }

    def test_rounding_is_part_of_recorded_lookup_with_explicit_overrides(self):
        config = load_config("core10")
        config["models"] = ["resnet18"]
        config["schemes"]["pot"]["percentile"] = "recorded"
        config["percentiles"] = {}
        config["schemes"] = {"pot": config["schemes"]["pot"]}
        config["conditions"] = {
            "check": {"scheme": "pot"},
            "fit": {"scheme": "pot", "adaround": True},
        }
        with patch(
            "opennpu_quant.configuration._recorded_percentiles",
            return_value=self.policies(config, "resnet18"),
        ):
            validate(config)
            self.assertEqual(
                99.999, scheme_for_condition(config, "resnet18", "check").percentile
            )
            self.assertEqual(
                99.99, scheme_for_condition(config, "resnet18", "fit").percentile
            )
            with self.assertRaisesRegex(ValueError, "Resolve a condition"):
                scheme_for(config, "resnet18", "pot")
            config["percentiles"] = {"resnet18": {"pot": 99.9}}
            self.assertEqual(
                99.9, scheme_for_condition(config, "resnet18", "fit").percentile
            )
            config["schemes"]["pot"]["percentile"] = "per_model"
            self.assertEqual(
                99.9, scheme_for_condition(config, "resnet18", "check").percentile
            )
            config["schemes"]["pot"]["percentile"] = 99.0
            self.assertEqual(
                99.0, scheme_for_condition(config, "resnet18", "check").percentile
            )

    def test_rtn_cache_is_keyed_by_resolved_policy_not_family(self):
        worker = _ModelRun.__new__(_ModelRun)
        worker.bundle = SimpleNamespace(paths={"network": "graph"})
        worker.statistics = object()
        worker.calibration_ort = worker.ort = object()
        worker.factory = lambda: iter(())
        worker.config = {"adaround": {"steps": 1}}
        worker.rtn = {}
        worker.generation = None
        first, second = object(), object()
        with (
            tempfile.TemporaryDirectory() as temp,
            patch(
                "opennpu_quant.runner.quantize", side_effect=[first, second]
            ) as quantize,
            patch("opennpu_quant.runner.apply_adaround", return_value=second) as fit,
        ):
            policy = {"scheme": "pot"}
            directory = Path(temp)
            worker.out = directory
            worker.name = "toy"
            a, b = Scheme(percentile=99.999), Scheme(percentile=99.99)
            self.assertIs(first, worker._quantize(policy, a, directory, "first"))
            self.assertIs(
                second,
                worker._quantize(dict(policy, adaround=True), b, directory, "second"),
            )
            self.assertIs(second, fit.call_args.args[1])
            self.assertIs(second, worker._quantize(policy, b, directory, "again"))
            self.assertEqual(2, quantize.call_count)

    def test_real_runner_and_failed_report_resolve_the_same_condition(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, config = fixture(Path(temp))
            config["conditions"]["pot_adaround"] = {"scheme": "pot", "adaround": True}
            config["adaround"].update(
                steps=1, batch_size=1, window_samples=1, window_steps=1, device="cpu"
            )
            config["percentiles"]["toy"].pop("pot")
            config["schemes"]["pot"]["percentile"] = "recorded"
            with patch(
                "opennpu_quant.configuration._recorded_percentiles",
                return_value=self.policies(config, "toy"),
            ):
                # Collect and quantize actual small tensors for both distinct policies.
                summary = run(config, paths)
                self.assertEqual(
                    [None, 99.999, 99.99],
                    [row["percentile"] for row in summary["rows"]],
                )
                for name, expected in (("pot_rtn", 99.999), ("pot_adaround", 99.99)):
                    value = read_json(
                        paths.runs / config["name"] / "toy" / name / "result.json"
                    )
                    self.assertEqual(expected, value["scheme"]["percentile"])
                with patch(
                    "opennpu_quant.runner._ModelRun._evaluate",
                    side_effect=ValueError("evaluation interrupted"),
                ):
                    with self.assertRaisesRegex(ValueError, "evaluation interrupted"):
                        run(config, paths, force=True)
                summary = read_json(paths.runs / config["name"] / "summary.json")
                self.assertEqual(
                    [None, 99.999, 99.99],
                    [row["percentile"] for row in summary["rows"]],
                )
                self.assertFalse(any(row["complete"] for row in summary["rows"]))
