"""Rules of the evaluation-config steps, checked without running inference."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from opennpu_quant._io import atomic_json
from opennpu_quant.exceptions import describe
from opennpu_quant.public_run import (
    _completed_record,
    _record_failure,
    _reuse_or_recover,
    _runtime_config,
    _sample_scope,
    _selection,
)
from opennpu_quant.results import metric_values
from opennpu_quant.run_storage import condition_counts


def prepared(identity="identity", preprocessing="preprocessing"):
    return SimpleNamespace(identity=identity, preprocessing=preprocessing)


class CountTests(unittest.TestCase):
    def test_reused_conditions_count_as_completed(self):
        status = dict(
            models=dict(
                a=dict(
                    conditions=dict(
                        fp32=dict(state="reused"), q=dict(state="completed")
                    )
                ),
                b=dict(
                    conditions=dict(fp32=dict(state="pending"), q=dict(state="failed"))
                ),
            )
        )
        self.assertEqual(
            dict(completed=2, pending=1, failed=1), condition_counts(status)
        )


class SelectionIdentityTests(unittest.TestCase):
    def test_catalog_location_and_gpu_index_are_not_settings(self):
        catalog = dict(
            models=dict(
                toy=dict(
                    recipe=dict(task="classification"),
                    variants=dict(fp32=dict(network={}, scheme=None)),
                    defaults=dict(fp32="fp32"),
                )
            )
        )
        config = dict(
            name="n",
            source="catalog.json",
            models=["toy"],
            conditions="defaults",
            runtime=dict(device="cuda:0"),
            evaluation=dict(limit=None),
            output={},
        )
        signatures = {}
        for source, device in (
            ("catalog.json", "cuda:0"),
            ("../elsewhere/catalog.json", "cuda:1"),
            ("catalog.json", "cpu"),
        ):
            config.update(source=source, runtime=dict(device=device))
            signatures[device] = _selection(config, catalog)[2]
        self.assertEqual(signatures["cuda:0"], signatures["cuda:1"])
        self.assertNotEqual(signatures["cuda:0"], signatures["cpu"])
        config["evaluation"]["limit"] = 3
        self.assertNotEqual(signatures["cpu"], _selection(config, catalog)[2])


class CatalogReuseIdentityTests(unittest.TestCase):
    def test_only_memory_capacity_is_excluded_without_mutating_settings(self):
        from copy import deepcopy
        from opennpu_quant.run_identity import provided_condition

        variant = dict(
            network=dict(sha256="network", external_data={"w": "weights"}),
            scheme=dict(percentile=99.9),
            adaround=dict(steps=10000),
        )
        catalog = dict(
            models=dict(
                toy=dict(
                    recipe=dict(task="classification"),
                    variants=dict(fp32=variant),
                    defaults=dict(fp32="fp32"),
                )
            )
        )
        config = dict(
            models=["toy"],
            conditions="defaults",
            runtime=dict(device="cpu"),
            evaluation=dict(limit=None),
            output={},
        )
        original = deepcopy(catalog)
        selection = _selection(config, catalog)[2]
        condition = provided_condition("eval", {"network": "hash"}, variant, {})
        for runtime in (
            {},
            {"gpu_mem_limit": 6 * 1024**3},
            {"gpu_mem_limit": 8 * 1024**3},
        ):
            changed = deepcopy(catalog)
            selected = changed["models"]["toy"]["variants"]["fp32"]
            selected["runtime"] = runtime
            before = deepcopy(changed)
            with self.subTest(runtime=runtime):
                self.assertEqual(selection, _selection(config, changed)[2])
                self.assertEqual(
                    condition,
                    provided_condition("eval", {"network": "hash"}, selected, {}),
                )
                self.assertEqual(before, changed)
        self.assertEqual(original, catalog)
        for field, value in (
            ("network", dict(sha256="other", external_data={"w": "weights"})),
            ("network", dict(sha256="network", external_data={"w": "other"})),
            ("scheme", dict(percentile=99.99)),
            ("adaround", dict(steps=20)),
        ):
            changed = deepcopy(catalog)
            selected = changed["models"]["toy"]["variants"]["fp32"]
            selected[field] = value
            with self.subTest(field=field, value=value):
                self.assertNotEqual(selection, _selection(config, changed)[2])
                self.assertNotEqual(
                    condition,
                    provided_condition("eval", {"network": "hash"}, selected, {}),
                )


class SampleScopeTests(unittest.TestCase):
    def test_rows_must_match_the_requested_scope(self):
        for expected, rows, limit, pair_scope in (
            (100, 100, None, False),
            (100, 10, 10, False),
            (100, 100, 500, False),
            (6000, 6000, 10, True),
        ):
            with self.subTest(expected=expected, rows=rows, limit=limit):
                _sample_scope(expected, range(rows), limit, pair_scope)
        for expected, rows, limit in ((100, 99, None), (100, 11, 10), (0, 0, None)):
            with (
                self.subTest(expected=expected, rows=rows, limit=limit),
                self.assertRaisesRegex(ValueError, "differs from prepared scope"),
            ):
                _sample_scope(expected, range(rows), limit, False)


class RuntimeConfigTests(unittest.TestCase):
    def test_device_and_gpu_memory_limit(self):
        self.assertEqual(
            ("CPUExecutionProvider",), _runtime_config("cpu", {}, False).providers
        )
        gpu = _runtime_config("cuda:1", {"runtime": {"gpu_mem_limit": 1024}}, True)
        options = dict(gpu.provider_options[0])
        self.assertEqual(
            ("1", "1024"), (options["device_id"], options["gpu_mem_limit"])
        )
        self.assertTrue(gpu.strict_versions)
        default = dict(_runtime_config("cuda:0", {}, False).provider_options[0])
        self.assertEqual(str(6 * 1024**3), default["gpu_mem_limit"])


class ReuseOrRecoverTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.state = Path(folder.name)
        self.directory = self.state / "toy/fp32"
        self.directory.mkdir(parents=True)

    def call(self, record=None):
        return _reuse_or_recover(
            self.directory,
            record or prepared(),
            state=self.state,
            attempt="attempt",
            model="toy",
            condition="fp32",
        )

    def test_a_new_condition_is_evaluated(self):
        self.assertFalse(self.call())

    def test_custom_preprocessing_is_never_reused(self):
        atomic_json(self.directory / "complete.json", dict(identity="identity"))
        with self.assertRaisesRegex(
            ValueError, "Preprocessing identity is unavailable"
        ):
            self.call(prepared(preprocessing=None))

    def test_a_changed_identity_is_refused(self):
        atomic_json(self.directory / "complete.json", dict(identity="other"))
        with self.assertRaisesRegex(ValueError, "Run execution identity changed"):
            self.call()

    def test_invalid_records_are_preserved_and_evaluated_once(self):
        # The identity still matches, but the proof is missing.
        atomic_json(
            self.directory / "result.json",
            dict(state="completed", evidence=dict(identity="identity")),
        )
        with self.assertLogs("opennpu_quant.public_run", "WARNING") as logs:
            self.assertFalse(self.call())
        self.assertFalse(self.directory.exists())
        archived = self.state / ".evaluations/attempt/toy/fp32/result.json"
        self.assertTrue(archived.is_file())
        self.assertIn("toy/fp32", logs.output[0])


class CompletedRecordTests(unittest.TestCase):
    def test_record_carries_metrics_and_identities(self):
        base = dict(state="pending", functional_passed=False, evidence=dict(run="r"))
        values = SimpleNamespace(
            metrics={"top1": 0.5},
            metric_units={"top1": "fraction"},
            metric_details={"classes": 2},
            samples=2,
            complete=True,
            runtime={"device": "cpu"},
            elapsed_s=1.5,
        )
        evaluation = SimpleNamespace(
            expected=2,
            evaluation="evaluation",
            identity="identity",
            preprocessing="preprocessing",
            bundle=SimpleNamespace(spec=SimpleNamespace(recipe={})),
        )
        definition = dict(
            variants=dict(fp32=dict(network=dict(sha256="fp32-network"))),
            defaults=dict(fp32="fp32"),
        )
        with patch(
            "opennpu_quant.models.evaluation_profiles.identity",
            return_value="profile",
        ):
            record = _completed_record(base, values, evaluation, definition)
        self.assertEqual("completed", record["state"])
        self.assertTrue(record["functional_passed"])
        self.assertEqual(
            metric_values(values.metrics, values.metric_units), record["metrics"]
        )
        self.assertEqual(
            (2, 2, True, True, "evaluation", "profile"),
            (
                record["samples"],
                record["expected_samples"],
                record["complete"],
                record["full_evaluation"],
                record["evaluation_identity"],
                record["evaluation_profile"],
            ),
        )
        self.assertEqual(
            dict(
                run="r",
                identity="identity",
                model_identity="fp32-network",
                preprocessing_identity="preprocessing",
            ),
            record["evidence"],
        )
        # The base record it starts from is left unchanged.
        self.assertEqual(
            dict(state="pending", functional_passed=False, evidence=dict(run="r")),
            base,
        )


class FailureRecordTests(unittest.TestCase):
    def test_state_error_record_and_traceback_log(self):
        current = dict(state="running")
        missing = FileNotFoundError("manifest.json")
        self.assertEqual(
            "pending",
            _record_failure(current, missing, preparing=True, model="m", condition="c"),
        )
        self.assertEqual(describe(missing), describe(current["error"]))
        with self.assertLogs("opennpu_quant.public_run", "ERROR") as logs:
            state = _record_failure(
                current, TypeError("bug"), preparing=False, model="m", condition="c"
            )
        self.assertEqual("failed", state)
        self.assertIn("m/c", logs.output[0])
        with self.assertNoLogs("opennpu_quant.public_run", "ERROR"):
            state = _record_failure(
                current, KeyboardInterrupt(), preparing=False, model="m", condition="c"
            )
        self.assertEqual(
            ("interrupted", "KeyboardInterrupt"), (state, describe(current["error"]))
        )


if __name__ == "__main__":
    unittest.main()
