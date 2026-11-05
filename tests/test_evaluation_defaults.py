"""Canonical model defaults, local recipe validation, and comparison isolation."""

import copy
import os
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

from opennpu_quant.models.adapter import ModelEvaluator
from opennpu_quant.models.contracts import EvaluationSample
from opennpu_quant.models.evaluation_profiles import (
    ALTERNATIVES,
    apply_profile,
    profile_name,
)
from opennpu_quant.models.spec import ModelSpec, model_names


class EvaluationDefaultTests(unittest.TestCase):
    def test_old_local_bundle_uses_default_without_changing_its_graph_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "recipe.yaml"
            recipe = copy.deepcopy(ModelSpec.load("yolox").recipe)
            recipe["decode"].pop("profile")
            recipe["graphs"]["postprocess"] = "local_host.onnx"
            recipe["network_input_map"] = {"images": "prepared_images"}
            recipe["preprocess"]["size"] = 320
            path.write_text(yaml.safe_dump(recipe))
            before = path.read_bytes()
            resolved = ModelSpec.for_run("yolox", temp).recipe
            self.assertEqual("yolox_torch_nms_v1", profile_name(resolved))
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(
                "yolox_torch_nms_v1", profile_name(ModelSpec.load(path).recipe)
            )
            self.assertEqual(recipe, {**resolved, "decode": recipe["decode"]})

    def test_explicit_comparison_takes_precedence_and_is_model_specific(self):
        with tempfile.TemporaryDirectory() as temp:
            default = ModelSpec.for_run("resnet18", temp)
            comparison = ModelSpec.for_run("resnet18", temp, profile="stable_topk_v1")
            self.assertEqual("stable_topk_v1", profile_name(default.recipe))
            self.assertEqual("stable_topk_v1", profile_name(comparison.recipe))
            self.assertEqual(
                "stable_topk_v1", profile_name(ModelSpec.load("resnet18").recipe)
            )
            with self.assertRaisesRegex(ValueError, "not verified"):
                ModelSpec.for_run("resnet18", temp, profile="yolox_torch_nms_v1")
            for model, alternative in (
                ("yolov7_face", "yolov7_face_clip_all_v1"),
                ("yolov5s_seg", "yolov5_seg_scaled_coeff_mask_area_v1"),
            ):
                spec = ModelSpec.for_run(model, temp, profile=alternative)
                self.assertEqual(alternative, profile_name(spec.recipe))

    def test_generation_profiles_must_name_packaged_or_local_models(self):
        from opennpu_quant.configuration import load_config, validate

        config = load_config("core10")
        config["models"] = ["resnet18"]
        with tempfile.TemporaryDirectory() as models:
            # A known profile does not make an unknown model valid.
            config["evaluation"]["profiles"] = {"resnet81": "stable_topk_v1"}
            with self.assertRaisesRegex(ValueError, "unknown models: resnet81"):
                validate(config, models_dir=models)
            local = Path(models) / "my_model"
            local.mkdir()
            (local / "recipe.yaml").write_text("{}")
            # Packaged or local models outside the selection remain allowed.
            for model in ("resnet50", "my_model"):
                config["evaluation"]["profiles"] = {model: "stable_topk_v1"}
                validate(config, models_dir=models)

    def test_local_explicit_profile_requires_experiment_override_for_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            recipe = copy.deepcopy(ModelSpec.load("yolov7_face").recipe)
            recipe["decode"]["profile"] = "yolov7_face_clip_all_v1"
            (Path(temp) / "recipe.yaml").write_text(yaml.safe_dump(recipe))
            with self.assertWarnsRegex(UserWarning, "evaluation.profiles.yolov7_face"):
                spec = ModelSpec.for_run("yolov7_face", temp)
            self.assertEqual("yolov7_face_clip_first4_v1", profile_name(spec.recipe))
            spec = ModelSpec.for_run(
                "yolov7_face", temp, profile="yolov7_face_clip_all_v1"
            )
            self.assertEqual("yolov7_face_clip_all_v1", profile_name(spec.recipe))

    def test_legacy_profile_names_are_rejected_without_modifying_recipes(self):
        from opennpu_quant.configuration import load_config, validate

        for model, names in (
            ("resnet18", ("current_v1", "numpy_argsort_v1", "historical_cnn33_v1")),
            ("yolox", ("current_v1", "historical_cnn33_v1")),
            ("yolov7_face", ("current_v1", "historical_cnn33_v1", "face_all_boxes_v1")),
            ("yolov5s_seg", ("current_v1", "historical_cnn33_v1", "mask_area_v1")),
        ):
            for old in names:
                with (
                    self.subTest(model=model, profile=old),
                    tempfile.TemporaryDirectory() as temp,
                ):
                    recipe = copy.deepcopy(ModelSpec.load(model).recipe)
                    recipe["decode"]["profile"] = old
                    path = Path(temp) / "recipe.yaml"
                    path.write_text(yaml.safe_dump(recipe))
                    before = path.read_bytes()
                    with self.assertRaisesRegex(
                        ValueError, "Unknown evaluation profile"
                    ):
                        ModelSpec.load(path)
                    with self.assertRaisesRegex(
                        ValueError, "Unknown evaluation profile"
                    ):
                        ModelSpec.for_run(model, temp)
                    with self.assertRaisesRegex(
                        ValueError, "Unknown evaluation profile"
                    ):
                        ModelSpec.for_run(model, Path(temp) / "empty", profile=old)
                    config = load_config("cnn30")
                    config["evaluation"]["profiles"] = {model: old}
                    with self.assertRaisesRegex(
                        ValueError, "Unknown evaluation profile"
                    ):
                        validate(config)
                    self.assertEqual(before, path.read_bytes())

    def test_current_local_profiles_preserve_their_contracts(self):
        from opennpu_quant.models.evaluation_contracts import resolve

        for model in model_names():
            recipe = ModelSpec.load(model).recipe
            default = profile_name(recipe)
            for profile in (default, ALTERNATIVES.get(default)):
                if profile is None:
                    continue
                with (
                    self.subTest(model=model, profile=profile),
                    tempfile.TemporaryDirectory() as temp,
                ):
                    explicit = apply_profile(recipe, profile)
                    path = Path(temp) / "recipe.yaml"
                    path.write_text(yaml.safe_dump(explicit))
                    before = path.read_bytes()
                    loaded = ModelSpec.load(path)
                    chosen = ModelSpec.for_run(model, temp, profile=profile)
                    self.assertEqual(explicit, loaded.recipe)
                    self.assertEqual(explicit, chosen.recipe)
                    self.assertEqual(
                        resolve(explicit).identity(), resolve(loaded.recipe).identity()
                    )
                    self.assertEqual(before, path.read_bytes())

    def test_non_string_profiles_raise_value_error(self):
        from opennpu_quant.configuration import load_config, validate

        for value in ([], {}, None, True, 1):
            with self.subTest(profile=value):
                config = load_config("core10")
                config["evaluation"]["profiles"] = {"resnet18": value}
                with self.assertRaisesRegex(ValueError, "Unknown evaluation profile"):
                    validate(config)
                with tempfile.TemporaryDirectory() as temp:
                    recipe = copy.deepcopy(ModelSpec.load("resnet18").recipe)
                    recipe["decode"]["profile"] = value
                    path = Path(temp) / "recipe.yaml"
                    path.write_text(yaml.safe_dump(recipe))
                    with self.assertRaisesRegex(
                        ValueError, "Unknown evaluation profile"
                    ):
                        ModelSpec.load(path)

    def test_invalid_profile_cli_returns_command_error_without_traceback(self):
        from opennpu_quant.configuration import load_config

        for value in ([], {}, "current_v1"):
            with self.subTest(profile=value), tempfile.TemporaryDirectory() as temp:
                config = load_config("core10")
                config["evaluation"]["profiles"] = {"resnet18": value}
                path = Path(temp) / "invalid.yaml"
                path.write_text(yaml.safe_dump(config))
                process = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "opennpu_quant",
                        "run",
                        "-c",
                        str(path),
                        "--dry-run",
                    ],
                    cwd=temp,
                    env=os.environ.copy(),
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
                self.assertEqual(2, process.returncode, process.stderr)
                self.assertIn("ValueError: Unknown evaluation profile", process.stderr)
                self.assertNotIn("Traceback", process.stderr)
                self.assertEqual("", process.stdout)

    def test_custom_local_models_keep_their_recipe_and_reject_wrong_name(self):
        with tempfile.TemporaryDirectory() as temp:
            recipe = copy.deepcopy(ModelSpec.load("resnet18").recipe)
            recipe.update(name="custom_classifier")
            recipe["decode"].pop("profile")
            (Path(temp) / "recipe.yaml").write_text(yaml.safe_dump(recipe))
            self.assertEqual(
                recipe, ModelSpec.for_run("custom_classifier", temp).recipe
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                ModelSpec.for_run("resnet18", temp)

    def test_default_classifier_uses_stable_ties_and_historical_override_is_preserved(
        self,
    ):
        logits = {"logits": np.ones((1, 6), np.float32)}
        sample = EvaluationSample("tie", {}, 5)
        with tempfile.TemporaryDirectory() as temp:
            for profile, expected in (
                (None, 0.0),
                ("stable_topk_v1", 0.0),
                ("numpy_topk_v1", 1.0),
            ):
                spec = ModelSpec.for_run("resnet18", temp, profile=profile)
                evaluator = ModelEvaluator(
                    SimpleNamespace(spec=spec, finish=lambda outputs, meta: outputs),
                    None,
                )
                evaluator.update(logits, sample)
                self.assertEqual(expected, evaluator.finalize()["top1"])
                np.testing.assert_array_equal(logits["logits"], np.ones((1, 6)))
