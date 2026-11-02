"""Full LFW pair order, folds, views and rejection before mixed-model work."""

import copy
import contextlib
import io
import re
import tempfile
import unittest
from tests.library_fixtures import small_config

from pathlib import Path
from unittest.mock import patch

import numpy as np

from opennpu_quant._io import atomic_json, read_json, sha256
from opennpu_quant.configuration import load_config, validate
from opennpu_quant.data.lfw_protocol import require_full, reject_partial_result
from opennpu_quant.evaluation.face_verification import verification, read_pairs


def write_pairs(path):
    path.write_text(
        "10\t300\n"
        + "".join("A\t1\t2\n" if i % 600 < 300 else "A\t1\tB\t1\n" for i in range(6000))
    )
    return read_pairs(path)


def full_manifest(root):
    from PIL import Image

    pair_path = root / "pairs.txt"
    pairs = write_pairs(pair_path)
    keys = sorted({p[k] for p in pairs for k in ("first", "second")})
    Image.new("RGB", (112, 112)).save(root / "image.png")
    atomic_json(
        root / "alignment.json",
        dict(kind="lfw_mobilefacenet", smoke=False, failures=[]),
    )
    info = dict(
        dataset="lfw_mobilefacenet",
        root=str(root),
        smoke=False,
        expected_samples=6,
        annotations=dict(pairs="pairs.txt", alignment="alignment.json"),
        evaluation_pairs=pairs,
        calibration=[
            dict(id="cal", path="image.png", image_key="C/C_0001.jpg", flip=0)
        ],
        validation=[
            dict(id=k + str(f), path="image.png", image_key=k, flip=f)
            for k in keys
            for f in (0, 1)
        ],
    )
    path = root / "manifest.json"
    atomic_json(path, info)
    return path, sha256(pair_path)


class LfwProtocolTests(unittest.TestCase):
    def test_folds_flip_views_and_full_only_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            pairs = write_pairs(Path(temp) / "pairs.txt")
        first, second = np.eye(2, dtype=np.float32)
        vectors = {
            "A/A_0001.jpg": {0: first, 1: first},
            "A/A_0002.jpg": {0: first, 1: first},
            "B/B_0001.jpg": {0: second, 1: second},
        }
        before = copy.deepcopy(vectors)
        metrics, details = verification(vectors, pairs, dimensions=2)
        self.assertEqual(1.0, metrics["verification_accuracy"])
        self.assertEqual(10, len(details["folds"]))
        self.assertTrue(all(row["threshold"] == 0.01 for row in details["folds"]))
        for key in vectors:
            for flip in (0, 1):
                self.assertEqual(
                    vectors[key][flip].tobytes(), before[key][flip].tobytes()
                )
        with self.assertRaisesRegex(ValueError, "full 6000"):
            verification(vectors, [pairs[0], pairs[300]], dimensions=2)
        with self.assertRaisesRegex(ValueError, "Incomplete/reordered"):
            verification(vectors, list(reversed(pairs)), dimensions=2)
        vectors["A/A_0001.jpg"].pop(1)
        with self.assertRaisesRegex(ValueError, "original and horizontal"):
            verification(vectors, pairs, dimensions=2)

    def test_scope_boundaries_and_removed_options(self):
        from opennpu_quant.cli import build_parser

        for limit in (None, "full", 6000, 6001):
            require_full(limit=limit)
        for limit in (0, 1, 3, 5999, True):
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(ValueError, "full 6000"),
            ):
                require_full(limit=limit)
        for args in (
            ["run", "--functional-check"],
            ["reproduce", "prepare", "--functional-check"],
            ["data", "prepare", "lfw_mobilefacenet", "--root", ".", "--smoke"],
        ):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                build_parser().parse_args(args)
            self.assertEqual(2, error.exception.code)
        config = small_config()
        config["evaluation"]["functional_check"] = True
        with self.assertRaisesRegex(ValueError, "Unknown setting.*functional_check"):
            validate(config)

    def test_legacy_records_are_rejected_without_numpy_or_artifact_rewrite(self):
        from opennpu_quant.results import tables, read_records

        variants = [
            dict(primary_metric="smoke_pair_accuracy"),
            dict(metrics={"smoke_pair_accuracy": {"value": 1.0, "unit": "fraction"}}),
            dict(evaluation_mode="functional_check"),
            dict(task="face_verification", limit=4),
            dict(task="face_verification", metric_details=dict(pair_count=2)),
        ]
        for row in variants:
            with (
                self.subTest(row=row),
                self.assertRaisesRegex(ValueError, "full 6000"),
            ):
                tables([row])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "records.json"
            atomic_json(
                path, dict(schema_version=1, records=[], functional_checks=[{}])
            )
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "full 6000"):
                read_records(path)
            self.assertEqual(before, path.read_bytes())
        reject_partial_result(dict(task="face_verification", limit=6000))

    def test_mixed_run_rejects_before_lock_calibration_or_inference(self):
        from opennpu_quant.paths import Paths
        from opennpu_quant.runner import run

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = load_config("cnn30")
            config["models"] = ["resnet18", "mobilefacenet_relu_foamliu"]
            config["evaluation"]["limit"] = 3
            paths = Paths.resolve(home=root)
            with (
                patch("opennpu_quant.runner.run_directory_lock") as lock,
                patch("opennpu_quant.runner._run_model") as work,
            ):
                with self.assertRaisesRegex(ValueError, "resnet18,yolov5s"):
                    run(config, paths)
            lock.assert_not_called()
            work.assert_not_called()
            self.assertEqual([], list(root.iterdir()))


class LfwManifestTests(unittest.TestCase):
    def test_full_dataset_order_views_identity_and_calibration(self):
        from opennpu_quant.data.samples import Dataset

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, digest = full_manifest(root)
            with patch("opennpu_quant.data.lfw.PAIR_HASH", digest):
                dataset = Dataset(path)
                before = copy.deepcopy(dataset.info)
                cal = dataset.identity("calibration", 1)
                for count in (6000, 6001, None):
                    self.assertEqual(
                        before["validation"], dataset.rows("validation", count)
                    )
                    self.assertEqual(
                        dataset.identity("validation"),
                        dataset.identity("validation", count),
                    )
                self.assertEqual(cal, dataset.identity("calibration", 1))
                with self.assertRaisesRegex(ValueError, "full 6000"):
                    dataset.rows("validation", 3)
                self.assertEqual(before, dataset.info)

    def test_smoke_and_pair_or_view_mutation_rejected(self):
        from opennpu_quant.data.samples import Dataset

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, digest = full_manifest(root)
            original = read_json(path)
            for mutation, message in (
                (lambda r: r.update(smoke=True), "full 6000"),
                (
                    lambda r: r.update(evaluation_pairs=r["evaluation_pairs"][:2]),
                    "full 6000",
                ),
                (
                    lambda r: r["evaluation_pairs"][0].update(same=False),
                    "pairs/fold order",
                ),
                (lambda r: r["validation"].pop(), "image view"),
            ):
                row = copy.deepcopy(original)
                mutation(row)
                atomic_json(path, row)
                with (
                    patch("opennpu_quant.data.lfw.PAIR_HASH", digest),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    Dataset(path)


class LfwAlignmentDependencyTests(unittest.TestCase):
    def test_missing_alignment_dependency_fails_before_download_or_output(self):
        import builtins
        from opennpu_quant.data.lfw import prepare_lfw

        original = builtins.__import__

        def without_skimage(name, *args, **kwargs):
            if name == "skimage" or name.startswith("skimage."):
                raise ModuleNotFoundError("No module named skimage")
            return original(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (
                patch("builtins.__import__", side_effect=without_skimage),
                patch("opennpu_quant.data.lfw.download_inputs") as download,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "scikit-image==0.25.2.*uv pip install"
                ):
                    prepare_lfw(
                        root / "source", root / "data/manifest.json", download=True
                    )
            download.assert_not_called()
            self.assertEqual([], list(root.iterdir()))

    def test_identity_landmarks_preserve_pixels_with_the_installed_dependency(self):
        from opennpu_quant.data.lfw import align_mobilefacenet

        image = np.random.default_rng(1).integers(0, 256, (112, 112, 3), dtype=np.uint8)
        points = np.array(
            [
                [38.29459953, 51.69630051],
                [73.53179932, 51.50139999],
                [56.02519989, 71.73660278],
                [41.54930115, 92.3655014],
                [70.72990036, 92.20410156],
            ],
            dtype=np.float32,
        )
        actual = align_mobilefacenet(image, points.T.reshape(-1))
        self.assertEqual(image.tobytes(), actual.tobytes())

    def test_shared_annotation_reader_rejects_changed_pins_and_duplicate_keys(self):
        from opennpu_quant.data.lfw import read_annotations

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pairs = write_pairs(root / "pairs.txt")
            row = dict(full_path="lfw_funneled/A/A_0001.jpg", landmarks=[1.0] * 10)
            atomic_json(root / "lfw_funneled.json", dict(samples=[row]))
            with (
                patch("opennpu_quant.data.lfw.PAIR_HASH", sha256(root / "pairs.txt")),
                patch(
                    "opennpu_quant.data.lfw.LANDMARK_HASH",
                    sha256(root / "lfw_funneled.json"),
                ),
            ):
                self.assertEqual(
                    (pairs, {"A/A_0001.jpg": row["landmarks"]}), read_annotations(root)
                )
                atomic_json(root / "lfw_funneled.json", dict(samples=[row, row]))
                with self.assertRaisesRegex(ValueError, "changed: lfw_funneled.json"):
                    read_annotations(root)
                with (
                    patch(
                        "opennpu_quant.data.lfw.LANDMARK_HASH",
                        sha256(root / "lfw_funneled.json"),
                    ),
                    self.assertRaisesRegex(ValueError, "Duplicate LFW landmark key"),
                ):
                    read_annotations(root)


class LfwAlignmentReferenceTests(unittest.TestCase):
    """Crops are compared with the earlier CNN30 inputs; host BLAS details are kept apart."""

    def test_crop_identity_is_order_independent_and_detects_any_crop_change(self):
        from opennpu_quant.data import lfw

        images = {
            "B/B_0001.jpg": dict(crop_sha256="2" * 64),
            "A/A_0001.jpg": dict(crop_sha256="1" * 64),
        }
        reordered = dict(reversed(list(images.items())))
        self.assertEqual(lfw.crop_identity(images), lfw.crop_identity(reordered))
        changed = dict(images, **{"A/A_0001.jpg": dict(crop_sha256="3" * 64)})
        self.assertNotEqual(lfw.crop_identity(images), lfw.crop_identity(changed))
        self.assertFalse(lfw.reference_comparison(images)["matches_reference_crops"])
        reference = dict(images=2, sha256=lfw.crop_identity(images))
        with patch.object(lfw, "REFERENCE_CROPS", reference):
            comparison = lfw.reference_comparison(reordered)
        self.assertTrue(comparison["matches_reference_crops"])
        self.assertEqual(comparison["crop_identity"], reference["sha256"])

    def test_reference_covers_calibration_and_all_evaluation_images(self):
        from importlib.resources import files
        from opennpu_quant.data.lfw import REFERENCE_CROPS

        calibration = [
            line
            for line in files("opennpu_quant.resources.splits")
            .joinpath("lfw.txt")
            .read_text()
            .splitlines()
            if line and not line.startswith("#")
        ]
        self.assertEqual(REFERENCE_CROPS["images"], len(calibration) + 7701)
        self.assertRegex(REFERENCE_CROPS["sha256"], "^[0-9a-f]{64}$")

    def test_mismatch_warns_with_kernel_and_guidance_but_match_is_silent(self):
        import warnings
        from opennpu_quant.data.lfw import ALIGNMENT_GUIDE, warn_if_not_reference

        environment = dict(openblas_core="Sandybridge")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_not_reference(dict(matches_reference_crops=True), environment)
            warn_if_not_reference(dict(matches_reference_crops=False), environment)
        self.assertEqual(1, len(caught))
        self.assertIs(caught[0].category, RuntimeWarning)
        message = str(caught[0].message)
        for text in ("Sandybridge", "OPENBLAS_CORETYPE=Haswell", ALIGNMENT_GUIDE):
            self.assertIn(text, message)
        guide_path, anchor = ALIGNMENT_GUIDE.split("#")
        guide = Path(__file__).parents[1] / guide_path
        if guide.is_file():  # the guide ships with the repository, not the wheel
            headings = re.findall(r"^#+ (.+)$", guide.read_text(), re.M)
            slugs = {
                re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-")
                for h in headings
            }
            self.assertIn(anchor, slugs)

    def test_alignment_environment_records_kernel_and_variables(self):
        import os
        from opennpu_quant.data.lfw import alignment_environment, openblas_core
        from opennpu_quant.ort.cpu_environment import cpu_environment

        core = openblas_core()
        self.assertTrue(core is None or (isinstance(core, str) and core))
        with patch.dict(
            os.environ, {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_NUM_THREADS": "1"}
        ):
            environment = alignment_environment()
            expected = cpu_environment()
        dispatch = environment["cpu_numerics"]["dispatch"]
        self.assertEqual("Haswell", dispatch["OPENBLAS_CORETYPE"])
        self.assertEqual("1", dispatch["OPENBLAS_NUM_THREADS"])
        self.assertEqual(expected, environment["cpu_numerics"])
        self.assertEqual(core, environment["openblas_core"])
        self.assertEqual(
            {"openblas_core", "cpu_numerics", "numpy", "opencv", "skimage"},
            set(environment),
        )

    def test_prepare_records_comparison_and_environment_then_warns(self):
        import warnings
        from importlib.resources import files
        from opennpu_quant.data import lfw

        calibration = (
            files("opennpu_quant.resources.splits")
            .joinpath("lfw.txt")
            .read_text()
            .splitlines()
        )
        pairs = [
            dict(first="P/P_0001.jpg", second="P/P_0002.jpg", same=True),
            dict(first="Q/Q_0001.jpg", second="R/R_0001.jpg", same=False),
        ]
        keys = calibration + ["P/P_0001.jpg", "P/P_0002.jpg", "Q/Q_0001.jpg"]
        keys.append("R/R_0001.jpg")

        def crop(image, landmarks, key):
            value = np.zeros((112, 112, 3), np.uint8)
            value.reshape(-1)[: len(key)] = list(key.encode())
            return value

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            for key in keys:
                path = root / "data/lfw_funneled" / key
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(key)  # distinct source bytes per image
            (root / "pairs.txt").write_text("pairs")
            destination = Path(temp) / "data/lfw_mobilefacenet/manifest.json"
            order, original_warn = [], lfw.warn_if_not_reference

            def warn(comparison, environment):
                order.append(destination.is_file())
                original_warn(comparison, environment)

            with (
                patch.object(lfw, "SOURCE_IMAGES", len(keys)),
                patch.object(lfw, "read_annotations", return_value=(pairs, {})),
                patch.object(lfw, "aligned_crop", side_effect=crop),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                with patch.object(lfw, "warn_if_not_reference", side_effect=warn):
                    manifest = lfw.prepare_lfw(root, destination)
                proof = read_json(destination.parent / "alignment.json")
                environment = read_json(
                    destination.parent / "alignment_environment.json"
                )
                self.assertEqual([True], order)  # warning after the manifest
                self.assertEqual(1, len(caught))
                self.assertIs(caught[0].category, RuntimeWarning)
                self.assertFalse(proof["matches_reference_crops"])
                self.assertEqual(
                    lfw.crop_identity(proof["images"]), proof["crop_identity"]
                )
                self.assertEqual(
                    lfw.REFERENCE_CROPS["sha256"], proof["reference_crop_identity"]
                )
                self.assertEqual(
                    dict(pairs="pairs.txt", alignment="alignment.json"),
                    manifest["annotations"],
                )
                self.assertIn("openblas_core", environment)
                self.assertIn("dispatch", environment["cpu_numerics"])

                # The same crops against a matching reference are silent.
                caught.clear()
                reference = dict(images=len(keys), sha256=proof["crop_identity"])
                with patch.object(lfw, "REFERENCE_CROPS", reference):
                    lfw.prepare_lfw(root, destination)
                self.assertEqual([], caught)
                self.assertTrue(
                    read_json(destination.parent / "alignment.json")[
                        "matches_reference_crops"
                    ]
                )

    def test_data_prepare_prints_reference_match_and_kernel(self):
        import json
        import os
        from opennpu_quant.cli import main

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def prepared(name, source_root, manifest, **kwargs):
                manifest.parent.mkdir(parents=True)
                atomic_json(
                    manifest.parent / "alignment.json",
                    dict(matches_reference_crops=False),
                )
                atomic_json(
                    manifest.parent / "alignment_environment.json",
                    dict(openblas_core="Sandybridge"),
                )
                return dict(validation=[0] * 15402, calibration=[0] * 128)

            stdout = io.StringIO()
            with (
                patch.dict(os.environ),  # main() sets thread defaults; restore them
                patch("opennpu_quant.data.prepare.prepare", side_effect=prepared),
                contextlib.redirect_stdout(stdout),
            ):
                main(
                    [
                        "data",
                        "prepare",
                        "lfw_mobilefacenet",
                        "--root",
                        str(root / "source"),
                        "--data-dir",
                        str(root / "data"),
                    ]
                )
            value = json.loads(stdout.getvalue())
        self.assertEqual(15402, value["validation"])
        self.assertIs(False, value["matches_reference_crops"])
        self.assertEqual("Sandybridge", value["openblas_core"])
