"""Auxiliary metrics, fixed-angle numeric accuracy and protocol arithmetic."""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from opennpu_quant.evaluation.face_verification import (
    _fixed_angle_verification,
    verification,
)
from tests.test_lfw import write_pairs


class FixedAngleArithmeticTests(unittest.TestCase):
    def test_reference_row_normalization_bits_and_input_preservation(self):
        rng = np.random.default_rng(143)
        first = rng.standard_normal((6, 128)).astype(np.float32)
        second = rng.standard_normal((6, 128)).astype(np.float32)
        before = first.tobytes(), second.tobytes()
        cosine = np.array(
            [
                -0.03949250280857086,
                -0.11342641711235046,
                -0.022165633738040924,
                0.1188855916261673,
                0.028921369463205338,
                0.07656341791152954,
            ],
            dtype=np.float32,
        )
        radians = np.arange(6, dtype=np.float32)
        expected = np.full(6, 90, dtype=np.float32)
        # Keep the FP32 normalization contract independent of CPU math dispatch.
        with (
            patch(
                "opennpu_quant.evaluation.face_verification.np.arccos",
                return_value=radians,
            ) as acos,
            patch(
                "opennpu_quant.evaluation.face_verification.np.degrees",
                return_value=expected,
            ) as degrees,
        ):
            accuracy, angles = _fixed_angle_verification(
                first, second, np.zeros(6, bool)
            )
        acos.assert_called_once()
        degrees.assert_called_once()
        self.assertEqual(np.float32, acos.call_args.args[0].dtype)
        self.assertEqual(cosine.tobytes(), acos.call_args.args[0].tobytes())
        self.assertIs(radians, degrees.call_args.args[0])
        self.assertIs(expected, angles)
        self.assertEqual(1.0, accuracy)
        self.assertEqual(before, (first.tobytes(), second.tobytes()))

    def test_angles_against_high_precision_reference(self):
        fixture = json.loads(
            (Path(__file__).parent / "fixtures/face_angle_reference.json").read_text()
        )
        rows = fixture["cases"]
        cosines = np.frombuffer(
            bytes.fromhex("".join(row["cosine_le_hex"] for row in rows)), dtype="<f4"
        ).astype(np.float32)
        reference_radians = np.array([float(row["radians"]) for row in rows])
        reference_degrees = np.array([float(row["degrees"]) for row in rows])
        first = np.zeros((len(rows), 128), np.float32)
        second = np.zeros_like(first)
        first[:, 0], second[:, 0] = cosines, 1
        observed = {}
        native_acos = np.arccos

        def capture_cosine(value):
            observed["cosine"] = value.copy()
            observed["radians"] = native_acos(value)
            return observed["radians"]

        # Feed exact cosine values to the real math functions. Normalization is
        # covered separately, so its rounding cannot hide an arccos error here.
        with (
            patch(
                "opennpu_quant.evaluation.face_verification.np.linalg.norm",
                return_value=np.ones((len(rows), 1), np.float32),
            ),
            patch(
                "opennpu_quant.evaluation.face_verification.np.arccos",
                side_effect=capture_cosine,
            ),
        ):
            _, angles = _fixed_angle_verification(
                first, second, np.zeros(len(rows), bool)
            )
        np.testing.assert_array_equal(cosines, observed["cosine"])
        for actual in (observed["radians"], angles):
            self.assertEqual(np.float32, actual.dtype)
            self.assertEqual((len(rows),), actual.shape)
            self.assertTrue(np.isfinite(actual).all())
        # One binary32 spacing from the high-precision acos result. Converting
        # radians to degrees adds one spacing for the constant and multiplication.
        rad_budget = np.abs(np.spacing(reference_radians.astype(np.float32))).astype(
            np.float64
        )
        deg_budget = rad_budget * (180 / math.pi) + np.abs(
            np.spacing(reference_degrees.astype(np.float32))
        ).astype(np.float64)
        for actual, reference, budget in (
            (observed["radians"], reference_radians, rad_budget),
            (angles, reference_degrees, deg_budget),
        ):
            np.testing.assert_array_less(
                np.abs(actual.astype(np.float64) - reference),
                np.nextafter(budget, np.inf),
            )

    def test_native_angles_repeat_bitwise_and_preserve_inputs(self):
        rng = np.random.default_rng(143)
        first = rng.standard_normal((6, 128)).astype(np.float32)
        second = rng.standard_normal((6, 128)).astype(np.float32)
        before = first.tobytes(), second.tobytes()
        accuracy, angles = _fixed_angle_verification(first, second, np.zeros(6, bool))
        repeated_accuracy, repeated = _fixed_angle_verification(
            first, second, np.zeros(6, bool)
        )
        self.assertEqual(np.float32, angles.dtype)
        self.assertEqual((6,), angles.shape)
        self.assertTrue(np.isfinite(angles).all())
        self.assertEqual(1.0, accuracy)
        self.assertEqual(accuracy, repeated_accuracy)
        self.assertEqual(angles.tobytes(), repeated.tobytes())
        self.assertEqual(before, (first.tobytes(), second.tobytes()))

    def test_threshold_compares_stored_angles_as_float64(self):
        first = np.ones((3, 128), np.float32)
        boundary = np.float32(74.18)
        angles = np.array(
            [np.nextafter(boundary, np.float32(0)), boundary, 90], np.float32
        )
        self.assertGreater(float(boundary), 74.18)
        with patch(
            "opennpu_quant.evaluation.face_verification.np.degrees", return_value=angles
        ):
            accuracy, _ = _fixed_angle_verification(
                first, first, np.array([True, False, False])
            )
        self.assertEqual(1.0, accuracy)

    def test_full_protocol_emits_canonical_metric_with_reference_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            pairs = write_pairs(Path(temp) / "pairs.txt")
        first = np.eye(1, 128, 0, dtype=np.float32)[0]
        second = np.eye(1, 128, 1, dtype=np.float32)[0]
        vectors = {
            "A/A_0001.jpg": {0: first, 1: first},
            "A/A_0002.jpg": {0: first, 1: first},
            "B/B_0001.jpg": {0: second, 1: second},
        }
        for pair in pairs[300:356]:
            pair["second"] = "A/A_0002.jpg"
        metrics, details = verification(vectors, pairs, dimensions=128)
        self.assertEqual(0.9906666666666666, metrics["fixed_angle_accuracy"])
        self.assertNotEqual(5944 / 6000, metrics["fixed_angle_accuracy"])
        self.assertEqual(74.18, details["fixed_angle_threshold_degrees"])
        self.assertFalse(any("qualcomm" in k for k in metrics.keys() | details.keys()))
        baseline, baseline_details = verification(
            {k: {i: v[:2] for i, v in views.items()} for k, views in vectors.items()},
            pairs,
            dimensions=2,
        )
        self.assertEqual(baseline_details["folds"], details["folds"])
        for key, value in baseline.items():
            self.assertEqual(
                np.float64(value).tobytes(), np.float64(metrics[key]).tobytes()
            )
