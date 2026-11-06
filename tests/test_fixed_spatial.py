import unittest
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image

from opennpu_quant.models.extended_decode import segmentation
from opennpu_quant.models.preprocess import preprocess
from opennpu_quant.models.spec import ModelSpec


class FixedSpatialTests(unittest.TestCase):
    def test_fixed_shapes_and_preserved_input(self):
        rng = np.random.default_rng(17)
        for kind, size in (
            ("mmseg_letterbox", 512),
            ("yolov5_letterbox", 640),
            ("yolov7_letterbox", 640),
        ):
            for w, h in ((640, 480), (480, 640), (513, 513), (1001, 321), (301, 299)):
                with self.subTest(kind=kind, wh=(w, h)):
                    pixels = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
                    image = Image.fromarray(pixels)
                    cfg = dict(kind=kind, size=size)
                    saved = deepcopy(cfg)
                    tensor, meta = preprocess(image, cfg)
                    self.assertEqual(tensor.shape, (1, 3, size, size))
                    self.assertEqual(tensor.dtype, np.float32)
                    self.assertTrue(tensor.flags.c_contiguous)
                    self.assertTrue(np.isfinite(tensor).all())
                    self.assertEqual(meta["input_size"], [size, size])
                    self.assertEqual(cfg, saved)
                    np.testing.assert_array_equal(np.asarray(image), pixels)

    def test_mmseg_normalizes_before_bottom_right_padding(self):
        pixels = np.full((30, 50, 3), 128, np.uint8)
        value, meta = preprocess(
            Image.fromarray(pixels), dict(kind="mmseg_letterbox", size=64)
        )
        self.assertEqual(meta["valid_size"], [64, 38])
        self.assertEqual(meta["pad"], [0, 0])
        expected = (
            np.array([128, 128, 128], np.float32)
            - np.array([123.675, 116.28, 103.53], np.float32)
        ) / np.array([58.395, 57.12, 57.375], np.float32)
        np.testing.assert_array_equal(value[0, :, 0, 0], expected)
        self.assertTrue((value[:, :, 38:] == 0).all())

    def test_yolov5_retains_area_downsampling_and_half_padding(self):
        pixels = np.random.default_rng(7).integers(
            0, 256, (479, 1001, 3), dtype=np.uint8
        )
        value, meta = preprocess(
            Image.fromarray(pixels), dict(kind="yolov5_letterbox", size=640)
        )
        width, height = meta["valid_size"]
        expected = cv2.resize(
            pixels, (640, int(479 * 640 / 1001)), interpolation=cv2.INTER_AREA
        )
        left, top = [round(v - 0.1) for v in meta["pad"]]
        actual = value[0, :, top : top + height, left : left + width].transpose(1, 2, 0)
        np.testing.assert_array_equal(actual, expected.astype(np.float32) / 255)
        self.assertEqual(meta["scale"], [height / 479, height / 479])
        self.assertEqual(meta["pad"], [0, (640 - height) / 2])

    def test_yolov7_unrounded_padding_restores_coordinates(self):
        import torch
        from opennpu_quant.models.extended_decode import restore_boxes
        from opennpu_quant.models.historical_yolo import restored_boxes

        _, meta = preprocess(
            Image.new("RGB", (1001, 321)), dict(kind="yolov7_letterbox", size=640)
        )
        gain = 640 / 1001
        pad = (640 - 321 * gain) / 2
        self.assertEqual(meta["pad"], [0, pad])
        original = np.array([[100, 50, 300, 200]], np.float32)
        network = original * gain + np.array([0, pad, 0, pad], np.float32)
        generic = restore_boxes(network, meta)
        historical = restored_boxes(
            torch.from_numpy(network), meta, "yolov7_face"
        ).numpy()
        historical[:, 2:] -= historical[:, :2]
        expected = np.array([[100, 50, 200, 150]], np.float32)
        np.testing.assert_allclose(generic, expected, rtol=0, atol=2e-5)
        np.testing.assert_array_equal(generic, historical)

    def test_mask_padding_is_removed_before_original_resize(self):
        # Padding deliberately strongly prefers a different class.
        logits = np.zeros((1, 3, 8, 8), np.float32)
        logits[:, 1, :4] = 10
        logits[:, 2, 4:] = 1000
        meta = dict(
            original_size=[16, 8], input_size=[8, 8], valid_size=[8, 4], pad=[0, 0]
        )
        result = segmentation(logits, meta, "mmseg_letterbox")
        self.assertEqual(result.shape, (8, 16))
        np.testing.assert_array_equal(result, np.ones((8, 16), np.uint8))
        for bad in ([0, 4], [9, 4], [8, True]):
            with self.assertRaisesRegex(ValueError, "Invalid fixed MMSeg"):
                segmentation(logits, dict(meta, valid_size=bad), "mmseg_letterbox")

    def test_bad_size_and_layout_rejected(self):
        image = Image.new("RGB", (100, 80))
        for kind in ("mmseg_letterbox", "yolov5_letterbox", "yolov7_letterbox"):
            for size in (True, 0, -1, 512.5):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    preprocess(image, dict(kind=kind, size=size))
            with self.assertRaisesRegex(ValueError, "NCHW"):
                preprocess(image, dict(kind=kind, size=512), "NHWC")
        with self.assertRaisesRegex(ValueError, "multiple of 32"):
            preprocess(image, dict(kind="yolov5_letterbox", size=641))

    def test_face_parameter_repeat_accepts_only_proven_static_spatial_shape(self):
        import onnx
        from onnx import helper as h, numpy_helper as n
        from opennpu_quant.export.broadcast import replace_parameter_tiles
        from opennpu_quant.export.finalize import outputs

        def graph(repeats):
            return h.make_model(
                h.make_graph(
                    [
                        h.make_node("Expand", ["weight", "ones"], ["expanded"]),
                        h.make_node("Tile", ["expanded", "repeats"], ["tiled"]),
                        h.make_node("Mul", ["images", "tiled"], ["output"]),
                    ],
                    "parameter",
                    [h.make_tensor_value_info("images", 1, [1, 3, 8, 12])],
                    [h.make_tensor_value_info("output", 1, [1, 3, 8, 12])],
                    [
                        n.from_array(
                            np.arange(3, dtype=np.float32).reshape(1, 3, 1, 1), "weight"
                        ),
                        n.from_array(np.ones(4, np.int64), "ones"),
                        n.from_array(np.array(repeats, np.int64), "repeats"),
                    ],
                ),
                opset_imports=[h.make_opsetid("", 17)],
                ir_version=10,
            )

        source = graph([1, 1, 8, 12])
        revised, proof = replace_parameter_tiles(source, expected={"Mul": 1})
        self.assertEqual({"Mul": 1}, proof["counts"])
        self.assertFalse(
            any(n.op_type in ("Tile", "Expand") for n in revised.graph.node)
        )
        feed = [
            {"images": np.linspace(-2, 2, 288, dtype=np.float32).reshape(1, 3, 8, 12)}
        ]
        np.testing.assert_array_equal(
            outputs(source, feed)[0]["output"], outputs(revised, feed)[0]["output"]
        )
        # A repeat of one would broadcast legally, but does not prove the required rewrite.
        with self.assertRaisesRegex(ValueError, "Tile repeats"):
            replace_parameter_tiles(graph([1, 1, 1, 12]))
        onnx.checker.check_model(revised)

    def test_face_parameter_repeat_keeps_dynamic_feature_contract(self):
        from onnx import helper as h, numpy_helper as n
        from opennpu_quant.export.broadcast import replace_parameter_tiles
        from opennpu_quant.export.finalize import outputs

        shape = [1, 3, "height", "width"]
        source = h.make_model(
            h.make_graph(
                [
                    h.make_node("Shape", ["images"], ["shape"]),
                    h.make_node("Gather", ["shape", "axes"], ["spatial"], axis=0),
                    h.make_node("Concat", ["prefix", "spatial"], ["repeats"], axis=0),
                    h.make_node("Expand", ["weight", "ones"], ["expanded"]),
                    h.make_node("Tile", ["expanded", "repeats"], ["tiled"]),
                    h.make_node("Add", ["images", "tiled"], ["output"]),
                ],
                "dynamic-parameter",
                [h.make_tensor_value_info("images", 1, shape)],
                [h.make_tensor_value_info("output", 1, shape)],
                [
                    n.from_array(np.array([2, 3], np.int64), "axes"),
                    n.from_array(np.ones(2, np.int64), "prefix"),
                    n.from_array(np.ones(4, np.int64), "ones"),
                    n.from_array(
                        np.arange(3, dtype=np.float32).reshape(1, 3, 1, 1), "weight"
                    ),
                ],
            ),
            opset_imports=[h.make_opsetid("", 17)],
            ir_version=10,
        )
        revised, _ = replace_parameter_tiles(source, expected={"Add": 1})
        for height, width in ((8, 12), (7, 9)):
            feed = [
                {
                    "images": np.linspace(
                        -3, 3, 3 * height * width, dtype=np.float32
                    ).reshape(1, 3, height, width)
                }
            ]
            np.testing.assert_array_equal(
                outputs(source, feed)[0]["output"], outputs(revised, feed)[0]["output"]
            )

    def test_export_worker_retains_fixed_shapes_for_all_fixture_aspects(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from opennpu_quant.export import worker
        from tools.make_fixed_spatial_bundle import POLICIES

        class Captured(Exception):
            pass

        for name, (_, size) in POLICIES.items():
            with self.subTest(model=name), tempfile.TemporaryDirectory() as temp:

                def capture(recipe, weight, source, feeds, raw, *, fixture_images):
                    self.assertEqual(len(feeds), 3)
                    self.assertEqual(
                        [im.size for im in fixture_images],
                        [(size, size), (size, size // 2), (size // 2, size)],
                    )
                    for feed, image in zip(feeds, fixture_images, strict=True):
                        expected, _ = preprocess(image, recipe["preprocess"])
                        self.assertEqual(feed["images"].shape, (1, 3, size, size))
                        np.testing.assert_array_equal(feed["images"], expected)
                    raise Captured()

                with (
                    patch(
                        "sys.argv",
                        ["export", name, "--destination", temp, "--weights-dir", temp],
                    ),
                    patch.object(
                        worker,
                        "checkpoint",
                        autospec=True,
                        return_value=Path(temp) / "weights",
                    ),
                    patch.object(
                        worker, "checkout", autospec=True, return_value=Path(temp)
                    ),
                    patch(
                        "opennpu_quant.export.torch_models.export_model",
                        autospec=True,
                        side_effect=capture,
                    ),
                    self.assertRaises(Captured),
                ):
                    worker.main()

    def test_packaged_recipes_match_fixed_preprocessing(self):
        from tools.make_fixed_spatial_bundle import POLICIES

        for name, (kind, size) in POLICIES.items():
            recipe = ModelSpec.load(name).recipe
            self.assertEqual(recipe["preprocess"]["kind"], kind)
            self.assertEqual(recipe["input"]["shape"], [1, 3, size, size])
            self.assertFalse(recipe["export"]["dynamic_spatial"])
            for image_size in ((640, 480), (321, 1001), (1001, 321)):
                value, _ = preprocess(
                    Image.new("RGB", image_size), recipe["preprocess"]
                )
                self.assertEqual(value.shape, (1, 3, size, size))


if __name__ == "__main__":
    unittest.main()
