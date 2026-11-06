"""Independent check against the pinned MMSeg inference pipeline."""

from copy import deepcopy

import numpy as np


def check(model, recipe, images, feeds, references):
    """Compare with upstream transforms configured for the declared input policy."""
    if recipe["preprocess"]["kind"] != "mmseg_letterbox":
        return _check(model, recipe, images, feeds, references)
    original_cfg = model.cfg
    original_padding = model.data_preprocessor.test_cfg
    size = recipe["preprocess"]["size"]
    model.cfg = deepcopy(original_cfg)
    try:
        resize = [t for t in model.cfg.test_pipeline if t["type"] == "Resize"]
        if len(resize) != 1 or not resize[0].get("keep_ratio"):
            raise ValueError("Unsupported MMSeg resize pipeline")
        resize[0]["scale"] = (size, size)
        model.data_preprocessor.test_cfg = dict(size=(size, size))
        return _check(model, recipe, images, feeds, references)
    finally:
        model.cfg = original_cfg
        model.data_preprocessor.test_cfg = original_padding


def _check(model, recipe, images, feeds, references):
    import torch
    from mmseg.apis import inference_model
    from mmseg.apis.utils import _preprare_data
    from ..models.extended_decode import segmentation
    from ..models.preprocess import preprocess

    if model.decode_head.align_corners or model.test_cfg["mode"] != "whole":
        raise ValueError("Unsupported MMSeg restore contract")
    checks = []
    for image, feed, output in zip(images, feeds, references):
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[..., ::-1])
        data, _ = _preprare_data(bgr, model)
        with torch.inference_mode():
            prepared = model.data_preprocessor(data, training=False)
            official = inference_model(model, bgr)
        meta = prepared["data_samples"][0].metainfo
        if recipe["preprocess"]["kind"] != "mmseg_letterbox" and any(
            meta.get("padding_size", meta.get("img_padding_size", [0] * 4))
        ):
            raise ValueError("MMSeg protocol unexpectedly pads input")
        actual = prepared["inputs"].cpu().numpy()
        np.testing.assert_allclose(actual, feed["images"], atol=1e-5, rtol=1e-4)
        _, own_meta = preprocess(image, recipe["preprocess"])
        mask = segmentation(output["logits"], own_meta, recipe["preprocess"]["kind"])
        expected = official.pred_sem_seg.data[0].cpu().numpy()
        if not np.array_equal(mask, expected):
            raise ValueError("MMSeg original restored masks differ")
        checks.append(
            dict(
                input_shape=list(actual.shape),
                input_max_abs=float(np.abs(actual - feed["images"]).max()),
                mask_equal=True,
            )
        )
    return dict(
        passed=True,
        independent_preprocess=True,
        independent_mask_restore=True,
        fixtures=checks,
        upstream_commit=recipe["source"]["commit"],
        input_policy=recipe["preprocess"],
    )
