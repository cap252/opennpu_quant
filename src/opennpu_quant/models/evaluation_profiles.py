"""Versioned evaluation contracts selected by model recipes or explicit overrides."""

import copy
from importlib.metadata import version, PackageNotFoundError

SPECIAL = {
    "yolox": "yolox_torch_nms_v1",
    "yolov7_face": "yolov7_face_clip_first4_v1",
    "yolov5s_seg": "yolov5_seg_scaled_coeff_bbox_area_v1",
}
GENERIC = {
    "logits": "stable_topk_v1",
    "yolo": "yolo_opencv_nms_v1",
    "yolo_no_objectness": "yolo_scores_opencv_nms_v1",
    "yolo_face": "yolo_face_opencv_nms_v1",
    "yolo_seg": "yolo_seg_unscaled_coeff_mask_area_v1",
    "tf_od_api": "tf_od_api_v1",
    "retinaface": "retinaface_v1",
    "retinanet": "retinanet_host_nms_v1",
    "nanodet": "nanodet_v1",
    "yolov4": "yolov4_opencv_nms_v1",
    "segmentation_labels": "segmentation_labels_v1",
    "segmentation_resize": "segmentation_resize_v1",
    "face_verification": "lfw_flip_ten_fold_v1",
}
ALTERNATIVES = {
    "stable_topk_v1": "numpy_topk_v1",
    "yolov7_face_clip_first4_v1": "yolov7_face_clip_all_v1",
    "yolov5_seg_scaled_coeff_bbox_area_v1": "yolov5_seg_scaled_coeff_mask_area_v1",
}
NAMES = frozenset((*SPECIAL.values(), *GENERIC.values(), *ALTERNATIVES.values()))


def default_profile(model, kind):
    return SPECIAL.get(model) or GENERIC[kind]


def validate_profile(model, profile, kind=None):
    if not isinstance(profile, str) or profile not in NAMES:
        raise ValueError("Unknown evaluation profile: " + str(profile))
    if kind is None:
        from .spec import ModelSpec, model_names

        if model in model_names():
            kind = ModelSpec.load(model).recipe["decode"]["kind"]
        else:
            # Check a custom model's decoder when its local recipe is loaded.
            return
    default = default_profile(model, kind)
    if profile not in (default, ALTERNATIVES.get(default)):
        raise ValueError(f"Evaluation profile {profile} is not verified for {model}")


def apply_profile(recipe, profile):
    validate_profile(recipe["name"], profile, recipe["decode"]["kind"])
    result = copy.deepcopy(recipe)
    result["decode"]["profile"] = profile
    return result


def profile_name(recipe):
    model = recipe.get("name", "")
    name = recipe["decode"].get(
        "profile", default_profile(model, recipe["decode"]["kind"])
    )
    validate_profile(model, name, recipe["decode"]["kind"])
    return name


def mask_area(recipe):
    from .evaluation_contracts import resolve

    return resolve(recipe).mask_area


def identity(recipe):
    dependencies = {}
    for name in (
        "numpy",
        "torch",
        "torchvision",
        "opencv-python",
        "opencv-python-headless",
        "pycocotools",
    ):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            pass
    from .evaluation_contracts import resolve

    return dict(
        contract=resolve(recipe).identity(),
        profile=profile_name(recipe),
        mask_area=mask_area(recipe),
        dependencies=dependencies,
    )
