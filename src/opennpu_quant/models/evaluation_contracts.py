"""Resolve model evaluation once, independently of host-graph execution."""

from dataclasses import dataclass, replace
from functools import partial
from typing import Callable

from .evaluation_profiles import SPECIAL, profile_name


@dataclass(frozen=True)
class EvaluationContract:
    """A decoder plus scoring policy. Custom callables are never reference evidence.

    postprocess receives (host_outputs, recipe, metadata, categories). Detection
    and face decoders return their existing prediction rows. Semantic
    segmentation decoders return a label array. Classification/verification use
    their dedicated evaluators and do not accept this hook.
    """

    name: str
    postprocess: Callable | None
    ranking: str = "stable"
    mask_area: str = "mask"
    bbox_metric: bool = False
    custom: bool = False

    def identity(self):
        return dict(
            contract=self.name,
            ranking=self.ranking,
            mask_area=self.mask_area,
            bbox_metric=self.bbox_metric,
            reusable=not self.custom,
            original_reproduction_eligible=not self.custom,
        )


def resolve(recipe, *, postprocess=None):
    """Resolve numerical behavior once when constructing an evaluator."""
    from . import model_postprocess as decoders
    from .historical_yolo import postprocess as torch_decode

    model, kind = recipe.get("name", ""), recipe["decode"]["kind"]
    name = profile_name(recipe)
    if name == "numpy_topk_v1":
        result = EvaluationContract(name, None, ranking="numpy_argsort")
    elif model in SPECIAL:
        result = EvaluationContract(
            name,
            partial(torch_decode, clip_all_boxes=name == "yolov7_face_clip_all_v1"),
            mask_area="bbox" if name == SPECIAL[model] else "mask",
            bbox_metric=kind == "yolo_seg",
        )
    else:
        result = EvaluationContract(name, decoders.for_recipe(recipe))
    if postprocess is not None:
        if not callable(postprocess):
            raise TypeError("postprocess must be a callable")
        if result.postprocess is None:
            raise ValueError(
                "Use a dedicated evaluator for classification or face verification"
            )
        result = replace(
            result, name="custom_postprocess", postprocess=postprocess, custom=True
        )
    return result
