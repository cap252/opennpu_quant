# Evaluation profiles

Models differ in how they break classification ties, run NMS, restore face coordinates and rebuild masks. An **evaluation profile** selects one version of this postprocessing and scoring behavior. The evaluator resolves it once when the model is loaded.

Generation runs use model recipes. Evaluation configs use catalog recipes. Both public defaults use stable classification tie ordering. Identical profiles do not guarantee identical results across different models, inputs or devices.

## Default profile per model

| Models | Profile | Behavior it preserves |
|---|---|---|
| 12 classifiers | `stable_topk_v1` | Exact ties are ordered by class index |
| YOLOX | `yolox_torch_nms_v1` | Original NMS and score handling, including out-of-image coordinates and more than 300 detections |
| YOLOv7-face | `yolov7_face_clip_first4_v1` | Original coordinate restoration, integer conversion, score serialization and clipping of only the first four boxes |
| YOLOv5s-seg | `yolov5_seg_scaled_coeff_bbox_area_v1` | Objectness-scaled mask coefficients, mask reconstruction order and bbox-area evaluation |
| Other models | Decoder-specific names, such as `retinaface_v1`, `nanodet_v1` or `segmentation_resize_v1` | Each model's decoder and evaluator as implemented |

Set `evaluation.profiles.<model>` in an experiment file using the names above. Unsupported model/profile combinations are rejected.

For classification, `numpy_topk_v1` retains the original NumPy ordering for comparison. The public `TopKClassification` API sorts stably by default.

## What a profile changes

A profile changes only the postprocessing and scoring behavior. Preprocessing, ONNX files, host graphs and input mappings stay the same. Graph partitions and tensor names belong to the exported model bundle, and evaluation catalogs declare the graph connections.

The default evaluation profiles for the YOLO models listed above use Torch/Torchvision for CPU postprocessing. `setup.sh` installs the required packages.

To replace a model's Python decoder entirely, use the Python API hook described in [Replace Python postprocessing](api.md#replace-python-postprocessing).

## Precedence and local recipes

For a packaged model, a generation run picks the profile in this order:

1. `evaluation.profiles.<model>` in the experiment file.
2. The default profile of the packaged recipe.

A local `<models-dir>/<model>/recipe.yaml` still controls the bundle settings, such as inputs, preprocessing, graphs and output names. The profile default comes from the package, and the tool warns when a local recipe names a different profile. It does not edit the file. Custom models that are not in the package use the profile in their own recipe.

Local recipes and experiment files use the profile names listed above. Unknown names are rejected.

<a id="changing-preserved-behavior"></a>

## Compare against another profile

Changing a profile can change predictions and evaluation scores even for the same model outputs. Use the same profile for FP32 and INT8 accuracy comparisons. Saved results record the profile they were evaluated with.

Use a new output directory and set the profile per model in the experiment file. This excerpt changes only face clipping and mask area:

```yaml
evaluation:
  limit: 32
  profiles:
    yolov7_face: yolov7_face_clip_all_v1
    yolov5s_seg: yolov5_seg_scaled_coeff_mask_area_v1
```

YOLOv7-face clips only the first four face boxes with its default profile. `yolov7_face_clip_all_v1` applies clipping to every face box. `yolov5_seg_scaled_coeff_mask_area_v1` keeps the objectness-scaled mask coefficients in YOLOv5s-seg and changes only the COCO area policy. These alternatives apply to their respective models. Set a supported profile for each model. Before any model runs, the tool validates every model/profile combination and rejects names that are not packaged models, local recipes or, for fixed-model runs, catalog models. An entry for a known model outside the current `--models` selection is accepted and unused. Fixed-model runs also accept an explicit profile override.

Profiles are part of the stored results and their identities. A profile change does not affect calibration inputs or statistics, so statistics caches remain valid. See [Reuse and caching](run-pipeline.md#reuse-and-caching) for what triggers recomputation.
