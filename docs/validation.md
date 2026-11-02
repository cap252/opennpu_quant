# Validation scope and limitations

This page explains how to read a result and records what has been verified so far. Other pages link here instead of repeating these points.

## How to read a result

A result answers one of three questions, and each question needs its own evidence.

1. **Did it run?** A `completed` state, a written report or a passing `demo` shows that the pipeline executes.
2. **Does it match a reference?** Reference agreement requires a separate comparison of raw tensors, predictions and metrics. Agreement at one level does not imply agreement at another.
3. **What is the benchmark accuracy?** A run on a few samples (`--limit`, `demo`) measures accuracy only on that subset and does not represent the benchmark. Benchmark accuracy comes from a full evaluation, for example all 50,000 ImageNet images or all 5,000 COCO images. The default table is `results.csv`. `--table-profile opennpu_raw` separates full and partial evaluations into `raw_data.csv` and `raw_data.partial.csv`. LFW supports full 6,000-pair evaluation only ([LFW](models-and-data.md#lfw)).

CPU runs check functionality, and their numbers can differ from GPU runs. A failed GPU request is not retried as a CPU run. This does not guarantee that every operator executes on GPU.

Hash checks establish the identity and integrity of models, inputs and saved files. They do not establish numerical agreement. Retain raw outputs with the optional capture settings when a separate numerical comparison is needed.

## What has been verified

The primary targets are core10 and cnn30. Historical validation covered cnn30's 210 selected conditions. Selecting cnn30 does not constitute a new execution of those conditions. Results on a new server must be judged from that server's own outputs and environment records.

| Historical check | Verified in those checks | Not established by those checks |
|---|---|---|
| Calibration | Statistics recollected from the full calibration inputs of every packaged model, and activation encodings of all 180 INT8 conditions, match the original measurements bit for bit | Collection on other GPUs and environments |
| Execution of the original models | Small-sample GPU execution, repeatability and QDQ audits for all 210 conditions | Agreement across GPU models and on new devices |
| Evaluation defaults | Saved-output comparisons verified bitwise-equal predictions and metric values for all 203 non-LFW conditions. Archived MobileFaceNet subset diagnostics are not part of this accuracy evidence | Fresh GPU inference and full-accuracy checks |
| Full accuracy | Full WIDER FACE evaluations of the seven YOLOv7-face conditions. Full FP32 evaluations of YOLOv3-Tiny, YOLOv4-Tiny, YOLOv7-Tiny, RetinaNet, RetinaFace, DeepLabV3-MobileNetV2 and MobileFaceNet reproduce the original metrics bit for bit, including the auxiliary `fixed_angle_accuracy` metric and the `pixel_accuracy` metric detail | Full evaluation of the other models and INT8 conditions, and the comparison of the MobileFaceNet prediction stream with the original serialization |
| Creating INT8 models | QDQ, PoT and AdaRound contracts and small functional regressions | Full regeneration of every model and long AdaRound training |
| Target hardware | Execution through ONNX Runtime | NPU kernels and native INT8 accumulation |

The full-LFW-only workflow was checked by rescoring saved FP32 and PoT RTN outputs. All 15,402 predictions per condition, primary and auxiliary metrics, and recovery matched bit for bit. The 128 calibration input digests also remained unchanged. CPU checks covered ResNet-18 and YOLOv5s with FP32 and PoT RTN, both fixed-model replay and small new RTN runs. Full LFW resume was checked with a small test graph using the complete pair protocol. These checks did not repeat full GPU inference or long AdaRound training.

These historical checks retain their original scope and source versions. They do not claim fresh inference on each subsequent source revision. Detailed logs and large outputs are not part of this repository.

## Known limitations

- QDQ parameter, PoT and runtime audits check the ONNX graph that ONNX Runtime executes. They do not validate an NPU's INT8 accumulation or kernel accuracy.
- Several models take variable input sizes ([Notes on specific models](model-reference.md#notes-on-specific-models)). Deploying them on fixed-input hardware needs a new fixed-shape export and a new FP32/INT8 evaluation, with a review of the CPU host graphs and the integer `Shape` outputs at the device boundary.
- [Export checks](models-and-data.md#export-fp32-onnx) record finite framework differences in both validation modes. Export success does not establish numerical agreement or accuracy.
- AdaRound compares Torch and ONNX Runtime layer outputs. `strict` stops on tolerance violations, while `functional` records finite differences.
- Matching CUDA package versions does not guarantee bitwise-equal results.
- LFW face alignment depends on the OpenBLAS kernel used during preparation, so the same source images can produce slightly different crops and LFW accuracy on another CPU ([Reproduce LFW alignment](models-and-data.md#reproduce-lfw-alignment)). LFW scoring also normalizes embeddings with BLAS, so pair distances can differ in their last bits between CPUs.
- FP32 accuracy follows each model's packaged evaluation and can differ from published numbers. Known differences in the evaluation conditions are listed under [Notes on specific models](model-reference.md#notes-on-specific-models).
- Some original measurements lack raw tensors or a compatible prediction stream. Matching metrics alone do not establish raw-output or prediction agreement.

## Repository tests

The [development checks](../tests/README.md) use small models and temporary datasets. They cover the CLI, QDQ construction, caching, resume, evaluation and the 15-column reports without any external assets. Measuring real-model accuracy requires the datasets and models described in the other guides.
