# Validation scope and limitations

This page explains how to read a result and the limitations to consider.

## How to read a result

A result answers one of three questions, and each question needs its own evidence.

1. **Did it run?** A `completed` state, a written report or a passing `demo` shows that the pipeline executes.
2. **Does it match a reference?** Reference agreement requires a separate comparison of raw tensors, predictions and metrics. Agreement at one level does not imply agreement at another.
3. **What is the benchmark accuracy?** A run on a few samples (`--limit`, `demo`) measures accuracy only on that subset and does not represent the benchmark. Benchmark accuracy comes from a full evaluation, for example all 50,000 ImageNet images or all 5,000 COCO images. The default table is `results.csv`. `--table-profile opennpu_raw` separates full and partial evaluations into `raw_data.csv` and `raw_data.partial.csv`. LFW supports full 6,000-pair evaluation only ([LFW](models-and-data.md#lfw)).

CPU runs check functionality, and their numbers can differ from GPU runs. A failed GPU request is not retried as a CPU run. This does not guarantee that every operator executes on GPU.

Hash checks establish the identity and integrity of models, inputs and saved files. They do not establish numerical agreement. Retain raw outputs with the optional capture settings when a separate numerical comparison is needed.

## Known limitations

- [Export checks](models-and-data.md#export-fp32-onnx) record finite framework differences in both validation modes. Export success does not establish numerical agreement or accuracy.
- AdaRound compares Torch and ONNX Runtime layer outputs. `strict` stops on tolerance violations, while `functional` records finite differences.
- Matching CUDA package versions does not guarantee bitwise-equal results.
- LFW face alignment depends on the OpenBLAS kernel used during preparation, so the same source images can produce slightly different crops and LFW accuracy on another CPU ([Reproduce LFW alignment](models-and-data.md#reproduce-lfw-alignment)). LFW scoring also normalizes embeddings with BLAS, so pair distances can differ in their last bits between CPUs.
- FP32 accuracy follows each model's packaged evaluation and can differ from published numbers. Known differences in the evaluation conditions are listed under [Notes on specific models](model-reference.md#notes-on-specific-models).
- Some original measurements lack raw tensors or a compatible prediction stream. Matching metrics alone do not establish raw-output or prediction agreement.

## Repository tests

The [development checks](../tests/README.md) use small models and temporary datasets. They cover the CLI, QDQ construction, caching, resume, evaluation and the 15-column reports without any external assets. Measuring real-model accuracy requires the datasets and models described in the other guides.
