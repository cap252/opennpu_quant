# Evaluate release models

Release evaluation uses existing ONNX models and performs no export, calibration or AdaRound training. Follow the [download and extraction commands](../README.md#2-check-supplied-onnx-models-on-a-few-images), prepare the [datasets](models-and-data.md#prepare-datasets), and pass a supplied evaluation config to `run`.

<a id="bind-the-assets"></a>

## Set model and data paths

```bash
opennpu-quant run -c bundle/configs/core10.yaml \
  --models resnet18,yolov5s --conditions fp32,pot_rtn \
  --limit 3 --device cuda:0 --data-dir workspace/data \
  --output workspace/evaluations
```

The config refers to `catalog.json`. Model paths are relative to that catalog, so the bundle can be moved as a unit. `data prepare` writes local manifests under `<data-dir>/<dataset>/manifest.json`. Use that same `--data-dir` for `run`. Original data remains outside the bundle, and no bindings file is required. See [workspace paths](configuration.md#workspace-paths) for path overrides.

### External ONNX weights

Keep each external weight file at the relative path recorded by its ONNX graph. The catalog records the graph and weight hashes. Missing, changed or escaping paths are rejected. External data in sparse ONNX tensors is not supported.

<a id="generate-the-lfw-input-cache"></a>

## Prepare LFW

For ordinary evaluation, prepare the complete LFW dataset with `data prepare lfw_mobilefacenet` as described in the [dataset guide](models-and-data.md#prepare-datasets). This prepares aligned images and the manifest used by `run`. LFW images and input tensors are not included in the release. Matching earlier CNN30 inputs also requires the [documented LFW alignment CPU setting](models-and-data.md#reproduce-lfw-alignment). Verify the aligned crop hashes before comparing accuracy. The same original JPEGs and pairs alone do not establish identical model inputs.

## Choose a model set

| Evaluation config | Scope |
|---|---|
| `bundle/configs/core10.yaml` | The core 10 models, seven default conditions each |
| `bundle/configs/cnn30.yaml` | All 30 models, seven default conditions each |
| `bundle/configs/core10-candidates.yaml` | All 131 supplied core10 candidates |

The release contains 271 candidates. The extra core10 RTN percentiles and YOLOv5s float AdaRound candidate are preserved without being evaluated automatically by the default configs. Classification uses stable top-k ranking. Detection, segmentation and LFW retain their recorded input and scoring conventions.

## Scale up from one model

```bash
opennpu-quant run -c bundle/configs/cnn30.yaml --limit full \
  --device cuda:0 --data-dir workspace/data --output workspace/evaluations
opennpu-quant report workspace/evaluations/cnn30
```

A complete default CNN30 evaluation contains 210 conditions. LFW requires all 6,000 pairs, so select other models for a small check. Read completion, sample scope and accuracy separately. The default output is `results.csv` with `settings.json` and internal `.state/` records.

## Resume an interrupted run

Repeat the same command. Completed conditions are reused after model, input, settings and result checks. An interrupted condition starts its evaluation again. Restoring a missing model permits the pending condition to run. Changing evaluation settings or scope requires a new output directory or `--force`.

Evaluation and its `--dry-run` preparation checks exit with code 1 if any condition failed or remains pending. Results from completed conditions are preserved. Prepare the missing data or restore the missing model, then repeat the command.

`--save-predictions`, `--raw-samples` and `--raw-sample-ids` retain optional diagnostics. Raw capture is bounded by `--raw-max-mib`. A normal run does not retain sample files or repeat every inference.

<a id="check-one-model"></a>
<a id="preparation-states"></a>
<a id="check-agreement-with-the-original-measurements"></a>
<a id="compare-two-servers"></a>
<a id="gpu-environment-and-numerical-verdicts"></a>
<a id="files-written-by-reproduce"></a>

## Diagnostics and execution errors

Environment and historical numerical differences are diagnostics. File corruption, invalid inputs, non-finite outputs, missing evaluation samples, a profiled first input whose outputs differ from the evaluated ones, and ONNX Runtime session or inference failures are execution errors.

With an evaluation config, an execution error affects only its condition. The condition is recorded as `failed`, or as `pending` when preparation is missing a file, together with its error in `.state/status.json`. Later conditions still run, completed results are kept, and the command exits with code 1 after printing the summary. Fix the cause and repeat the command to run the remaining conditions. A generation config stops at the first execution error, records it in `.state/status.json` and exits with code 2.

Unexpected errors, such as a programming error, are recorded the same way. An evaluation config also logs the traceback and continues with the next condition. A generation config stops and shows the traceback with exit code 1. Only a stop request, such as Ctrl-C, records `interrupted`.

An error while saving the run state or its reports, such as a full disk, stops an evaluation config as well. Completed conditions remain reusable. When the run state can still be written, the run is recorded as `failed` with that error in `.state/status.json`. A handled error exits with code 2, and an unexpected error shows its traceback with code 1, as with generation configs.
