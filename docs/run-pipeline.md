# Run experiments

`opennpu-quant run` creates and evaluates INT8 models from an experiment config, or evaluates existing ONNX files from an evaluation config. This guide continues from the [quick start](../README.md#quick-start).

## Choose an experiment

Select a packaged preset or your own YAML/JSON experiment file with `-c`:

| Preset | Models | Conditions | Data |
|---|---|---|---|
| `core10` (default) | 10 core models | `fp32`, `pot_*`, `float_*`, `sym_pot_*` | Full validation sets |
| `cnn30` | 30 models, including core10 | `fp32`, `pot_*`, `float_*`, `sym_pot_*` | Full validation sets |

The full-evaluation presets use `evaluation.limit: null`. An explicit numeric limit also covers the full dataset when it includes every evaluation sample. The [model reference](model-reference.md#model-groups) lists which models belong to `core10` and `cnn30`.

Each INT8 condition combines a scheme with RTN (`_rtn`) or AdaRound (`_adaround`):

| Scheme | Activations | Weights | Scales |
|---|---|---|---|
| `pot` | Asymmetric, per-tensor | Per-channel | `pot_ceil` |
| `float` | Asymmetric, per-tensor | Per-channel | `float` |
| `sym_pot` | Symmetric, per-tensor | Per-tensor | `pot_ceil` |

`float` means INT8 with unconstrained scales, not FP32.

To change settings, copy a preset as shown in [Example: change a percentile](configuration.md#example-change-a-percentile). Presets specify a percentile for each model and condition. [Original per-model settings](model-reference.md#original-per-model-settings) lists these values and the remaining differences from recorded settings.

## Run fixed models

Use an evaluation config from the model bundle:

```bash
opennpu-quant run -c bundle/configs/core10.yaml \
  --models resnet18,yolov5s --conditions fp32,pot_rtn --limit 3 \
  --data-dir workspace/data --output workspace/evaluations
```

This path uses existing ONNX models and does not export, calibrate or train. See [Evaluate release models](reproducing-results.md) for the full dataset commands. Generation-only options such as `--calibration-samples` and `--save-models` are rejected.

## Select models, conditions and samples

- `--models` selects packaged or locally defined models. When adding a model to an experiment, provide its per-model percentile settings as described in [Add a custom model](models-and-data.md#add-a-custom-model). `--conditions` selects defined conditions. Generation configs prepend FP32 only when it was omitted, otherwise preserving your order. Evaluation configs use exactly the requested conditions and put FP32 first only when you selected it.
- `--limit` sets the number of evaluation samples. For LFW it counts pairs, and the full manifest and at least 6,000 pairs are required. A smaller request stops before any model calibrates or infers.
- `--calibration-samples` sets the number of calibration inputs separately.
- `--dry-run` performs no inference. Generation configs validate settings, including the LFW limit, and print the effective settings, resolved percentiles and paths without checking model or dataset availability. Evaluation configs check preparation, print the selected condition IDs and exit with code 1 if any condition failed or remains pending.

[Command-line overrides](configuration.md#command-line-overrides) lists the other options, such as AdaRound steps and cache sizes.

## Paths and devices

For generation configs, use the same model and data directories for preparation and execution. `--home` defaults to `./workspace`, and individual path options override its subdirectories ([Workspace paths](configuration.md#workspace-paths)). Here `--output` is the parent of `<output>/<experiment name>`. Evaluation configs use the same `<output>/<name>` layout.

The main presets run `network.onnx` on `cuda:0`. An experiment file sets the device with `runtime.device`, and `--device` overrides it. CPU host graphs and postprocessing always run on the CPU, also during GPU execution.

## Evaluate an existing ONNX model

If `<models-dir>/resnet18/network.onnx` and the ImageNet manifest already exist, skip the export. Models with CPU host graphs also need those files and their recipe mappings. Replace the paths with your bundle and data locations.

```bash
opennpu-quant run -c core10 --models resnet18 --conditions fp32 \
  --limit 2 --device cuda:0 \
  --models-dir workspace/models --data-dir workspace/data \
  --output workspace/fp32-check
```

FP32-only evaluation needs no calibration data or quantization. It runs inference, postprocessing and evaluation, and verifies completed results before reusing them.

| Goal | Path |
|---|---|
| Evaluate an existing FP32 model | The `run --conditions fp32` command above |
| Create a new INT8 model from an existing FP32 model | Add INT8 conditions to `run`. Valid statistics caches are reused |
| Evaluate supplied FP32 or INT8 models | [`run -c` with an evaluation config](#run-fixed-models) |
| Rebuild tables from completed evaluations | [`report`](reporting.md) |

An ONNX file by itself does not tell the tool which dataset, preprocessing or decoder to use. To evaluate your own model, give it a recipe as described in [Add a custom model](models-and-data.md#add-a-custom-model).

## Monitor progress and read results

A run first verifies its settings, model bundle and dataset. FP32 conditions go straight to evaluation. INT8 conditions reuse or collect statistics, build RTN QDQ, optionally train AdaRound, audit the graph and evaluate it.

`.state/status.json` tracks each model and condition as `pending`, `running`, `completed`, `reused`, `failed` or `interrupted`, together with any error. `interrupted` means the run was stopped, for example with Ctrl-C. In evaluation configs, a missing file during preparation leaves the condition `pending`. Other errors mark the condition `failed`. A directory lock keeps two runs from writing to the same run directory. Reports are refreshed after each condition in evaluation configs, after each model in generation configs, at the end and after a failure.

When the run finishes, the console shows each condition's metric, accuracy, recovery, sample count and `partial` or `full` scope. `run --json` prints `output`, `device`, `rows` and `mean_recovery` as JSON and sends progress messages to stderr. Evaluation configs also include `execution` with the state and completed, pending and failed counts. The console format does not change which report files are written. Use `--format` and `--table-profile` to select files, with the same options as [report commands](reporting.md#output-files). TSV and Markdown are optional.

With an evaluation config, exit code 0 means all requested conditions completed or were reused, including evaluations with a sample limit. Failed or pending conditions give exit code 1 after results are saved and the summary is printed. An evaluation config also stops when its run state or reports cannot be saved, with the same exit codes as a generation config. Generation configs stop at the first error. Handled execution errors exit with code 2 without a final console or JSON summary. Invalid arguments and other handled command errors also exit with code 2.

To write the Excel-compatible `raw_data.*` tables or rebuild reports later, use `report` as described in [Reports and Excel export](reporting.md). [How to read a result](validation.md#how-to-read-a-result) explains what partial and full results can show.

Use `--raw-samples N` or `--raw-sample-ids IDS` to save selected network inputs, outputs and predictions under each condition's `.state/` directory. `--raw-max-mib` limits their uncompressed tensor bytes. Raw capture is off by default.

## Reuse and caching

Reuse checks the code, graphs, inputs and order, expected and listed evaluation counts, runtime settings, evaluation profiles, and quantization and training settings. Generation configs preserve earlier evaluations when evaluation options change. Evaluation configs require a new output directory or `--force` for changed settings. Forced evaluations preserve earlier condition records in `.state/`. Identical settings can reuse verified results without inference. With `--save-models`, a new evaluation can reuse the saved QDQ model without calibration or training when generation inputs and settings still match. Changes that affect model generation or dataset declarations require a new `--output` or `--force`.

| Layer | Location | Lookup and verification | `--force` | `--recalibrate` | Disable with |
|---|---|---|---|---|---|
| Configuration guard | `<run>/.state/settings.json` | Generation settings | Bypassed and overwritten | No effect | — |
| Completed results | `<run>/.state/<model>/<cond>/complete.json` | Condition identity and file hashes | Ignored and recomputed | INT8 recomputed with a new generation | — |
| Input disk cache | `<cache>/inputs/<model>/<identity>/` | Calibration data identity, preprocessing settings and source files, recipe input specification, input mapping, network input definitions, and NumPy, Pillow and OpenCV versions. With a CPU `preprocess.onnx`, also its graph, ORT execution environment and the source hashes of the ORT runtime modules | Kept | Kept | `--no-input-cache` |
| Statistics cache | `<cache>/calibration/<model>/<key>/` | Model, input digest, bins, ORT environment, and source hashes of the statistics code and ORT runtime modules | Kept | Collected again under `requests/<generation>/` | — |
| AdaRound checkpoints | `<run>/.state/.generated/<model>/<generation identity>/adaround/` (with a request subdirectory after `--recalibrate`) | Generation identity | Kept and resumed | New generation | — |
| AdaRound activation cache | RAM and GPU memory | — | — | — | `--no-cache` |

- **`--force`** skips completed-result and generated-model reuse and the configuration guard. Statistics caches and AdaRound checkpoints are kept. Results from other settings are excluded from the current reports.
- **`--recalibrate`** writes a per-model `recalibration.json` whose generation is the current execution ID. Later runs keep using that generation while the file exists, even without the flag.
- **Code changes** also change the identity, because it covers every packaged `.py` file. After updating the package, even for a comment, earlier results and AdaRound checkpoints are no longer reused. Use a new `--output` or rerun with `--force`, and never edit stored hashes to force reuse.
- **Caches** use identities separate from completed results. An input or statistics cache is reused when its inputs, settings, environment and source hashes listed in the table match. The ORT runtime modules are session creation, tensor fetch, native error translation, CPU and GPU environment probes and CUDA library loading under `ort/`. A code change outside those source files does not invalidate it on its own, while any edit to one of them, even a comment, does.
- **CPU environment changes** can affect preprocessing and metrics even during GPU runs, so CPU numerical settings are checked before result reuse.
- **GPU environment and memory settings:** GPU specifications, libraries and computation settings are checked for reuse. Device indices and the ORT `gpu_mem_limit` are excluded. AdaRound cache and learning-window settings remain part of reuse checks.
- **Concurrent runs** can read completed input and statistics caches together. Cache creation and publication use an exclusive lock, so a competing cache access waits and then validates the data. A wait can be interrupted, and process exit releases the lock. Concurrent writers to the same run directory or AdaRound checkpoint still fail immediately.
- There is no cache-cleaning command. Delete a cache directory when no run is using it.

With `--save-models`, generated QDQ models are saved before evaluation and can be reused after an evaluation failure. Public model files are in `models/`. Generation proofs, checkpoints and earlier evaluations are kept inside `.state/`. These directories are not automatically cleaned. Removing them loses the corresponding reuse or resume data.

## Output files

```text
<output>/<name>/
├── models/       Generated ONNX and required companion files
├── results.csv   Accuracy, auxiliary metrics and recovery
├── settings.json Effective settings, model connections and diagnostics
└── .state/       Internal completion and reuse records
```

Evaluation-only runs use the supplied models in place and do not create `models/`. A `.lock` file protects concurrent writers. The presets save generated models before evaluation, so an evaluation failure does not discard a completed model.

Raw tensors and predictions are saved only when requested. An interrupted evaluation restarts that condition, while verified completed conditions are reused. Keep `.state/` to retain reuse evidence. `report` reads saved records without inference and writes only the requested formats.

## Troubleshooting

| Problem | What to do |
|---|---|
| GPU execution does not start | Run `nvidia-smi` and `opennpu-quant info`. An NVML error can indicate a driver/library mismatch or another driver problem. Check that error separately from the Python CUDA packages. The tool does not modify drivers or replace a failed GPU run with a CPU run |
| "Run configuration changed" | Model-generation settings differ from the saved configuration. Use a new `--output` or `--force`. In generation configs, evaluation-only changes preserve previous results separately |
| Missing dataset manifest | Prepare the dataset with the same `--home` or `--data-dir` as the run |
| Unknown model or condition | Check `opennpu-quant models` and the conditions defined in the selected experiment |
