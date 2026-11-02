# Configuration reference

Use an experiment config to create and evaluate models, or a [fixed-model config](#evaluate-fixed-models) to evaluate supplied ONNX files. Both use `run -c`.

<a id="use-a-preset-or-an-experiment-file"></a>

## Start with a preset

The repository includes complete experiment YAML files:

| Preset | YAML file | Models | Conditions per model |
|---|---|---|---|
| `core10` | [core10.yaml](../src/opennpu_quant/resources/experiments/core10.yaml) | 10 | 7: FP32 and RTN/AdaRound for `pot`, `float` and `sym_pot` |
| `cnn30` | [cnn30.yaml](../src/opennpu_quant/resources/experiments/cnn30.yaml) | 30 | 7: the same conditions as core10 |

`run` defaults to `core10`. Choose `-c cnn30` to use the 30-model preset. Both presets use `runtime.device: cuda:0` and `evaluation.limit: null` for full evaluation. You can inspect a preset without starting an experiment:

```bash
opennpu-quant run -c cnn30 --dry-run
```

For settings with CLI options, combine the preset with those options. No new YAML or JSON is needed. The [common settings](#common-settings) and [multiple-model example](#run-multiple-models-and-conditions) show this approach. To save your own settings or change an option that has no CLI flag, copy a preset and edit it. From the repository root:

```bash
cp src/opennpu_quant/resources/experiments/cnn30.yaml experiment.yaml
opennpu-quant run -c experiment.yaml --dry-run
```

The file selected with `-c` must contain the required experiment settings. A file containing only changed keys is not automatically filled from a preset. You can keep editing your own copy. A new file is not required for each change. Use a new `name` or `--output` when changing a completed experiment so its saved results remain separate.

If you installed only the package, [Example: change a percentile](#example-change-a-percentile) shows how to load and edit the full preset through Python. The same approach works for other supported experiment settings.

## Evaluate fixed models

An evaluation config selects existing ONNX files without calibration or training:

```yaml
mode: evaluate
name: core10
source: ../catalog.json
models: [resnet18, yolov5s]
conditions: [fp32, pot_rtn]
runtime:
  device: cuda:0
evaluation:
  limit: 3
```

`source` is relative to the YAML file. Model paths in the catalog are relative to the catalog directory. The catalog contains graph hashes, host connections, input and evaluation conventions, candidate conditions and default choices. Use `conditions: defaults` for the default choices or `conditions: all` for every candidate of the selected models. Data manifests come from `--data-dir`.

Omitting the evaluation limit selects the full dataset. LFW requires all 6,000 pairs. Generation-only options are rejected.

## Common settings

The sections below describe generation experiments. One file can select several models and conditions. The [detailed settings](#detailed-settings) cover less common options.

| What to change | Experiment key | Command-line option |
|---|---|---|
| Models to evaluate | `models` | `--models resnet18,resnet50` |
| Conditions to apply | `conditions` | `--conditions fp32,pot_rtn`, using names defined in the experiment |
| Execution device | `runtime.device` | `--device cuda:0` or `--device cpu` |
| Calibration input count | `calibration.samples.<dataset>` | `--calibration-samples 8` sets the count for every dataset |
| Evaluation sample count | `evaluation.limit` | `--limit 3` for an image-based check. Full evaluation uses `null` in YAML |
| Save generated INT8 ONNX files | `output.save_qdq_models` | `--save-models` |

Calibration inputs determine quantization ranges. Evaluation inputs measure accuracy. Set these counts separately. LFW evaluates face pairs and requires all 6,000 pairs, so a three-image limit is not applicable to that model. A small evaluation checks execution. Use the full evaluation set to measure accuracy.

The keys control concrete stages of execution: `models` selects the outer loop, `conditions` selects the work for each model, and the remaining settings configure calibration, quantization and evaluation. Model-specific input and output rules come from model recipes, so a separate experiment file for every model is unnecessary.

## Run multiple models and conditions

This example selects two models and two conditions, giving four model/condition combinations:

```bash
opennpu-quant run -c cnn30 --models resnet18,resnet50 \
  --conditions fp32,pot_rtn --output workspace/two-model-runs --dry-run
```

For generation configs, `--dry-run` validates and prints the effective configuration, paths and percentile of each selected condition for each model. It checks the LFW limit but does not check model files or datasets. With fixed models, it checks preparation and prints the selected condition IDs without running inference.

Execution visits each selected model in order and then each selected condition. Results are written to `<output>/<experiment name>/results.csv`. Condition evidence is kept in `.state/`. For generation configs, `--conditions` prepends the defined FP32 baseline if omitted and preserves its position if explicitly selected. Evaluation configs do not add FP32 automatically.

A generation error stops `run`. Evaluation configs retain successful conditions and record failed or pending conditions separately. Running the same configuration again can reuse verified completed results. It does not necessarily repeat every measurement. See [Reuse and caching](run-pipeline.md#reuse-and-caching) for rerunning or changing an experiment.

<a id="where-settings-live"></a>

## Where to change settings

Use this table to choose the file or option for the change you want to make.

| What to change | Where to change it |
|---|---|
| Models, quantization conditions, calibration counts and evaluation scope | Experiment YAML/JSON selected with `-c`, or a supported `run` option |
| A model's input size, preprocessing or output interpretation | Model recipe. The packaged files are in `resources/models/*.yaml`, and an exported bundle can have a local `recipe.yaml` |
| Model, data, cache and result directories | Path options or the optional `opennpu_quant.local.yaml` file. See [Workspace paths](#workspace-paths) |
| Existing model candidates and graph connections | The catalog referenced by an evaluation config |

For example, changing a percentile in an experiment file changes the INT8 model created by `run`. An evaluation config selects existing files. Editing a generation preset does not change those files. Reports use the settings saved at execution time, so editing a preset also leaves existing evaluations unchanged.

## Detailed settings

Use the following sections when you need an option beyond the common settings. Tables distinguish preset values from defaults supplied by configuration objects. Unqualified scheme defaults refer to `Scheme`. They may differ from a preset's explicit values. A listed default does not mean every YAML key can be omitted, which is why copying a complete preset is the starting point.

### Command-line overrides

The experiment file is validated when loaded. Supported CLI options replace the corresponding values, and the effective configuration is validated again. Options that select several models or datasets can update more than one key.

A local `recipe.yaml` supplies the model bundle's input and graph settings. Packaged models still use their packaged evaluation profile by default. Choose another profile explicitly with `evaluation.profiles.<model>` or `--evaluation-profile`. See [Precedence and local recipes](evaluation-profiles.md#precedence-and-local-recipes).

| Option | Configuration key |
|---|---|
| `--device` | `runtime.device` |
| `--limit` | `evaluation.limit` |
| `--evaluation-profile` | `evaluation.profiles` for all selected models (comparison only) |
| `--calibration-samples` | All `calibration.samples.*` |
| `--adaround-steps` | `adaround.steps` |
| `--save-models` | `output.save_qdq_models` |
| `--save-predictions` | `output.save_predictions` |
| `--raw-samples`, `--raw-sample-ids` | `output.raw_samples` or `output.raw_sample_ids` (choose one) |
| `--raw-max-mib` | `output.raw_max_bytes`, converted from MiB |
| `--host-cache-gib`, `--gpu-cache-mib`, `--no-cache` | `activation_cache.*` |
| `--no-input-cache` | `calibration.input_cache.enabled` |

`--models`, `--conditions` and `--dry-run` are described in [Select models, conditions and samples](run-pipeline.md#select-models-conditions-and-samples).

Environment variables such as `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` control CPU numerical-library thread counts. The CLI sets unset variables to 1 and preserves existing values. Python API calls leave these variables unchanged. To compare Python and CLI runs with matching thread settings, set the variables before importing NumPy, ONNX Runtime or `opennpu_quant`, because libraries can read them during initialization.

### Workspace paths

A local path file is optional. Paths resolve per key in this order: CLI option, environment variable, local YAML, then the default derived from home. The default local filename is `opennpu_quant.local.yaml`. `--local-config` selects another file. Relative paths resolve from the working directory. Local `datasets.<name>` settings override only the image root recorded in a dataset manifest.

| Directory | CLI | Environment variable | Default |
|---|---|---|---|
| home | `--home` | `OPENNPU_QUANT_HOME` | `./workspace` |
| models | `--models-dir` | `OPENNPU_QUANT_MODELS` | `<home>/models` |
| data | `--data-dir` | `OPENNPU_QUANT_DATA` | `<home>/data` |
| cache | `--cache-dir` | `OPENNPU_QUANT_CACHE` | `<home>/cache` |
| runs | `--output` (run) | `OPENNPU_QUANT_RUNS` | `<home>/runs` |
| weights | `--weights-dir` | `OPENNPU_QUANT_WEIGHTS` | `<home>/weights` |

Precedence applies to each key separately. For example, `models:` in the local YAML wins over a default derived from `--home`.

`--output` means something different for each command. For `run` it is the parent of `<output>/<experiment name>`, and for `results` it is the export directory. Evaluation model paths come from the catalog.

### Preprocessing and calibration

`input`, `network_input_map`, `graphs` and `preprocess.*` belong to the model recipe or bundle. The `calibration.*` keys belong to the experiment file.

| Key | Default and allowed values | Effect of a change |
| --- | --- | --- |
| `input`, `network_input_map`, `graphs` | Model-specific shape, layout and names. Network feeds are float32 | Export boundaries and input shapes are revalidated, and the input, statistics and evaluation identities change |
| `preprocess.kind`, resize, crop, size, mean, std | See the [model reference](model-reference.md). Each kind supports its own keys | Inputs, statistics and evaluations are revalidated. Editing a declared shape does not change the ONNX file |
| `calibration.samples.<dataset>` | ImageNet 1,000 and others 128 in both presets. Positive integer | The input prefix and statistics identity change, including the order |
| `calibration.expected_samples_identity.<model>` | Optional SHA256 | Collection fails early if the input order or bytes differ from the original |
| `calibration.histogram_bins` | Default 2048. Integer of at least 2 and at least `quantized_bins` | New histograms are collected |
| `calibration.device` | Default `run`. Allowed: `run`, `cpu`, `cuda:N` | The statistics runtime identity changes. CPU and GPU statistics can differ |
| `calibration.input_cache.enabled` | `true` in presets and `false` if omitted | Changes caching only, not the numerical policy |
| `calibration.input_cache.max_bytes` | Default 2 GiB. Positive integer number of bytes | Storage budget. Fixed and per-sample variable shapes are both kept |

The manifest written by `data prepare` defines the files, annotations, calibration lists and split order ([Prepare datasets](models-and-data.md#prepare-datasets)).

### Quantization and AdaRound

These keys belong to the experiment file. Some keys abbreviate a repeated prefix with a leading dot. For example, `.weight_granularity` means `schemes.<family>.weight_granularity`.

| Key | Default and allowed values | Effect of a change |
| --- | --- | --- |
| `conditions.<name>.scheme`, `.adaround` | Scheme name and boolean. `fp32: {}` is the FP32 baseline | Creates a separate QDQ, training and evaluation condition |
| `schemes.<family>.scope` | Default `all`. Allowed: `basic`, `all` | Feature boundaries change, so the statistics plan and audit are revalidated |
| `.activation_symmetric` | Default `false`. Boolean | Statistics can be reused, but QDQ is rebuilt and reevaluated |
| `.weight_granularity` | Default `per_channel`. Allowed: `per_channel`, `per_tensor` | QDQ, biases and AdaRound are revalidated |
| `.method`, `.percentile` | Default `percentile` with 99.99. Use `null` for `minmax` and `entropy`. Presets use `per_model` with explicit condition values. `recorded` is a compatibility value that reads the explicit cnn30 preset for that model and condition | New ranges use the same statistics. Changed ranges invalidate model generation and training reuse |
| `percentiles.<model>.<family or condition>` | Required for `per_model`, optional overrides for `recorded`. Range (0, 100]. A condition key such as `pot_rtn` can differ from `pot_adaround`. A family key such as `pot` takes precedence over both | Sets the percentile used by each condition |
| `.activation_scale`, `.weight_scale` | Default `float`. Allowed: `float`, `pot_nearest`, `pot_ceil`, and `pot_mse` for weights | QDQ is rebuilt and reevaluated. The original PoT conditions use `pot_ceil` |
| `.quantized_bins` | Default 128. Integer from 2 to `histogram_bins` | Ranges and QDQ are revalidated |
| `adaround.steps` | Default 10,000. Positive integer | The training and checkpoint identity changes |
| `adaround.learning_rate`, `.warmup_fraction`, `.regularization` | 0.001, 0.2 and 0.01. Ranges: >0, [0, 1) and ≥0 | The training trajectory changes |
| `adaround.beta_start`, `.beta_end`, `.seed` | 20, 2 and 20260928. Start ≥ end > 0, integer seed ≥ 0 | Settings for reproducible training |
| `adaround.batch_size`, `.window_samples`, `.window_steps` | 8, 32 and 100. Positive integers | Sample grouping, order and checkpoint identity change |
| `adaround.window_bytes` | 512 MiB, or 4 GiB for some original settings. Positive integer number of bytes | The window composition can change. This is separate from the activation cache budget |
| `adaround.device` | Default `auto`. Allowed: `auto`, `cpu`, `cuda` | The Torch device and numerical path change |
| `adaround.validation`, `.atol`, `.rtol` | `strict`, 1e-5 and 1e-4. Allowed: `strict` or `functional`, and non-negative tolerances | Functional mode records finite numerical differences. Shape and dtype errors still fail |
| `activation_cache.*` | `enabled` true, host 4 GiB, device 512 MiB and reserve 2 GiB. Integer bytes ≥ 0 | Part of the completion and training identity. Disable with `--no-cache` |

`--host-cache-gib` and `--gpu-cache-mib` set `activation_cache.host_bytes` and `device_bytes`. `--gpu-window-mib` is an alias of `--gpu-cache-mib`, and neither changes `adaround.window_bytes`. These cache settings apply only to AdaRound. The Python API supports more detailed `QuantizationConfig` policies, and not every API field is an experiment key ([Quantization logic](quantization.md)).

### Postprocessing, evaluation and reporting

`decode.*` belongs to the model recipe. Other configuration keys below belong to the experiment. `--table-profile` is a command-line report option of `run`, `report` and `results`.

Generation runs use model-recipe profiles. Evaluation configs use their catalog profiles. `--evaluation-profile` explicitly overrides either ([Evaluation profiles](evaluation-profiles.md)).

| Key | Default and allowed values | Effect of a change |
| --- | --- | --- |
| `decode.*` | Model-specific confidence, IoU, maximum detections and output mapping | Identical network outputs can give different predictions and metrics. Reevaluate FP32 and INT8 |
| `evaluation.profiles.<model>` | Default of the packaged model. See [Evaluation profiles](evaluation-profiles.md) | The profile identity changes. Unsupported combinations and unknown model names are rejected before any model runs |
| `evaluation.limit` | `null` (full) in both presets. Positive integer or `null`. LFW requires all 6,000 pairs | The evaluation input identity and FP32 pairing change |
| `runtime.device`, `runtime.cuda.gpu_mem_limit` | Presets use `cuda:0`. Select the CPU explicitly. GPU memory limit: 6 GiB by default, 6 or 8 GiB in recorded settings | The numerical runtime is checked for reuse. The ORT memory limit is excluded |
| `runtime.intra_op_threads`, `.inter_op_threads` | Generation configs: 1 and 1. Non-negative integers, where 0 lets ORT choose. Evaluation configs use one thread | The numerical runtime is checked for reuse |
| `output.save_qdq_models`, `.save_predictions` | `true` and `false` respectively in generation presets. Boolean | Part of the completed-result identity |
| `report.primary_metric.<task>` | Recipe or preset metric | Selects among saved metrics. Only ratio units are allowed, and no inference runs |
| `report.target_mean_recovery` | `null` in both presets. Positive number or `null` | The summary marks whether each condition's cohort mean recovery meets this target ([FP32 pairing and recovery](reporting.md#fp32-pairing-and-recovery)) |
| `--table-profile opennpu_raw` | Optional for `run`, `report` and `results` | Replaces the default `results.*` table with `raw_data.*` tables split into full and partial evaluations. The evaluation identity is unchanged |

Use `report` with `--format` or `--table-profile` to change presentation without inference. To change execution settings, edit your experiment file and run it again. Do not edit saved settings or completion evidence to force reuse. See [Reuse and caching](run-pipeline.md#reuse-and-caching) for when earlier work can be reused.

## Example: change a percentile

`load_config()` loads the full preset from an installed package. Edit the returned configuration and save it to change any supported experiment setting, including model selection, `calibration.histogram_bins` or `adaround.learning_rate`. This example changes a percentile and validates the configuration without running the experiment:

```bash
python - <<'PY'
from opennpu_quant.configuration import load_config
import json
config = load_config('core10')
config['name'] = 'resnet18_p999'
config['models'] = ['resnet18']
config['conditions'] = {key: config['conditions'][key] for key in ('fp32', 'pot_rtn')}
config['calibration']['samples'] = dict.fromkeys(config['calibration']['samples'], 8)
config['evaluation']['limit'] = 32
config['percentiles']['resnet18'] = {'pot': 99.9}
with open('experiment.json', 'w') as stream:
    json.dump(config, stream, indent=2)
PY
opennpu-quant run -c experiment.json --dry-run
```

<a id="example-add-an-adaround-condition"></a>

## Example: run a short AdaRound check

`core10` already defines AdaRound conditions such as `pot_adaround`. `--adaround-steps` changes their training length but does not select them. This example copies the preset, shortens the AdaRound windows and selects the existing condition explicitly:

```bash
python - <<'PY'
import json
from opennpu_quant.configuration import load_config

config = load_config('core10')
config['name'] = 'resnet18_adaround_smoke'
config['models'] = ['resnet18']
config['adaround'].update(batch_size=1, window_samples=1, window_steps=1)
with open('adaround-smoke.json', 'w', encoding='utf-8') as stream:
    json.dump(config, stream, indent=2)
PY
opennpu-quant run -c adaround-smoke.json --conditions fp32,pot_adaround \
  --calibration-samples 2 --limit 2 --adaround-steps 2 --device cuda:0 \
  --output workspace/adaround-example-runs --dry-run
```

Without `--dry-run`, the command calibrates, trains two AdaRound steps per target layer and evaluates FP32 and INT8 in the same run, so the report has a recovery denominator. The preset already saves the generated model, so `--save-models` is not needed. These settings check that training works. They are not the original training settings.

## Use the original per-model settings

The generation presets contain explicit per-condition percentiles. Release evaluation configs select existing ONNX candidates and their evaluation profiles. Neither reads the historical selection during ordinary execution.
