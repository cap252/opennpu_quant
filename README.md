# opennpu-quant

opennpu-quant turns FP32 ONNX models into INT8 QDQ models, with optional power-of-two scales and AdaRound, and measures their accuracy against FP32. The default `core10` group covers 10 models, and `cnn30` covers 30 across classification, object and face detection, semantic and instance segmentation, and face verification.

Preprocessing prepares network inputs. Calibration collects FP32 activation statistics, and quantization builds the INT8 graph. Network outputs pass through postprocessing and task evaluation, then a report pairs INT8 with FP32 to calculate recovery. An evaluation config lets `run` use existing ONNX models without calibration or training. [Architecture](docs/architecture.md) explains the stages.

## Requirements

- Linux x86_64 and Python 3.11.13 for the tested setup.
- Bash, `curl`, `tar` and `sha256sum` for setup and release downloads.
- An NVIDIA GPU with a compatible driver for CUDA execution. A system CUDA toolkit is not needed.
- CPU execution is available for installation and pipeline checks. Its numbers can differ from GPU results.

## Install

```bash
./setup.sh --device cuda
source .venv/bin/activate
opennpu-quant info
```

`setup.sh` installs Python 3.11.13 and pinned dependencies into `.venv`, using `uv` from `PATH` or a checksum-verified copy in `.tools/`. For CPU checks, use `./setup.sh --device cpu` and pass `--device cpu` below. Re-running setup re-syncs the environment, so repeat the intended device profile. Add `--data` when preparing ImageNet from Hugging Face. The base profiles include LFW alignment dependencies. Model-specific export dependencies and their installation commands are listed in [Extra export environments](docs/models-and-data.md#extra-export-environments). The CPU profile uses the CPU provider in the pinned GPU-capable ORT package.

If a large dependency download times out, retry the same device profile with `UV_HTTP_TIMEOUT=300 ./setup.sh --device cuda`. This changes the network timeout, not the pinned packages.

## Quick start

### 1. Check the installation

```bash
opennpu-quant demo --device cuda:0
```

The demo calibrates, quantizes and evaluates a small built-in model without downloading assets.

<a id="2-check-supplied-onnx-models-on-a-few-images"></a>

### 2. Check supplied ONNX models

Download the core10 archive from [opennpu_quant v1.0.1](https://github.com/cap252/opennpu_quant/releases/tag/v1.0.1) and extract it from the repository root:

```bash
curl -fL -O https://github.com/cap252/opennpu_quant/releases/download/v1.0.1/cnn30-core10-01-of-01.tar.gz &&
  sha256sum --check --ignore-missing release-v1.0.1.sha256 &&
  tar -xzf cnn30-core10-01-of-01.tar.gz
```

The repository’s [checksum list](release-v1.0.1.sha256) verifies the downloaded archives before extraction. Stop if verification fails. This creates `bundle/`, which Git ignores, with the core10 models and evaluation configs. To keep the bundle elsewhere, run these commands in that directory with the path to the checksum list, and pass that location's `bundle/configs/` files to `-c`. The additional20 archives are not needed for core10.

Prepare ImageNet and COCO using the [dataset layouts](docs/models-and-data.md#prepare-datasets). Replace the source paths below with your own. Preparation checks calibration inputs and the full validation sets, even for a short evaluation. The evaluation itself performs no calibration or training.

```bash
opennpu-quant data prepare imagenet --root /data/imagenet --source folder --data-dir workspace/data
opennpu-quant data prepare coco --root /data/coco --data-dir workspace/data
```

Use the same data directory to evaluate three images each for ResNet-18 and YOLOv5s:

```bash
opennpu-quant run -c bundle/configs/core10.yaml \
  --models resnet18,yolov5s --conditions fp32,pot_rtn \
  --limit 3 --device cuda:0 --data-dir workspace/data \
  --output workspace/evaluations
```

Check that all four conditions completed with none failed or pending. `workspace/evaluations/core10/results.csv` contains partial accuracy, auxiliary metrics and recovery. `settings.json` records the effective settings and diagnostics. Set `--limit full` and use a new output directory for full accuracy.

Add `--dry-run` to check model and data readiness without inference. Evaluation and preparation checks exit with code 1 if any condition failed or remains pending, and code 0 when all requested conditions completed or are ready.

### 3. Evaluate all selected conditions

Download all three additional20 archives from the same release and extract them alongside core10:

```bash
(
  for part in 01 02 03
  do
    archive="cnn30-additional20-${part}-of-03.tar.gz"
    curl -fL -O "https://github.com/cap252/opennpu_quant/releases/download/v1.0.1/$archive" || exit 1
  done
  sha256sum --check --ignore-missing release-v1.0.1.sha256 || exit 1
  for part in 01 02 03
  do
    tar -xzf "cnn30-additional20-${part}-of-03.tar.gz" || exit 1
  done
)
```

Each archive extracts independently into `bundle/`. Prepare all datasets, including the complete LFW data. When matching earlier CNN30 inputs, follow the [LFW CPU numerical settings](docs/models-and-data.md#reproduce-lfw-alignment) before preparing LFW. Images and input tensors are not included in the release. The default CNN30 config selects FP32 and six INT8 conditions for each of 30 models:

```bash
opennpu-quant run -c bundle/configs/cnn30.yaml --limit full \
  --device cuda:0 --data-dir workspace/data --output workspace/evaluations
opennpu-quant report workspace/evaluations/cnn30
```

Check 210 completed conditions and their full evaluation scope. Import `workspace/evaluations/cnn30/results.csv` into Excel. The release also preserves extra core10 candidates, which are selected only by `bundle/configs/core10-candidates.yaml`.

LFW requires all 6,000 pairs. A small request that includes MobileFaceNet is rejected before inference. The [release evaluation guide](docs/reproducing-results.md) explains paths, optional raw capture and reuse.

<a id="create-new-models-when-needed"></a>

## Create new INT8 models

Use this workflow to create an INT8 model from an existing trained checkpoint: prepare the dataset, export FP32 ONNX, collect calibration statistics, then quantize and evaluate. `export resnet18` downloads the pinned pretrained checkpoint and writes the FP32 bundle to `workspace/models/resnet18/`. `run` evaluates FP32 and creates and evaluates the selected INT8 model.

First download and extract ImageNet using the [dataset guide](docs/models-and-data.md#prepare-datasets). The `data prepare` command below checks the local files and writes a manifest. It does not download images. This example uses two calibration images and two evaluation images for a quick execution check:

```bash
opennpu-quant data prepare imagenet --root /data/imagenet --source folder
opennpu-quant export resnet18
opennpu-quant run -c core10 --models resnet18 --conditions fp32,pot_rtn \
  --calibration-samples 2 --limit 2 --device cuda:0 \
  --output workspace/example-runs
opennpu-quant report workspace/example-runs/core10
```

Preparation requires all 50,000 ImageNet validation images and the calibration inputs, even when this example evaluates only two images. The presets save generated models under `workspace/example-runs/core10/models/resnet18/pot_rtn/network.onnx`. `results.csv` records the partial accuracy and recovery. See [model and data preparation](docs/models-and-data.md) and [run experiments](docs/run-pipeline.md) for full evaluation and AdaRound training.

| Preset for `run` | Models | Conditions | Evaluation scope |
|---|---|---|---|
| `core10` (default) | 10 | 7: FP32 and RTN/AdaRound for Asym/PC float, Asym/PC PoT and Sym/PT PoT | Full datasets |
| `cnn30` | 30 | 7: FP32 and RTN/AdaRound for Asym/PC float, Asym/PC PoT and Sym/PT PoT | Full datasets |

A successful fixed-model evaluation verifies the supplied graphs and evaluation path. Recreating their quantization statistics and trained weights requires separate regeneration checks. The [configuration reference](docs/configuration.md) explains generation and evaluation settings.

## Validation scope

Assess execution completion, reference agreement and benchmark accuracy separately. Benchmark accuracy requires full evaluation of each selected model and condition. For LFW, use all 6,000 pairs. Rescoring saved outputs checks postprocessing and metric calculation. Verifying inference on a new GPU or newly generated INT8 models requires a new evaluation. See [validation scope](docs/validation.md) for the evidence and limits.

## Python API

The Python API is available when you need to integrate quantization into your own Python workflow. With an FP32 model and a calibration input function prepared:

```python
from opennpu_quant import OrtConfig, Scheme, calibrate, quantize, save_model

ort = OrtConfig.cpu()
stats = calibrate(model, feeds, ort=ort)
scheme = Scheme(activation_scale="pot_ceil", weight_scale="pot_ceil")
result = quantize(model, stats, scheme, ort=ort)
save_model(result.model, "model_int8.onnx")
```

`feeds` supplies preprocessed calibration inputs as dictionaries mapping model input names to NumPy arrays. Pass the function itself. Each call must return a fresh iterator with the same values and order, because calibration reads the inputs twice. See the [Python API guide](docs/api.md) for a complete example.

## Documentation

- **Guides:** [Prepare models and data](docs/models-and-data.md), [Run experiments](docs/run-pipeline.md), [Reports and Excel export](docs/reporting.md), [Python API](docs/api.md), [Evaluate release models](docs/reproducing-results.md)
- **Reference:** [Configuration reference](docs/configuration.md), [Model reference](docs/model-reference.md), [Evaluation profiles](docs/evaluation-profiles.md), [Validation scope and limitations](docs/validation.md)
- **Internals:** [How opennpu-quant works](docs/architecture.md), [Quantization logic](docs/quantization.md), [Development checks](tests/README.md)

The [documentation index](docs/README.md) describes each page.

## License

Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY.md).
