# Development checks

The test suite builds small models and datasets in temporary directories, so it needs no downloads or external assets. See [Validation scope and limitations](../docs/validation.md) for how to interpret real-model results.

## Run the checks

Run from the repository root after setup:

```bash
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/ruff check src tests tools
.venv/bin/ruff format --check src tests tools
.venv/bin/python tools/build_presets.py --check
.venv/bin/python -m unittest discover -s tests -t . -v
```

Use `./.tools/uv` if setup installed uv locally. Tests that need Torch, Torchvision or pycocotools run when those packages are installed, as they are in the `setup.sh` environments.

## Fixtures

| File | Contents |
|---|---|
| `fixtures/pot/golden_40.json` | Expected graph and output hashes for a demo model |
| `fixtures/public_api_signatures.json` | The public API contract |
| `fixtures/preset_contracts.json` | Ordered effective configurations and condition lists for the two standalone presets. Update only for an intentional settings change |
| `fixtures/export_settings.json` | Export settings for all 30 recipes. Update only for a reviewed, intentional export change |
| `fixtures/documented_internals.json` | Signatures of the non-public names that README and docs import |
| `fixtures/opennpu_raw_format.json` | Columns, display names and schemes of the 15-column table |
| `fixtures/face_angle_reference.json` | 20 high-precision arccos reference cases for the fixed-angle metric |
| `library_fixtures.py` | A toy model and feeds shared by tests |
| `dataset_fixtures.py` | A small embedding graph for the full LFW protocol test |

## Documented API compatibility

The API guide imports model and dataset helpers from their implementation modules. These names are not exported by the top-level `opennpu_quant` package, but the documented imports are part of the examples users rely on. `test_documented_internal_imports_keep_their_signatures` checks both the complete set of these imports and their signatures against `fixtures/documented_internals.json`. Keep the fixture in sync when intentionally adding a documented helper, and review compatibility before changing a listed signature.

The Python postprocessing example uses `ModelEvaluator` and `resolve` as extension points. Behavioral tests in `test_evaluation_contracts.py` also check execution order, argument forwarding, reuse of the existing metric, and the flags that exclude custom postprocessing from automatic result reuse and original-measurement reproduction. The basic package import is checked separately to ensure that it creates no files and does not load numerical runtimes.

## What each file covers

### Quantization core

| File | Covers |
|---|---|
| `core/test_pot_scales.py` | PoT scale math: exponents, ties, non-finite values, zero-point clipping, MSE candidates (NumPy only) |
| `ort/test_pot_quantization.py` | 480 scale/policy combinations, golden hashes, shared weights, bias scales, INT32 limits, tamper detection |
| `ort/test_adaround.py` | AdaRound changes only weight codes. Conv/Gemm/MatMul variants, checkpoints and resume, budgets, empty bias names, CUDA checks |
| `ort/test_adaround_cache.py` | AdaRound activation cache: LRU, budgets, GPU fallback |
| `ort/test_adaround_memory.py` | Chunked regularizer gradients equal the dense computation |
| `ort/test_input_cache.py` | `prepare_feed_cache` contract, tampering, locks, budgets, cached feeds in AdaRound, CLI cache switches |
| `ort/test_safe_fetch.py` | `ort.fetch.run_tensors` copies and pass-through inputs |
| `ort/test_errors.py` | Every native ORT exception class, including session and inference errors, is translated with its cause, while unrelated exceptions and interrupts propagate |

### API, CLI and configuration

| File | Covers |
|---|---|
| `test_api_usability.py` | Public names and signatures, signatures of non-public names used in docs, docstrings and doctests, light import, configuration-change messages |
| `test_cli_usability.py` | CLI help, input errors, demo output, local recipes, device validation |
| `test_user_distribution.py` | User commands, core10/cnn30 scope and independence from archived replay tools and data |
| `test_preset_generation.py` | Standalone preset generation, drift detection, independent configs and policy conversion |
| `test_input_validation.py` | Experiment validation with exact key paths, YAML error locations, filesystem errors, error messages |
| `test_recorded_conditions.py` | Condition-specific recorded percentile resolution and template validation |
| `test_recipe_settings.py` | Removed recipe keys are rejected by YAML loading and direct construction, while active settings and custom preprocessing are kept |
| `test_release_inputs.py` | Local path settings, dataset roots and precedence |
| `test_interface_reduction.py` | Segmentation metrics, range selection, recipe keys, removed CLI options, version |

### Models, export and evaluation

| File | Covers |
|---|---|
| `test_fixed_spatial.py` | Fixed recipes, resize/padding and box/mask restoration, export fixtures and static/dynamic parameter broadcasting |
| `test_fixed_spatial_bundle.py` | Input-only ONNX conversion, candidate preservation, graph hashes and atomic bundle publication |
| `test_cnn30.py` | The exact cnn30 model order and task counts, core10 and cnn30 settings with explicit condition percentiles, strict/functional validation, export rewrites (face grids, fixed upsample, DFL split, constant Floor), dynamic input caches and AdaRound windows, decoders, float/PoT scale audits, DepthToSpace |
| `test_cnn30_support.py` | CNN30 support and retired-model rejection without blocking custom models |
| `test_tensorflow_export.py` | Mocked SSD/DeepLab converter calls forward the recipe opset (no real TensorFlow conversion) |
| `test_export_contract.py` | Framework diagnostics, mandatory raw/final and partition checks, and the export settings snapshot. Historical contract checks are recorded as not executed |
| `test_export_boundaries.py` | Network/host boundaries, export/runtime input agreement and recorded export settings |
| `test_source_archive.py` | Pinned archives: reuse, tampering, extraction recovery, links and path safety |
| `test_protobuf_compat.py` | Current and legacy descriptor APIs, and ONNX traversal with the installed protobuf runtime |
| `test_graph_assets.py` | External weights, required hashes, reuse after asset recovery, sparse tensors and path safety |
| `test_repeat_outputs.py` | Input shapes, dtypes and overridable initializer inputs |
| `test_final_reproduction.py` | Calibration input guards and the explicit legacy top-k profile |
| `test_model_bundle.py` | Exported folders with and without `model.json`, mapping precedence, tampering, explicit graph paths and output aliases |
| `test_preprocessing_reliability.py` | Custom preprocessing never reuses cached feeds, and built-in caches invalidate on setting or dependency changes, or on ORT code changes when a CPU graph runs |
| `test_metrics.py` | Exact integer Top-K ordering, ties at top-1/top-5, input preservation, WIDER FACE AP edge cases, COCO undefined values |
| `test_segmentation_predictions.py` | A VOC-style run whose saved predictions replay to the same metrics |
| `test_recovery.py` | Common recovery operation order, exact FP32 self-comparison, mixed units, missing denominators and serialized precision |
| `test_lfw.py` | Full LFW pair order, ten folds, flip views, unchanged calibration input selection, early rejection of partial evaluation requests, manifest tampering, the reference crop identity, mismatch warnings and recorded OpenBLAS alignment settings |
| `test_auxiliary_metrics.py` | Fixed-angle normalization bits, high-precision arccos references, same-environment repeatability, FP64 angle threshold boundary, full-protocol metric names and serialization |
| `test_evaluation_contracts.py` | Descriptive profiles for supported recipes, unsupported profile rejection and the Python postprocessing hook |
| `test_evaluation_defaults.py` | Generation defaults and packaged recipe profiles, local bundles, override precedence, classifier tie order and rejection of profiles for unknown model names |
| `test_evaluation_profiles.py` | YOLO profile behavior (NMS, clipping, mask arithmetic, metric area), real evaluator routing and evaluation changes. Needs Torch and Torchvision |

The fixed-angle math check uses [20 high-precision reference cases](fixtures/face_angle_reference.json), including endpoints and inputs near the 74.18-degree threshold. The references were computed from exact FP32 inputs at 100 and 200 decimal digits, with matching stored digits. The test requires arccos error within one FP32 spacing of the mathematical reference. Its degree-conversion budget includes that error propagated by 180/pi and one additional FP32 spacing. This is a numeric test contract for these cases, not a guarantee about every input or CPU. Normalization, threshold semantics and same-environment repeatability are checked separately. Runtime metric comparisons and bitwise reproduction verdicts are unchanged.

### Runs, reuse and reports

| File | Covers |
|---|---|
| `test_portable.py` | End-to-end local run with resume, reuse and tampering, statistics relocation, downloads and archives, checkout cleanup and ImageNet preparation |
| `test_dataset_download.py` | COCO/VOC downloads from small local archives, CLI preparation, offline reuse, VOC layout, reuse and missing-file repair, checksums, conflicting files, unsafe archives and existing manifest validation |
| `test_evaluation_recovery.py` | Corrupt evaluation records are preserved and evaluated once, saved models remain reusable, and failed inference does not enter a retry loop. Covers both run modes and CLI outcomes |
| `test_evaluation_steps.py` | Rules of the evaluation-config steps without inference: condition counts, the selection and condition identities, sample scope, runtime settings, reuse or one-time recovery of saved records, completed records and failure records |
| `test_condition_outcomes.py` | One rule for every run path: stop requests interrupt, missing files during preparation stay pending, and any other exception fails only its condition in evaluation configs and stops generation. CLI exit codes, logged tracebacks of unexpected errors, the profiled-output comparison, and save failures of state or reports that stop the run while completed conditions stay reusable |
| `test_run_records.py` | Shared generation and evaluation evidence, incomplete status, archived cache rejection and reporting without execution imports |
| `test_result_reliability.py` | Result reuse evidence, invalidation by graph or data changes, forced recompute, stale files, original errors, undefined primary metric |
| `test_unified_run.py` | Saved QDQ reuse, evaluation archives and restoration, metadata-only reports and concurrent report locking |
| `test_runtime_boundaries.py` | CPU numerical identity, FP32-only input requirements, environment evidence and GPU profile limits |
| `test_gpu_environment.py` | GPU specification, library and calculation identities, UUID-independent reuse and hashed runtime device records, and the kernel driver and driver library records of CUDA evaluations |
| `test_code_identity.py` | Identity changes for nested Python edits, renames, additions and deletions, and ignores non-Python files |
| `test_cache_concurrency.py` | Concurrent input and statistics readers, one input-cache builder, waiting writers, failure recovery, process exit, interruption and integrity checks after waiting |
| `test_recalibrate.py` | `--recalibrate` generations, partial and aborted runs, the source files in the statistics cache key |
| `test_report_output.py` | One writer for every report folder: destinations inside `.state` or another run are rejected for `report` and `results`, settings.json follows the reported records, and filters do not change environment diagnostics |
| `test_results.py` | `results` tables: units, FP32 pairing, identity mismatches, historical-origin rejection before export, Excel/Unicode, partial denominators, obsolete metrics tables preserved and warned about by standalone reports |
| `test_final_evidence.py` | Evaluation-identity mismatches |
| `test_raw_exports.py` | 15-column raw tables of run exports, `results --stdout` and `report` rebuilds, including the percentile and scheme columns |
| `test_run_capture.py` | Selected raw capture, budgets, input preservation, corruption and aborted publication |
| `test_replay_diagnostics.py` | Documented CSV examples, provenance hashes and recovery values |
| `test_raw_report.py` | Excel-compatible tables for ordinary `report`: formats, units, full/partial split, denominators, exclusions and metadata-only execution |
| `test_review_fixes.py` | Report formats and policies, export Top-5 parity, recipe layout checks, audit block size, gate provenance, optional Torch suite, COCO details |

### Public run storage

| File | Covers |
|---|---|
| `test_public_run.py` | Generation and existing-ONNX evaluation, minimal outputs, reuse including across GPU memory limit changes, one obsolete-table warning per run, the `settings.json` of separate report folders, missing or corrupted assets, run-format conflicts, native ORT failures and profiled-sample mismatches as condition failures, functional-pass flags of failed conditions, profile validation before any work, exit codes and LFW dry-run limits |
