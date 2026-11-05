# Reports and Excel export

Report commands turn saved evaluations into tables. They check the provenance and completion evidence of each result, and they never run inference, calibration or training. `run` already writes its default reports, so use these commands to select a table format or rebuild reports later.

## Choose a report command

```bash
opennpu-quant report workspace/example-runs/core10 --format csv,json --table-profile opennpu_raw
opennpu-quant results workspace/example-runs/core10/records.json \
  --output workspace/tables --table-profile opennpu_raw
```

| You have | Command | What it reads |
|---|---|---|
| A generation or evaluation directory from `run` | `report` | Saved settings, status and completed evaluation files |
| An exported `records.json` | `results` | The portable records only, without the original result files |

`report RUN_DIR` reads both ordinary run modes.

`run` and `report` write `results.csv` by default. `settings.json` retains the effective settings and diagnostics. When `report --output` writes to another folder, that folder also receives a `settings.json` for the results it contains, while the run folder's `settings.json` keeps describing the whole run. Select `json`, `tsv` or `md` explicitly for additional reports. Portable `results` exports default to CSV and JSON. `results --format` requires `--output`. No XLSX file is generated.

`--table-profile opennpu_raw` is optional for these commands. It replaces the primary table with the full/partial tables below, using display names that match the workbook. With `results`, this option requires `--output`. With `--stdout csv` or `--stdout tsv` and this profile, stdout shows only the full-evaluation table.

## Output files

| File | Contents |
|---|---|
| `results.csv` | The 15-column primary table |
| `settings.json` | Effective configuration, model connections, runtime and diagnostics |
| `records.json` (`--format json`) | Portable measurement records for `results` |
| `summary.json` (`--format json`) | Per-condition results and complete-cohort recovery means |

With `--table-profile opennpu_raw`, these files replace `results.csv`:

| File | Contents |
|---|---|
| `raw_data.csv` | Verified full evaluations: `full_evaluation` and `complete` are true and the sample count equals the full expected count |
| `raw_data.partial.csv` | Valid metrics from evaluations that do not meet the full-evaluation requirements, such as `--limit` runs |
| `raw_data.metadata.json` | Provenance, row locations, and pairing and exclusion reasons. Always retained with this profile |

Records without explicit full-evaluation evidence remain in the partial tables and are excluded from full-cohort means. This classification does not establish that the original evaluation was partial.

`--format tsv` writes TSV tables instead of CSV, `json` writes the two JSON reports, and `md` writes `summary.md`. Formats can be combined. Empty tables retain their headers. CSV uses UTF-8 with a BOM, and TSV uses plain UTF-8.

`report` also accepts `--models` and `--conditions`. If a condition is missing from the profile tables, look up its reason in `raw_data.metadata.json`. Standalone report commands preserve files in unselected formats. `run` also refreshes existing report files after each condition or model and on success, failure or interruption. Use a new report directory for a clean output set.

`--output` takes the run folder itself or a separate folder. A folder inside `.state` or another run's folder is rejected, so a report never overwrites run files. `results --output` follows the same rule.

## The 15-column table

The column names and order follow the `raw data` sheet of the OpenNPU results workbook.

| Order | Column | Value |
|---:|---|---|
| 1 | `model` | Model display name |
| 2 | `task` | Task name |
| 3 | `dataset` | Dataset name |
| 4 | `metric` | Primary metric of this evaluation |
| 5 | `fp32_accuracy` | FP32 metric × 100. INT8 rows show their matching FP32 value |
| 6 | `int8_accuracy` | INT8 metric × 100, blank for FP32 rows |
| 7 | `recovery_percent` | 100 × INT8 / FP32, blank when unavailable |
| 8 | `scheme` | `FP32` or the policy string, without the percentile value |
| 9 | `percentile` | Percentile value when the Percentile method is used |
| 10 | `model_source_url` | Source URL from the recipe |
| 11 | `model_source_revision` | Source commit or framework version |
| 12 | `top5_percent` | top5 × 100, blank if absent |
| 13 | `AP50_percent` | AP50 × 100, blank if absent |
| 14 | `AP75_percent` | AP75 × 100, blank if absent |
| 15 | `metrics_json` | Original value and unit of every metric the evaluator reports |

Accuracy and recovery are plain numbers without a `%` sign or extra rounding. For example, the fraction `0.69684` appears as `69.684`, while `metrics_json` keeps the fraction. Metrics keep their own names in `metrics_json`, so instance segmentation reports `mask_AP50` rather than `AP50`. Semantic segmentation reports only `mIoU` there. Its pixel accuracy, evaluated class count and per-class IoU are metric details, which the CSV and TSV tables do not include. Use `--format json` and read `metric_details` in `records.json` for them.

A typical `scheme` value:

```text
scope=all | activation=asymmetric | weight=per_channel | activation_scale=pot_ceil | weight_scale=pot_ceil | calibration=percentile | rounding=AdaRound
```

The table profile changes only these task and dataset names, to match the workbook's lookups:

| Internal ID or name | Display name |
|---|---|
| classification, detection, face | image_classification, object_detection, face_detection |
| segmentation | semantic_segmentation |
| lfw_mobilefacenet | lfw |

Models keep their recipe titles. Internal IDs, evaluation profiles and metrics are unchanged. The [format fixture](../tests/fixtures/opennpu_raw_format.json) stores the columns, names and schemes, so the tests do not need the workbook.

## FP32 pairing and recovery

Recovery needs a successful FP32 result for the same model in the same run. The two results must match in model, preprocessing and evaluation identity, metric, data split, sample count, full or partial scope, profile, sample unit, view count and full expected count.

Without a compatible FP32 result, the denominator and recovery stay blank, and the INT8 measurement is still reported. If FP32 measures 0, its zero is kept and only the recovery is blank. A valid FP32 self-comparison is exactly `100.0`. INT8 recovery is computed as `100 × INT8 accuracy / FP32 accuracy` before display-unit conversion, without rounding. Matching fraction or percent units use their original values, and mixed units are normalized to fractions. Metrics in dB have no recovery ratio. The filters of `report` do not add FP32 automatically, so include it whenever you need recovery, for example `--conditions fp32,pot_rtn`.

Reports use only the current result's `metrics` and `metric_units`. They never copy the original `measurement.metrics`, fill missing values with zero or borrow FP32 from another run directory. A current measurement that differs from the original is reported as measured.

The summary shows a cohort mean per condition over the selected models in one run. The mean is computed only when every selected model has a full evaluation with a matching FP32 result, so partial runs show `mean_recovery: null` even though each row has its own recovery. Records from separate runs are not averaged together.

For LFW, ordinary runs count input views in `samples` and record `pair_count` in metric details. The optional raw table metadata labels the view count and sample unit ([LFW](models-and-data.md#lfw)).

Environment diagnostics compare each result with its FP32 baseline and are recorded in `settings.json` and, when requested, `summary.json` and `raw_data.metadata.json`. A diagnostic does not remove recovery values whose pairing passed. Ordinary run pairing requires matching numerical runtime settings. Report filters do not change the diagnostics: a result is compared with its run's FP32 runtime even when `--conditions` leaves the FP32 row out. Recovery still needs the FP32 row in the selection.

CUDA evaluations also record the NVIDIA kernel driver text from `/proc/driver/nvidia/version` and the SHA256 of the driver libraries mapped by the inference process, such as `libcuda.so.1`. The GPU diagnostic needs both to pass. A record without them remains `pending`. These fields are diagnostics only and do not change result reuse.

## Exclusions and verification

Failed, unexecuted, unverifiable or metric-less results get no numerical row in the `raw_data` tables, and `raw_data.metadata.json` records why. `results.csv` keeps their row with blank INT8 and recovery values. A valid INT8 measurement without a compatible FP32 denominator still appears, with blank recovery. Partial evaluations go to the partial files instead of being excluded. Partial LFW artifacts are rejected. Prepare a new full LFW evaluation.

`report` checks saved settings, completion markers and attested file hashes. A completed condition whose files fail these checks is reported as `unverified` without metrics. `results` validates portable record structure and units but cannot reauthenticate the original files. Use `report` when source-file verification is needed.

## Import into Excel

1. Import `raw_data.csv` with Excel's text/CSV import.
2. Keep the accuracy and recovery columns as plain numbers such as `69.684`. Do not multiply them by 100 or apply percentage formatting.
3. Paste the rows into your workbook and update any summary sheets yourself. A report has one row per executed condition and never edits workbook formulas or ranges.

## Examples

The [raw data examples](examples/raw-data/README.md) contain a full CSV table and a partial CSV table from saved evaluations, with recovery recalculated using the current formula. They illustrate the format and are not evidence of a full reproduction.
