# Raw data examples

These CSV tables retain measurements exported with an earlier version’s `reproduce report --table-profile opennpu_raw` command. For current runs, use `report --table-profile opennpu_raw` as described in [Reports](../../reporting.md). Recovery has been recalculated from their retained `metrics_json` using the current report formula. No inference was rerun. They illustrate the [15-column format](../../reporting.md#the-15-column-table).

The YOLOv8n rows retain an empty source revision from the earlier export. Current reports include a recorded code revision when it is available.

- `raw_data.csv`: two completed conditions from one full run on all 5,000 COCO images, FP32 and Sym/PT PoT AdaRound of YOLOv8n.
- `raw_data.partial.csv`: ResNet-18 FP32 and Asym/PC PoT RTN from a three-sample run.
- [provenance.json](provenance.json): original run and package provenance, the table revision, and current CSV file hashes. The source runs were not revalidated during the table revision.

The two tables come from separate runs and do not share FP32 denominators. The report commands normally also write `raw_data.metadata.json` and the headers of empty groups, which are omitted here.
