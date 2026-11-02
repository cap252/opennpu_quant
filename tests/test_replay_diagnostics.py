"""Failed repeatability measurements stay inspectable without entering recovery."""

import csv
import json
from pathlib import Path
import unittest

from opennpu_quant._io import read_json, sha256
from opennpu_quant.results import recovery_percent, RAW_FIELDS


class DocumentationExampleTests(unittest.TestCase):
    def test_csv_examples_match_current_recovery_and_recorded_hashes(self):
        folder = Path(__file__).resolve().parents[1] / "docs/examples/raw-data"
        provenance = read_json(folder / "provenance.json")
        self.assertFalse(list(folder.glob("*.tsv")))
        for item in provenance.values():
            for filename, expected in item["files"].items():
                with self.subTest(file=filename):
                    path = folder / filename
                    self.assertEqual(".csv", path.suffix)
                    self.assertEqual(expected, sha256(path))
                    with path.open(encoding="utf-8-sig", newline="") as stream:
                        reader = csv.DictReader(stream)
                        self.assertEqual(RAW_FIELDS, reader.fieldnames)
                        rows = list(reader)
                    baseline = next(r for r in rows if r["scheme"] == "FP32")
                    base = json.loads(baseline["metrics_json"])[baseline["metric"]]
                    for row in rows:
                        metric = json.loads(row["metrics_json"])[row["metric"]]
                        expected = recovery_percent(
                            metric, base, is_baseline=row is baseline
                        )
                        self.assertEqual(repr(expected), row["recovery_percent"])
