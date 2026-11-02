import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from opennpu_quant.report import report as run_report
from opennpu_quant._io import atomic_json, object_hash, read_json, sha256
from opennpu_quant.raw_report import build_report, write_report
from opennpu_quant.results import RAW_FIELDS, export_records
from tests.test_results import record

FIXTURE = read_json(Path(__file__).parent / "fixtures/opennpu_raw_format.json")


class RawReportTests(unittest.TestCase):
    def test_portable_summary_does_not_average_separate_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            export_records(
                [
                    record(run_id="first", model="a", full_evaluation=True),
                    record(run_id="second", model="b", full_evaluation=True),
                ],
                temp,
            )
            self.assertIsNone(
                read_json(Path(temp) / "summary.json")["mean_recovery"]["fp32"]
            )

    def test_precision_units_full_partial_and_default_unchanged(self):
        value = 0.6968412345678912
        base = record(value=value, full_evaluation=True)
        quantized = record("q", 0.67890123456789, full_evaluation=True)
        partial = record(model="partial", samples=2, full_evaluation=False)
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            export_records([base, quantized, partial], p)
            before = (p / "results.csv").read_bytes()
            self.assertFalse((p / "raw_data.csv").exists())
            export_records([base, quantized, partial], p, table_profile="opennpu_raw")
            self.assertEqual(before, (p / "results.csv").read_bytes())
            self.assertTrue(
                (p / "raw_data.csv").read_bytes().startswith(b"\xef\xbb\xbf")
            )
            rows = list(
                csv.DictReader(
                    io.StringIO((p / "raw_data.csv").read_text(encoding="utf-8-sig"))
                )
            )
            self.assertEqual(2, len(rows))
            self.assertEqual(value * 100, float(rows[0]["fp32_accuracy"]))
            self.assertEqual(
                value, json.loads(rows[0]["metrics_json"])["top1"]["value"]
            )
            self.assertEqual(
                100 * 0.67890123456789 / value,
                float(rows[1]["recovery_percent"]),
            )
            self.assertEqual(
                2, len((p / "raw_data.partial.csv").read_text().splitlines())
            )

    def test_missing_mismatched_zero_or_unverified_denominator(self):
        for case in (
            "missing",
            "identity",
            "profile",
            "scope",
            "zero",
            "proof",
            "reference",
            "sample_unit",
            "expected_samples",
            "view_count",
        ):
            with self.subTest(case=case):
                b, q = record(), record("q", 0.78)
                records = [b, q]
                if case == "missing":
                    records = [q]
                if case == "identity":
                    q["evaluation_identity"] = "changed"
                if case == "profile":
                    q["evaluation_profile"] = "changed"
                if case == "scope":
                    b["full_evaluation"] = True
                if case == "zero":
                    b["metrics"]["top1"]["value"] = 0
                if case == "proof":
                    b["evidence"]["result_hash_verified"] = False
                if case == "reference":
                    b["reference"] = True
                if case in {"sample_unit", "expected_samples", "view_count"}:
                    q[case] = "changed"
                result = build_report(records)
                row = result["partial"][-1]
                self.assertIsNone(row["recovery_percent"])
                self.assertEqual(78, row["int8_accuracy"])

    def test_failures_missing_metrics_and_original_values_are_not_exported(self):
        values = [
            record(state="fail"),
            record(state="pending"),
            record(metrics={}),
            record(reference=True),
            record(origin="custom"),
        ]
        for i, r in enumerate(values):
            r["model"] = str(i)
            r["measurement"] = {"metrics": {"top1": 0.99}}
        result = build_report(values)
        self.assertEqual([], result["full"] + result["partial"])
        self.assertEqual(5, result["metadata"]["counts"]["excluded"])
        self.assertTrue(all(r["reason"] for r in result["metadata"]["records"]))

    def test_standard_run_report_and_cli_are_metadata_only(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            config = dict(
                name="test",
                models=["m"],
                conditions={"fp32": {}},
                evaluation={"split": "validation"},
            )
            atomic_json(p / "effective_config.json", config)
            path = p / "m/fp32/result.json"
            atomic_json(
                path,
                dict(
                    title="M",
                    metrics={"top1": 0.7},
                    metric="top1",
                    samples=2,
                    expected_samples=2,
                    complete=True,
                    full_evaluation=True,
                    config_identity=object_hash(config),
                    model_identity="m",
                    preprocessing_identity="p",
                    evaluation_identity="e",
                    model_metadata={
                        "task": "classification",
                        "metric": "top1",
                        "dataset": "imagenet",
                    },
                ),
            )
            atomic_json(path.parent / "complete.json", dict(result_sha256=sha256(path)))
            result = run_report(p, table_profile="opennpu_raw")
            self.assertEqual(1, len(result["raw_data"]["full"]))
            for args in (
                ["report", str(p), "--table-profile", "opennpu_raw"],
                [
                    "results",
                    str(p / "records.json"),
                    "--output",
                    str(p / "tables"),
                    "--table-profile",
                    "opennpu_raw",
                ],
            ):
                self.metadata_cli(args)
            path.write_text("{}")
            self.assertEqual(
                1,
                run_report(p, table_profile="opennpu_raw")["raw_data"]["metadata"][
                    "counts"
                ]["excluded"],
            )

    def metadata_cli(self, args):
        code = """
import sys, json, contextlib, io
sys.path.insert(0, sys.argv[2])
from opennpu_quant.cli import main
with contextlib.redirect_stdout(io.StringIO()):
    main(json.loads(sys.argv[1]))
print(json.dumps(sorted({'numpy','onnx','onnxruntime','torch','tensorflow','cv2'} & set(sys.modules))))
"""
        result = subprocess.run(
            [
                sys.executable,
                "-O",
                "-c",
                code,
                json.dumps(args),
                str(Path(__file__).resolve().parents[1]),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual([], json.loads(result.stdout))

    def test_empty_report_has_headers_and_metadata(self):
        face = record(
            model_metadata={
                "task": "face_verification",
                "decode": {"kind": "face_verification"},
            },
            metric_details={"pair_count": 6000},
        )
        meta = build_report([face])["metadata"]["records"][0]
        self.assertEqual("views", meta["sample_unit"])
        self.assertEqual(100, meta["view_count"])
        self.assertEqual(6000, meta["pair_count"])
        self.assertEqual("current_v1", meta["evaluation_profile"])
        with tempfile.TemporaryDirectory() as temp:
            write_report(build_report([]), temp, formats="csv,tsv")
            self.assertEqual(
                "\t".join(RAW_FIELDS) + "\n", (Path(temp) / "raw_data.tsv").read_text()
            )
