import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from opennpu_quant._io import atomic_json, read_json, sha256
from opennpu_quant.cli import THREAD_VARIABLES, main
from opennpu_quant.public_run import evaluate_config, generate
from opennpu_quant.run_records import current_records
from opennpu_quant.report import report
from tests.test_result_reliability import fixture


def evaluation_fixture(root):
    paths, config = fixture(root)
    config["output"]["save_qdq_models"] = True
    config["evaluation"]["limit"] = None
    from opennpu_quant.runner import run

    run(config, paths, formats="csv")
    recipe = __import__("yaml").safe_load(
        (paths.models / "toy/recipe.yaml").read_text()
    )
    fp = paths.models / "toy/network.onnx"
    q = paths.runs / config["name"] / "models/toy/pot_rtn/network.onnx"
    from opennpu_quant.configuration import scheme_for_condition

    catalog = {
        "schema_version": 1,
        "models": {
            "toy": {
                "recipe": recipe,
                "host_graphs": {},
                "variants": {
                    "fp32": {
                        "network": {
                            "path": str(fp.relative_to(root)),
                            "sha256": sha256(fp),
                        },
                        "scheme": None,
                    },
                    "pot_rtn": {
                        "network": {
                            "path": str(q.relative_to(root)),
                            "sha256": sha256(q),
                        },
                        "scheme": scheme_for_condition(
                            config, "toy", "pot_rtn"
                        ).to_dict(),
                    },
                },
                "defaults": {"fp32": "fp32", "pot_rtn": "pot_rtn"},
            }
        },
    }
    settings = {
        "mode": "evaluate",
        "name": "evaluated",
        "source": "catalog.json",
        "models": ["toy"],
        "conditions": "defaults",
        "runtime": {"device": "cpu"},
        "evaluation": {"limit": None},
        "output": {},
    }
    atomic_json(root / "catalog.json", catalog)
    import yaml

    (root / "evaluate.yaml").write_text(yaml.safe_dump(settings))
    return paths, settings, catalog


@contextlib.contextmanager
def first_profile_drifts():
    """Make the first profiled sample differ from its evaluated outputs."""
    from opennpu_quant.evaluation import loop  # noqa: F401  binds the original first
    from opennpu_quant.ort import fetch

    original = fetch.run_tensors
    calls = []

    def run(session, names, feeds):
        outputs = original(session, names, feeds)
        calls.append(names)
        return [value + 1 for value in outputs] if len(calls) == 1 else outputs

    # profile_sample looks the function up in ort.fetch each time it runs.
    with patch.object(fetch, "run_tensors", side_effect=run):
        yield calls


class PublicRunTests(unittest.TestCase):
    def test_raw_cli_selection_replaces_yaml_selection(self):
        cases = (
            ({"raw_sample_ids": ["3"]}, ["--raw-samples", "1"], ["2"]),
            ({"raw_samples": 1}, ["--raw-sample-ids", "3"], ["3"]),
            ({"raw_sample_ids": ["2"]}, ["--raw-samples", "0"], []),
            ({"raw_samples": 1}, ["--raw-samples", "2"], ["2", "3"]),
            ({}, ["--raw-samples", "1"], ["2"]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, config, _ = evaluation_fixture(root)
            config["conditions"] = ["fp32"]
            for i, (output, flags, wanted_ids) in enumerate(cases):
                with self.subTest(output=output, flags=flags):
                    config["output"] = output
                    config["name"] = "capture-" + str(i)
                    config_file = root / "evaluate.json"
                    atomic_json(config_file, config)
                    command = [
                        sys.executable,
                        "-B",
                        "-m",
                        "opennpu_quant",
                        "run",
                        "-c",
                        str(config_file),
                        "--home",
                        str(root),
                        "--local-config",
                        str(root / "local.yaml"),
                        "--json",
                        *flags,
                    ]
                    for dry_run in (True, False):
                        process = subprocess.run(
                            command + (["--dry-run"] if dry_run else []),
                            cwd=root,
                            capture_output=True,
                            text=True,
                            timeout=60,
                        )
                        self.assertEqual(0, process.returncode, process.stderr)
                        json.loads(process.stdout)
                    out = paths.runs / config["name"]
                    captures = list(out.rglob("raw.json"))
                    if wanted_ids:
                        self.assertEqual(1, len(captures))
                        saved = read_json(captures[0])
                        self.assertEqual(
                            wanted_ids, [row["id"] for row in saved["samples"]]
                        )
                        self.assertTrue(
                            all(
                                (captures[0].parent / f).is_file()
                                for f in saved["files"]
                            )
                        )
                    else:
                        self.assertEqual([], captures)
                        self.assertFalse(list(out.rglob("*.npz")))

    def test_evaluation_rejects_empty_raw_id_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, config, _ = evaluation_fixture(root)
            config["output"] = {"raw_samples": 1}
            config_file = root / "evaluate.json"
            atomic_json(config_file, config)
            process = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "opennpu_quant",
                    "run",
                    "-c",
                    str(config_file),
                    "--home",
                    str(root),
                    "--local-config",
                    str(root / "local.yaml"),
                    "--raw-sample-ids",
                    "",
                    "--json",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(2, process.returncode, process.stderr)
            self.assertIn("--raw-sample-ids must not be empty", process.stderr)
            self.assertFalse((paths.runs / config["name"]).exists())

    def test_generation_has_minimal_outputs_and_models_before_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, config = fixture(Path(tmp))
            config["output"]["save_qdq_models"] = True
            with patch(
                "opennpu_quant.runner._ModelRun._evaluate",
                side_effect=RuntimeError("evaluation failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                    generate(config, paths)
            # FP32 fails first. Retry only INT8 to check publication before its failed evaluation.
            config["conditions"].pop("fp32")
            with patch(
                "opennpu_quant.runner._ModelRun._evaluate",
                side_effect=RuntimeError("evaluation failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                    generate(config, paths, force=True)
            out = paths.runs / config["name"]
            self.assertTrue((out / "models/toy/pot_rtn/network.onnx").is_file())
            self.assertEqual(
                {p.name for p in out.iterdir()},
                {".lock", ".state", "models", "results.csv", "settings.json"},
            )
            with patch(
                "opennpu_quant.runner.calibrate",
                side_effect=AssertionError("must reuse"),
            ):
                value = generate(config, paths)
            self.assertEqual(value["rows"][0]["state"], "completed")
            recorded = read_json(out / "settings.json")["models"]["toy"]
            self.assertEqual(recorded["recipe"]["name"], "toy")
            self.assertIn("network", recorded["source_graphs"])
            self.assertTrue(recorded["network_input_map"])
            self.assertEqual(
                recorded["generated_networks"]["pot_rtn"],
                "models/toy/pot_rtn/network.onnx",
            )

    # The CPU identity records thread variables, and main() sets unset ones to 1.
    # Fix them for the API and CLI calls alike, independent of test order.
    @patch.dict(os.environ, {key: "1" for key in THREAD_VARIABLES})
    def test_plain_evaluation_reuse_report_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, cfg, cat = evaluation_fixture(root)
            with patch(
                "opennpu_quant.runner.calibrate",
                side_effect=AssertionError("no training"),
            ):
                value = evaluate_config(cfg, cat, root, paths)
            self.assertEqual(value["execution"]["completed"], 2)
            self.assertEqual(value["execution"]["failed"], 0)
            self.assertEqual(value["mean_recovery"]["pot_rtn"], 50)
            out = paths.runs / cfg["name"]
            self.assertEqual(
                {p.name for p in out.iterdir()},
                {".lock", ".state", "results.csv", "settings.json"},
            )
            self.assertFalse(list(out.rglob("*.prediction.json")))
            with patch(
                "opennpu_quant.evaluation.loop.evaluate",
                side_effect=AssertionError("no inference"),
            ):
                reused = evaluate_config(cfg, cat, root, paths)
                self.assertEqual(reused["execution"]["completed"], 2)
                self.assertEqual(report(out)["rows"][1]["recovery_pct"], 50)
                self.assertFalse((out / "summary.json").exists())
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                main(
                    [
                        "run",
                        "-c",
                        str(root / "evaluate.yaml"),
                        "--home",
                        str(root),
                        "--local-config",
                        str(root / "local.yaml"),
                        "--json",
                    ]
                )
            self.assertEqual(json.loads(stdout.getvalue())["execution"]["failed"], 0)

    def test_separate_report_folder_carries_its_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, cfg, cat = evaluation_fixture(root)
            evaluate_config(cfg, cat, root, paths)
            out = paths.runs / cfg["name"]
            whole = (out / "settings.json").read_bytes()
            tables = root / "tables"
            report(out, output=tables, conditions=["fp32"])
            self.assertEqual(
                {".lock", "results.csv", "settings.json"},
                {p.name for p in tables.iterdir()},
            )
            run_settings = json.loads(whole)
            selected = read_json(tables / "settings.json")
            for key in ("measurements", "environment_diagnostics"):
                self.assertEqual(
                    [("toy", "fp32")],
                    [(r["model"], r["condition"]) for r in selected[key]],
                )
                self.assertEqual(run_settings[key][0], selected[key][0])
            derived = {"measurements", "environment_diagnostics", "mean_recovery"}
            self.assertEqual(
                {k: v for k, v in run_settings.items() if k not in derived},
                {k: v for k, v in selected.items() if k not in derived},
            )
            # The run folder's settings.json still describes the whole run.
            self.assertEqual(whole, (out / "settings.json").read_bytes())
            self.assertEqual(2, len(run_settings["measurements"]))

    def test_memory_capacity_changes_reuse_completed_results_and_record_new_settings(
        self,
    ):
        from copy import deepcopy
        from opennpu_quant.public_run import _prepare_evaluation
        from opennpu_quant.models.spec import ModelSpec

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, config, catalog = evaluation_fixture(root)
            initial = evaluate_config(config, catalog, root, paths)
            self.assertEqual(2, initial["execution"]["completed"])
            state = paths.runs / config["name"] / ".state"
            files = {p: p.read_bytes() for p in state.glob("toy/*/*.json")}
            for capacity in (6 * 1024**3, 8 * 1024**3):
                changed = deepcopy(catalog)
                for variant in changed["models"]["toy"]["variants"].values():
                    variant["runtime"] = dict(gpu_mem_limit=capacity)
                definition = changed["models"]["toy"]
                actual = _prepare_evaluation(
                    ModelSpec(definition["recipe"]),
                    definition,
                    definition["variants"]["fp32"],
                    config,
                    root,
                    paths,
                    False,
                )
                stored = read_json(state / "toy/fp32/result.json")
                self.assertEqual(stored["evidence"]["identity"], actual.identity)
                with patch(
                    "opennpu_quant.public_run._evaluate_prepared",
                    side_effect=AssertionError("must reuse"),
                ):
                    result = evaluate_config(config, changed, root, paths)
                self.assertEqual(2, result["execution"]["completed"])
                status = read_json(state / "status.json")["models"]["toy"]["conditions"]
                self.assertEqual({"reused"}, {row["state"] for row in status.values()})
                saved = read_json(state / "settings.json")["models"]["toy"]["variants"]
                self.assertEqual(
                    {capacity}, {v["runtime"]["gpu_mem_limit"] for v in saved.values()}
                )
                self.assertEqual(files, {p: p.read_bytes() for p in files})

    def test_each_run_warns_once_about_obsolete_metrics_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, config = fixture(root)
            folder = paths.runs / config["name"]
            folder.mkdir(parents=True)
            (folder / "metrics.csv").write_text("run_id,model\n")
            with self.assertLogs("opennpu_quant.results", "WARNING") as logs:
                generate(config, paths, formats="csv")
            self.assertEqual(1, len(logs.records))
            self.assertIn(str(folder / "metrics.csv"), logs.records[0].getMessage())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, config, catalog = evaluation_fixture(root)
            folder = paths.runs / config["name"]
            folder.mkdir(parents=True)
            (folder / "metrics.tsv").write_text("run_id\tmodel\n")
            # Each run in one process warns again, including a reusing rerun.
            for _ in range(2):
                with self.assertLogs("opennpu_quant.results", "WARNING") as logs:
                    value = evaluate_config(config, catalog, root, paths)
                self.assertEqual(2, value["execution"]["completed"])
                self.assertEqual(1, len(logs.records))
                self.assertIn(str(folder / "metrics.tsv"), logs.records[0].getMessage())
            self.assertEqual("run_id\tmodel\n", (folder / "metrics.tsv").read_text())

    def test_changed_expected_count_cannot_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, cfg, cat = evaluation_fixture(root)
            evaluate_config(cfg, cat, root, paths)
            p = paths.data / "imagenet/manifest.json"
            data = read_json(p)
            data["expected_samples"] = 3
            atomic_json(p, data)
            result = evaluate_config(cfg, cat, root, paths)
            self.assertEqual(result["execution"]["failed"], 2)
            self.assertTrue(all(r["state"] == "failed" for r in result["rows"]))
            self.assertIsNone(result["mean_recovery"]["pot_rtn"])

    def test_missing_model_restores_and_evaluation_records_recover(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, cfg, cat = evaluation_fixture(root)
            p = root / cat["models"]["toy"]["variants"]["pot_rtn"]["network"]["path"]
            backup = p.with_suffix(".backup")
            p.rename(backup)
            result = evaluate_config(cfg, cat, root, paths)
            self.assertEqual(result["execution"]["pending"], 1)
            backup.rename(p)
            result = evaluate_config(cfg, cat, root, paths)
            self.assertEqual(result["execution"]["completed"], 2)
            complete = paths.runs / cfg["name"] / ".state/toy/pot_rtn/complete.json"
            complete.write_text("{broken")
            result = evaluate_config(cfg, cat, root, paths)
            self.assertEqual(result["execution"]["failed"], 0)
            self.assertTrue(result["rows"][1]["complete"])
            archived = list(
                (complete.parents[2] / ".evaluations").glob(
                    "*/toy/pot_rtn/complete.json"
                )
            )
            self.assertEqual(["{broken"], [p.read_text() for p in archived])

    def test_wrong_mode_does_not_add_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, cfg, cat = evaluation_fixture(root)
            evaluate_config(cfg, cat, root, paths)
            old = {
                str(p.relative_to(root)): sha256(p)
                for p in (paths.runs / cfg["name"]).rglob("*")
                if p.is_file()
            }
            _, generation = fixture(root / "other")
            generation["name"] = cfg["name"]
            with self.assertRaisesRegex(ValueError, "different execution format"):
                generate(generation, paths)
            self.assertEqual(
                old,
                {
                    str(p.relative_to(root)): sha256(p)
                    for p in (paths.runs / cfg["name"]).rglob("*")
                    if p.is_file()
                },
            )

    def test_catalog_validation_and_no_public_reproduce(self):
        from opennpu_quant.evaluation_config import load

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, cfg, cat = evaluation_fixture(root)
            load(root / "evaluate.yaml")
            cat["models"]["toy"]["variants"]["fp32"]["network"]["path"] = (
                "../escape.onnx"
            )
            atomic_json(root / "catalog.json", cat)
            with self.assertRaisesRegex(ValueError, "escapes"):
                load(root / "evaluate.yaml")
        from opennpu_quant.cli import build_parser

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["reproduce", "run"])

    def test_reviewed_integrity_and_scope_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, cfg, cat = evaluation_fixture(root)
            cfg["evaluation"]["limit"] = 3
            cfg["output"]["save_predictions"] = True
            first = evaluate_config(cfg, cat, root, paths)
            self.assertEqual(first["execution"]["completed"], 2)
            self.assertTrue(all(r["full_evaluation"] for r in first["rows"]))
            directory = paths.runs / cfg["name"] / ".state/toy/pot_rtn"
            (directory / "predictions.jsonl.gz").unlink()
            rows = report(paths.runs / cfg["name"])["rows"]
            self.assertEqual(rows[1]["state"], "unverified")
            self.assertIsNone(rows[1]["recovery_pct"])
            result = evaluate_config(cfg, cat, root, paths)
            self.assertEqual(result["execution"]["failed"], 0)
            self.assertEqual(result["execution"]["pending"], 0)
            self.assertTrue((directory / "predictions.jsonl.gz").is_file())
            atomic_json(directory / "complete.json", {})
            self.assertEqual(
                evaluate_config(cfg, cat, root, paths)["execution"]["failed"], 0
            )

    def test_generation_rejected_config_preserves_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, cfg = fixture(root)
            generate(cfg, paths)
            stored = paths.runs / cfg["name"] / ".state/settings.json"
            previous = stored.read_bytes()
            cfg["calibration"]["samples"]["imagenet"] = 1
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                generate(cfg, paths)
            self.assertEqual(previous, stored.read_bytes())

    def test_all_candidates_preserve_fp32_alias(self):
        from opennpu_quant.evaluation_config import select

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, cfg, cat = evaluation_fixture(root)
            row = cat["models"]["toy"]
            row["variants"]["fp32_graph"] = row["variants"].pop("fp32")
            row["defaults"]["fp32"] = "fp32_graph"
            cfg["conditions"] = "all"
            self.assertIn("fp32", select(cfg, cat)["toy"])
            result = evaluate_config(cfg, cat, root, paths)
            self.assertEqual(result["execution"]["completed"], 2)
            self.assertEqual(result["mean_recovery"]["pot_rtn"], 50)

    def test_all_candidates_do_not_resolve_default_aliases(self):
        from copy import deepcopy
        from opennpu_quant.evaluation_config import select

        with tempfile.TemporaryDirectory() as tmp:
            _, cfg, cat = evaluation_fixture(Path(tmp))
            row = cat["models"]["toy"]
            old = row["variants"]["pot_rtn"]
            new = deepcopy(old)
            new["scheme"]["percentile"] = 99.9
            row["variants"]["pot_rtn_p99_9"] = new
            row["defaults"]["pot_rtn"] = "pot_rtn_p99_9"
            cfg["conditions"] = "all"
            selected = select(cfg, cat)["toy"]
            self.assertEqual(len(selected), 3)
            self.assertEqual(selected["pot_rtn"], old)
            self.assertEqual(selected["pot_rtn_p99_9"], new)
            cfg["conditions"] = "defaults"
            self.assertEqual(select(cfg, cat)["toy"]["pot_rtn"], new)

    def test_bad_format_does_not_change_completed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, cfg, cat = evaluation_fixture(root)
            evaluate_config(cfg, cat, root, paths)
            output = paths.runs / cfg["name"]
            before = {
                str(p.relative_to(output)): sha256(p)
                for p in output.rglob("*")
                if p.is_file()
            }
            with self.assertRaisesRegex(ValueError, "Supported report formats"):
                evaluate_config(cfg, cat, root, paths, formats="bogus")
            self.assertEqual(
                before,
                {
                    str(p.relative_to(output)): sha256(p)
                    for p in output.rglob("*")
                    if p.is_file()
                },
            )

    def test_generated_model_is_republished_on_verified_reuse(self):
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            paths, cfg = fixture(Path(tmp))
            cfg["output"]["save_qdq_models"] = True
            generate(cfg, paths)
            output = paths.runs / cfg["name"]
            network = output / "models/toy/pot_rtn/network.onnx"
            digest = sha256(network)
            shutil.rmtree(output / "models")
            with (
                patch(
                    "opennpu_quant.runner.calibrate",
                    side_effect=AssertionError("recalibrated"),
                ),
                patch(
                    "opennpu_quant.runner.evaluate",
                    side_effect=AssertionError("reevaluated"),
                ),
            ):
                generate(cfg, paths)
            self.assertEqual(sha256(network), digest)

    def test_list_beyond_expected_count_is_not_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, cfg, cat = evaluation_fixture(root)
            manifest = paths.data / "imagenet/manifest.json"
            data = read_json(manifest)
            data["expected_samples"] = 1
            atomic_json(manifest, data)
            cfg["evaluation"]["limit"] = 1
            result = evaluate_config(cfg, cat, root, paths)
            self.assertEqual(result["execution"]["failed"], 2)
            self.assertFalse(any(r["full_evaluation"] for r in result["rows"]))

    def test_malformed_catalog_values_have_clear_errors(self):
        from copy import deepcopy
        from opennpu_quant.evaluation_config import load, select

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, cfg, original = evaluation_fixture(root)
            for scheme in ([], {"extra": True}, {"method": []}):
                cat = deepcopy(original)
                cat["models"]["toy"]["variants"]["pot_rtn"]["scheme"] = scheme
                atomic_json(root / "catalog.json", cat)
                with self.assertRaises(ValueError):
                    load(root / "evaluate.yaml")
            for aliases in (True, [], {"scores": 1}, {"": "logits"}, {"scores": ""}):
                cat = deepcopy(original)
                cat["models"]["toy"]["output_aliases"] = aliases
                atomic_json(root / "catalog.json", cat)
                with self.assertRaisesRegex(ValueError, "output_aliases"):
                    load(root / "evaluate.yaml")
            cat = deepcopy(original)
            cat["models"]["toy"]["output_aliases"] = {"scores": "logits"}
            atomic_json(root / "catalog.json", cat)
            self.assertEqual(load(root / "evaluate.yaml")[1], cat)
            for key in ("models", "conditions"):
                malformed = deepcopy(cfg)
                malformed[key] = [[]]
                with self.assertRaises(ValueError):
                    select(malformed, original)

    def test_public_lfw_counts_views_and_rejects_partial_requests(self):
        import onnx
        import numpy as np
        from opennpu_quant.paths import Paths
        from opennpu_quant.models.spec import ModelSpec
        from tests.dataset_fixtures import lfw_assets
        from tests.test_lfw import full_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = lfw_assets(root)
            manifest, pair_hash = full_manifest(assets)
            recipe = ModelSpec.load("mobilefacenet_relu_foamliu").recipe
            recipe["input"]["shape"] = [1, 3, 4, 4]
            recipe["preprocess"] = dict(
                kind="center_crop",
                interpolation="bilinear",
                resize=4,
                crop=4,
                mean=[0.1] * 3,
                std=[1.0] * 3,
            )
            model = onnx.load(assets / "model.onnx")
            model.graph.output[0].name = "embedding"
            model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 128
            model.graph.node[-1].output[0] = "embedding"
            model.graph.initializer[0].CopyFrom(
                onnx.numpy_helper.from_array(
                    np.arange(384, dtype=np.float32).reshape(3, 128), "weight"
                )
            )
            onnx.save(model, assets / "model.onnx")
            name = recipe["name"]
            catalog = dict(
                schema_version=1,
                models={
                    name: dict(
                        recipe=recipe,
                        variants={
                            "fp32": dict(
                                network=dict(
                                    path=str((assets / "model.onnx").relative_to(root)),
                                    sha256=sha256(assets / "model.onnx"),
                                ),
                                scheme=None,
                            )
                        },
                        defaults={"fp32": "fp32"},
                    )
                },
            )
            paths = Paths.resolve(home=root / "public")
            atomic_json(
                paths.data / "lfw_mobilefacenet/manifest.json", read_json(manifest)
            )
            cfg = dict(
                name="lfw",
                mode="evaluate",
                models=[name],
                conditions="defaults",
                runtime=dict(device="cpu"),
                evaluation=dict(limit=6000),
                output={},
            )
            with patch("opennpu_quant.data.lfw.PAIR_HASH", pair_hash):
                result = evaluate_config(cfg, catalog, root, paths)
            self.assertEqual(result["execution"]["completed"], 1, result)
            row = result["rows"][0]
            self.assertEqual(row["samples"], 6)
            self.assertTrue(row["full_evaluation"])
            details = json.loads(row["metric_details_json"])
            self.assertEqual(details["pair_count"], 6000)
            cfg["evaluation"]["limit"] = 3
            with patch("opennpu_quant.public_run.run_directory_lock") as lock:
                with self.assertRaisesRegex(ValueError, "full 6000"):
                    evaluate_config(cfg, catalog, root, paths)
            lock.assert_not_called()

    def test_report_lock_spans_public_file_publication(self):
        import os
        import subprocess
        import sys
        import time

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, cfg, cat = evaluation_fixture(root)
            evaluate_config(cfg, cat, root, paths)
            output = paths.runs / cfg["name"]
            ready = root / "ready"
            release = root / "release"
            script = """
import sys,time
from pathlib import Path
from opennpu_quant import results
from opennpu_quant.report import report
ready,release=map(Path,sys.argv[2:])
original=results._text
def pause(*args,**kwargs):
    ready.touch()
    deadline=time.monotonic()+20
    while not release.exists():
        if time.monotonic()>deadline: raise RuntimeError('parent did not release publication')
        time.sleep(.01)
    return original(*args,**kwargs)
results._text=pause
report(sys.argv[1])
"""
            env = dict(
                os.environ,
                PYTHONPATH=os.pathsep.join(
                    str(Path(p).resolve()) for p in sys.path if p
                ),
            )
            first = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    script,
                    str(output),
                    str(ready),
                    str(release),
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 20
                while (
                    not ready.exists()
                    and first.poll() is None
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                self.assertTrue(ready.exists(), "report did not reach publication")
                second = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "opennpu_quant",
                        "report",
                        str(output),
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(second.returncode, 2, second.stderr)
                self.assertIn("another process owns", second.stderr.lower())
            finally:
                release.touch()
                stdout, stderr = first.communicate(timeout=20)
            self.assertEqual(first.returncode, 0, stderr)
            self.assertTrue(
                all(r["state"] == "completed" for r in report(output)["rows"])
            )


class PublicRunExitTests(unittest.TestCase):
    def cli(self, root, *options, code=0, config=None, json_output=True):
        env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p),
        )
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "opennpu_quant",
                "run",
                "-c",
                str(config or root / "evaluate.yaml"),
                "--home",
                str(root),
                "--local-config",
                str(root / "local.yaml"),
                "--device",
                "cpu",
                "--limit",
                "1",
                *(["--json"] if json_output else []),
                *options,
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        if code == 2 or not json_output:
            return result
        return json.loads(result.stdout)

    def test_pending_exit_preserves_results_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, config, catalog = evaluation_fixture(root)
            output = paths.runs / config["name"]
            manifest = paths.data / "imagenet/manifest.json"
            backup = manifest.with_suffix(".backup")
            manifest.rename(backup)
            value = self.cli(root, "--dry-run", code=1)
            self.assertEqual(value["validation"], dict(ready=0, pending=2, failed=0))
            self.assertFalse(output.exists())
            value = self.cli(root, code=1)
            self.assertEqual(
                value["execution"],
                dict(state="pending", completed=0, pending=2, failed=0),
            )
            self.assertTrue((output / "results.csv").is_file())
            self.assertTrue((output / "settings.json").is_file())
            self.assertEqual(
                read_json(output / ".state/status.json")["state"], "pending"
            )
            self.assertFalse(list(output.rglob("complete.json")))
            console = self.cli(root, code=1, json_output=False)
            self.assertIn("pending", console.stdout)
            backup.rename(manifest)

            network = (
                root
                / catalog["models"]["toy"]["variants"]["pot_rtn"]["network"]["path"]
            )
            backup = network.with_suffix(".backup")
            network.rename(backup)
            value = self.cli(root, "--dry-run", code=1)
            self.assertEqual(value["validation"], dict(ready=1, pending=1, failed=0))
            value = self.cli(root, code=1)
            self.assertEqual(
                value["execution"],
                dict(state="pending", completed=1, pending=1, failed=0),
            )
            proof = output / ".state/toy/fp32/complete.json"
            saved = (sha256(proof), proof.stat().st_mtime_ns)
            self.assertFalse((proof.parent.parent / "pot_rtn/complete.json").exists())
            backup.rename(network)

            value = self.cli(root, "--dry-run")
            self.assertEqual(value["validation"], dict(ready=2, pending=0, failed=0))
            value = self.cli(root)
            self.assertEqual(
                value["execution"],
                dict(state="completed", completed=2, pending=0, failed=0),
            )
            self.assertEqual(
                [r["state"] for r in value["rows"]], ["reused", "completed"]
            )
            self.assertTrue(
                all(r["samples"] == 1 and not r["complete"] for r in value["rows"])
            )
            self.assertEqual(saved, (sha256(proof), proof.stat().st_mtime_ns))
            value = self.cli(root)
            self.assertTrue(all(r["state"] == "reused" for r in value["rows"]))

    def test_failed_exit_keeps_successful_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, config, catalog = evaluation_fixture(root)
            variants = catalog["models"]["toy"]["variants"]
            network = root / variants["pot_rtn"]["network"]["path"]
            network.write_bytes(network.read_bytes() + b"changed")
            value = self.cli(root, "--dry-run", code=1)
            self.assertEqual(value["validation"], dict(ready=1, pending=0, failed=1))
            self.assertFalse((paths.runs / config["name"]).exists())
            value = self.cli(root, code=1)
            self.assertEqual(
                value["execution"],
                dict(state="failed", completed=1, pending=0, failed=1),
            )
            self.assertEqual(
                [r["state"] for r in value["rows"]], ["completed", "failed"]
            )
            output = paths.runs / config["name"]
            self.assertTrue((output / ".state/toy/fp32/complete.json").is_file())
            self.assertFalse((output / ".state/toy/pot_rtn/complete.json").exists())
            network = root / variants["fp32"]["network"]["path"]
            network.write_bytes(network.read_bytes() + b"changed")
            value = self.cli(root, "--dry-run", code=1)
            self.assertEqual(value["validation"], dict(ready=0, pending=0, failed=2))
            value = self.cli(root, code=1)
            self.assertEqual(value["execution"]["failed"], 2)
            self.assertTrue(all(r["state"] == "failed" for r in value["rows"]))
            self.assertTrue((output / "results.csv").is_file())
            self.assertEqual(
                read_json(output / ".state/status.json")["state"], "failed"
            )
            error = self.cli(root, "--limit", "0", code=2)
            self.assertIn("limit", error.stderr)

    def test_native_ort_failure_is_recorded_and_later_conditions_run(self):
        import copy
        import onnx
        from tests.ort.test_errors import runtime_error_model

        for phase in ("session", "inference"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths, config, catalog = evaluation_fixture(root)
                row = catalog["models"]["toy"]
                row["variants"]["later"] = copy.deepcopy(row["variants"]["fp32"])
                row["defaults"]["later"] = "later"
                broken = root / "broken.onnx"
                model = runtime_error_model()
                if phase == "session":
                    model.graph.node[0].op_type = "MissingOperator"
                onnx.save_model(model, broken)
                row["variants"]["pot_rtn"]["network"] = dict(
                    path=broken.name, sha256=sha256(broken)
                )
                atomic_json(root / "catalog.json", catalog)
                value = self.cli(root, code=1)
                self.assertEqual(
                    value["execution"],
                    dict(state="failed", completed=2, pending=0, failed=1),
                )
                out = paths.runs / config["name"]
                state = read_json(out / ".state/status.json")["models"]["toy"][
                    "conditions"
                ]
                self.assertIn("ONNX Runtime", state["pot_rtn"]["error"]["message"])
                self.assertEqual(state["pot_rtn"]["state"], "failed")
                self.assertFalse((out / ".state/toy/pot_rtn/complete.json").exists())
                self.assertTrue((out / ".state/toy/fp32/complete.json").is_file())
                self.assertTrue((out / ".state/toy/later/complete.json").is_file())
                self.assertEqual(
                    [r["state"] for r in report(out)["rows"]],
                    ["completed", "failed", "completed"],
                )
                # Only completed evaluations claim a functional pass.
                self.assertNotIn("record", state["pot_rtn"])
                passed = {
                    r["condition"]: r["functional_passed"]
                    for r in current_records(out)[0]
                }
                self.assertEqual(dict(fp32=True, pot_rtn=False, later=True), passed)
                # A status file written before this rule is corrected when read.
                status = read_json(out / ".state/status.json")
                status["models"]["toy"]["conditions"]["pot_rtn"]["record"] = {
                    "functional_passed": True
                }
                atomic_json(out / ".state/status.json", status)
                failed = [
                    r for r in current_records(out)[0] if r["condition"] == "pot_rtn"
                ]
                self.assertFalse(failed[0]["functional_passed"])
                # A repeated failure still reuses the two successful conditions.
                value = self.cli(root, code=1)
                self.assertEqual(
                    [r["state"] for r in value["rows"]], ["reused", "failed", "reused"]
                )
                console = self.cli(root, code=1, json_output=False)
                self.assertNotIn("Traceback", console.stderr)
                self.assertIn("failed", console.stdout)

    @patch.dict(os.environ, {key: "1" for key in THREAD_VARIABLES})
    def test_profiled_sample_mismatch_fails_only_its_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, config, catalog = evaluation_fixture(root)
            with first_profile_drifts() as calls:
                value = evaluate_config(config, catalog, root, paths)
            self.assertEqual(2, len(calls))
            self.assertEqual(
                value["execution"],
                dict(state="failed", completed=1, pending=0, failed=1),
            )
            out = paths.runs / config["name"]
            state = read_json(out / ".state/status.json")["models"]["toy"]["conditions"]
            self.assertEqual("failed", state["fp32"]["state"])
            self.assertIn("Profiled sample output", state["fp32"]["error"]["message"])
            self.assertIn(
                "max absolute difference 1", state["fp32"]["error"]["message"]
            )
            self.assertEqual("completed", state["pot_rtn"]["state"])
            self.assertFalse((out / ".state/toy/fp32/complete.json").exists())
            self.assertTrue((out / ".state/toy/pot_rtn/complete.json").is_file())
            # The command reports the failed condition instead of a traceback.
            stdout = io.StringIO()
            with (
                first_profile_drifts(),
                contextlib.redirect_stdout(stdout),
                self.assertRaises(SystemExit) as exited,
            ):
                main(
                    [
                        "run",
                        "-c",
                        str(root / "evaluate.yaml"),
                        "--home",
                        str(root),
                        "--local-config",
                        str(root / "local.yaml"),
                        "--json",
                    ]
                )
            self.assertEqual(1, exited.exception.code)
            self.assertEqual(
                ["failed", "reused"],
                [r["state"] for r in json.loads(stdout.getvalue())["rows"]],
            )

    def test_evaluation_profiles_are_validated_before_any_work(self):
        import copy

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, config, catalog = evaluation_fixture(root)
            toy2 = copy.deepcopy(catalog["models"]["toy"])
            toy2["recipe"]["name"] = "toy2"
            catalog["models"]["toy2"] = toy2
            for name, models, profiles, message in (
                ("typo", ["toy"], {"toyy": "numpy_topk_v1"}, "catalog: toyy"),
                (
                    "late",
                    ["toy", "toy2"],
                    {"toy2": "yolov7_face_clip_all_v1"},
                    "not verified for toy2",
                ),
                ("type", ["toy"], {"toy": 3}, "must be a profile name"),
            ):
                cfg = dict(
                    config,
                    name=name,
                    models=models,
                    evaluation=dict(limit=None, profiles=profiles),
                )
                with (
                    self.subTest(name=name),
                    patch(
                        "opennpu_quant.evaluation.loop.evaluate",
                        side_effect=AssertionError("no inference"),
                    ),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    evaluate_config(cfg, catalog, root, paths)
                # Rejected before the lock: no partial run or running status.
                self.assertFalse((paths.runs / name).exists())
            # A valid entry for an unselected catalog model serves other subsets.
            cfg = dict(
                config,
                name="subset",
                models=["toy"],
                evaluation=dict(limit=None, profiles={"toy2": "stable_topk_v1"}),
            )
            value = evaluate_config(cfg, catalog, root, paths)
            self.assertEqual(value["execution"]["completed"], 2)

    def test_generation_native_ort_failure_has_command_error_exit(self):
        import onnx
        from tests.ort.test_errors import runtime_error_model

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, config = fixture(root)
            config["conditions"] = {"fp32": {}}
            onnx.save_model(runtime_error_model(), paths.models / "toy/network.onnx")
            atomic_json(root / "generation.json", config)
            result = self.cli(root, config=root / "generation.json", code=2)
            self.assertIn("ONNX Runtime RuntimeException", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(result.stdout, "")
            out = paths.runs / config["name"]
            self.assertEqual(read_json(out / ".state/status.json")["state"], "failed")
            self.assertFalse(list(out.rglob("complete.json")))

    @patch.dict(os.environ, {key: "1" for key in THREAD_VARIABLES})
    def test_generation_profiled_sample_mismatch_has_command_error_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, config = fixture(root)
            config["conditions"] = {"fp32": {}}
            atomic_json(root / "generation.json", config)
            stderr = io.StringIO()
            with (
                first_profile_drifts(),
                contextlib.redirect_stderr(stderr),
                self.assertRaises(SystemExit) as exited,
            ):
                main(
                    [
                        "run",
                        "-c",
                        str(root / "generation.json"),
                        "--home",
                        str(root),
                        "--local-config",
                        str(root / "local.yaml"),
                        "--device",
                        "cpu",
                        "--limit",
                        "1",
                    ]
                )
            self.assertEqual(2, exited.exception.code)
            self.assertIn("Profiled sample output", stderr.getvalue())
            out = paths.runs / config["name"]
            self.assertEqual(read_json(out / ".state/status.json")["state"], "failed")
            self.assertFalse(list(out.rglob("complete.json")))

    def test_generation_lfw_dry_run_validates_limit_without_assets(self):
        for models in (
            "mobilefacenet_relu_foamliu",
            "resnet18,mobilefacenet_relu_foamliu",
        ):
            with self.subTest(models=models), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "local.yaml").write_text("{}")
                result = self.cli(
                    root,
                    "--dry-run",
                    "--models",
                    models,
                    "--conditions",
                    "fp32",
                    "--limit",
                    "3",
                    config="cnn30",
                    code=2,
                )
                self.assertIn("6000", result.stderr)
                for limit in ("6000", "full"):
                    value = self.cli(
                        root,
                        "--dry-run",
                        "--models",
                        models,
                        "--conditions",
                        "fp32",
                        "--limit",
                        limit,
                        config="cnn30",
                    )
                    self.assertNotIn("execution", value)
                    self.assertNotIn("validation", value)
                self.assertFalse((root / "runs").exists())

    def test_generation_dry_run_does_not_require_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "local.yaml").write_text("{}")
            value = self.cli(
                root,
                "--dry-run",
                "--models",
                "resnet18",
                "--conditions",
                "fp32,pot_rtn",
                config="core10",
            )
            self.assertEqual(value["config"]["models"], ["resnet18"])
            self.assertIn("percentiles", value)
            self.assertNotIn("validation", value)
            self.assertNotIn("execution", value)
            self.assertFalse((root / "runs").exists())
