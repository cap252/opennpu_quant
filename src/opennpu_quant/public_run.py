"""Public run storage and existing-ONNX evaluation without replay evidence."""

from copy import deepcopy
from pathlib import Path
import contextlib
import gzip
from uuid import uuid4

from ._identity import code_identity
from ._io import atomic_json, read_json, object_hash, sha256, now
from ._locking import run_directory_lock, report_output_lock
from .results import export_records, report_policy, summarize, normalize_formats


def _guard(root, mode):
    if any(
        (root / n).exists() for n in ("effective_config.json", "run.json", "prepared")
    ):
        raise ValueError(
            "Output contains an older execution format. Use a new output directory"
        )
    stored = root / ".state/settings.json"
    if stored.exists() and read_json(stored).get("mode") != mode:
        raise ValueError(
            "Output directory contains a different execution format. Use a new output directory"
        )


def _settings(root, value):
    atomic_json(root / ".state/settings.json", value)


def _public_settings(root, records, summary, destination=None):
    """Write settings.json for these records next to the tables that report them."""
    value = read_json(root / ".state/settings.json")
    value["measurements"] = [
        dict(
            model=r["model"],
            condition=r["condition"],
            state=r["state"],
            samples=r.get("samples"),
            expected_samples=r.get("expected_samples"),
            runtime=r.get("runtime", {}),
            finished_at=r.get("finished_at"),
        )
        for r in records
    ]
    value["environment_diagnostics"] = [
        dict(model=r["model"], condition=r["condition"], **r["environment_diagnostic"])
        for r in summary["rows"]
    ]
    value["mean_recovery"] = summary["mean_recovery"]
    atomic_json(Path(destination or root) / "settings.json", value)


def _generation_records(root):
    from .report import _record

    state = root / ".state"
    config = read_json(state / "effective_config.json")
    status = read_json(state / "status.json")
    policy = report_policy(config.get("report"))
    return [
        _record(state, config, status, policy, model, condition)
        for model in config["models"]
        for condition in config["conditions"]
    ], policy


def _completed(directory, identity):
    """One integrity gate for both reuse and reporting, without inference."""
    from .report import verify_artifacts

    try:
        proof = read_json(directory / "complete.json")
        if (
            not isinstance(proof, dict)
            or not isinstance(proof.get("identity"), str)
            or proof["identity"] != identity
            or not verify_artifacts(directory, proof)
        ):
            raise ValueError(
                "Saved evaluation evidence is corrupt or its identity changed"
            )
        record = read_json(directory / "result.json")
        if (
            record["evidence"]["identity"] != identity
            or record["state"] != "completed"
            or type(record["samples"]) is not int
            or record["samples"] <= 0
            or type(record["expected_samples"]) is not int
            or record["complete"] != (record["samples"] == record["expected_samples"])
            or record["full_evaluation"] != record["complete"]
        ):
            raise ValueError("Saved evaluation result is inconsistent")
        from .results import percent

        if not record["metrics"] or record["primary_metric"] not in record["metrics"]:
            raise ValueError("Saved evaluation metrics are missing")
        for metric in record["metrics"].values():
            if percent(metric) is None:
                raise ValueError("Saved evaluation metric is missing")
        return record
    except (OSError, KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError(
            "Saved evaluation evidence is invalid. Use --force or a new output directory"
        ) from error


def _evaluation_records(root):
    state = root / ".state"
    status = read_json(state / "status.json")
    rows = []
    for model, conditions in status["models"].items():
        for condition, current in conditions.items():
            record = deepcopy(current["record"])
            complete = state / model / condition / "complete.json"
            result = complete.with_name("result.json")
            verified = False
            try:
                verified = current["state"] in ("completed", "reused")
                if verified:
                    record = _completed(complete.parent, current["identity"])
            except (OSError, ValueError, KeyError, TypeError):
                verified = False
            if verified:
                record["state"] = current["state"]
                record["evidence"].update(
                    result_hash_verified=True, result_sha256=sha256(result)
                )
            else:
                record.update(
                    state="unverified"
                    if current["state"] in ("completed", "reused")
                    else current["state"],
                    metrics={},
                    functional_passed=False,
                    complete=False,
                    full_evaluation=False,
                )
                record["evidence"]["result_hash_verified"] = False
                record["failure_reason"] = current.get("error")
            rows.append(record)
    return rows, report_policy(None)


def report_locked(
    root,
    *,
    output=None,
    formats=None,
    table_profile=None,
    models=None,
    conditions=None,
    refresh=False,
):
    root = Path(root)
    settings = read_json(root / ".state/settings.json")
    records, policy = (
        _generation_records(root)
        if settings["mode"] == "generate"
        else _evaluation_records(root)
    )
    for requested, key in ((models, "model"), (conditions, "condition")):
        if requested is not None and (
            not requested or set(requested) - {r[key] for r in records}
        ):
            raise ValueError("Unknown or empty " + key + " filter")
    records = [
        r
        for r in records
        if (models is None or r["model"] in models)
        and (conditions is None or r["condition"] in conditions)
    ]
    summary = summarize(records, policy=policy)
    destination = Path(output) if output else root
    with report_output_lock(root, destination):
        export_records(
            records,
            destination,
            formats="csv" if formats is None else formats,
            table_profile=table_profile,
            metadata=dict(kind="public_run", report_policy=policy),
            summary=summary,
            _refresh_existing=refresh,
            _primary_only=True,
        )
        if destination.resolve() != root.resolve():
            # A separate report folder carries the settings and environment
            # diagnostics of the results it contains.
            _public_settings(root, records, summary, destination)
        elif models is None and conditions is None:
            # The run folder's settings.json always describes the whole run.
            _public_settings(root, records, summary)
    if table_profile:
        from .raw_report import build_report

        summary["raw_data"] = build_report(records, profile=table_profile)
    return summary


def _report_options(formats, table_profile):
    normalize_formats("csv" if formats is None else formats)
    if table_profile not in (None, "opennpu_raw"):
        raise ValueError("Unknown table profile: " + str(table_profile))


def _dataset_scope(dataset):
    expected = dataset.info["expected_samples"]
    listed = len(dataset.rows("validation"))
    if type(expected) is not int or expected <= 0 or listed > expected:
        raise ValueError(
            "Prepared validation list exceeds or has an invalid expected sample count"
        )
    return expected


def generate(config, paths, **kwargs):
    from .runner import run, _ModelRun
    from .run_storage import link_model
    from .graph.model import external_data_files

    _report_options(kwargs.get("formats"), kwargs.get("table_profile"))
    root = paths.runs / config["name"]
    connections = {}

    class PublicModelRun(_ModelRun):
        def __init__(self, *args, **options):
            super().__init__(*args, **options)
            _dataset_scope(self.dataset)
            connections[self.name] = dict(
                recipe=self.spec.recipe,
                network_input_map=self.bundle.mapping,
                output_aliases=self.bundle.aliases,
                source_graphs={
                    role: dict(path=str(path.resolve()), identity=self.graphs[role])
                    for role, path in self.bundle.paths.items()
                },
                generated_networks={
                    condition: str(
                        Path("models") / self.name / condition / "network.onnx"
                    )
                    for condition, policy in self.config["conditions"].items()
                    if policy and self.config["output"]["save_qdq_models"]
                },
            )

        def _publish_model(self, source, condition):
            destination = root / "models" / self.name / condition
            link_model(source, destination / "network.onnx")
            for filename, path in external_data_files(source).items():
                link_model(path, destination / filename)
            for role, path in self.bundle.paths.items():
                if role != "network":
                    link_model(path, root / "models" / self.name / path.name)
                    for filename, extra in external_data_files(path).items():
                        link_model(extra, root / "models" / self.name / filename)

        def _generated(self, policy, scheme, directory, force):
            result = super()._generated(policy, scheme, directory, force)
            if self.generated_path is not None:
                self._publish_model(self.generated_path, directory.name)
            return result

        def _reuse(self, directory, condition, identity, policy, current):
            reused = super()._reuse(directory, condition, identity, policy, current)
            if reused and policy and self.config["output"]["save_qdq_models"]:
                self._publish_model(directory / "model.onnx", condition)
            return reused

    with run_directory_lock(root, fixed=False):
        _guard(root, "generate")

        def refresh(*unused):
            # runner publishes effective_config only after accepting the change.
            effective = read_json(root / ".state/effective_config.json")
            _settings(
                root,
                dict(
                    schema_version=1,
                    mode="generate",
                    config=effective,
                    source_identity=code_identity(),
                    models={
                        model: {
                            "source_directory": str((paths.models / model).resolve()),
                            "generated_directory": "models/" + model,
                            **connections.get(model, {}),
                        }
                        for model in effective["models"]
                    },
                ),
            )
            return report_locked(
                root,
                formats=kwargs.get("formats"),
                table_profile=kwargs.get("table_profile"),
                refresh=True,
            )

        return run(
            config,
            paths,
            **kwargs,
            _output=root / ".state",
            _model_runner=PublicModelRun,
            _reporter=refresh,
        )


def _assets(folder, descriptors):
    from .evaluation_config import relative
    from .graph.model import external_data_files, model_identity

    paths, identities = {}, {}
    for role, desc in descriptors.items():
        path = relative(folder, desc["path"])
        if sha256(path) != desc["sha256"]:
            raise ValueError("Model file changed: " + str(path))
        external = external_data_files(path)
        if {n: sha256(p) for n, p in external.items()} != desc.get("external_data", {}):
            raise ValueError("External model weights differ: " + str(path))
        paths[role], identities[role] = path, model_identity(path)
    return paths, identities


def _base_record(name, model, condition, definition, variant):
    from .results import normalize_scheme

    recipe = definition["recipe"]
    return dict(
        run_id=name,
        model=model,
        model_name=recipe["title"],
        condition=condition,
        state="pending",
        origin="standalone_run",
        primary_metric=recipe["metric"],
        metrics={},
        metric_details={},
        functional_passed=False,  # set only by a completed evaluation
        full_evaluation=False,
        complete=False,
        reference=False,
        dataset=recipe["dataset"],
        split="validation",
        model_source=recipe["source"],
        model_metadata=recipe,
        quantization=normalize_scheme(variant["scheme"], variant.get("adaround")),
        evidence=dict(
            result_hash_verified=False, source_binding="recorded_at_measurement"
        ),
    )


def evaluate_config(
    config,
    catalog,
    folder,
    paths,
    *,
    force=False,
    dry_run=False,
    formats=None,
    table_profile=None,
    strict_versions=False,
):
    from .evaluation_config import select
    from .ort.config import OrtConfig
    from .ort.session import environment
    from .models.spec import ModelSpec
    from .models.bundle import ModelBundle
    from .models.adapter import ModelEvaluator
    from .models.evaluation_profiles import apply_profile, identity as profile_identity
    from .data.samples import Dataset, preprocessing_implementation
    from .evaluation.loop import evaluate
    from .evaluation.capture import RawCapture, settings as capture_settings
    from .results import metric_values
    from .data.lfw_protocol import require_full

    _report_options(formats, table_profile)
    selected = select(config, catalog)
    for model in selected:
        if catalog["models"][model]["recipe"]["task"] == "face_verification":
            require_full(limit=config["evaluation"]["limit"])
    root = paths.runs / config["name"]
    requested = dict(
        config=deepcopy(config), models={m: catalog["models"][m] for m in selected}
    )
    requested["config"].pop("source", None)  # location is not measurement identity
    signature = object_hash(requested)
    device = config["runtime"].get("device", "cuda:0")
    limit = config["evaluation"]["limit"]
    if not dry_run:
        guard = run_directory_lock(root, fixed=False)
    else:
        guard = contextlib.nullcontext()
    with guard:
        _guard(root, "evaluate")
        saved = root / ".state/settings.json"
        if (
            saved.exists()
            and read_json(saved)["selection_identity"] != signature
            and not force
        ):
            raise ValueError(
                "Evaluation settings changed. Use a new output directory or --force"
            )
        status = dict(
            state="running",
            attempt=uuid4().hex,
            models={
                m: {
                    c: dict(
                        state="pending",
                        record=_base_record(
                            config["name"], m, c, catalog["models"][m], v
                        ),
                    )
                    for c, v in rows.items()
                }
                for m, rows in selected.items()
            },
        )

        def publish():
            atomic_json(root / ".state/status.json", status)
            return report_locked(
                root, formats=formats, table_profile=table_profile, refresh=True
            )

        if not dry_run:
            _settings(
                root,
                dict(
                    schema_version=1,
                    mode="evaluate",
                    config=config,
                    models=requested["models"],
                    model_root=str(Path(folder).resolve()),
                    selection_identity=signature,
                    source_identity=code_identity(),
                ),
            )
            publish()
        preparations = []
        summary = dict(rows=[], mean_recovery={})
        for model, variants in selected.items():
            definition = catalog["models"][model]
            override = config["evaluation"].get("profiles", {}).get(model)
            recipe = (
                apply_profile(definition["recipe"], override)
                if override is not None
                else deepcopy(definition["recipe"])
            )
            spec = ModelSpec(recipe)
            if recipe["task"] == "face_verification":
                require_full(limit=limit)
            for condition, variant in variants.items():
                current = status["models"][model][condition]
                directory = root / ".state" / model / condition
                phase = "preparation"
                try:
                    ort = (
                        OrtConfig.cpu(strict_versions=strict_versions)
                        if device == "cpu"
                        else OrtConfig.cuda(
                            int(device.split(":")[1]),
                            memory_limit=variant.get("runtime", {}).get(
                                "gpu_mem_limit", 6 * 1024**3
                            ),
                            strict_versions=strict_versions,
                        )
                    )
                    descriptors = dict(
                        definition.get("host_graphs", {}), network=variant["network"]
                    )
                    graph_paths, graph_ids = _assets(folder, descriptors)
                    bundle = ModelBundle(
                        spec, graph_paths, aliases=definition.get("output_aliases")
                    )
                    dataset = Dataset(
                        paths.data / recipe["dataset"] / "manifest.json",
                        root=paths.datasets.get(recipe["dataset"]),
                    )
                    expected = _dataset_scope(dataset)
                    rows = dataset.rows("validation", limit)
                    pair_scope = recipe["task"] == "face_verification"
                    if pair_scope:
                        require_full(dataset.info, limit)
                    wanted = (
                        expected
                        if limit is None or pair_scope
                        else min(limit, expected)
                    )
                    if (
                        type(expected) is not int
                        or expected <= 0
                        or len(rows) != wanted
                    ):
                        raise ValueError(
                            "Evaluation sample count differs from prepared scope"
                        )
                    impl = preprocessing_implementation(recipe["preprocess"])
                    preprocessing = (
                        object_hash(
                            [
                                recipe["preprocess"],
                                recipe["input"],
                                bundle.mapping,
                                graph_ids.get("preprocess"),
                                impl,
                            ]
                        )
                        if impl
                        else None
                    )
                    evaluation = object_hash(
                        [
                            dataset.identity("validation", limit),
                            expected,
                            len(dataset.rows("validation")),
                            recipe,
                            profile_identity(recipe),
                            preprocessing,
                            {k: v for k, v in graph_ids.items() if k != "network"},
                            environment(ort),
                            code_identity(),
                        ]
                    )
                    identity = object_hash(
                        [evaluation, graph_ids, variant, config["output"]]
                    )
                    current["identity"] = identity
                    preparations.append(
                        dict(model=model, condition=condition, state="ready")
                    )
                    if dry_run:
                        continue
                    complete = directory / "complete.json"
                    phase = "evaluation"
                    if complete.exists() and not force:
                        if not impl:
                            raise ValueError(
                                "Preprocessing identity is unavailable. Use --force or a new output directory"
                            )
                        try:
                            _completed(directory, identity)
                        except ValueError as error:
                            raise ValueError(
                                "Evaluation identity changed or evidence is corrupt. Use a new output directory or --force"
                            ) from error
                        current["state"] = "reused"
                        summary = publish()
                        continue
                    # Keep prior evaluations readable, including failed force attempts.
                    if force and directory.exists():
                        archived = (
                            root
                            / ".state/.evaluations"
                            / status["attempt"]
                            / model
                            / condition
                        )
                        archived.parent.mkdir(parents=True, exist_ok=True)
                        directory.rename(archived)
                    current["state"] = "running"
                    publish()
                    evaluator = ModelEvaluator(bundle, dataset)
                    capture = (
                        RawCapture(evaluator, directory, config["output"])
                        if capture_settings(config["output"])
                        else None
                    )
                    if capture:
                        evaluator = capture
                    directory.mkdir(parents=True, exist_ok=True)
                    stream = (
                        gzip.open(directory / "predictions.jsonl.gz.pending", "wt")
                        if config["output"].get("save_predictions")
                        else None
                    )
                    try:
                        if stream:
                            from .runner import _tee_predictions

                            _tee_predictions(evaluator, stream)
                        values = evaluate(
                            graph_paths["network"],
                            dataset.samples(bundle, limit=limit),
                            evaluator=evaluator,
                            ort=ort,
                            expected_samples=expected,
                            limit=None if limit is None else len(rows),
                        )
                    finally:
                        if stream:
                            stream.close()
                    if recipe["metric"] not in values.metrics:
                        raise ValueError("Primary metric is absent")
                    if capture:
                        capture.finish()
                        capture.publish()
                    if stream:
                        (directory / "predictions.jsonl.gz.pending").replace(
                            directory / "predictions.jsonl.gz"
                        )
                    record = deepcopy(current["record"])
                    fp = definition["variants"][definition["defaults"]["fp32"]][
                        "network"
                    ]
                    record.update(
                        state="completed",
                        functional_passed=True,
                        metrics=metric_values(values.metrics, values.metric_units),
                        metric_details=values.metric_details,
                        samples=values.samples,
                        expected_samples=expected,
                        complete=values.complete,
                        full_evaluation=values.complete,
                        evaluation_identity=evaluation,
                        evaluation_profile=profile_identity(recipe),
                        runtime=values.runtime,
                        finished_at=now(),
                        elapsed_s=values.elapsed_s,
                    )
                    record["evidence"].update(
                        identity=identity,
                        model_identity=fp["sha256"],
                        preprocessing_identity=preprocessing,
                    )
                    atomic_json(directory / "result.json", record)
                    files = {}
                    if capture:
                        files["raw.json"] = sha256(directory / "raw.json")
                    if stream:
                        files["predictions.jsonl.gz"] = sha256(
                            directory / "predictions.jsonl.gz"
                        )
                    atomic_json(
                        complete,
                        dict(
                            identity=identity,
                            result_sha256=sha256(directory / "result.json"),
                            files=files,
                        ),
                    )
                    current["state"] = "completed"
                except (OSError, ValueError, RuntimeError, ImportError) as error:
                    current.update(
                        state="pending"
                        if phase == "preparation"
                        and isinstance(error, FileNotFoundError)
                        else "failed",
                        error=f"{type(error).__name__}: {error}",
                    )
                    if dry_run:
                        preparations.append(
                            dict(
                                model=model,
                                condition=condition,
                                state=current["state"],
                                reason=current["error"],
                            )
                        )
                except BaseException:
                    current["state"] = "interrupted"
                    if not dry_run:
                        status["state"] = "interrupted"
                        publish()
                    raise
                if not dry_run:
                    summary = publish()
        counts = {
            s: sum(
                c["state"] in (s, "reused") if s == "completed" else c["state"] == s
                for rows in status["models"].values()
                for c in rows.values()
            )
            for s in ("completed", "pending", "failed")
        }
        if dry_run:
            return dict(
                output=str(root),
                device=device,
                preparation=preparations,
                validation=dict(
                    ready=sum(p["state"] == "ready" for p in preparations),
                    failed=counts["failed"],
                    pending=sum(p["state"] == "pending" for p in preparations),
                ),
                will_generate_models=False,
            )
        status["state"] = (
            "failed"
            if counts["failed"]
            else "pending"
            if counts["pending"]
            else "completed"
        )
        summary = publish()
        return dict(
            output=str(root),
            device=device,
            rows=summary["rows"],
            mean_recovery=summary["mean_recovery"],
            execution=dict(state=status["state"], **counts),
        )
