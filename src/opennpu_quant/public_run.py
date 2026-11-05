"""Existing-ONNX evaluation runs. Records are stored through run_storage and read
through run_records."""

from copy import deepcopy
from dataclasses import dataclass
from . import run_identity
from pathlib import Path
import contextlib
import gzip
import logging

from ._identity import code_identity, run_config_identity
from ._io import atomic_json, read_json, object_hash, sha256, now
from ._locking import run_directory_lock
from .results import normalize_formats, warn_obsolete_reports
from .run_records import ATTESTED_FILES, reusable_evaluation, base_record
from .run_storage import (
    guard_run,
    archive_condition,
    new_status,
    fail_condition,
    condition_counts,
    finish_status,
    model_state,
    write_completed,
)
from .report import report_from_settings_locked
from .exceptions import (
    condition_state,
    describe,
    error_record,
    expected as expected_error,
)

logger = logging.getLogger(__name__)


def _settings(root, value):
    atomic_json(root / ".state/settings.json", value)


def _report_options(formats, table_profile):
    normalize_formats("csv" if formats is None else formats)
    if table_profile not in (None, "opennpu_raw"):
        raise ValueError("Unknown table profile: " + str(table_profile))


def generate(config, paths, **kwargs):
    from .runner import run

    return run(config, paths, **kwargs)


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


def _selection(config, catalog):
    """Selected catalog variants, their catalog entries and the selection identity."""
    from .evaluation_config import select
    from .data.lfw_protocol import require_full

    selected = select(config, catalog)
    for model in selected:
        if catalog["models"][model]["recipe"]["task"] == "face_verification":
            require_full(limit=config["evaluation"]["limit"])
    models = {m: catalog["models"][m] for m in selected}
    # Keep recorded runtime settings while excluding arena capacity from reuse.
    identity_models = deepcopy(models)
    for definition in identity_models.values():
        definition["variants"] = {
            key: run_identity.provided_variant(variant)
            for key, variant in definition["variants"].items()
        }
    # The catalog location and the GPU index are not measurement identity.
    requested = {k: v for k, v in config.items() if k != "source"}
    return (
        selected,
        models,
        object_hash([run_config_identity(requested), identity_models]),
    )


def _check_saved_selection(root, signature, force):
    """Refuse a different selection in an existing evaluation folder unless forced."""
    guard_run(root, "evaluate")
    saved = root / ".state/settings.json"
    if (
        saved.exists()
        and read_json(saved)["selection_identity"] != signature
        and not force
    ):
        raise ValueError(
            "Evaluation settings changed. Use a new output directory or --force"
        )


def _model_recipe(config, definition, model):
    """The catalog recipe with the config's evaluation profile override, if any."""
    from .models.evaluation_profiles import apply_profile

    override = config["evaluation"].get("profiles", {}).get(model)
    if override is not None:
        return apply_profile(definition["recipe"], override)
    return deepcopy(definition["recipe"])


def _runtime_config(device, variant, strict_versions):
    """ORT settings for ``cpu`` or ``cuda:N`` and the variant's GPU memory limit."""
    from .ort.config import OrtConfig

    if device == "cpu":
        return OrtConfig.cpu(strict_versions=strict_versions)
    return OrtConfig.cuda(
        int(device.split(":")[1]),
        memory_limit=variant.get("runtime", {}).get("gpu_mem_limit", 6 * 1024**3),
        strict_versions=strict_versions,
    )


def _sample_scope(expected, rows, limit, pair_scope):
    """Check that the prepared rows match the requested evaluation scope."""
    wanted = expected if limit is None or pair_scope else min(limit, expected)
    if type(expected) is not int or expected <= 0 or len(rows) != wanted:
        raise ValueError("Evaluation sample count differs from prepared scope")


@dataclass(frozen=True)
class PreparedEvaluation:
    """One condition with its graphs, data and runtime resolved and checked."""

    bundle: object
    dataset: object
    ort: object
    expected: int
    selected_count: int
    preprocessing: str | None
    evaluation: str
    identity: str


def _prepare_evaluation(
    spec, definition, variant, config, folder, paths, strict_versions
):
    """Resolve one condition and compute its identities without running inference."""
    from .ort.session import environment
    from .models.bundle import ModelBundle
    from .models.evaluation_profiles import identity as profile_identity
    from .data.samples import Dataset, preprocessing_implementation, validation_scope
    from .data.lfw_protocol import require_full

    recipe = spec.recipe
    limit = config["evaluation"]["limit"]
    ort = _runtime_config(
        config["runtime"].get("device", "cuda:0"), variant, strict_versions
    )
    descriptors = dict(definition.get("host_graphs", {}), network=variant["network"])
    graph_paths, graph_ids = _assets(folder, descriptors)
    bundle = ModelBundle(spec, graph_paths, aliases=definition.get("output_aliases"))
    dataset = Dataset(
        paths.data / recipe["dataset"] / "manifest.json",
        root=paths.datasets.get(recipe["dataset"]),
    )
    expected = validation_scope(dataset)
    rows = dataset.rows("validation", limit)
    pair_scope = recipe["task"] == "face_verification"
    if pair_scope:
        require_full(dataset.info, limit)
    _sample_scope(expected, rows, limit, pair_scope)
    impl = preprocessing_implementation(recipe["preprocess"])
    preprocessing = run_identity.preprocessing(
        recipe, bundle.mapping, graph_ids.get("preprocess"), impl
    )
    evaluation = run_identity.provided_evaluation(
        inputs=dataset.identity("validation", limit),
        expected=expected,
        listed=len(dataset.rows("validation")),
        recipe=recipe,
        profile=profile_identity(recipe),
        preprocessing=preprocessing,
        graphs=graph_ids,
        runtime=environment(ort),
        code=code_identity(),
    )
    identity = run_identity.provided_condition(
        evaluation, graph_ids, variant, config["output"]
    )
    return PreparedEvaluation(
        bundle, dataset, ort, expected, len(rows), preprocessing, evaluation, identity
    )


def _reuse_or_recover(directory, prepared, *, state, attempt, model, condition):
    """Reuse verified saved records, or preserve invalid ones for one new evaluation.

    Returns True when the condition is reused. A changed identity raises, as
    reusable_evaluation does, so an unreadable record never bypasses a change.
    """
    if (
        not (directory / "complete.json").exists()
        and not (directory / "result.json").exists()
    ):
        return False
    if not prepared.preprocessing:
        raise ValueError(
            "Preprocessing identity is unavailable. Use --force or a new output directory"
        )
    if reusable_evaluation(directory, prepared.identity) is not None:
        return True
    archived = archive_condition(directory, state, attempt)
    logger.warning(
        "Saved evaluation records are invalid. Preserved %s and evaluating %s/%s once",
        archived,
        model,
        condition,
    )
    return False


def _completed_record(base, values, prepared, definition):
    """The result record of a completed evaluation."""
    from .models.evaluation_profiles import identity as profile_identity
    from .results import metric_values

    record = deepcopy(base)
    fp = definition["variants"][definition["defaults"]["fp32"]]["network"]
    record.update(
        state="completed",
        functional_passed=True,
        metrics=metric_values(values.metrics, values.metric_units),
        metric_details=values.metric_details,
        samples=values.samples,
        expected_samples=prepared.expected,
        complete=values.complete,
        full_evaluation=values.complete,
        evaluation_identity=prepared.evaluation,
        evaluation_profile=profile_identity(prepared.bundle.spec.recipe),
        runtime=values.runtime,
        finished_at=now(),
        elapsed_s=values.elapsed_s,
    )
    record["evidence"].update(
        identity=prepared.identity,
        model_identity=fp["sha256"],
        preprocessing_identity=prepared.preprocessing,
    )
    return record


def _evaluate_prepared(prepared, directory, definition, base, output, limit):
    """Evaluate one condition and save its result, writing completion evidence last."""
    from .models.adapter import ModelEvaluator
    from .evaluation.loop import evaluate
    from .evaluation.capture import RawCapture, settings as capture_settings

    bundle, dataset = prepared.bundle, prepared.dataset
    evaluator = ModelEvaluator(bundle, dataset)
    capture = (
        RawCapture(evaluator, directory, output) if capture_settings(output) else None
    )
    if capture:
        evaluator = capture
    directory.mkdir(parents=True, exist_ok=True)
    # Files from an earlier attempt must not be attested by this result.
    for stale in ATTESTED_FILES:
        (directory / stale).unlink(missing_ok=True)
    stream = (
        gzip.open(directory / "predictions.jsonl.gz.pending", "wt")
        if output.get("save_predictions")
        else None
    )
    try:
        if stream:
            from .evaluation.capture import tee_predictions

            tee_predictions(evaluator, stream)
        values = evaluate(
            bundle.paths["network"],
            dataset.samples(bundle, limit=limit),
            evaluator=evaluator,
            ort=prepared.ort,
            expected_samples=prepared.expected,
            limit=None if limit is None else prepared.selected_count,
        )
    finally:
        if stream:
            stream.close()
    if bundle.spec.recipe["metric"] not in values.metrics:
        raise ValueError("Primary metric is absent")
    if capture:
        capture.finish()
        capture.publish()
    if stream:
        (directory / "predictions.jsonl.gz.pending").replace(
            directory / "predictions.jsonl.gz"
        )
    record = _completed_record(base, values, prepared, definition)
    files = {}
    if capture:
        files["raw.json"] = sha256(directory / "raw.json")
    if stream:
        files["predictions.jsonl.gz"] = sha256(directory / "predictions.jsonl.gz")
    write_completed(directory, record, files=files)


def _record_failure(current, error, *, preparing, model, condition):
    """Record the state an exception leaves one condition in, and return it."""
    fail_condition(current, error, preparing=preparing)
    if current["state"] != "interrupted" and not expected_error(error):
        # Fail only this condition, but keep the traceback of an
        # unexpected error such as a programming error.
        logger.error(
            "Unexpected error in %s/%s, recorded as failed",
            model,
            condition,
            exc_info=error,
        )
    return current["state"]


def _dry_run_result(root, device, preparations, counts):
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
    from .models.spec import ModelSpec

    _report_options(formats, table_profile)
    selected, models, signature = _selection(config, catalog)
    root = paths.runs / config["name"]
    state = root / ".state"
    device = config["runtime"].get("device", "cuda:0")
    limit = config["evaluation"]["limit"]
    guard = (
        contextlib.nullcontext() if dry_run else run_directory_lock(root, fixed=False)
    )
    with guard:
        _check_saved_selection(root, signature, force)
        status = new_status(selected, config)

        report_failed = False

        def publish():
            nonlocal report_failed
            try:
                atomic_json(state / "status.json", status)
                return report_from_settings_locked(
                    root, formats=formats, table_profile=table_profile, refresh=True
                )
            except BaseException:
                report_failed = True
                raise

        try:
            if not dry_run:
                _settings(
                    root,
                    dict(
                        schema_version=2,
                        mode="evaluate",
                        config=config,
                        models=models,
                        conditions=selected,
                        model_root=str(Path(folder).resolve()),
                        selection_identity=signature,
                        source_identity=code_identity(),
                    ),
                )
                warn_obsolete_reports(root)
                publish()
            preparations = []
            for model, variants in selected.items():
                definition = catalog["models"][model]
                spec = ModelSpec(_model_recipe(config, definition, model))
                status["models"][model]["state"] = "running"
                for condition, variant in variants.items():
                    current = status["models"][model]["conditions"][condition]
                    directory = state / model / condition
                    preparing = True
                    try:
                        prepared = _prepare_evaluation(
                            spec,
                            definition,
                            variant,
                            config,
                            folder,
                            paths,
                            strict_versions,
                        )
                        current["identity"] = prepared.identity
                        preparations.append(
                            dict(model=model, condition=condition, state="ready")
                        )
                        if dry_run:
                            continue
                        preparing = False
                        if not force and _reuse_or_recover(
                            directory,
                            prepared,
                            state=state,
                            attempt=status["attempt_id"],
                            model=model,
                            condition=condition,
                        ):
                            current["state"] = "reused"
                            current["result_sha256"] = sha256(directory / "result.json")
                        else:
                            # Keep prior evaluations readable, including failed force attempts.
                            if force and directory.exists():
                                archive_condition(
                                    directory, state, status["attempt_id"]
                                )
                            current["state"] = "running"
                            publish()
                            _evaluate_prepared(
                                prepared,
                                directory,
                                definition,
                                base_record(
                                    config["name"],
                                    model,
                                    condition,
                                    definition,
                                    variant,
                                ),
                                config["output"],
                                limit,
                            )
                            current["state"] = "completed"
                            current["result_sha256"] = sha256(directory / "result.json")
                    except BaseException as error:
                        if report_failed:
                            raise
                        outcome = _record_failure(
                            current,
                            error,
                            preparing=preparing,
                            model=model,
                            condition=condition,
                        )
                        if outcome == "interrupted":
                            raise
                        if dry_run:
                            preparations.append(
                                dict(
                                    model=model,
                                    condition=condition,
                                    state=outcome,
                                    reason=describe(error),
                                )
                            )
                    if not dry_run:
                        publish()
                status["models"][model]["state"] = model_state(status["models"][model])
            counts = condition_counts(status)
            if dry_run:
                return _dry_run_result(root, device, preparations, counts)
            finish_status(status)
            summary = publish()
            return dict(
                output=str(root),
                device=device,
                rows=summary["rows"],
                mean_recovery=summary["mean_recovery"],
                execution=dict(state=status["state"], **counts),
            )
        except BaseException as error:
            if not dry_run:
                for model_status in status["models"].values():
                    for current in model_status["conditions"].values():
                        if current["state"] == "running":
                            fail_condition(current, error)
                finish_status(status)
                status.update(state=condition_state(error), error=error_record(error))
                try:
                    # Preserve the original exception and never retry a failed report.
                    atomic_json(state / "status.json", status)
                    if not report_failed:
                        publish()
                except BaseException:
                    logger.error(
                        "Could not save aborted evaluation state or report",
                        exc_info=True,
                    )
            raise
