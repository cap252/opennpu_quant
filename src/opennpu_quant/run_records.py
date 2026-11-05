"""Read and verify saved measurements without importing an execution driver.

Older result layouts are read only. All report entrypoints use this boundary.

Condition state "completed" means the requested evaluation finished. The complete
and full_evaluation flags mean all expected dataset samples were evaluated.
complete.json attests published files even for a subset. functional_passed records
successful evaluation, not framework parity or agreement with earlier accuracy.
"""

from pathlib import Path
from copy import deepcopy
from ._io import read_json, sha256, object_hash
from ._identity import run_config_identity
from .results import metric_values, normalize_scheme, report_policy, percent
from .exceptions import describe

ATTESTED_FILES = ("quantization.json", "predictions.jsonl.gz", "raw.json")


def verify_artifacts(directory, proof, *, model_verified=False):
    """True when result.json, model.onnx and the extra files match complete.json.

    ``model_verified`` skips hashing a model.onnx the caller just checked.
    """
    directory = Path(directory)
    if not isinstance(proof, dict):
        return False
    result = directory / "result.json"
    if not result.is_file() or sha256(result) != proof.get("result_sha256"):
        return False
    files = proof.get("files", {})
    if not isinstance(files, dict) or set(files) - set(ATTESTED_FILES):
        return False
    for name, digest in files.items():
        if not (directory / name).is_file() or sha256(directory / name) != digest:
            return False
    if "raw.json" in files:
        from .evaluation.capture import verify_capture

        if not verify_capture(directory):
            return False
    model = directory / "model.onnx"
    return (
        model_verified
        or not proof.get("model_sha256")
        or (model.is_file() and sha256(model) == proof["model_sha256"])
    )


def _condition_status(status, model, condition):
    return (
        status.get("models", {}).get(model, {}).get("conditions", {}).get(condition, {})
    )


def _verified(path, config, status):
    """Unreadable completion evidence cannot keep an earlier success report alive."""
    try:
        return _verified_result(path, config, status)
    except (ValueError, KeyError, TypeError, AttributeError, OSError):
        return False


def _verified_result(path, config, status):
    proof_path = path.parent / "complete.json"
    if not path.is_file() or not proof_path.is_file():
        return False
    try:
        proof = read_json(proof_path)
        if not verify_artifacts(path.parent, proof):
            return False
        measured = read_json(path)
    except (ValueError, FileNotFoundError, IsADirectoryError):
        # Corrupt evidence must not prevent stale successful reports being replaced.
        return False
    if not isinstance(measured, dict):
        return False
    current = _condition_status(status, path.parent.parent.name, path.parent.name)
    rebound = (
        current.get("state") == "reused"
        and current.get("verification") == "same_execution_identity_and_artifact_hashes"
        and current.get("reused_config_identity") == run_config_identity(config)
        and bool(status.get("attempt_id"))
    )
    if not rebound and measured.get("config_identity") not in (
        object_hash(config),
        run_config_identity(config),
    ):
        return False
    if status.get("attempt_id"):
        current = _condition_status(status, path.parent.parent.name, path.parent.name)
        if (
            current.get("state") not in {"completed", "reused"}
            or current.get("result_sha256") != proof.get("result_sha256")
            or current.get("identity") != measured.get("identity")
            or proof.get("identity") != measured.get("identity")
            or current.get("measurement_attempt_id") != measured.get("attempt_id")
        ):
            return False
    return True


def _record(directory, config, status, policy, model, condition):
    """One portable record per model/condition; unverified results carry no measurement."""
    path = directory / model / condition / "result.json"
    ok = _verified(path, config, status)
    current = _condition_status(status, model, condition)
    row = read_json(path) if ok else {}
    from .data.lfw_protocol import reject_partial_result

    reject_partial_result(row)
    reject_partial_result(current)
    metadata = row.get("model_metadata", {})
    binding = "recorded_at_measurement" if metadata else "not_recorded"
    if not metadata:
        # A reference recipe is not measured provenance.
        try:
            from .models.spec import ModelSpec

            recipe = ModelSpec.load(model).recipe
            metadata = {
                k: recipe[k] for k in ("source", "dataset", "task", "input", "metric")
            }
            binding = "packaged_recipe_reference"
        except (FileNotFoundError, ValueError):
            pass
    condition_policy = config["conditions"][condition]
    scheme = row.get("scheme")
    if not ok and condition_policy:
        scheme = dict(config["schemes"][condition_policy["scheme"]])
        if scheme.get("percentile") in ("per_model", "recorded"):
            from .configuration import scheme_for_condition

            scheme["percentile"] = scheme_for_condition(
                config, model, condition
            ).percentile
    adaround = (row.get("adaround") or {}).get("config")
    if adaround is None and condition_policy.get("adaround"):
        adaround = config["adaround"]
    task = metadata.get("task")
    if task == "face_verification":
        from .data.lfw_protocol import require_full

        require_full(limit=config.get("evaluation", {}).get("limit"))
    if ok and task is None and policy["primary_metric"]:
        raise ValueError("Report primary_metric requires model task metadata: " + model)
    primary = policy["primary_metric"].get(
        task, row.get("metric", metadata.get("metric", "unknown"))
    )
    metrics = metric_values(row.get("metrics", {}), row.get("metric_units"))
    if ok and (primary not in metrics or percent(metrics[primary]) is None):
        raise ValueError(
            "Missing or incompatible report metric: " + model + "/" + primary
        )
    if ok:
        state = "completed"
    elif current.get("state") in {"completed", "reused"}:
        state = "unverified"
    else:
        state = current.get("state", "unverified" if path.exists() else "pending")
    return dict(
        run_id=config["name"],
        model=model,
        model_name=row.get("title", model),
        condition=condition,
        state=state,
        origin="standalone_run",
        primary_metric=primary,
        metrics=metrics,
        metric_details=row.get("metric_details", {}),
        functional_passed=row.get("functional_passed", False),
        export_validation=row.get("export_validation", {}),
        full_evaluation=row.get("full_evaluation", False),
        dataset=metadata.get("dataset"),
        split="validation",
        samples=row.get("samples"),
        expected_samples=row.get("expected_samples"),
        complete=row.get("complete", False),
        reference=row.get("reference", False),
        evaluation_identity=row.get("evaluation_identity"),
        quantization=normalize_scheme(scheme, adaround),
        model_source=metadata.get("source", {}),
        model_metadata=metadata,
        runtime=row.get("runtime", {}),
        finished_at=row.get("finished_at"),
        elapsed_s=row.get("elapsed_s"),
        evidence=dict(
            result_sha256=sha256(path) if ok else None,
            result_hash_verified=ok,
            source_binding=binding,
            model_identity=row.get("model_identity"),
            preprocessing_identity=row.get("preprocessing_identity"),
            identity=row.get("identity"),
            attempt_id=status.get("attempt_id"),
            measurement_attempt_id=row.get("attempt_id"),
            current_attempt=current,
        ),
    )


def reusable_evaluation(directory, identity):
    """Return a verified evaluation, or None for a new/recoverable evaluation.

    Surviving identities must agree with the current execution. An unreadable
    record alone is not a reason to bypass an input, model or settings change.
    This only inspects stored evidence. It never executes or retries evaluation.
    """
    saved = {}
    for name in ("complete.json", "result.json"):
        path = directory / name
        if not path.exists():
            continue
        try:
            saved[name] = read_json(path)
        except ValueError:
            saved[name] = None
    if not saved:
        return None
    proof = saved.get("complete.json")
    record = saved.get("result.json")
    identities = []
    if isinstance(proof, dict):
        identities.append(proof.get("identity"))
    if isinstance(record, dict) and isinstance(record.get("evidence"), dict):
        identities.append(record["evidence"].get("identity"))
    identities = [value for value in identities if isinstance(value, str) and value]
    if identities and any(value != identity for value in identities):
        raise ValueError(
            f"Run execution identity changed: {directory}. "
            "The stored identity does not identify the exact cause. "
            "Possible causes include code, model/host graphs, inputs or "
            "preprocessing, runtime or quantization settings. "
            "Use a new --output to keep previous results, or --force to recompute "
            "selected results (calibration caches and AdaRound checkpoints may still be reused)."
        )
    if not identities:
        raise ValueError(
            f"Saved evaluation identity is unavailable: {directory}. "
            "Use a new --output to keep previous results, or --force to recompute selected results"
        )
    # A damaged ONNX is not an evaluation-record repair. Generation validates
    # its independent saved-model evidence as well before running inference.
    if isinstance(proof, dict) and proof.get("model_sha256"):
        model = directory / "model.onnx"
        if not model.is_file() or sha256(model) != proof["model_sha256"]:
            raise ValueError(
                f"Saved model artifact changed: {model}. Restore the model or request regeneration"
            )
    try:
        return _completed(directory, identity, model_verified=True)
    except ValueError:
        return None


def _completed(directory, identity, *, model_verified=False):
    """One integrity gate for both reuse and reporting, without inference."""
    try:
        proof = read_json(directory / "complete.json")
        if (
            not isinstance(proof, dict)
            or not isinstance(proof.get("identity"), str)
            or proof["identity"] != identity
            or not verify_artifacts(directory, proof, model_verified=model_verified)
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
            or not 0 < record["samples"] <= record["expected_samples"]
            or type(record["complete"]) is not bool
            or type(record["full_evaluation"]) is not bool
            or record["complete"] != (record["samples"] == record["expected_samples"])
            or record["full_evaluation"] != record["complete"]
        ):
            raise ValueError("Saved evaluation result is inconsistent")
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


def _generation_records(state):
    """Schema-1 generation records of a state directory or archived snapshot."""
    config = read_json(state / "effective_config.json")
    status = read_json(state / "status.json")
    policy = report_policy(config.get("report"))
    return [
        _record(state, config, status, policy, model, condition)
        for model in config["models"]
        for condition in config["conditions"]
    ], policy


def _evaluation_records(state):
    """Schema-1 evaluation records of a state directory or archived snapshot."""
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


def base_record(name, model, condition, definition, variant):
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


def record_directory(root):
    """Locate the active state or an archived run snapshot, or None for neither.

    A report folder also holds a settings.json, so a snapshot needs status.json.
    """
    root = Path(root)
    if (root / ".state/settings.json").is_file():
        return root / ".state"
    if (root / "settings.json").is_file() and (root / "status.json").is_file():
        return root
    return None


def verified_record(directory, current):
    record = _completed(directory, current["identity"])
    if current.get("result_sha256") != sha256(directory / "result.json"):
        raise ValueError("Saved condition status does not attest this result")
    return record


def current_records(root, *, state=None, settings=None):
    """Read both current modes with one verifier. Prior schemas are read-only.

    Callers that already located the state directory or read its settings.json
    pass them in, so each report reads them once.
    """
    state = state or record_directory(root)
    if state is None:
        raise ValueError(
            f"Not a run directory: {root}. Pass a run folder or an archived run snapshot"
        )
    if settings is None:
        settings = read_json(state / "settings.json")
    status_path = state / "status.json"
    status = read_json(status_path) if status_path.is_file() else {}
    if settings.get("schema_version") != 2:
        return (
            _generation_records(state)
            if settings["mode"] == "generate"
            else _evaluation_records(state)
        )
    policy = report_policy(settings["config"].get("report"))
    records = []
    request_verified = status.get("config_identity") == run_config_identity(
        settings["config"]
    )
    for model, variants in settings["conditions"].items():
        for condition in variants:
            current = (
                _condition_status(status, model, condition) if request_verified else {}
            )
            if not current:
                current = dict(
                    state="unverified", error="Missing or changed run status"
                )
            record = base_record(
                settings["config"]["name"],
                model,
                condition,
                settings["models"][model],
                settings["conditions"][model][condition],
            )
            directory = state / model / condition
            verified = False
            if current["state"] in ("completed", "reused"):
                try:
                    record = verified_record(directory, current)
                    verified = True
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            if verified:
                record["state"] = current["state"]
                primary = policy["primary_metric"].get(
                    record["model_metadata"].get("task"), record["primary_metric"]
                )
                if (
                    primary not in record["metrics"]
                    or percent(record["metrics"][primary]) is None
                ):
                    raise ValueError(
                        "Missing or incompatible report metric: "
                        + model
                        + "/"
                        + primary
                    )
                record["primary_metric"] = primary
                record["evidence"].update(
                    result_hash_verified=True,
                    result_sha256=sha256(directory / "result.json"),
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
                error = current.get("error")
                record["failure_reason"] = (
                    describe(error) if isinstance(error, dict) else error
                )
            records.append(record)
    return records, policy


def generation_record(measured, config):
    """Normalize a generated-model measurement before storing it."""
    recipe = dict(measured["model_metadata"], title=measured["title"])
    record = base_record(
        config["name"],
        measured["model"],
        measured["condition"],
        {"recipe": recipe},
        {
            "scheme": measured["scheme"],
            "adaround": (measured.get("adaround") or {}).get("config"),
        },
    )
    record.update(measured)
    record.update(
        state="completed",
        metrics=metric_values(measured["metrics"], measured["metric_units"]),
        primary_metric=measured["metric"],
    )
    record["evidence"].update(
        identity=measured["identity"],
        model_identity=measured["model_identity"],
        preprocessing_identity=measured["preprocessing_identity"],
        measurement_attempt_id=measured["attempt_id"],
    )
    return record
