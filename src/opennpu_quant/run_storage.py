"""Run-local generated models and preserved evaluation attempts."""

import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from ._io import atomic_json, object_hash, read_json, sha256, now
from .graph.model import load_model, save_model
from .quant.api import QuantizationResult
from .quant.config import QuantizationConfig
from .quant.parameters import verify_parameters


def read_generated(directory, identity, source_identity, input_identity):
    """Read a verified QDQ model as (result, record), or None without completion proof.

    Changed model bytes, calibration inputs or generation evidence raise an error.
    This verifies model generation, not a completed evaluation.
    """
    proof_path = directory / "complete.json"
    if not proof_path.is_file():
        return None
    proof = read_json(proof_path)
    if proof.get("identity") != identity or set(proof.get("files", {})) != {
        "model.onnx",
        "generation.json",
    }:
        raise ValueError("Generated model completion evidence differs")
    for name, digest in proof["files"].items():
        if not (directory / name).is_file() or sha256(directory / name) != digest:
            raise ValueError("Generated model artifact changed: " + name)
    record = read_json(directory / "generation.json")
    if (
        record["identity"] != identity
        or record["source_model_identity"] != source_identity
        or record["input_identity"] != input_identity
    ):
        raise ValueError("Generated model source or calibration inputs differ")
    model = load_model(directory / "model.onnx")
    verify_parameters(model, record["audit"])
    result = QuantizationResult(
        model,
        QuantizationConfig.from_dict(record["config"]),
        record["range_identity"],
        record["audit"],
    )
    return result, record


def save_generated(directory, identity, result, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(
        metadata,
        identity=identity,
        config=result.config.to_dict(),
        range_identity=result.range_identity,
        audit=result.audit,
    )
    verify_parameters(result.model, result.audit)
    with tempfile.TemporaryDirectory(dir=directory.parent) as temp:
        temp = Path(temp)
        save_model(result.model, temp / "model.onnx")
        atomic_json(temp / "generation.json", record)
        proof = dict(
            identity=identity,
            files={
                name: sha256(temp / name) for name in ("model.onnx", "generation.json")
            },
        )
        (directory / "complete.json").unlink(missing_ok=True)
        for name in proof["files"]:
            (temp / name).replace(directory / name)
        atomic_json(directory / "complete.json", proof)
    return record


def link_model(source, destination):
    """Share unchanged bytes inside a run, without duplicating large QDQ files."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".pending")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    temporary.replace(destination)


def can_archive_for_config_changes(changes):
    """Allow these config changes after preserving the preceding evaluation.

    This is not approval to reuse a generated model. Changes to runtime or
    selected conditions can also change generation identity, checked separately.
    """
    return all(
        path.split(".")[0]
        in {
            "models",
            "conditions",
            "evaluation",
            "runtime",
            "report",
            "output",
        }
        for path in changes
    )


def finish_archive(output):
    journal = output / ".evaluation-transition.json"
    if not journal.exists():
        return
    transition = read_json(journal)
    destination = (output / transition["destination"]).resolve()
    if not destination.is_relative_to((output / ".evaluations").resolve()):
        raise ValueError("Unsafe evaluation archive path")
    destination.mkdir(parents=True, exist_ok=True)
    if not transition.get("archive_complete"):
        for name in transition["names"]:
            if Path(name).name != name or name.startswith("."):
                raise ValueError("Unsafe evaluation archive member")
            source, target = output / name, destination / name
            if source.is_file() and name not in (
                "effective_config.json",
                "status.json",
            ):
                if not target.exists():
                    temporary = target.with_name(target.name + ".pending")
                    try:
                        shutil.copyfile(source, temporary)
                        if sha256(source) != sha256(temporary):
                            raise ValueError(
                                "Report changed during evaluation archive: " + name
                            )
                        temporary.replace(target)
                    finally:
                        temporary.unlink(missing_ok=True)
                elif sha256(source) != sha256(target):
                    raise ValueError(
                        "Report changed during evaluation archive: " + name
                    )
                continue
            if source.exists():
                if target.exists():
                    # Recover an older journal interrupted after restoring its request.
                    request = transition.get("recalibration", {}).get(name)
                    if (
                        request is not None
                        and source.is_dir()
                        and target.is_dir()
                        and {p.name for p in source.iterdir()} == {"recalibration.json"}
                        and read_json(source / "recalibration.json") == request
                    ):
                        continue
                    raise ValueError("Evaluation archive already contains " + name)
                source.replace(target)
        transition["archive_complete"] = True
        atomic_json(journal, transition)
    for name, request in transition.get("recalibration", {}).items():
        if Path(name).name != name or name.startswith("."):
            raise ValueError("Unsafe model name in evaluation archive")
        atomic_json(output / name / "recalibration.json", request)
    journal.unlink()


def archive_evaluation(output, config):
    names = list(config["models"]) + [
        "settings.json",
        "effective_config.json",
        "status.json",
    ]
    names += [
        p.name
        for p in output.iterdir()
        if p.is_file()
        and (
            p.stem in {"records", "summary", "results", "accuracy", "metrics"}
            or p.name.startswith("raw_data.")
        )
    ]
    names = sorted({name for name in names if (output / name).exists()})
    destination = ".evaluations/" + object_hash(config) + "-" + uuid4().hex[:8]
    requests = {
        name: read_json(output / name / "recalibration.json")
        for name in config["models"]
        if (output / name / "recalibration.json").is_file()
    }
    atomic_json(
        output / ".evaluation-transition.json",
        dict(destination=destination, names=names, recalibration=requests),
    )
    finish_archive(output)


def restore_evaluation(output, model, condition, identity):
    """Restore a verified archived condition without changing its measurement record."""
    from .run_records import verified_record

    destination = output / model / condition
    if destination.exists():
        return False
    for archive in sorted((output / ".evaluations").glob("*"), reverse=True):
        source = archive / model / condition
        if not (source / "complete.json").is_file():
            continue
        try:
            proof = read_json(source / "complete.json")
            if proof.get("identity") != identity:
                continue
            status = read_json(archive / "status.json")
            if status.get("schema_version") != 2:
                # Older measurements remain reportable, not a new model cache.
                continue
            current = status["models"][model]["conditions"][condition]
            if (
                current["state"] not in ("completed", "reused")
                or current["identity"] != identity
            ):
                continue
            verified_record(source, current)
        except (ValueError, KeyError, TypeError, OSError):
            continue
        # Never follow a substituted archive directory or artifact symlink.
        if any(p.is_symlink() for p in (archive, source.parent, source)) or any(
            p.is_symlink() for p in source.rglob("*")
        ):
            raise ValueError("Evaluation archive contains a symbolic link")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=destination.parent, prefix=".restore-"
        ) as tmp:
            pending = Path(tmp) / "condition"

            def copy_artifact(source_file, destination_file):
                if Path(source_file).name == "model.onnx":
                    link_model(Path(source_file), Path(destination_file))
                else:
                    shutil.copyfile(source_file, destination_file)

            shutil.copytree(source, pending, copy_function=copy_artifact)
            from .run_records import verify_artifacts

            verified_record(source, current)
            if not verify_artifacts(pending, proof):
                raise ValueError("Evaluation archive changed during restoration")
            pending.replace(destination)
        return True
    return False


def archive_condition(directory, state, attempt):
    """Preserve a condition before replacement under the caller's run lock."""
    destination = (
        state / ".evaluations" / attempt / directory.parent.name / directory.name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    directory.rename(destination)
    return destination


def new_status(conditions, config):
    """The same condition hierarchy for generated and supplied models."""
    from ._identity import run_config_identity

    return dict(
        config_identity=run_config_identity(config),
        schema_version=2,
        state="running",
        attempt_id=uuid4().hex,
        started=now(),
        finished=None,
        models={
            model: dict(
                state="pending",
                conditions={name: dict(state="pending") for name in selected},
            )
            for model, selected in conditions.items()
        },
    )


def fail_condition(current, error, *, preparing=False):
    from .exceptions import condition_state, error_record

    current.update(
        state=condition_state(error, preparing=preparing), error=error_record(error)
    )


def condition_counts(status):
    return {
        state: sum(
            row["state"] in ("completed", "reused")
            if state == "completed"
            else row["state"] == state
            for model in status["models"].values()
            for row in model["conditions"].values()
        )
        for state in ("completed", "pending", "failed")
    }


def _combined_state(states):
    return next(
        (s for s in ("interrupted", "failed", "running", "pending") if s in states),
        "completed",
    )


def model_state(model):
    """The state of a model, from the states of its conditions."""
    return _combined_state({row["state"] for row in model["conditions"].values()})


def finish_status(status):
    counts = condition_counts(status)
    for model in status["models"].values():
        model["state"] = model_state(model)
    status.update(
        state=_combined_state({m["state"] for m in status["models"].values()}),
        finished=now(),
    )
    return counts


def write_completed(directory, record, *, model_hash=None, files=None, generation=None):
    """Publish a finished evaluation's evidence last under the caller's run lock.

    A requested subset also has this proof. Dataset coverage remains in the
    record's complete and full_evaluation fields.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "complete.json").unlink(missing_ok=True)
    atomic_json(directory / "result.json", record)
    proof = dict(
        identity=record["evidence"]["identity"],
        result_sha256=sha256(directory / "result.json"),
        files=files or {},
    )
    if model_hash is not None:
        proof["model_sha256"] = model_hash
    if generation:
        proof["calibration_generation"] = generation
    atomic_json(directory / "complete.json", proof)


def guard_run(root, mode):
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
