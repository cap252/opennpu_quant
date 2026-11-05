"""Run an experiment: for each model and condition, quantize, evaluate and record.

Every condition gets an execution identity (code, graphs, inputs, settings). A finished
condition with the same identity and intact files is reused instead of recomputed.
Layout of a run folder: docs/run-pipeline.md.
"""

import contextlib
import gzip
import hashlib
import logging
from pathlib import Path
import shlex
import tempfile
import time

from ._feeds import feed_digest
from ._identity import code_identity, run_config_identity
from . import run_identity
from ._io import atomic_json, read_json, object_hash, sha256, now
from ._locking import run_directory_lock
from .configuration import (
    DEFAULT_INPUT_CACHE_BYTES,
    validate,
    runtime,
    scheme_for_condition,
)
from .models.spec import ModelSpec
from .models.bundle import ModelBundle
from .models.adapter import ModelEvaluator
from .data.samples import Dataset, preprocessing_implementation, validation_scope
from .graph.model import model_identity, save_model
from .api import calibrate, quantize, apply_adaround
from .quant.storage import save_statistics, read_statistics, statistics_key
from .quant.config import AdaroundConfig
from .quant.adaround_cache import ActivationCacheConfig
from .ort.session import environment, create_session
from .evaluation.loop import evaluate
from .evaluation.capture import tee_predictions
from .report import report_from_settings_locked
from .run_records import ATTESTED_FILES, generation_record, reusable_evaluation
from .run_storage import (
    archive_condition,
    archive_evaluation,
    can_archive_for_config_changes,
    fail_condition,
    finish_archive,
    finish_status,
    guard_run,
    link_model,
    new_status,
    read_generated,
    restore_evaluation,
    save_generated,
    write_completed,
)
from .exceptions import condition_state, error_record


logger = logging.getLogger(__name__)


_REUSE_GUIDANCE = (
    "Use a new --output to keep previous results, or --force to recompute selected "
    "results (calibration caches and AdaRound checkpoints may still be reused)."
)


def _config_changes(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        paths = []
        for key in sorted(before.keys() | after.keys(), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                paths.append(path)
            else:
                paths.extend(_config_changes(before[key], after[key], path))
        return sorted(paths)
    return [prefix or "<root>"] if before != after else []


def runtime_audit(result, ort):
    from .quant.engine import audit_runtime

    with tempfile.TemporaryDirectory(
        dir=ort.scratch, prefix="opennpu_audit_"
    ) as folder:
        original = Path(folder) / "qdq.onnx"
        executed = Path(folder) / "executed.onnx"
        save_model(result.model, original)
        session = create_session(original, ort, optimized=executed)
        del session
        return audit_runtime(original, executed, result.audit)


def run(
    config, paths, *, force=False, recalibrate=False, formats=None, table_profile=None
):
    """Generate and evaluate under one run lock, preserving saved models."""
    from .results import normalize_formats, warn_obsolete_reports
    from .data.lfw_protocol import require_full

    formats = normalize_formats("csv" if formats is None else formats)
    if table_profile is not None:
        from .raw_report import validate_profile

        validate_profile(table_profile)
    validate(config, models_dir=paths.models)
    if recalibrate and not any(config["conditions"].values()):
        raise ValueError("--recalibrate requires a quantized condition")
    definitions, conditions, specs = {}, {}, {}
    for name in config["models"]:
        spec = specs[name] = ModelSpec.for_run(
            name,
            paths.models / name,
            profile=config["evaluation"].get("profiles", {}).get(name),
        )
        if spec.recipe["task"] == "face_verification":
            require_full(
                read_json(paths.data / spec.recipe["dataset"] / "manifest.json")
            )
        definitions[name] = dict(
            recipe=spec.recipe,
            source_directory=str((paths.models / name).resolve()),
            generated_directory="models/" + name,
        )
        conditions[name] = {}
        for condition, policy in config["conditions"].items():
            scheme = scheme_for_condition(config, name, condition)
            conditions[name][condition] = dict(
                scheme=scheme.to_dict() if scheme else None,
                adaround=config["adaround"] if policy.get("adaround") else None,
            )
    root = paths.runs / config["name"]
    out = root / ".state"
    ort, code = runtime(config), code_identity()
    with run_directory_lock(root, fixed=False):
        guard_run(root, "generate")
        finish_archive(out)
        recorded = out / "settings.json"
        if recorded.exists():
            previous = read_json(recorded)["config"]
            if run_config_identity(previous) != run_config_identity(config):
                changed = _config_changes(previous, config)
                if not force and not can_archive_for_config_changes(changed):
                    detail = ", ".join(changed[:10])
                    if len(changed) > 10:
                        detail += f" (+{len(changed) - 10} more)"
                    raise ValueError(
                        "Run configuration changed. Changed keys: "
                        + detail
                        + ". "
                        + _REUSE_GUIDANCE
                    )
                archive_evaluation(out, previous)
        atomic_json(
            recorded,
            dict(
                schema_version=2,
                mode="generate",
                config=config,
                source_identity=code,
                models=definitions,
                conditions=conditions,
            ),
        )
        status = new_status(conditions, config)
        atomic_json(out / "status.json", status)
        warn_obsolete_reports(root)
        try:
            if recalibrate:
                for name in config["models"]:
                    atomic_json(
                        out / name / "recalibration.json",
                        dict(generation=status["attempt_id"], state="pending"),
                    )
            for name in config["models"]:
                status["models"][name]["state"] = "running"
                atomic_json(out / "status.json", status)
                _run_model(
                    name, config, paths, out, ort, code, force, status, spec=specs[name]
                )
                status["models"][name]["state"] = "completed"
                atomic_json(out / "status.json", status)
                _refresh_reports(root, formats, table_profile)
        except BaseException as error:
            status.update(
                state=condition_state(error),
                error=error_record(error),
                finished=now(),
            )
            for model in status["models"].values():
                if model["state"] == "running":
                    fail_condition(model, error)
                for condition in model["conditions"].values():
                    if condition["state"] == "running":
                        fail_condition(condition, error)
            atomic_json(out / "status.json", status)
            with contextlib.suppress(Exception):
                _refresh_reports(root, formats, table_profile)
            raise
        finish_status(status)
        atomic_json(out / "status.json", status)
        return _refresh_reports(root, formats, table_profile)


def _refresh_reports(directory, formats, table_profile):
    return report_from_settings_locked(
        directory, formats=formats, table_profile=table_profile, refresh=True
    )


def _run_model(name, config, paths, out, ort, code, force, status, spec=None):
    job = _ModelRun(name, config, paths, out, ort, code, status, spec=spec)
    settings = read_json(out / "settings.json")
    settings["models"][name].update(job.connections())
    atomic_json(out / "settings.json", settings)
    job.run(force)


class _ModelRun:
    """One model's conditions in one run.

    Calibration statistics, the calibration input factory and RTN results are shared
    by the model's conditions; statistics are collected only when a condition needs them.
    """

    def __init__(self, name, config, paths, out, ort, code, status, spec=None):
        self.name, self.config, self.paths, self.out = name, config, paths, out
        self.ort, self.code, self.status = ort, code, status
        from .models.evaluation_profiles import identity as profile_identity

        # run() passes the recipe it already resolved, so warnings appear once.
        self.spec = spec or ModelSpec.for_run(
            name,
            paths.models / name,
            profile=config["evaluation"].get("profiles", {}).get(name),
        )
        self.evaluation_profile = profile_identity(self.spec.recipe)
        self.bundle = ModelBundle.from_directory(self.spec, paths.models / name)
        dataset_name = self.spec.recipe["dataset"]
        manifest = paths.data / dataset_name / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' is not prepared (missing {manifest}). "
                f"Run: opennpu-quant data prepare {shlex.quote(dataset_name)} "
                f"--root '<dataset folder>' --data-dir {shlex.quote(str(paths.data))}"
            )
        self.dataset = Dataset(manifest, root=paths.datasets.get(dataset_name))
        validation_scope(self.dataset)
        # Hash the materialized graph, including the bytes actually read from external data.
        # Bundle location and external-data filenames are not model identity.
        self.graphs = {k: model_identity(v) for k, v in self.bundle.paths.items()}
        self._identify_inputs(config["evaluation"]["limit"])
        self.count = self.factory = self.calibration_data = None
        if any(config["conditions"].values()):
            self.count = config["calibration"]["samples"][dataset_name]
            self.factory = self.dataset.feeds(self.bundle, self.count)
            self.calibration_data = self.dataset.identity("calibration", self.count)
        self.recalibration_request_path = out / name / "recalibration.json"
        self.recalibration_request = (
            read_json(self.recalibration_request_path)
            if self.recalibration_request_path.exists()
            else None
        )
        # The persisted recalibration request ID is distinct from a model's
        # generation identity. Keep it across later runs without --recalibrate.
        self.recalibration_generation = (
            self.recalibration_request["generation"]
            if self.recalibration_request
            else None
        )
        self.statistics = None
        self.calibration_ort = None
        self.input_cache = None
        self.raw_capture = self.capture_writer = None
        self.generation_record = None
        self.generated_path = None
        self.model_generation = None
        self.rtn = {}  # resolved policy -> RTN QuantizationResult

    def connections(self):
        return dict(
            recipe=self.spec.recipe,
            network_input_map=self.bundle.mapping,
            output_aliases=self.bundle.aliases,
            source_graphs={
                role: dict(path=str(path.resolve()), identity=self.graphs[role])
                for role, path in self.bundle.paths.items()
            },
            generated_networks={
                condition: str(Path("models") / self.name / condition / "network.onnx")
                for condition, policy in self.config["conditions"].items()
                if policy and self.config["output"]["save_qdq_models"]
            },
        )

    def _publish_model(self, source, condition):
        from .graph.model import external_data_files

        root = self.out.parent
        destination = root / "models" / self.name / condition
        link_model(source, destination / "network.onnx")
        for filename, path in external_data_files(source).items():
            link_model(path, destination / filename)
        for role, path in self.bundle.paths.items():
            if role != "network":
                link_model(path, root / "models" / self.name / path.name)
                for filename, extra in external_data_files(path).items():
                    link_model(extra, root / "models" / self.name / filename)

    def _identify_inputs(self, limit):
        """Preprocessing and evaluation identities; None for unidentifiable plugins."""
        recipe = self.spec.recipe
        implementation = preprocessing_implementation(recipe["preprocess"])
        self.preprocessing_identity = run_identity.preprocessing(
            recipe, self.bundle.mapping, self.graphs.get("preprocess"), implementation
        )
        self.evaluation_identity = (
            run_identity.generation_evaluation(
                inputs=self.dataset.identity("validation", limit),
                expected=self.dataset.info.get("expected_samples"),
                listed=len(self.dataset.rows("validation")),
                recipe=recipe,
                profile=self.evaluation_profile,
                runtime=environment(self.ort),
                code=self.code,
                graphs=self.graphs,
                mapping=self.bundle.mapping,
                preprocessing=self.preprocessing_identity,
            )
            if self.preprocessing_identity
            else None
        )

    def _save_status(self):
        atomic_json(self.out / "status.json", self.status)

    def run(self, force):
        for condition, policy in self.config["conditions"].items():
            self._run_condition(condition, policy, force)

    def _primary_metric(self):
        recipe = self.spec.recipe
        return recipe["metric"]

    def _run_condition(self, condition, policy, force):
        current = self.status["models"][self.name]["conditions"][condition]
        current["state"] = "running"
        self._save_status()
        start = time.monotonic()
        directory = self.out / self.name / condition
        scheme = scheme_for_condition(self.config, self.name, condition)
        generation = self._generation_identity(policy, scheme) if policy else None
        identity = self._identity(condition, policy, generation)
        if not force and not (
            policy
            and self.recalibration_request
            and self.recalibration_request["state"] == "pending"
        ):
            restore_evaluation(self.out, self.name, condition, identity)
        if not force and self._reuse(
            directory, condition, identity, policy, current, generation
        ):
            self._save_status()
            return
        result = audit = None
        self.generation_record = self.generated_path = self.model_generation = None
        if policy:
            result = self._generated(policy, scheme, directory, force, generation)
            audit = runtime_audit(result, self.ort)
        network = result.model if result else self.bundle.paths["network"]
        values = self._evaluate(network, directory)
        metric = self._primary_metric()
        if metric not in values.metrics:
            raise ValueError(
                f"Primary metric '{metric}' is undefined for {self.name}/{condition}; "
                f"available: {', '.join(sorted(values.metrics)) or 'none'}. "
                "Check the recipe metric or evaluate more samples."
            )
        # Files from an earlier attempt must not be attested by this result.
        for stale in ("model.onnx", *ATTESTED_FILES):
            (directory / stale).unlink(missing_ok=True)
        if self.config["output"]["save_predictions"]:
            (directory / "predictions.jsonl.gz.pending").replace(
                directory / "predictions.jsonl.gz"
            )
        record = self._record(condition, policy, scheme, identity, values, result)
        record.update(
            runtime_audit=audit, elapsed_s=time.monotonic() - start, finished_at=now()
        )
        if policy and self.recalibration_generation:
            record["calibration_generation"] = self.recalibration_generation
        self._write(directory, policy, record, result)
        current.update(
            state="completed",
            identity=identity,
            result_sha256=sha256(directory / "result.json"),
            measurement_attempt_id=self.status["attempt_id"],
        )
        self._save_status()

    def _generation_identity(self, policy, scheme):
        recipe = self.spec.recipe
        return run_identity.generated_model(
            code=self.code,
            graphs=self.graphs,
            mapping=self.bundle.mapping,
            recipe=recipe,
            implementation=preprocessing_implementation(recipe["preprocess"]),
            inputs=self.calibration_data,
            count=self.count,
            calibration=run_config_identity(
                {"calibration": self.config["calibration"]}
            ),
            scheme=scheme.to_dict(),
            runtime=self._generation_environment(policy),
            adaround=self.config["adaround"] if policy.get("adaround") else None,
            evaluation_runtime=environment(self.ort)
            if policy.get("adaround")
            else None,
            activation_cache=self.config.get("activation_cache")
            if policy.get("adaround")
            else None,
            request=self.recalibration_generation,
        )

    def _identity(self, condition, policy, generation):
        return run_identity.generated_condition(
            code=self.code,
            graphs=self.graphs,
            mapping=self.bundle.mapping,
            evaluation=self.evaluation_identity,
            condition=condition,
            policy=policy,
            generation=generation,
            output=self.config["output"],
        )

    def _generation_environment(self, policy):
        calibration = self.config["calibration"]["device"]
        ort = self.ort if calibration == "run" else runtime(self.config, calibration)
        result = {"calibration": environment(ort)}
        if policy.get("adaround"):
            import torch
            from .ort.gpu_environment import training_environment

            device = self.config["adaround"].get("device", "auto")
            cuda = device == "cuda" or (
                device == "auto" and "CUDAExecutionProvider" in self.ort.providers
            )
            index = next(
                (
                    int(dict(self.ort.provider_options[i]).get("device_id", 0))
                    if self.ort.provider_options
                    else 0
                    for i, provider in enumerate(self.ort.providers)
                    if provider == "CUDAExecutionProvider"
                ),
                0,
            )
            with torch.cuda.device(index) if cuda else contextlib.nullcontext():
                result["training"] = dict(
                    torch=torch.__version__,
                    cuda=torch.version.cuda,
                    device="cuda" if cuda else "cpu",
                    numerics=training_environment(torch, "cuda" if cuda else "cpu"),
                )
        return result

    def _generated(self, policy, scheme, directory, force, identity):
        """Reuse or build QDQ, then publish saved models before evaluation."""
        cache = self.out / ".generated" / self.name / identity
        saved = self.config["output"]["save_qdq_models"]
        pending = (
            self.recalibration_request is not None
            and self.recalibration_request["state"] == "pending"
        )
        reused = None
        if not force and not pending and (cache / "complete.json").is_file():
            if preprocessing_implementation(self.spec.recipe["preprocess"]) is None:
                raise ValueError(
                    "Cannot reuse generated models with unidentified custom preprocessing"
                )
            digest = hashlib.sha256()
            for feed in self.factory():
                digest.update(feed_digest(feed).encode())
            reused = read_generated(
                cache, identity, self.graphs["network"], digest.hexdigest()
            )
        if reused is not None:
            result, self.generation_record = reused
            self.input_cache = self.generation_record.get("input_cache")
        else:
            self._collect_statistics(scheme.config().calibration)
            result = self._quantize(policy, scheme, identity)
            if saved:
                self.generation_record = save_generated(
                    cache,
                    identity,
                    result,
                    dict(
                        source_model_identity=self.graphs["network"],
                        input_identity=self.statistics.samples_identity,
                        calibration_identity=self.statistics.statistics_identity,
                        input_cache=self.input_cache,
                    ),
                )
        retained = saved or reused is not None
        self.generated_path = cache / "model.onnx" if retained else None
        self.model_generation = dict(
            identity=identity,
            state="reused" if reused else "generated",
            saved=retained,
            reuse="verified_saved_model"
            if retained
            else "model_not_saved_regeneration_required",
        )
        if self.generated_path is not None:
            self._publish_model(self.generated_path, directory.name)
        return result

    def _reuse(self, directory, condition, identity, policy, current, generation):
        """Reuse verified results or fall through to one evaluation attempt.

        Recoverable evaluation records do not trigger a retry of failed inference.
        Model and calibration reuse remain independent checks.
        """
        complete = directory / "complete.json"
        if not complete.exists() and not (directory / "result.json").exists():
            return False
        if not self.preprocessing_identity:
            raise ValueError(
                "Cannot reuse results with unidentified custom preprocessing. "
                + _REUSE_GUIDANCE
            )
        try:
            proof = read_json(complete)
        except (ValueError, FileNotFoundError):
            proof = None
        if policy and self._predates_request(proof, directory):
            return False
        previous = reusable_evaluation(directory, identity)
        if previous is None:
            archived = archive_condition(directory, self.out, self.status["attempt_id"])
            logger.warning(
                "Saved evaluation records cannot be reused. Preserved %s and evaluating %s/%s once",
                archived,
                self.name,
                condition,
            )
            if (
                policy
                and not (
                    self.out / ".generated" / self.name / generation / "complete.json"
                ).is_file()
            ):
                logger.warning(
                    "No saved QDQ is available for %s/%s. Model generation is required before evaluation",
                    self.name,
                    condition,
                )
            return False
        current.update(
            state="reused",
            identity=identity,
            result_sha256=proof["result_sha256"],
            measurement_attempt_id=previous.get("attempt_id"),
            measurement_finished_at=previous.get("finished_at"),
            verification="same_execution_identity_and_artifact_hashes",
            reused_config_identity=run_config_identity(self.config),
        )
        if policy and self.config["output"]["save_qdq_models"]:
            self._publish_model(directory / "model.onnx", condition)
        return True

    def _predates_request(self, proof, directory):
        """True when a saved INT8 condition belongs to another recalibration request.

        complete.json records the request. When it is missing or unreadable,
        result.json does, so a damaged proof cannot block a requested rebuild.
        """
        saved = proof
        if not (isinstance(saved, dict) and isinstance(saved.get("identity"), str)):
            try:
                saved = read_json(directory / "result.json")
            except (ValueError, OSError):
                return False
            if not isinstance(saved, dict):
                return False
        return saved.get("calibration_generation") != self.recalibration_generation

    def _collect_statistics(self, options):
        """Collect shared histograms using a validated policy, before range selection."""
        if self.statistics is not None:
            return
        calibration = self.config["calibration"]
        input_policy = calibration.get("input_cache", {})
        if input_policy.get("enabled", False):
            cache_start = time.monotonic()
            self.factory = self.dataset.cached_feeds(
                self.bundle,
                self.count,
                self.paths.cache / "inputs" / self.name,
                max_bytes=input_policy.get("max_bytes", DEFAULT_INPUT_CACHE_BYTES),
            )
            manifest = self.factory.manifest
            self.input_cache = dict(
                identity=manifest["identity"],
                samples_identity=manifest["samples_identity"],
                sample_count=manifest["sample_count"],
                bytes=sum(v["bytes"] for v in manifest["tensors"]),
                reused=self.factory.reused,
                prepare_seconds=time.monotonic() - cache_start,
                manifest_sha256=sha256(self.factory.directory / "manifest.json"),
            )
            atomic_json(self.out / self.name / "input_cache.json", self.input_cache)
        signatures = hashlib.sha256()
        for feed in self.factory():
            signatures.update(feed_digest(feed).encode())
        feed_id = signatures.hexdigest()
        expected = calibration.get("expected_samples_identity", {}).get(self.name)
        if expected is not None and feed_id != expected:
            raise ValueError(
                f"Calibration inputs differ from the selected source for {self.name}: "
                f"expected {expected}, observed {feed_id}. Check image order and preprocessing."
            )
        self.calibration_ort = (
            self.ort
            if calibration["device"] == "run"
            else runtime(self.config, calibration["device"])
        )
        key = statistics_key(
            self.graphs["network"],
            feed_id,
            calibration["histogram_bins"],
            self.calibration_ort,
        )
        cache = self.paths.cache / "calibration" / self.name / key
        if self.recalibration_request:
            cache = cache / "requests" / self.recalibration_generation
        pending = (
            self.recalibration_request is not None
            and self.recalibration_request["state"] == "pending"
        )
        network = self.bundle.paths["network"]
        if (cache / "complete.json").exists() and not pending:
            self.statistics = read_statistics(
                cache,
                network,
                config=options,
                ort=self.calibration_ort,
                samples_identity=feed_id,
            )
        else:
            self.statistics = calibrate(
                network, self.factory, config=options, ort=self.calibration_ort
            )
            save_statistics(self.statistics, cache)
        if self.recalibration_request:
            identity = self.statistics.statistics_identity
            if (
                not pending
                and self.recalibration_request["statistics_identity"] != identity
            ):
                raise ValueError(
                    "Recalibration statistics changed; request --recalibrate again"
                )
            self.recalibration_request.update(
                state="completed", statistics_identity=identity
            )
            atomic_json(self.recalibration_request_path, self.recalibration_request)

    def _quantize(self, policy, scheme, identity):
        """RTN result of the resolved policy, then AdaRound when requested."""
        network = self.bundle.paths["network"]
        key = object_hash(scheme.to_dict())
        if key not in self.rtn:
            # Use the collection execution profile to validate encodings, independent of evaluation device.
            self.rtn[key] = quantize(
                network, self.statistics, scheme, ort=self.calibration_ort
            )
        if not policy.get("adaround"):
            return self.rtn[key]
        cache = dict(self.config.get("activation_cache", {}))
        enabled = cache.pop("enabled", True)
        checkpoint = self.out / ".generated" / self.name / identity / "adaround"
        return apply_adaround(
            network,
            self.rtn[key],
            self.factory,
            AdaroundConfig(**self.config["adaround"]),
            ort=self.ort,
            checkpoint_dir=checkpoint / self.recalibration_generation
            if self.recalibration_generation
            else checkpoint,
            activation_cache=ActivationCacheConfig(**cache) if enabled else None,
        )

    def _evaluate(self, network, directory):
        evaluator = ModelEvaluator(self.bundle, self.dataset)
        self.raw_capture = self.capture_writer = None
        output = self.config["output"]
        capture = None
        if output.get("raw_samples") or output.get("raw_sample_ids"):
            from .evaluation.capture import RawCapture

            capture = RawCapture(evaluator, directory, output)
            evaluator = capture
        stream = None
        if self.config["output"]["save_predictions"]:
            # Predictions are streamed; the preprocessed dataset is never held in memory.
            directory.mkdir(parents=True, exist_ok=True)
            stream = gzip.open(directory / "predictions.jsonl.gz.pending", "wt")
            tee_predictions(evaluator, stream)
        limit = self.config["evaluation"]["limit"]
        sample_limit = limit
        if limit is not None and "evaluation_pairs" in self.dataset.info:
            # LFW limits count pairs. Other datasets must retain the requested count.
            sample_limit = len(self.dataset.rows("validation", limit))
        try:
            measured = evaluate(
                network,
                self.dataset.samples(self.bundle, limit=limit),
                evaluator=evaluator,
                ort=self.ort,
                expected_samples=self.dataset.info["expected_samples"],
                limit=sample_limit,
            )
            if capture is not None:
                self.raw_capture = capture.finish()
                self.capture_writer = capture
            return measured
        finally:
            if stream:
                stream.close()

    def _record(self, condition, policy, scheme, identity, values, result):
        recipe = self.spec.recipe
        metric = self._primary_metric()
        return dict(
            attempt_id=self.status["attempt_id"],
            model=self.name,
            title=recipe["title"],
            model_metadata={
                k: recipe[k]
                for k in (
                    "source",
                    "dataset",
                    "task",
                    "input",
                    "metric",
                    "preprocess",
                    "decode",
                )
            },
            condition=condition,
            metric=metric,
            value=values.metrics[metric],
            metrics=values.metrics,
            metric_units=values.metric_units,
            metric_details=values.metric_details,
            samples=values.samples,
            expected_samples=self.dataset.info["expected_samples"],
            complete=values.complete,
            functional_passed=True,
            export_validation=self.bundle.validation,
            full_evaluation=values.samples == self.dataset.info["expected_samples"],
            reference=bool(self.dataset.info.get("reference_bytes_verified"))
            and values.complete,
            config_identity=run_config_identity(self.config),
            evaluation_identity=self.evaluation_identity,
            evaluation_profile=self.evaluation_profile,
            preprocessing_identity=self.preprocessing_identity,
            identity=identity,
            model_identity=self.graphs["network"],
            graph_identities=self.graphs,
            scheme=scheme.to_dict() if scheme else None,
            calibration_samples=self.count if policy else None,
            calibration_identity=(
                self.generation_record["calibration_identity"]
                if self.generation_record
                else self.statistics.statistics_identity
            )
            if policy
            else None,
            model_generation=self.model_generation,
            raw_capture=self.raw_capture,
            input_cache=self.input_cache if policy else None,
            adaround=result.audit.get("adaround") if result else None,
            runtime=values.runtime,
        )

    def _write(self, directory, policy, record, result):
        """Write result.json and its files, then complete.json with their hashes."""
        if self.capture_writer is not None:
            self.capture_writer.publish()
        model_hash = None
        if result and self.config["output"]["save_qdq_models"]:
            if self.generated_path is not None:
                link_model(self.generated_path, directory / "model.onnx")
            else:
                save_model(result.model, directory / "model.onnx")
            model_hash = sha256(directory / "model.onnx")
        if result:
            atomic_json(directory / "quantization.json", result.audit)
        record = generation_record(record, self.config)
        files = {
            name: sha256(directory / name)
            for name in ATTESTED_FILES
            if (directory / name).is_file()
        }
        write_completed(
            directory,
            record,
            model_hash=model_hash,
            files=files,
            generation=self.recalibration_generation if policy else None,
        )
