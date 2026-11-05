from dataclasses import dataclass, field
from copy import deepcopy
import time
import numpy as np
from ..ort.config import OrtConfig
from ..ort.session import create_session, environment
from ..ort.fetch import run_tensors
from ..exceptions import InvalidOutputError


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict
    samples: int
    complete: bool
    elapsed_s: float
    runtime: dict
    metric_details: dict = field(default_factory=dict)
    metric_units: dict = field(default_factory=dict)


def evaluate(model, samples, *, evaluator, ort=None, expected_samples=None, limit=None):
    """Stream samples through ONNX Runtime and a caller-supplied evaluator.

    Parameters
    ----------
    model : path-like or onnx.ModelProto
        Network to evaluate with the supplied input dictionaries.
    samples : iterable of EvaluationSample
        One pass of uniquely identified, preprocessed samples; not a factory.
    evaluator : object
        Must implement reset(), update(outputs, sample), and finalize().
        finalize returns finite scalar metrics; metric_units and metric_details
        are optional. This function does not choose preprocessing or a metric.
    ort : OrtConfig, optional
        CPU by default; controls execution and optional placement profiling.
    expected_samples : int, optional
        Full evaluation count. Missing requested samples raise an error.
    limit : int, optional
        Positive maximum to consume. A limited subset is not a complete result.

    Returns
    -------
    EvaluationResult
        Metrics, sample count, elapsed time, runtime, units and details.
        complete is true only when expected_samples is given and reached.
        Empty/duplicate samples or nonfinite outputs/metrics are rejected.

    Examples
    --------
    With model and feeds prepared first:

    >>> from opennpu_quant import evaluate, EvaluationSample, TopKClassification
    >>> items = [EvaluationSample(str(i), feed, 5) for i, feed in enumerate(feeds())]
    >>> measured = evaluate(model, items, evaluator=TopKClassification(), expected_samples=len(items))
    >>> assert measured.complete
    """
    ort = ort or OrtConfig.cpu()
    _validate_counts(expected_samples, limit)
    session = create_session(model, ort)
    return _evaluate_stream(
        session,
        samples,
        evaluator=evaluator,
        ort=ort,
        expected_samples=expected_samples,
        limit=limit,
        profile_model=model if ort.profile_samples else None,
    )


def _validate_counts(expected_samples, limit):
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be positive")
    if expected_samples is not None and (
        type(expected_samples) is not int or expected_samples < 1
    ):
        raise ValueError("expected_samples must be a positive integer")


def _evaluate_stream(
    session,
    samples,
    *,
    evaluator,
    ort,
    expected_samples=None,
    limit=None,
    fetch=None,
    record=None,
    profile_model=None,
):
    """Shared inference/scoring loop. ``fetch`` and ``record`` wrap each sample's I/O."""
    _validate_counts(expected_samples, limit)
    names = [v.name for v in session.get_outputs()]
    seen = set()
    evaluator.reset()
    start = time.monotonic()
    placement = None
    from itertools import islice

    for index, sample in enumerate(islice(iter(samples), limit)):
        if sample.sample_id in seen:
            raise ValueError("Duplicate sample: " + str(sample.sample_id))
        seen.add(sample.sample_id)
        values = (
            fetch(index, sample)
            if fetch is not None
            else dict(zip(names, run_tensors(session, names, sample.feeds)))
        )
        if set(values) != set(names):
            raise InvalidOutputError("Output names differ from the session")
        if any(not np.isfinite(v).all() for v in values.values()):
            raise InvalidOutputError("Nonfinite prediction")
        if placement is None and profile_model is not None:
            from ..ort.profile import profile_sample

            placement = profile_sample(profile_model, sample.feeds, values, ort)
        evaluator.update(values, sample)
        if record is not None:
            record(index, sample, values)
    if not seen:
        raise ValueError("No evaluation samples")
    requested = (
        min(expected_samples, limit)
        if expected_samples is not None and limit
        else expected_samples
    )
    if requested is not None and len(seen) != requested:
        raise ValueError("Missing evaluation samples")
    metrics = dict(evaluator.finalize())
    if any(not np.isfinite(v) for v in metrics.values()):
        raise InvalidOutputError("Nonfinite metric")
    runtime = dict(**environment(ort), placement=placement)
    if hasattr(evaluator, "contract"):
        runtime["evaluation_contract"] = evaluator.contract.identity()
    if "CUDAExecutionProvider" in ort.providers:
        from ..ort.cuda_libs import loaded_driver_libraries, loaded_libraries
        from ..ort.diagnostics import kernel_driver

        runtime["loaded_libraries"] = loaded_libraries()
        # Read by the inference process itself, where ORT has mapped the driver;
        # environment diagnostics need both fields to compare CUDA results.
        runtime["driver_libraries"] = loaded_driver_libraries()
        runtime["kernel_driver"] = kernel_driver()
    return EvaluationResult(
        metrics,
        len(seen),
        expected_samples is not None and len(seen) == expected_samples,
        time.monotonic() - start,
        runtime,
        deepcopy(getattr(evaluator, "metric_details", {})),
        {
            key: getattr(evaluator, "metric_units", {}).get(key, "fraction")
            for key in metrics
        },
    )
