import collections


MAIN_COMPUTE = frozenset(
    (
        "Conv",
        "FusedConv",
        "NhwcConv",
        "MatMul",
        "Gemm",
        "FusedMatMul",
        "QLinearConv",
        "QLinearMatMul",
        "QGemm",
        "ConvInteger",
        "MatMulInteger",
    )
)


def summarize_profile(events):
    counts = collections.Counter()
    types = {}
    for event in events:
        a = event.get("args", {})
        if event.get("cat") != "Node" or not a.get("provider"):
            continue
        key = (a["provider"], a.get("op_name", "unknown"))
        counts[key] += 1
        types.setdefault(
            key,
            {
                "input_type_shape": a.get("input_type_shape"),
                "output_type_shape": a.get("output_type_shape"),
            },
        )
    rows = [
        {"provider": p, "op_type": o, "calls": n, **types[(p, o)]}
        for (p, o), n in sorted(counts.items())
    ]
    cuda = sum(r["calls"] for r in rows if r["provider"] == "CUDAExecutionProvider")
    cpu = sum(r["calls"] for r in rows if r["provider"] == "CPUExecutionProvider")
    integer = sum(
        r["calls"]
        for r in rows
        if r["provider"] == "CUDAExecutionProvider"
        and r["op_type"].startswith(
            ("QLinearConv", "QLinearMatMul", "QGemm", "ConvInteger", "MatMulInteger")
        )
    )
    cpu_compute = [
        r
        for r in rows
        if r["provider"] == "CPUExecutionProvider" and r["op_type"] in MAIN_COMPUTE
    ]
    return {
        "cuda_observed": bool(cuda),
        "cpu_main_compute": cpu_compute,
        # A sampled profile describes executed operators, not every possible path.
        "sampled_main_compute_placement": "cpu_compute_observed"
        if cpu_compute
        else "cuda_only_observed"
        if any(
            r["provider"] == "CUDAExecutionProvider" and r["op_type"] in MAIN_COMPUTE
            for r in rows
        )
        else "not_observed",
        "full_network_placement": "not_verified",
        "operators": rows,
        "cuda_calls": cuda,
        "cpu_calls": cpu,
        "cuda_integer_compute_calls": integer,
        "cuda_fraction_by_node_calls": cuda / (cuda + cpu) if cuda + cpu else 0.0,
    }


def assert_cuda_observed(providers, profile):
    if "CUDAExecutionProvider" not in providers or not profile["cuda_calls"]:
        raise RuntimeError(
            "CUDA was requested but no CUDA kernel executed; CPU fallback is not GPU success"
        )
    if not any(
        r["provider"] == "CUDAExecutionProvider" and r["op_type"] in MAIN_COMPUTE
        for r in profile["operators"]
    ):
        raise RuntimeError("No main neural-network computation was observed on CUDA")


#: Tolerances for comparing the profiled rerun with the evaluated outputs.
ATOL = 1e-5
RTOL = 1e-4


def output_difference(output, expected):
    """Describe how ``output`` differs from ``expected``, or return None if it matches.

    The test is ``|output - expected| <= ATOL + RTOL * |expected|`` per value,
    with NaN equal to NaN, as in ``numpy.testing.assert_allclose``.

    >>> import numpy as np
    >>> output_difference(np.zeros(3), np.zeros(3)) is None
    True
    >>> output_difference(np.zeros((1, 3)), np.zeros((1, 4)))
    'shape (1, 3) instead of (1, 4)'
    >>> output_difference(np.array([np.nan, 1.0]), np.array([0.0, 1.0]))
    '1 of 2 values differ, 1 of them non-finite'
    """
    import numpy as np

    output, expected = np.asarray(output), np.asarray(expected)
    if output.shape != expected.shape:
        return f"shape {output.shape} instead of {expected.shape}"
    differing = ~np.isclose(output, expected, atol=ATOL, rtol=RTOL, equal_nan=True)
    if not differing.any():
        return None
    actual, reference = output.astype(np.float64), expected.astype(np.float64)
    parts = [f"{int(differing.sum())} of {differing.size} values differ"]
    nonfinite = int((differing & ~np.isfinite(actual)).sum())
    if nonfinite:
        parts.append(f"{nonfinite} of them non-finite")
    finite = differing & np.isfinite(actual) & np.isfinite(reference)
    if finite.any():
        delta = np.abs(actual[finite] - reference[finite])
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = np.max(delta / np.abs(reference[finite]))
        parts.append(f"max absolute difference {delta.max():.6g}")
        parts.append(f"max relative difference {relative:.6g}")
    return ", ".join(parts)


def profile_sample(model, feeds, expected, config):
    """Profile one real input; do not confuse configured providers with execution."""
    import json, tempfile
    from pathlib import Path
    from ..exceptions import OutputMismatchError
    from .session import create_session
    from .fetch import run_tensors

    with tempfile.TemporaryDirectory(
        dir=config.scratch, prefix="opennpu_profile_"
    ) as folder:
        session = create_session(model, config, profile=Path(folder) / "profile")
        names = list(expected)
        outputs = run_tensors(session, names, feeds)
        path = Path(session.end_profiling())
        for name, output in zip(names, outputs):
            difference = output_difference(output, expected[name])
            if difference is not None:
                raise OutputMismatchError(
                    f"Profiled sample output {name!r} differs from the evaluated "
                    f"output (atol {ATOL:g}, rtol {RTOL:g}): {difference}"
                )
        summary = summarize_profile(json.loads(path.read_text()))
        if "CUDAExecutionProvider" in config.providers:
            assert_cuda_observed(session.get_providers(), summary)
        return summary
