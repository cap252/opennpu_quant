"""Explicit identity payloads for run reuse. Supplied and generated models differ.

The payload order preserves the preceding run format. Source edits still change
code_identity, so moving this code does not authorize reuse of earlier runs.
"""

from copy import deepcopy

from ._io import object_hash


def preprocessing(recipe, mapping, graph, implementation):
    return (
        object_hash(
            [recipe["preprocess"], recipe["input"], mapping, graph, implementation]
        )
        if implementation
        else None
    )


def generation_evaluation(
    *,
    inputs,
    expected,
    listed,
    recipe,
    profile,
    runtime,
    code,
    graphs,
    mapping,
    preprocessing,
):
    if preprocessing is None:
        return None
    return object_hash(
        [
            inputs,
            dict(expected_samples=expected, validation_samples=listed),
            recipe,
            profile,
            runtime,
            code,
            {k: v for k, v in graphs.items() if k != "network"},
            mapping,
            preprocessing,
        ]
    )


def provided_evaluation(
    *, inputs, expected, listed, recipe, profile, preprocessing, graphs, runtime, code
):
    return object_hash(
        [
            inputs,
            expected,
            listed,
            recipe,
            profile,
            preprocessing,
            {k: v for k, v in graphs.items() if k != "network"},
            runtime,
            code,
        ]
    )


def generated_model(
    *,
    code,
    graphs,
    mapping,
    recipe,
    implementation,
    inputs,
    count,
    calibration,
    scheme,
    runtime,
    adaround,
    evaluation_runtime,
    activation_cache,
    request,
):
    return object_hash(
        [
            code,
            graphs["network"],
            graphs.get("preprocess"),
            mapping,
            recipe["input"],
            recipe["preprocess"],
            implementation,
            inputs,
            count,
            calibration,
            scheme,
            runtime,
            adaround,
            evaluation_runtime,
            activation_cache,
            request,
        ]
    )


def generated_condition(
    *, code, graphs, mapping, evaluation, condition, policy, generation, output
):
    return object_hash(
        [code, graphs, mapping, evaluation, condition, policy, generation, output]
    )


def provided_variant(variant):
    """Identity inputs for a catalog variant, excluding ORT arena capacity."""
    value = deepcopy(variant)
    runtime = value.get("runtime")
    if runtime is not None:
        runtime.pop("gpu_mem_limit", None)
        if not runtime:
            value.pop("runtime")
    return value


def provided_condition(evaluation, graphs, variant, output):
    return object_hash([evaluation, graphs, provided_variant(variant), output])
