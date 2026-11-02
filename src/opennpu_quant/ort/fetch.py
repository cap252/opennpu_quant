import numpy as np

from .errors import runtime_errors


def run_tensors(session, names, feed):
    """Preserve requested order and independent passthrough input copies.

    ORT 1.20.2 CUDA input-as-output fetches can race the input H2D copy.
    Actual node outputs still execute through ORT. Only names declared as
    graph inputs are read from caller feeds; arbitrary extra feed keys cannot
    replace computed tensors. Pure passthrough requests need no session run.
    """
    required = list(session.get_inputs())
    optional = list(session.get_overridable_initializers())
    inputs = {value.name: feed[value.name] for value in required}
    inputs.update(
        {value.name: feed[value.name] for value in optional if value.name in feed}
    )
    computed = list(dict.fromkeys(name for name in names if name not in inputs))
    if not computed:
        for value in required + optional:
            if value.name in inputs:
                _validate_input(value, inputs[value.name])
    passthrough = {
        name: np.array(inputs[name], copy=True) for name in names if name in inputs
    }
    values = []
    if computed:
        with runtime_errors():
            values = session.run(computed, inputs)
    if len(values) != len(computed):
        raise ValueError("ORT returned an unexpected tensor count")
    outputs = dict(zip(computed, values))
    return [
        passthrough[name].copy() if name in passthrough else outputs[name]
        for name in names
    ]


def _validate_input(value, array):
    kinds = {
        "float": "float32",
        "double": "float64",
        "float16": "float16",
        "bool": "bool",
        "string": "object",
        **{
            k: k
            for k in (
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
            )
        },
    }
    expected = kinds.get(value.type.removeprefix("tensor(").removesuffix(")"))
    if (
        expected is None
        or not isinstance(array, np.ndarray)
        or array.dtype != np.dtype(expected)
    ):
        raise ValueError("Input dtype differs from ONNX declaration: " + value.name)
    if value.shape is not None and (
        array.ndim != len(value.shape)
        or any(
            isinstance(dim, int) and dim != size
            for dim, size in zip(value.shape, array.shape)
        )
    ):
        raise ValueError("Input shape differs from ONNX declaration: " + value.name)
