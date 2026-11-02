"""Translate native ORT failures without hiding unrelated Python exceptions."""

from contextlib import contextmanager


@contextmanager
def runtime_errors():
    # Delay the optional runtime import until a session is created or executed.
    from onnxruntime.capi import onnxruntime_pybind11_state as native

    try:
        yield
    except Exception as error:
        # Every exception class registered by the native module, including ones
        # added by ORT versions newer than the verified release.
        if type(error).__module__ != native.__name__:
            raise
        raise RuntimeError(f"ONNX Runtime {type(error).__name__}: {error}") from error
