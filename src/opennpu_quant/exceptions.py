"""Execution errors and the state an exception leaves a condition in.

Generation runs, evaluation runs and the command line share one rule:

* Only a stop request interrupts a run: ``KeyboardInterrupt``, ``SystemExit``
  and other exceptions that are not ``Exception`` subclasses.
* A file missing while an evaluation config prepares a condition leaves it
  ``pending``. Generation configs do not prepare conditions separately.
* Every other exception fails the condition that raised it. Evaluation
  configs continue with the next condition; generation configs stop.
"""

#: Errors reported as a message instead of a traceback: invalid input, file,
#: runtime and optional-dependency errors, including every ExecutionError.
EXPECTED_ERRORS = (OSError, ValueError, RuntimeError, ImportError)


class ExecutionError(RuntimeError):
    """A condition ran but produced a result that cannot be used."""


class OutputMismatchError(ExecutionError):
    """Running the same input again gave outputs that differ from the evaluated ones."""


class InvalidOutputError(ExecutionError, ValueError):
    """Outputs or metrics are unusable, for example non-finite.

    It is also a ValueError, the type these checks raised before.
    """


def condition_state(error, *, preparing=False):
    """Return the state ``error`` leaves its condition in.

    >>> condition_state(KeyboardInterrupt())
    'interrupted'
    >>> condition_state(FileNotFoundError("manifest.json"), preparing=True)
    'pending'
    >>> condition_state(TypeError("bug"))
    'failed'
    """
    if not isinstance(error, Exception):
        return "interrupted"
    if preparing and isinstance(error, FileNotFoundError):
        return "pending"
    return "failed"


def expected(error):
    """True when ``error`` is reported as a message rather than a traceback."""
    return isinstance(error, EXPECTED_ERRORS)


def error_record(error):
    """The ``{type, message}`` form of an exception kept in status.json."""
    return dict(type=type(error).__name__, message=str(error))


def describe(error):
    """``Type: message`` for an exception or its error record, or the type alone.

    >>> describe(ValueError("bad input"))
    'ValueError: bad input'
    >>> describe(error_record(KeyboardInterrupt()))
    'KeyboardInterrupt'
    """
    if isinstance(error, dict):
        name, message = error.get("type") or "Error", error.get("message") or ""
    else:
        name, message = type(error).__name__, str(error)
    return f"{name}: {message}" if message else name
