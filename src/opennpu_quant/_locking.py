from contextlib import contextmanager
import os
from pathlib import Path


class RunLock:
    """Hold a nonblocking writer lock that the OS releases on process exit."""

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError("Another process owns: " + str(self.path)) from None
        return self

    def __exit__(self, *args):
        import fcntl

        fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()


@contextmanager
def _exclusive_lock(descriptor, path):
    import fcntl

    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another process owns: " + str(path)) from None
    try:
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


@contextmanager
def _directory_lock(directory, *, create=False):
    # Readers need a lock even when a read-only snapshot has no .lock file.
    # Writers acquire the same directory inode before creating their .lock file.
    directory = Path(directory)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with _exclusive_lock(descriptor, directory):
            yield
    finally:
        os.close(descriptor)


@contextmanager
def report_source_lock(directory):
    """Lock an existing source without creating or writing files in it."""
    directory = Path(directory)
    with _directory_lock(directory):
        path = directory / ".lock"
        try:
            stream = path.open("rb")
        except FileNotFoundError:
            yield
        else:
            # Respect a pre-existing file lock, including one held by older code.
            with stream, _exclusive_lock(stream, path):
                yield


@contextmanager
def run_directory_lock(directory, *, fixed):
    """Check execution format under the writer lock, before publishing any files."""
    directory = Path(directory)
    incompatible = (
        ("effective_config.json", ".evaluation-transition.json")
        if fixed
        else ("run.json", "prepared/manifest.json", ".prepared-previous/manifest.json")
    )
    with _directory_lock(directory, create=True), RunLock(directory / ".lock"):
        if any((directory / name).exists() for name in incompatible):
            raise ValueError(
                "Output directory contains a different execution format. "
                "Use a new output directory"
            )
        yield


# Files that mark a folder as a run folder or archived run snapshot of any
# execution format. Report folders never contain them.
RUN_MARKERS = (
    ".state/settings.json",
    "effective_config.json",
    "run.json",
    "status.json",
)


def check_report_destination(source, destination):
    """Reject report outputs that would overwrite another run's files."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source:
        return
    if ".state" in destination.parts:
        raise ValueError(
            f"Report output is inside a run's .state folder: {destination}. "
            "Choose a separate folder for reports"
        )
    if any((destination / marker).exists() for marker in RUN_MARKERS):
        raise ValueError(
            f"Report output is another run folder: {destination}. "
            "Choose a separate folder for reports"
        )


@contextmanager
def report_output_lock(source, destination):
    """Lock a distinct report destination while the caller holds the source lock."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source:
        yield
    else:
        check_report_destination(source, destination)
        with _directory_lock(destination, create=True), RunLock(destination / ".lock"):
            # The destination may have become a run while acquiring its lock.
            check_report_destination(source, destination)
            yield


@contextmanager
def cache_lock(path, *, shared=False):
    """Wait for a cache reader/writer lock, released on interruption or process exit."""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
