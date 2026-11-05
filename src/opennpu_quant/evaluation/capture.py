"""Optional, bounded network input/output evidence for ordinary evaluations."""

from pathlib import Path
import json
import tempfile
from .._io import atomic_json, read_json, sha256

DEFAULT_RAW_BYTES = 256 * 1024**2


def settings(output):
    """Validate bounded capture options and return whether capture is requested."""
    from ..configuration import _integer

    _integer(output.get("raw_samples", 0), "output.raw_samples", 0)
    _integer(output.get("raw_max_bytes", DEFAULT_RAW_BYTES), "output.raw_max_bytes", 1)
    ids = output.get("raw_sample_ids", [])
    if (
        not isinstance(ids, list)
        or any(not isinstance(v, str) or not v for v in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("output.raw_sample_ids must contain unique nonempty strings")
    if ids and output.get("raw_samples", 0):
        raise ValueError("Choose output.raw_samples or output.raw_sample_ids")
    return bool(ids or output.get("raw_samples", 0))


class RawCapture:
    def __init__(self, evaluator, directory, options):
        self.evaluator = evaluator
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pending = Path(
            tempfile.mkdtemp(prefix=".raw-pending-", dir=self.directory)
        )
        self.destination = Path("raw") / self.pending.name.removeprefix(".raw-pending-")
        self.limit = options.get("raw_samples", 0)
        self.ids = set(options.get("raw_sample_ids", []))
        self.budget = options.get("raw_max_bytes", DEFAULT_RAW_BYTES)
        self.rows, self.skipped = [], []
        self.files = {}
        self.observed = set()
        self.bytes = self.index = 0

    def __getattr__(self, name):
        return getattr(self.evaluator, name)

    def update(self, outputs, sample):
        import numpy as np
        from .._feeds import feed_digest

        selected = (
            str(sample.sample_id) in self.ids if self.ids else self.index < self.limit
        )
        self.index += 1
        self.observed.add(str(sample.sample_id))
        capture = None
        if selected:
            size = sum(v.nbytes for v in (*sample.feeds.values(), *outputs.values()))
            if self.bytes + size <= self.budget:
                capture = {
                    role: {
                        key: np.array(value, copy=True) for key, value in values.items()
                    }
                    for role, values in (("inputs", sample.feeds), ("outputs", outputs))
                }
            else:
                self.skipped.append(str(sample.sample_id))
        self.evaluator.update(outputs, sample)
        if capture is None:
            return
        arrays, mapping = {}, {}
        for role, values in capture.items():
            mapping[role] = {}
            for name, value in values.items():
                slot = "array_" + str(len(arrays))
                mapping[role][name] = slot
                arrays[slot] = value
        name = f"{len(self.rows):06d}.npz"
        path = self.pending / name
        np.savez_compressed(path, **arrays)
        prediction_name = path.with_suffix(".prediction.json").name
        atomic_json(self.pending / prediction_name, self.evaluator.prediction)
        for filename in (name, prediction_name):
            self.files[str(self.destination / filename)] = sha256(
                self.pending / filename
            )
        self.bytes += sum(a.nbytes for a in arrays.values())
        self.rows.append(
            dict(
                id=str(sample.sample_id),
                tensors=str(self.destination / name),
                prediction=str(self.destination / prediction_name),
                mapping=mapping,
                input_digest=feed_digest(capture["inputs"]),
                output_digest=feed_digest(capture["outputs"]),
            )
        )

    def finish(self):
        self.summary = dict(
            samples=len(self.rows),
            tensor_bytes=self.bytes,
            max_bytes=self.budget,
            budget_skipped=self.skipped,
            requested_ids_not_seen=sorted(self.ids - self.observed),
        )
        return self.summary

    def publish(self):
        destination = self.directory / self.destination
        destination.parent.mkdir(exist_ok=True)
        self.pending.rename(destination)
        atomic_json(
            self.directory / "raw.json",
            dict(
                schema_version=1,
                summary=self.summary,
                samples=self.rows,
                files=self.files,
            ),
        )


def verify_capture(directory):
    directory = Path(directory).resolve()
    try:
        record = read_json(directory / "raw.json")
        for name, digest in record["files"].items():
            file = directory / name
            if (
                not name.startswith("raw/")
                or not file.resolve().is_relative_to(directory / "raw")
                or not file.is_file()
                or sha256(file) != digest
            ):
                return False
        wanted = {
            name
            for row in record["samples"]
            for name in (row["tensors"], row["prediction"])
        }
        return (
            wanted == set(record["files"])
            and len(record["samples"]) == record["summary"]["samples"]
        )
    except (ValueError, KeyError, TypeError, AttributeError, OSError):
        return False


def tee_predictions(evaluator, stream):
    """Make evaluator.update also write each sample's prediction as one JSON line."""
    update = evaluator.update

    def update_and_write(outputs, sample):
        update(outputs, sample)
        prediction = dict(id=sample.sample_id, prediction=evaluator.prediction)
        stream.write(json.dumps(prediction) + "\n")

    evaluator.update = update_and_write
