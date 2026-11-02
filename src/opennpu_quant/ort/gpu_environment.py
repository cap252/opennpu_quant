"""CUDA numerical identity, independent of UUIDs and interpreter import order."""

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import uuid


def device_identifier(value):
    """Stable non-raw device identifier for diagnostic records, not cache identity."""
    import hashlib

    if not value:
        return None
    value = str(value).lower().removeprefix("gpu-")
    return hashlib.sha256(("opennpu-device-v1:" + value).encode()).hexdigest()


def recorded_device(record):
    """Keep device specifications and a digest without publishing the hardware UUID."""
    if record is None:
        return None
    result = {k: v for k, v in record.items() if k != "uuid"}
    result["uuid_sha256"] = record.get("uuid_sha256") or device_identifier(
        record.get("uuid")
    )
    return result


def recorded_visibility(value):
    """Redact UUID selectors while preserving visible-device order."""
    import re

    if value is None:
        return None
    return re.sub(
        r"(?:GPU-|MIG-)[A-Za-z0-9/_-]+",
        lambda match: "sha256:" + device_identifier(match.group()),
        value,
    )


def _checked(status):
    if status:
        raise RuntimeError(f"CUDA device identification failed (driver error {status})")


def device_record(ordinal):
    """Query the visible logical device without importing Torch or changing devices."""
    driver = ctypes.CDLL("libcuda.so.1")
    _checked(driver.cuInit(0))
    device = ctypes.c_int()
    _checked(driver.cuDeviceGet(ctypes.byref(device), int(ordinal)))
    name = ctypes.create_string_buffer(256)
    _checked(driver.cuDeviceGetName(name, len(name), device))

    def attribute(key):
        value = ctypes.c_int()
        _checked(driver.cuDeviceGetAttribute(ctypes.byref(value), key, device))
        return value.value

    identifier = (ctypes.c_ubyte * 16)()
    get_uuid = getattr(driver, "cuDeviceGetUuid_v2", driver.cuDeviceGetUuid)
    _checked(get_uuid(ctypes.byref(identifier), device))
    version = ctypes.c_int()
    _checked(driver.cuDriverGetVersion(ctypes.byref(version)))
    return dict(
        name=name.value.decode(),
        capability=[attribute(75), attribute(76)],
        multiprocessors=attribute(16),
        driver_api=version.value,
        ordinal=int(ordinal),
        uuid="GPU-" + str(uuid.UUID(bytes=bytes(identifier))),
    )


@lru_cache(maxsize=128)
def _file_hash(path, stat_key):
    from .._io import sha256

    return sha256(path)


def library_identity(*, include_provider=True):
    """Hash the selected runtime binaries, also before ORT loads its provider."""
    from .cuda_libs import library_paths, load_cuda_libraries
    from .session import runtime

    load_cuda_libraries()
    paths = library_paths()
    if include_provider:
        paths["onnxruntime_providers_cuda"] = (
            Path(runtime().__file__).parent / "capi/libonnxruntime_providers_cuda.so"
        )
    result = {}
    for name, path in paths.items():
        path = path.resolve(strict=True)
        st = path.stat()
        result[name] = _file_hash(
            str(path),
            (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns),
        )
    # Loaded files can differ from wheel metadata when LD_LIBRARY_PATH or an
    # earlier framework import selected another copy. Never bless that mismatch.
    maps = Path("/proc/self/maps")
    if maps.exists():
        mapped = set()
        for line in maps.read_text().splitlines():
            parts = line.split(maxsplit=5)
            if len(parts) == 6 and parts[5].startswith("/"):
                mapped.add(Path(parts[5]))
        for name, expected in paths.items():
            for actual in mapped:
                if not actual.name.startswith(expected.name):
                    continue
                st = actual.stat()
                digest = _file_hash(
                    str(actual),
                    (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns),
                )
                if digest != result[name]:
                    raise RuntimeError(
                        "Loaded CUDA library differs from selected runtime: "
                        + expected.name
                    )
    return result


def numerical_flags():
    return {
        name: os.environ.get(name)
        for name in ("NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG")
    }


def gpu_environment(ordinal, *, include_provider=True):
    record = device_record(ordinal)
    return dict(
        schema_version=1,
        device={k: v for k, v in record.items() if k not in ("uuid", "ordinal")},
        libraries=library_identity(include_provider=include_provider),
        flags=numerical_flags(),
    )


def training_environment(torch, device):
    """Identify Torch's actual training device, which can differ from ORT's device."""
    if device != "cuda":
        return None
    ordinal = torch.cuda.current_device()
    result = gpu_environment(ordinal, include_provider=False)
    # _torch_policy fixes this value before any reconstruction operation.
    result["flags"]["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    result["torch"] = dict(
        version=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
    )
    return result
