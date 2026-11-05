"""Record installed versions separately from libraries mapped by this process."""

import copy
import importlib.metadata
import platform
import os
import subprocess
from pathlib import Path

from .cpu_environment import cpu_environment
from .gpu_environment import recorded_visibility
from .cuda_libs import loaded_libraries, loaded_driver_libraries, compare_libraries

EXPECTED = {
    "onnxruntime-gpu": "1.20.2",
    "nvidia-cuda-runtime-cu12": "12.4.127",
    "nvidia-cublas-cu12": "12.4.5.8",
    "nvidia-cudnn-cu12": "9.1.0.70",
}
KERNEL_DRIVER = Path("/proc/driver/nvidia/version")


def kernel_driver(path=KERNEL_DRIVER):
    """Text of the loaded NVIDIA kernel module, or None when it cannot be read."""
    try:
        return Path(path).read_text().strip() or None
    except OSError:
        return None


def capture(*, libraries=False):
    packages = {}
    for name in (
        *EXPECTED,
        "numpy",
        "onnx",
        "torch",
        "torchvision",
        "pillow",
        "opencv-python-headless",
        "pycocotools",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu = dict(
            available=proc.returncode == 0,
            devices=proc.stdout.strip().splitlines(),
            diagnostic=recorded_visibility((proc.stderr + proc.stdout).strip())
            if proc.returncode
            else None,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        gpu = dict(available=False, diagnostic=str(error), devices=[])
    result = dict(
        packages=packages,
        python=platform.python_version(),
        system=platform.system(),
        machine=platform.machine(),
        cpu=platform.processor(),
        cpu_numerics=cpu_environment(),
        libc=platform.libc_ver(),
        cuda_visible_devices=recorded_visibility(
            os.environ.get("CUDA_VISIBLE_DEVICES")
        ),
        gpu=gpu,
        kernel_driver=kernel_driver(),
    )
    result["version_mismatches"] = {
        k: dict(expected=v, actual=packages[k])
        for k, v in EXPECTED.items()
        if packages[k] != v
    }
    if libraries:
        result["loaded_libraries"] = loaded_libraries()
        result["driver_libraries"] = loaded_driver_libraries()
    return result


def _recorded_environment(value):
    """Read both replay and ordinary evaluation records without probing this host."""
    value = copy.deepcopy(value or {})
    if "providers" in value and "execution" not in value:
        value["device"] = (
            "cuda" if "CUDAExecutionProvider" in value["providers"] else "cpu"
        )
        value["packages"] = {
            key: value[key] for key in ("numpy", "onnx", "onnxruntime") if key in value
        }
        value["execution"] = {
            key: value[key]
            for key in (
                "providers",
                "provider_options",
                "optimization",
                "intra_op_threads",
                "inter_op_threads",
            )
            if key in value
        }
    return value


def compare(actual, expected):
    actual, expected = map(_recorded_environment, (actual, expected))
    devices = [str(env.get("device", "")).split(":")[0] for env in (actual, expected)]
    gpu = devices == ["cuda", "cuda"]
    mismatches = []

    def normalized(value):
        value = copy.deepcopy(value)
        if isinstance(value, dict):
            value["provider_options"] = [
                sorted(
                    (k, v)
                    for k, v in dict(options).items()
                    if k not in {"device_id", "gpu_mem_limit"}
                )
                for options in value.get("provider_options", [])
            ]
        return value

    unverified_fields = []
    for key in ("packages", "execution", "implementation", "python"):
        left, right = actual.get(key), expected.get(key)
        if (left is None) != (right is None):
            unverified_fields.append(key)
            continue
        if key == "execution":
            left, right = normalized(left), normalized(right)
        if left != right:
            mismatches.append(key)
    if not all(devices):
        unverified_fields.append("device")
    elif devices[0] != devices[1]:
        mismatches.append("device")
    for key in ("cpu_numerics",):
        if actual.get(key) is None or expected.get(key) is None:
            unverified_fields.append(key)
        elif actual[key] != expected[key]:
            mismatches.append(key)
    drivers = {}
    if gpu:
        if not actual.get("kernel_driver") or not expected.get("kernel_driver"):
            unverified_fields.append("kernel_driver")
        elif actual["kernel_driver"] != expected["kernel_driver"]:
            mismatches.append("kernel_driver")
        drivers = compare_libraries(
            actual.get("driver_libraries", {"libraries": []}),
            expected.get("driver_libraries", {"libraries": []}),
        )
        if "content_changed" in drivers.values():
            mismatches.append("driver library contents")
        if not any(name.startswith("libcuda.so") for name in drivers) or any(
            state in {"not_loaded", "unverifiable"} for state in drivers.values()
        ):
            unverified_fields.append("driver_libraries")
    libraries = compare_libraries(
        actual.get("loaded_libraries", {"libraries": []}),
        expected.get("loaded_libraries", {"libraries": []}),
    )
    if (
        "content_changed" in libraries.values()
        or gpu
        and bool(actual.get("version_mismatches"))
        or gpu
        and bool(expected.get("version_mismatches"))
    ):
        mismatches.append("CUDA versions/library contents")
    required = ("libcudart", "libcublas", "libcudnn", "libonnxruntime_providers_cuda")
    missing = {
        side: [
            prefix
            for prefix in required
            if not any(
                row["name"].startswith(prefix)
                for row in env.get("loaded_libraries", {}).get("libraries", [])
            )
        ]
        for side, env in (("actual", actual), ("expected", expected))
    }
    unverified = (
        not libraries
        or any(missing.values())
        or any(v in ("not_loaded", "unverifiable") for v in libraries.values())
    )
    if gpu and unverified:
        unverified_fields.append("loaded_libraries")
    return dict(
        state="warning" if mismatches else "pending" if unverified_fields else "pass",
        mismatches=mismatches,
        libraries=libraries,
        drivers=drivers,
        unverified_fields=unverified_fields,
        not_loaded=missing,
        gpu_comparison=gpu,
    )


def diagnostic_lines(rows):
    """Describe recorded environment differences separately from numerical verdicts."""
    lines = []
    for row in rows:
        env = row.get("environment_diagnostic", row.get("environment", {}))
        if not env or env["state"] == "pass":
            continue
        details = list(env["mismatches"]) + [
            "unverified " + key for key in env["unverified_fields"]
        ]
        lines.append(
            f"{row['model']}/{row['condition']}: {env['state']} ({', '.join(details)})"
        )
    return lines


def warn_diagnostics(rows):
    import logging

    lines = diagnostic_lines(rows)
    if lines:
        message = " | ".join(lines[:3])
        if len(lines) > 3:
            message += f" | {len(lines) - 3} more conditions"
        logging.getLogger(__name__).warning(
            "Recorded environment diagnostics do not determine numerical acceptance. %s",
            message,
        )
