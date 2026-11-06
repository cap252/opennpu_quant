"""Create a separate fixed-input evaluation bundle for CNN30, preserving every Release candidate."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile

import onnx

from opennpu_quant._io import sha256
from opennpu_quant.evaluation_config import load, relative
from opennpu_quant.graph.model import external_data_files, load_model
from opennpu_quant.models.spec import ModelSpec


POLICIES = {
    "yolox": ("yolox", 640),
    "yolov7_face": ("yolov7_letterbox", 640),
    "yolov5s_seg": ("yolov5_letterbox", 640),
    "deeplabv3_r50_voc": ("mmseg_letterbox", 512),
    "fcn_r50_voc": ("mmseg_letterbox", 512),
}


def fixed_model(source, size):
    """Fix the batch-one three-channel input without changing nodes, weights or encodings."""
    if type(size) is not int or size < 1:
        raise ValueError("Fixed input size must be a positive integer")
    original = load_model(source)
    model = deepcopy(original)
    if len(model.graph.input) != 1:
        raise ValueError("Expected one network input")
    tensor = model.graph.input[0].type.tensor_type
    dims = tensor.shape.dim
    if (
        tensor.elem_type != onnx.TensorProto.FLOAT
        or len(dims) != 4
        or dims[0].dim_value != 1
        or dims[1].dim_value != 3
    ):
        raise ValueError("Expected a batch-one NCHW three-channel float32 network")
    for dim in dims[2:]:
        if dim.HasField("dim_value") and dim.dim_value != size:
            raise ValueError("Existing fixed input conflicts with requested size")
        dim.ClearField("dim_param")
        dim.dim_value = size
    onnx.checker.check_model(model)
    restored = deepcopy(model)
    restored.graph.input[0].CopyFrom(original.graph.input[0])
    if restored.SerializeToString() != original.SerializeToString():
        raise ValueError("Fixing the input changed other model contents")
    return model


def verified_asset(root, descriptor):
    path = relative(root, descriptor["path"])
    if sha256(path) != descriptor["sha256"]:
        raise ValueError(f"Source graph hash mismatch: {path}")
    external = external_data_files(path)
    expected = descriptor.get("external_data", {})
    if set(external) != set(expected) or any(
        sha256(asset) != expected[name] for name, asset in external.items()
    ):
        raise ValueError(f"Source external weights differ: {path}")
    return path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def copy_graph(source, target, stage, descriptor):
    shutil.copyfile(source, target)
    for location, asset in external_data_files(source).items():
        dest = relative(target.parent, location)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(asset, dest)
    return dict(descriptor, path=str(target.relative_to(stage)))


def create(catalog_path, output):
    """Publish a complete new bundle. Existing bundles are never overwritten."""
    catalog_path = Path(catalog_path).resolve(strict=True)
    output = Path(output).absolute()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="." + output.name + "-", dir=output.parent
    ) as temp:
        stage = Path(temp)
        # Validate the complete source catalog with the same reader as run.
        check = stage / "source.json"
        write_json(
            check, dict(mode="evaluate", name="source", source=str(catalog_path))
        )
        _, catalog, source_root = load(check)
        check.unlink()
        selected = dict(schema_version=1, models={})
        evidence = dict(source_catalog_sha256=sha256(catalog_path), models={})
        if not set(POLICIES).issubset(catalog["models"]):
            raise ValueError("Missing source model for fixed-input conversion")
        for name, original_row in catalog["models"].items():
            row = deepcopy(original_row)
            model_evidence = dict(variants={})
            if name in POLICIES:
                kind, size = POLICIES[name]
                declared = ModelSpec.load(name).recipe
                if declared["preprocess"]["kind"] != kind:
                    raise ValueError(
                        f"Recipe does not use the fixed-input policy: {name}"
                    )
                for field in ("input", "preprocess", "export"):
                    row["recipe"][field] = deepcopy(declared[field])
                model_evidence["input_shape"] = [1, 3, size, size]
            for key, variant in row["variants"].items():
                desc = variant["network"]
                source = verified_asset(source_root, desc)
                target = relative(stage, f"models/{name}/{key}/network.onnx")
                target.parent.mkdir(parents=True, exist_ok=True)
                if name in POLICIES:
                    onnx.save_model(fixed_model(source, size), target)
                    replacement = dict(
                        path=str(target.relative_to(stage)), sha256=sha256(target)
                    )
                else:
                    replacement = copy_graph(source, target, stage, desc)
                variant["network"] = replacement
                model_evidence["variants"][key] = dict(
                    source=desc,
                    fixed=replacement,
                    input_changed=name in POLICIES,
                    parameters_and_nodes_unchanged=True,
                )
            for role, desc in row.get("host_graphs", {}).items():
                source = verified_asset(source_root, desc)
                target = relative(stage, f"models/{name}/{role}/{source.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                row["host_graphs"][role] = copy_graph(source, target, stage, desc)
            selected["models"][name] = row
            evidence["models"][name] = model_evidence
        write_json(stage / "catalog.json", selected)
        from importlib.resources import files
        import yaml

        core = yaml.safe_load(
            files("opennpu_quant.resources.experiments")
            .joinpath("core10.yaml")
            .read_text()
        )["models"]
        core = [name for name in core if name in selected["models"]]
        configs = stage / "configs"
        configs.mkdir()
        for name, models, conditions in (
            ("cnn30", list(selected["models"]), "defaults"),
            ("core10", core, "defaults"),
            ("core10-candidates", core, "all"),
        ):
            if not models:
                continue
            config = dict(
                mode="evaluate",
                name=name,
                source="../catalog.json",
                models=models,
                conditions=conditions,
                runtime=dict(device="cuda:0"),
                evaluation=dict(limit=None),
                output={},
            )
            path = configs / (name + ".yaml")
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            load(path)
        write_json(stage / "conversion.json", evidence)
        stage.rename(output)
    return output / "configs" / "cnn30.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(create(args.catalog, args.output))
