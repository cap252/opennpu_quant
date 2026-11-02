"""Explicit evaluation inputs, independent of recorded research selections."""

from copy import deepcopy
from dataclasses import fields
from pathlib import Path
import re

from ._io import read_json, read_yaml
from .configuration import _mapping, _integer, _boolean, check_device


def name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Model, condition and experiment names must be simple names")
    return value


def relative(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Model paths must be nonempty relative paths")
    path = (Path(root) / value).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Model path escapes the bundle")
    return path


def load(path):
    path = Path(path).resolve()
    config = read_yaml(path)
    _mapping(
        config,
        "evaluation config",
        allowed={
            "mode",
            "name",
            "source",
            "models",
            "conditions",
            "runtime",
            "evaluation",
            "output",
        },
        required={"mode", "name", "source"},
    )
    if config["mode"] != "evaluate":
        raise ValueError("Evaluation config requires mode: evaluate")
    name(config["name"])
    if not isinstance(config["source"], str) or not config["source"]:
        raise ValueError("source must name a model catalog")
    catalog_path = (path.parent / config["source"]).resolve()
    catalog = read_json(catalog_path)
    _mapping(
        catalog,
        "catalog",
        allowed={"schema_version", "models"},
        required={"schema_version", "models"},
    )
    if type(catalog["schema_version"]) is not int or catalog["schema_version"] != 1:
        raise ValueError("Unsupported model catalog version")
    _mapping(catalog["models"], "catalog.models")
    if not catalog["models"]:
        raise ValueError("Catalog must contain models")
    for model, row in catalog["models"].items():
        name(model)
        _mapping(
            row,
            model,
            allowed={"recipe", "host_graphs", "variants", "defaults", "output_aliases"},
            required={"recipe", "variants", "defaults"},
        )
        _mapping(
            row["recipe"],
            model + ".recipe",
            required={
                "name",
                "title",
                "task",
                "dataset",
                "metric",
                "input",
                "graphs",
                "preprocess",
                "decode",
                "source",
            },
        )
        from .models.spec import ModelSpec

        ModelSpec(row["recipe"])
        aliases = row.get("output_aliases", {})
        _mapping(aliases, model + ".output_aliases")
        if any(
            not isinstance(alias, str)
            or not alias
            or not isinstance(target, str)
            or not target
            for alias, target in aliases.items()
        ):
            raise ValueError("output_aliases requires nonempty string names")
        _mapping(
            row.get("host_graphs", {}),
            model + ".host_graphs",
            allowed={"preprocess", "postprocess"},
        )
        if set(row["recipe"]["graphs"]) != {"network", *row.get("host_graphs", {})}:
            raise ValueError("Catalog graph roles differ from the recipe")
        if row["recipe"].get("name") != model:
            raise ValueError("Catalog recipe name differs from model")
        _mapping(row["variants"], model + ".variants")
        _mapping(row["defaults"], model + ".defaults")
        for condition, variant in row["variants"].items():
            name(condition)
            _mapping(
                variant,
                model + "." + condition,
                allowed={"network", "scheme", "adaround", "runtime"},
                required={"network", "scheme"},
            )
            if variant["scheme"] is None and variant.get("adaround"):
                raise ValueError("FP32 cannot have AdaRound settings")
            if variant["scheme"] is not None:
                from .quant.scheme import Scheme

                _mapping(
                    variant["scheme"],
                    "variant.scheme",
                    allowed={f.name for f in fields(Scheme)},
                )
                try:
                    Scheme(**variant["scheme"])
                except (TypeError, AttributeError) as error:
                    raise ValueError("Invalid variant quantization scheme") from error
            if variant.get("adaround") is not None:
                _mapping(variant["adaround"], "variant.adaround")
            for desc in [variant["network"], *row.get("host_graphs", {}).values()]:
                _mapping(
                    desc,
                    "graph",
                    allowed={"path", "sha256", "external_data"},
                    required={"path", "sha256"},
                )
                relative(catalog_path.parent, desc["path"])
                if not isinstance(desc["sha256"], str) or not re.fullmatch(
                    "[0-9a-f]{64}", desc["sha256"]
                ):
                    raise ValueError("Graph requires a SHA256 digest")
                _mapping(desc.get("external_data", {}), "graph.external_data")
                for location, digest in desc.get("external_data", {}).items():
                    relative(
                        relative(catalog_path.parent, desc["path"]).parent, location
                    )
                    if not isinstance(digest, str) or not re.fullmatch(
                        "[0-9a-f]{64}", digest
                    ):
                        raise ValueError("External weights require a SHA256 digest")
            options = variant.get("runtime", {})
            _mapping(options, "variant.runtime", allowed={"gpu_mem_limit"})
            if "gpu_mem_limit" in options:
                _integer(options["gpu_mem_limit"], "gpu_mem_limit", 1)
        for alias, target in row["defaults"].items():
            name(alias)
            if not isinstance(target, str) or target not in row["variants"]:
                raise ValueError("Default condition is not in the model catalog")
        fp32 = row["defaults"].get("fp32")
        if fp32 is None or row["variants"][fp32]["scheme"] is not None:
            raise ValueError("Each model requires a genuine FP32 default")
    config.setdefault("models", list(catalog["models"]))
    config.setdefault("conditions", "defaults")
    config.setdefault("runtime", {"device": "cuda:0"})
    _mapping(config["runtime"], "runtime", allowed={"device"})
    check_device(config["runtime"].get("device", "cuda:0"), "runtime.device")
    config.setdefault("evaluation", {})
    _mapping(config["evaluation"], "evaluation", allowed={"limit", "profiles"})
    config["evaluation"].setdefault("limit", None)
    if config["evaluation"]["limit"] == "full":
        config["evaluation"]["limit"] = None
    if config["evaluation"]["limit"] is not None:
        _integer(config["evaluation"]["limit"], "evaluation.limit", 1)
    _mapping(config["evaluation"].get("profiles", {}), "evaluation.profiles")
    config.setdefault("output", {})
    _mapping(
        config["output"],
        "output",
        allowed={"save_predictions", "raw_samples", "raw_sample_ids", "raw_max_bytes"},
    )
    _boolean(config["output"].get("save_predictions", False), "output.save_predictions")
    from .evaluation.capture import settings

    settings(config["output"])
    return config, catalog, catalog_path.parent


def _validate_profiles(config, catalog):
    """Reject unknown model names and unsupported profiles before any work starts.

    Entries for catalog models outside the selection stay allowed, so one config can
    serve several --models subsets, but every entry must name a catalog model.
    """
    from .models.evaluation_profiles import apply_profile

    profiles = config.get("evaluation", {}).get("profiles", {})
    unknown = sorted(str(model) for model in profiles if model not in catalog["models"])
    if unknown:
        raise ValueError(
            "evaluation.profiles names models that are not in the catalog: "
            + ", ".join(unknown)
        )
    for model, profile in profiles.items():
        if not isinstance(profile, str):
            raise ValueError(f"evaluation.profiles.{model} must be a profile name")
        apply_profile(catalog["models"][model]["recipe"], profile)


def select(config, catalog):
    models = config["models"]
    if (
        not isinstance(models, list)
        or not models
        or any(not isinstance(m, str) for m in models)
        or len(set(models)) != len(models)
    ):
        raise ValueError("models must contain unique model names")
    if set(models) - catalog["models"].keys():
        raise ValueError("Unknown evaluation model")
    _validate_profiles(config, catalog)
    conditions = config["conditions"]
    if not (isinstance(conditions, str) and conditions in ("all", "defaults")):
        if (
            not isinstance(conditions, list)
            or not conditions
            or any(not isinstance(c, str) for c in conditions)
            or len(set(conditions)) != len(conditions)
        ):
            raise ValueError("conditions must be defaults, all or unique names")
    result = {}
    for model in models:
        row = catalog["models"][model]
        names = (
            row["defaults"]
            if conditions == "defaults"
            else row["variants"]
            if conditions == "all"
            else conditions
        )
        result[model] = {}
        for condition in names:
            name(condition)
            key = (
                condition
                if conditions == "all"
                else row["defaults"].get(condition, condition)
            )
            if key not in row["variants"]:
                raise ValueError(f"Unknown condition {model}/{condition}")
            output_name = "fp32" if key == row["defaults"]["fp32"] else condition
            if output_name in result[model]:
                raise ValueError("Duplicate evaluation condition")
            result[model][output_name] = deepcopy(row["variants"][key])
        if "fp32" in result[model]:
            result[model] = {"fp32": result[model].pop("fp32"), **result[model]}
    return result
