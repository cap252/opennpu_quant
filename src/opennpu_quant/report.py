"""Rebuild a run folder's summary and tables from its verified result files.

Only results whose hashes match their complete.json (and the recorded run status) are
reported as measured; missing or failed measurements are never treated as zeros.
"""

from pathlib import Path
from ._io import atomic_json, read_json
from ._identity import run_config_identity
from ._locking import report_source_lock, report_output_lock
from .results import (
    export_records,
    normalize_formats,
    report_policy,
    summarize,
    tables,
)

from .run_records import _record, current_records, record_directory


def report(
    directory,
    *,
    formats=None,
    table_profile=None,
    output=None,
    models=None,
    conditions=None,
):
    """Compare matched FP32 denominators without treating missing results as zero."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(
            f"Not a run directory: {directory}. Pass an existing run folder"
        )
    with report_source_lock(directory):
        return _report_from_saved_format_locked(
            directory,
            formats=formats,
            table_profile=table_profile,
            output=output,
            models=models,
            conditions=conditions,
        )


def _report_from_saved_format_locked(
    directory,
    *,
    formats=None,
    table_profile=None,
    output=None,
    models=None,
    conditions=None,
):
    """Select a saved run format under the caller's source lock."""
    if record_directory(directory) is not None:
        return report_from_settings_locked(
            directory,
            output=output,
            formats=formats,
            table_profile=table_profile,
            models=models,
            conditions=conditions,
        )
    selected_formats = normalize_formats(formats)
    directory = Path(directory)
    if (directory / "run.json").is_file():
        if (directory / "effective_config.json").is_file():
            raise ValueError("Ambiguous run folder: both execution formats are present")
        raise ValueError(
            "Archived replay results are not supported by this report command"
        )
    if not (directory / "effective_config.json").is_file():
        raise ValueError(
            f"Not a run directory: {directory} (effective_config.json is missing). "
            "Pass <output>/<experiment name>, e.g. workspace/runs/core10"
        )
    config = read_json(directory / "effective_config.json")
    policy = report_policy(config.get("report"))
    status_path = directory / "status.json"
    status = read_json(status_path) if status_path.is_file() else {}
    for requested, available, label in (
        (models, config["models"], "models"),
        (conditions, config["conditions"], "conditions"),
    ):
        if requested is not None and (not requested or set(requested) - set(available)):
            raise ValueError("Unknown or empty " + label)
    every = [
        _record(directory, config, status, policy, model, condition)
        for model in config["models"]
        for condition in config["conditions"]
    ]
    metadata = dict(
        kind="standalone_run",
        config_identity=run_config_identity(config),
        report_policy=policy,
    )
    return _publish_report(
        select_records(every, models, conditions),
        directory,
        output or directory,
        selected_formats,
        table_profile,
        metadata,
        baselines=every,
    )


def select_records(records, models=None, conditions=None):
    """Records of the requested models and conditions; None selects all."""
    return [
        r
        for r in records
        if (models is None or r["model"] in models)
        and (conditions is None or r["condition"] in conditions)
    ]


def settings_document(settings, records, summary):
    """settings.json of a report folder: run settings and the reported measurements."""
    value = dict(settings)
    value["measurements"] = [
        dict(
            model=r["model"],
            condition=r["condition"],
            state=r["state"],
            samples=r.get("samples"),
            expected_samples=r.get("expected_samples"),
            runtime=r.get("runtime", {}),
            finished_at=r.get("finished_at"),
        )
        for r in records
    ]
    value["environment_diagnostics"] = [
        dict(model=r["model"], condition=r["condition"], **r["environment_diagnostic"])
        for r in summary["rows"]
    ]
    value["mean_recovery"] = summary["mean_recovery"]
    return value


def _publish_report(
    records, source, output, formats, table_profile, metadata, refresh=False, **options
):
    """Write one report folder under its lock; every run report goes through here."""
    with report_output_lock(source, output):
        return _write_report(
            records, output, formats, table_profile, metadata, refresh, **options
        )


def _write_report(
    records,
    directory,
    formats,
    table_profile,
    metadata,
    refresh=False,
    *,
    baselines=None,
    settings=None,
):
    """Write the tables, JSON reports and settings.json of one report folder.

    ``baselines`` are all records of the run, so environment diagnostics do not
    depend on a model or condition filter. ``settings`` is the run's stored
    settings; when given, settings.json describes the reported records.
    """
    rows = tables(records, baselines=baselines)[0]
    result = summarize(records, rows=rows, policy=metadata.get("report_policy"))
    export_records(
        records,
        directory,
        formats=formats,
        table_profile=table_profile,
        metadata=metadata,
        summary=result,
        refresh=refresh,
        baselines=baselines,
    )
    if settings is not None:
        atomic_json(
            Path(directory) / "settings.json",
            settings_document(settings, records, result),
        )
    if table_profile:
        from .raw_report import build_report

        result["raw_data"] = build_report(
            records, profile=table_profile, baselines=baselines
        )
    return result


def report_from_settings_locked(
    root,
    *,
    output=None,
    formats=None,
    table_profile=None,
    models=None,
    conditions=None,
    refresh=False,
):
    """Report a settings-based run under the caller's source lock.

    current_records handles its supported record schemas. A different output
    directory is locked separately while publishing the report.
    """
    root = Path(root)
    state = record_directory(root)
    if state is None:
        raise ValueError(f"Not a run directory: {root}. Pass an existing run folder")
    settings = read_json(state / "settings.json")
    records, policy = current_records(root, state=state, settings=settings)
    for requested, key in ((models, "model"), (conditions, "condition")):
        if requested is not None and (
            not requested or set(requested) - {r[key] for r in records}
        ):
            raise ValueError("Unknown or empty " + key + " filter")
    destination = (Path(output) if output else root).resolve()
    # A separate report folder gets the settings of the results it contains.
    # The run folder's settings.json always describes the whole run, and an
    # archived snapshot keeps its own settings.json.
    describes_results = destination != root.resolve() or (
        models is None and conditions is None
    )
    return _publish_report(
        select_records(records, models, conditions),
        root,
        destination,
        "csv" if formats is None else formats,
        table_profile,
        dict(kind="public_run", report_policy=policy),
        refresh,
        baselines=records,
        settings=settings
        if describes_results and destination != state.resolve()
        else None,
    )
