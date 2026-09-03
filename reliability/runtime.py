from dataclasses import asdict, fields
import json
from pathlib import Path
import platform
import subprocess
import sys

from reliability.config import CoreConfig


def core_config_from_namespace(namespace):
    defaults = CoreConfig()
    values = {
        field.name: getattr(namespace, field.name, getattr(defaults, field.name))
        for field in fields(CoreConfig)
    }
    return CoreConfig(**values)


def select_training_path(config):
    config.validate()
    return "core" if config.uses_core_path() else "legacy"


def build_checkpoint_payload(
    gaussian_state, iteration, config, core_state=None
):
    if select_training_path(config) == "legacy":
        return gaussian_state, iteration
    return {
        "schema_version": 1,
        "gaussian_state": gaussian_state,
        "iteration": iteration,
        "core_state": core_state,
    }


def _jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _namespace_values(namespace):
    return {
        key: _jsonable(value)
        for key, value in sorted(vars(namespace).items())
    }


def build_resolved_config(dataset, optimization, pipeline, core_config):
    core_values = _jsonable(asdict(core_config))
    core_values["enabled_features"] = list(core_config.enabled_features())
    return {
        "schema_version": 1,
        "model": _namespace_values(dataset),
        "optimization": _namespace_values(optimization),
        "pipeline": _namespace_values(pipeline),
        "core": core_values,
        "training_path": select_training_path(core_config),
    }


def _git_output(repository, *arguments):
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def collect_run_identity(repository):
    repository = Path(repository).resolve()
    identity = {
        "git_commit": _git_output(repository, "rev-parse", "HEAD"),
        "git_branch": _git_output(repository, "branch", "--show-current"),
        "git_dirty": bool(
            _git_output(repository, "status", "--porcelain", "--untracked-files=all")
        ),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
    except ImportError:
        identity.update(
            torch_version=None,
            torch_cuda_version=None,
            cuda_available=False,
            gpu_name=None,
        )
    else:
        cuda_available = torch.cuda.is_available()
        identity.update(
            torch_version=torch.__version__,
            torch_cuda_version=torch.version.cuda,
            cuda_available=cuda_available,
            gpu_name=torch.cuda.get_device_name(0) if cuda_available else None,
        )
    return identity


def write_run_metadata(output_directory, resolved_config, run_identity):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    documents = {
        "resolved_config.json": resolved_config,
        "run_identity.json": run_identity,
    }
    for filename, document in documents.items():
        with (output_directory / filename).open("w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
