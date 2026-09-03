"""Read-only comparison of baseline and feature-off AmbiSuR runs."""

from argparse import ArgumentParser
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re


_EVALUATION_PATTERN = re.compile(
    r"\[ITER\s+(?P<iteration>\d+)\]\s+Evaluating\s+(?P<split>[^:]+):\s+"
    r"L1\s+(?P<l1>[-+0-9.eE]+)\s+PSNR\s+(?P<psnr>[-+0-9.eE]+)"
)
_POINT_PATTERN = re.compile(r"\bPoints[=:]\s*(?P<points>\d+)")


def _read_optional_int(path):
    return int(path.read_text(encoding="utf-8").strip()) if path.is_file() else None


def _read_time(path):
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ply_vertex_count(path):
    with path.open("rb") as stream:
        for _ in range(256):
            line = stream.readline()
            if not line:
                break
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("element vertex "):
                return int(text.split()[-1])
            if text == "end_header":
                break
    raise ValueError(f"PLY vertex element missing from {path}")


def load_run(run_directory):
    run_directory = Path(run_directory).resolve()
    log_path = run_directory / "train.log"
    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file()
        else ""
    )
    evaluations = [
        {
            "iteration": int(match.group("iteration")),
            "split": match.group("split").strip(),
            "l1": float(match.group("l1")),
            "psnr": float(match.group("psnr")),
        }
        for match in _EVALUATION_PATTERN.finditer(log_text)
    ]
    point_matches = list(_POINT_PATTERN.finditer(log_text))
    point_clouds = {}
    for path in sorted(run_directory.glob("point_cloud/iteration_*/point_cloud.ply")):
        iteration = path.parent.name.removeprefix("iteration_")
        point_clouds[iteration] = {
            "vertices": _ply_vertex_count(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }

    start = _read_time(run_directory / "start_utc.txt")
    end = _read_time(run_directory / "end_utc.txt")
    peak_paths = (
        run_directory / "gpu_peak_mib.txt",
        run_directory / "peak_gpu_mib.txt",
    )
    gpu_peak = next(
        (_read_optional_int(path) for path in peak_paths if path.is_file()), None
    )
    return {
        "run_directory": str(run_directory),
        "exit_code": _read_optional_int(run_directory / "exit_code.txt"),
        "gpu_peak_mib": gpu_peak,
        "wall_time_seconds": (end - start).total_seconds() if start and end else None,
        "evaluations": evaluations,
        "final_points": (
            int(point_matches[-1].group("points")) if point_matches else None
        ),
        "point_clouds": point_clouds,
    }


def _evaluation_map(snapshot):
    return {
        (entry["iteration"], entry["split"]): entry
        for entry in snapshot["evaluations"]
    }


def compare_runs(baseline, candidate, *, rtol=1e-5, atol=1e-7):
    differences = []
    if baseline["exit_code"] != candidate["exit_code"]:
        differences.append("exit_code")
    if baseline["final_points"] != candidate["final_points"]:
        differences.append("final_points")

    baseline_evaluations = _evaluation_map(baseline)
    candidate_evaluations = _evaluation_map(candidate)
    for key in sorted(set(baseline_evaluations) | set(candidate_evaluations)):
        label = f"evaluation[{key[0]}:{key[1]}]"
        if key not in baseline_evaluations or key not in candidate_evaluations:
            differences.append(label)
            continue
        for metric in ("l1", "psnr"):
            if not math.isclose(
                baseline_evaluations[key][metric],
                candidate_evaluations[key][metric],
                rel_tol=rtol,
                abs_tol=atol,
            ):
                differences.append(f"{label}.{metric}")

    baseline_clouds = baseline["point_clouds"]
    candidate_clouds = candidate["point_clouds"]
    for iteration in sorted(set(baseline_clouds) | set(candidate_clouds), key=int):
        label = f"point_cloud[{iteration}]"
        if iteration not in baseline_clouds or iteration not in candidate_clouds:
            differences.append(label)
            continue
        for field in ("vertices", "sha256"):
            if baseline_clouds[iteration][field] != candidate_clouds[iteration][field]:
                differences.append(f"{label}.{field}")

    resource_deltas = {
        "gpu_peak_mib": (
            None
            if baseline["gpu_peak_mib"] is None or candidate["gpu_peak_mib"] is None
            else candidate["gpu_peak_mib"] - baseline["gpu_peak_mib"]
        ),
        "wall_time_seconds": (
            None
            if baseline["wall_time_seconds"] is None
            or candidate["wall_time_seconds"] is None
            else candidate["wall_time_seconds"] - baseline["wall_time_seconds"]
        ),
    }
    return {
        "equivalent": not differences,
        "rtol": rtol,
        "atol": atol,
        "differences": differences,
        "resource_deltas": resource_deltas,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("candidate_run", type=Path)
    args = parser.parse_args()
    result = compare_runs(load_run(args.baseline_run), load_run(args.candidate_run))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
