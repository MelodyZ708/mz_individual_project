#!/usr/bin/env python3
"""Evaluate selected Conv1 channel sets on the complete TUM lightswitch sequence.

This is deliberately separate from the MVS search database.  The MVS stage is
used for candidate discovery; this stage tests whether the resulting ordering
generalises to all 573 matched RGB-D timestamps in fr1/desk_lightswitch.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import queue
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
COMO_DIR = PROJECT_ROOT / "como"
CORE_SCRIPT = (
    PROJECT_ROOT
    / "channel_selection_pipeline/scripts/step_d_fail_fast_evaluation/"
    "run_r070_bruteforce.py"
)
DEFAULT_DATASET = Path(
    "/home/melody/data/tum/rgbd_dataset_freiburg1_desk_lightswitch"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "channel_selection_results/step_e_full_sequence_evaluation/"
    "fr1_desk_lightswitch"
)
DEFAULT_PYTHON = Path("/home/melody/anaconda3/envs/como/bin/python")
DEFAULT_EVO_APE = Path("/home/melody/anaconda3/envs/como/bin/evo_ape")
DEFAULT_EVO_RPE = Path("/home/melody/anaconda3/envs/como/bin/evo_rpe")
SHARED_CONFIG_LOCK = (
    PROJECT_ROOT
    / "channel_selection_results/step_d_fail_fast_evaluation/"
    "r070_bruteforce_v2/.como_config.lock"
)


def load_search_core():
    spec = importlib.util.spec_from_file_location("channel_search_core", CORE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import search metric helpers: {CORE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = load_search_core()


@dataclass(frozen=True)
class EvaluationSpec:
    label: str
    channels: tuple[int, ...] | None
    role: str

    @property
    def candidate_key(self) -> str:
        if self.channels is None:
            return "gray"
        return ",".join(str(channel) for channel in self.channels)

    @property
    def display(self) -> str:
        return "gray" if self.channels is None else f"[{self.candidate_key}]"


EVALUATION_SPECS = (
    EvaluationSpec("gray_baseline", None, "photometric baseline control"),
    EvaluationSpec(
        "known_cnn_baseline",
        (5, 29, 40, 52),
        "historical four-channel CNN control",
    ),
    EvaluationSpec(
        "mvs_best_ate",
        (8, 40, 50, 59),
        "lowest RPE-safe MVS SE(3) ATE",
    ),
    EvaluationSpec(
        "mvs_second_ate",
        (5, 8, 24, 30),
        "second-lowest RPE-safe MVS SE(3) ATE",
    ),
    EvaluationSpec(
        "mvs_balanced",
        (5, 24, 30, 59),
        "low MVS ATE with lower translation/rotation jumps",
    ),
    EvaluationSpec(
        "r080_rescue",
        (5, 17, 19, 59),
        "candidate recovered by the r=0.80 rescue audit",
    ),
    EvaluationSpec(
        "mvs_mid_low_jump",
        (5, 8, 20, 60),
        "moderate MVS ATE with lower local jumps",
    ),
    EvaluationSpec(
        "mvs_low_translation_jump",
        (0, 24, 30, 56),
        "translation-RPE stability control",
    ),
    EvaluationSpec(
        "mvs_low_rotation_jump",
        (6, 17, 18, 43),
        "rotation-RPE stability control",
    ),
)


@dataclass
class FullSequenceResult:
    label: str
    role: str
    candidate_key: str
    channels_json: str | None
    replicate: int
    status: str
    reason: str
    elapsed_seconds: float
    exit_code: int | None = None
    failure_timestamp: float | None = None
    failure_frame_index: int | None = None
    last_timestamp: float | None = None
    trajectory_poses: int | None = None
    associated_poses: int | None = None
    expected_matched_frames: int | None = None
    coverage_ratio: float | None = None
    se3_ate_rmse_m: float | None = None
    se3_ate_mean_m: float | None = None
    se3_ate_median_m: float | None = None
    se3_ate_max_m: float | None = None
    se3_ate_std_m: float | None = None
    rotation_ape_rmse_deg: float | None = None
    rotation_ape_mean_deg: float | None = None
    rotation_ape_max_deg: float | None = None
    translation_rpe_rmse_m: float | None = None
    translation_rpe_max_m: float | None = None
    rotation_rpe_rmse_deg: float | None = None
    rotation_rpe_max_deg: float | None = None
    allframe_sim3_rmse_m: float | None = None
    allframe_sim3_mean_m: float | None = None
    allframe_sim3_scale: float | None = None
    keyframe_sim3_rmse_m: float | None = None
    keyframe_sim3_mean_m: float | None = None
    keyframe_sim3_scale: float | None = None
    keyframe_associated_poses: int | None = None
    historical_evo_ape_rmse_m: float | None = None
    historical_evo_ape_mean_m: float | None = None
    historical_evo_rpe_rmse_m: float | None = None
    diagnostic_frames: int = 0
    photo_mse_median: float | None = None
    photo_mse_p95: float | None = None
    photo_mse_nonfinite_count: int = 0
    valid_ratio_min: float | None = None
    valid_ratio_median: float | None = None
    h_cond_max: float | None = None
    delta_norm_max: float | None = None
    crazy_affine_count: int = 0
    trajectory_path: str | None = None
    keyframe_trajectory_path: str | None = None
    log_path: str | None = None
    created_at: str = ""


class Console:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8", buffering=1)

    def say(self, message: str = "") -> None:
        line = (
            f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            if message
            else ""
        )
        print(line, flush=True)
        self.handle.write(line + "\n")

    def close(self) -> None:
        self.handle.close()


class ResultStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                role TEXT NOT NULL,
                candidate_key TEXT NOT NULL,
                channels_json TEXT,
                replicate INTEGER NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                elapsed_seconds REAL NOT NULL,
                exit_code INTEGER,
                failure_timestamp REAL,
                failure_frame_index INTEGER,
                last_timestamp REAL,
                trajectory_poses INTEGER,
                associated_poses INTEGER,
                expected_matched_frames INTEGER,
                coverage_ratio REAL,
                se3_ate_rmse_m REAL,
                se3_ate_mean_m REAL,
                se3_ate_median_m REAL,
                se3_ate_max_m REAL,
                se3_ate_std_m REAL,
                rotation_ape_rmse_deg REAL,
                rotation_ape_mean_deg REAL,
                rotation_ape_max_deg REAL,
                translation_rpe_rmse_m REAL,
                translation_rpe_max_m REAL,
                rotation_rpe_rmse_deg REAL,
                rotation_rpe_max_deg REAL,
                allframe_sim3_rmse_m REAL,
                allframe_sim3_mean_m REAL,
                allframe_sim3_scale REAL,
                keyframe_sim3_rmse_m REAL,
                keyframe_sim3_mean_m REAL,
                keyframe_sim3_scale REAL,
                keyframe_associated_poses INTEGER,
                historical_evo_ape_rmse_m REAL,
                historical_evo_ape_mean_m REAL,
                historical_evo_rpe_rmse_m REAL,
                diagnostic_frames INTEGER NOT NULL,
                photo_mse_median REAL,
                photo_mse_p95 REAL,
                photo_mse_nonfinite_count INTEGER NOT NULL,
                valid_ratio_min REAL,
                valid_ratio_median REAL,
                h_cond_max REAL,
                delta_norm_max REAL,
                crazy_affine_count INTEGER NOT NULL,
                trajectory_path TEXT,
                keyframe_trajectory_path TEXT,
                log_path TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(label, replicate)
            )
            """
        )
        existing_columns = {
            row[1]
            for row in self.connection.execute("PRAGMA table_info(evaluations)")
        }
        for column in (
            "historical_evo_ape_rmse_m",
            "historical_evo_ape_mean_m",
            "historical_evo_rpe_rmse_m",
        ):
            if column not in existing_columns:
                self.connection.execute(
                    f"ALTER TABLE evaluations ADD COLUMN {column} REAL"
                )
        self.connection.commit()

    def has(self, label: str, replicate: int) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM evaluations WHERE label=? AND replicate=?",
                (label, replicate),
            ).fetchone()
            is not None
        )

    def add(self, result: FullSequenceResult) -> None:
        payload = asdict(result)
        payload["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        columns = list(payload)
        self.connection.execute(
            f"INSERT OR REPLACE INTO evaluations ({','.join(columns)}) "
            f"VALUES ({','.join('?' for _ in columns)})",
            [payload[column] for column in columns],
        )
        self.connection.commit()

    def rows(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM evaluations ORDER BY id"
        ).fetchall()

    def update_historical_evo_ape(
        self, label: str, replicate: int, rmse_m: float, mean_m: float
    ) -> None:
        self.connection.execute(
            """
            UPDATE evaluations
            SET historical_evo_ape_rmse_m=?, historical_evo_ape_mean_m=?
            WHERE label=? AND replicate=?
            """,
            (rmse_m, mean_m, label, replicate),
        )
        self.connection.commit()

    def update_historical_evo_rpe(
        self, label: str, replicate: int, rmse_m: float
    ) -> None:
        self.connection.execute(
            """
            UPDATE evaluations
            SET historical_evo_rpe_rmse_m=?
            WHERE label=? AND replicate=?
            """,
            (rmse_m, label, replicate),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Full-sequence validation of the final Conv1 channel shortlist.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--candidate-plan",
        type=Path,
        help="JSON candidate plan generated by prepare_second_round_candidates.py.",
    )
    parser.add_argument("--como-dir", type=Path, default=COMO_DIR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--evo-ape", type=Path, default=DEFAULT_EVO_APE)
    parser.add_argument("--evo-rpe", type=Path, default=DEFAULT_EVO_RPE)
    parser.add_argument(
        "--refresh-historical-metrics",
        action="store_true",
        help=(
            "Recompute the historical run_random_channel_search.sh ATE from "
            "saved keyframe trajectories without rerunning COMO."
        ),
    )
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--batch-hours",
        type=float,
        help="Stop cleanly between candidates after this many wall-clock hours.",
    )
    parser.add_argument("--terminate-grace-seconds", type=float, default=3.0)
    parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=0.90,
        help="Minimum saved all-frame trajectory poses / matched RGB timestamps.",
    )
    parser.add_argument(
        "--completion-tolerance-seconds",
        type=float,
        default=0.10,
        help="Allowed gap between final trajectory and final dataset timestamp.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional subset of named configurations.",
    )
    return parser.parse_args()


def load_candidate_plan(path: Path) -> tuple[EvaluationSpec, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Candidate plan has no non-empty 'candidates' list: {path}")
    specs: list[EvaluationSpec] = []
    labels: set[str] = set()
    candidate_keys: set[str] = set()
    for index, item in enumerate(candidates, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Candidate {index} is not a JSON object")
        label = str(item.get("label", "")).strip()
        raw_channels = item.get("channels")
        role = str(item.get("role", "second-round MVS-qualified candidate"))
        if not label or label in labels:
            raise ValueError(f"Candidate {index} has a missing/duplicate label: {label!r}")
        if raw_channels is None:
            channels = None
        else:
            if not isinstance(raw_channels, list) or len(raw_channels) != 4:
                raise ValueError(
                    f"Candidate {index} must contain exactly four channels or "
                    "use null for the gray control"
                )
            channels = tuple(sorted(int(channel) for channel in raw_channels))
            if len(set(channels)) != 4 or channels[0] < 0 or channels[-1] >= 64:
                raise ValueError(f"Candidate {index} has invalid channels: {channels}")
        spec = EvaluationSpec(label, channels, role)
        if spec.candidate_key in candidate_keys:
            raise ValueError(
                f"Candidate {index} duplicates candidate key {spec.candidate_key}"
            )
        declared_key = item.get("candidate_key")
        if declared_key is not None and str(declared_key) != spec.candidate_key:
            raise ValueError(
                f"Candidate {index} declares key {declared_key!r}, but its "
                f"channels resolve to {spec.candidate_key!r}"
            )
        labels.add(label)
        candidate_keys.add(spec.candidate_key)
        specs.append(spec)
    expected_count = payload.get("selection", {}).get("selected_count")
    if expected_count is not None and int(expected_count) != len(specs):
        raise ValueError(
            f"Candidate plan expected {expected_count} rows but contains {len(specs)}"
        )
    return tuple(specs)


def read_timestamp_index(path: Path) -> list[float]:
    timestamps: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            timestamps.append(float(stripped.split()[0]))
    if not timestamps:
        raise ValueError(f"No timestamps found in {path}")
    if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
        raise ValueError(f"Timestamps are not strictly increasing: {path}")
    return timestamps


def nearest_frame_index(timestamp: float | None, timestamps: Sequence[float]) -> int | None:
    if timestamp is None:
        return None
    return int(np.argmin(np.abs(np.asarray(timestamps) - timestamp)))


def evaluate_historical_evo_ape(
    evo_ape: Path, groundtruth_path: Path, keyframe_trajectory_path: Path
) -> dict[str, float]:
    """Run the exact ATE command used by run_random_channel_search.sh."""
    command = [
        str(evo_ape),
        "tum",
        str(groundtruth_path),
        str(keyframe_trajectory_path),
        "--align",
        "--correct_scale",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"evo_ape exited with status {completed.returncode}: {detail}"
        )
    statistics: dict[str, float] = {}
    for line in completed.stdout.splitlines():
        fields = line.strip().split()
        if len(fields) == 2 and fields[0] in {"mean", "rmse"}:
            statistics[fields[0]] = float(fields[1])
    if "mean" not in statistics or "rmse" not in statistics:
        raise ValueError(f"Could not parse evo_ape statistics:\n{completed.stdout}")
    return statistics


def evaluate_historical_evo_rpe(
    evo_rpe: Path, groundtruth_path: Path, keyframe_trajectory_path: Path
) -> float:
    """Run the exact RPE command used by run_ate_multi_seq_local.sh."""
    command = [
        str(evo_rpe),
        "tum",
        str(groundtruth_path),
        str(keyframe_trajectory_path),
        "--align",
        "--correct_scale",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"evo_rpe exited with status {completed.returncode}: {detail}"
        )
    for line in completed.stdout.splitlines():
        fields = line.strip().split()
        if len(fields) == 2 and fields[0] == "rmse":
            return float(fields[1])
    raise ValueError(f"Could not parse evo_rpe RMSE:\n{completed.stdout}")


def evaluate_all_frames(
    trajectory_path: Path, groundtruth_path: Path
) -> tuple[dict[str, float], int, int, float, bool]:
    estimate_t, estimate_xyz, estimate_q = core.read_tum_poses(trajectory_path)
    if len(estimate_t) < 3:
        raise ValueError(f"Only {len(estimate_t)} all-frame trajectory poses were saved")
    reference_t, reference_xyz, reference_q = core.read_tum_poses(groundtruth_path)
    reference, estimate, reference_quat, estimate_quat = core.associate_poses(
        reference_t,
        reference_xyz,
        reference_q,
        estimate_t,
        estimate_xyz,
        estimate_q,
    )
    if len(estimate) < 3:
        raise ValueError(
            f"Only {len(estimate)} all-frame poses could be associated to ground truth"
        )
    metrics = core.trajectory_error_metrics(
        reference, estimate, reference_quat, estimate_quat
    )
    frozen = False
    if len(estimate) >= 30:
        estimate_motion = float(
            np.max(np.linalg.norm(estimate[-30:] - estimate[-1], axis=1))
        )
        reference_motion = float(
            np.max(np.linalg.norm(reference[-30:] - reference[-1], axis=1))
        )
        frozen = estimate_motion < 1e-4 and reference_motion > 1e-3
    return metrics, len(estimate_t), len(estimate), float(estimate_t[-1]), frozen


def populate_diagnostics(
    result: FullSequenceResult,
    diagnostics: Sequence[dict[str, float | int | bool | str]],
) -> None:
    result.diagnostic_frames = len(diagnostics)
    photo_values = core.finite_diag_values(diagnostics, "photo_mse")
    valid_ratios = core.finite_diag_values(diagnostics, "valid_ratio")
    h_conditions = core.finite_diag_values(diagnostics, "h_cond")
    delta_norms = core.finite_diag_values(diagnostics, "delta_norm")
    result.photo_mse_nonfinite_count = sum(
        1
        for item in diagnostics
        if "photo_mse" in item
        and isinstance(item["photo_mse"], (int, float))
        and not math.isfinite(float(item["photo_mse"]))
    )
    if len(photo_values):
        result.photo_mse_median = float(np.median(photo_values))
        result.photo_mse_p95 = float(np.percentile(photo_values, 95))
    if len(valid_ratios):
        result.valid_ratio_min = float(np.min(valid_ratios))
        result.valid_ratio_median = float(np.median(valid_ratios))
    if len(h_conditions):
        result.h_cond_max = float(np.max(h_conditions))
    if len(delta_norms):
        result.delta_norm_max = float(np.max(delta_norms))


def evaluate_one(
    args: argparse.Namespace,
    console: Console,
    store: ResultStore,
    guard,
    spec: EvaluationSpec,
    replicate: int,
    run_index: int,
    total_runs: int,
    matched_timestamps: Sequence[float],
) -> FullSequenceResult | None:
    if store.has(spec.label, replicate) and not args.rerun_existing:
        console.say(
            f"[SKIP] {spec.label} replicate={replicate} already has a saved result"
        )
        return None

    console.say("")
    console.say(
        f"[RUN {run_index}/{total_runs}] label={spec.label} replicate={replicate} "
        f"channels={spec.display} role={spec.role}"
    )
    candidate = core.Candidate(spec.channels, spec.label)
    config = guard.apply(candidate)
    config_digest = hashlib.sha256(
        yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    console.say(
        f"[CONFIG] tracking={config['tracking']['color']} mapping=gray "
        f"config_sha256={config_digest} timeout={args.timeout_seconds:.1f}s"
    )

    trajectory_source = args.como_dir / "results/data_tum_all_frames.txt"
    keyframe_source = args.como_dir / "results/data_tum.txt"
    trajectory_source.unlink(missing_ok=True)
    keyframe_source.unlink(missing_ok=True)
    command = [
        str(args.python),
        "-u",
        "como/como_dataset.py",
        "--dataset_type=tum",
        f"--dataset_dir={args.dataset_dir}",
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["COMO_SAVE_ALL_FRAME_TRAJECTORY"] = "1"
    started = time.monotonic()
    lines: list[str] = []
    diagnostics: list[dict[str, float | int | bool | str]] = []
    crazy_affine_count = 0
    failure_status: str | None = None
    failure_reason = ""
    failure_timestamp: float | None = None
    last_tracking_timestamp: float | None = None
    process: subprocess.Popen[str] | None = None
    exit_code: int | None = None

    try:
        process = subprocess.Popen(
            command,
            cwd=args.como_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
            env=environment,
        )
        assert process.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            try:
                assert process is not None and process.stdout is not None
                for output_line in process.stdout:
                    output_queue.put(output_line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(
            target=read_output,
            name=f"full-sequence-{spec.label}-rep{replicate}",
            daemon=True,
        )
        reader.start()
        eof = False
        while not eof:
            elapsed = time.monotonic() - started
            if elapsed > args.timeout_seconds:
                failure_status = "TIMEOUT"
                failure_reason = f"Exceeded {args.timeout_seconds:.1f}s wall-clock timeout"
                console.say(f"[FAIL-FAST] {failure_reason}")
                core.terminate_process_group(process, args.terminate_grace_seconds)
                break
            try:
                line = output_queue.get(timeout=0.2)
            except queue.Empty:
                if process.poll() is not None:
                    reader.join(timeout=1.0)
                continue
            if line is None:
                eof = True
                continue
            stripped = line.rstrip("\n")
            lines.append(stripped)
            timestamp_match = core.TIMESTAMP_RE.search(stripped)
            current_timestamp = (
                float(timestamp_match.group(1)) if timestamp_match else None
            )
            if current_timestamp is not None and (
                "[KF aff received]" in stripped or "[TRACK_DIAG]" in stripped
            ):
                last_tracking_timestamp = current_timestamp
            if "Crazy affine detected" in stripped:
                crazy_affine_count += 1
            diagnostic = core.parse_tracking_diag(stripped)
            if diagnostic is not None:
                diagnostics.append(diagnostic)
            if core.TRACKING_NAN_RE.search(stripped) or core.TRACKING_DIAG_NONFINITE_RE.search(
                stripped
            ):
                failure_status = "FAIL_TRACKING_NAN"
                failure_reason = "Tracking emitted non-finite affine/pose diagnostics"
                failure_timestamp = current_timestamp or last_tracking_timestamp
            elif any(marker in stripped for marker in core.KNOWN_RUNTIME_MARKERS):
                failure_status = "FAIL_TRACKING_RUNTIME"
                failure_reason = "Recognised COMO/Open3D tracking runtime failure"
                failure_timestamp = current_timestamp or last_tracking_timestamp
            if failure_status is not None:
                frame = nearest_frame_index(failure_timestamp, matched_timestamps)
                console.say(
                    f"[FAIL-FAST] status={failure_status} frame={frame} "
                    f"ts={failure_timestamp if failure_timestamp is not None else 'n/a'}; "
                    "terminating this run"
                )
                core.terminate_process_group(process, args.terminate_grace_seconds)
                break
        if process.poll() is None:
            core.terminate_process_group(process, args.terminate_grace_seconds)
        exit_code = process.returncode
    except BaseException:
        if process is not None:
            core.terminate_process_group(process, args.terminate_grace_seconds)
        raise
    finally:
        guard.restore()

    elapsed = time.monotonic() - started
    result = FullSequenceResult(
        label=spec.label,
        role=spec.role,
        candidate_key=spec.candidate_key,
        channels_json=json.dumps(spec.channels) if spec.channels is not None else None,
        replicate=replicate,
        status=failure_status or "PENDING",
        reason=failure_reason,
        elapsed_seconds=elapsed,
        exit_code=exit_code,
        failure_timestamp=failure_timestamp,
        failure_frame_index=nearest_frame_index(
            failure_timestamp, matched_timestamps
        ),
        expected_matched_frames=len(matched_timestamps),
        crazy_affine_count=crazy_affine_count,
    )
    populate_diagnostics(result, diagnostics)

    artifact_dir = args.output_dir / "artifacts" / spec.label
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / f"replicate_{replicate}.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result.log_path = str(log_path)

    if result.status == "PENDING":
        if exit_code != 0:
            result.status = "ERROR_RUNTIME"
            result.reason = (
                f"COMO exited with code {exit_code} without a recognised failure signature"
            )
        elif not trajectory_source.is_file():
            result.status = "FAIL_INVALID_TRAJECTORY"
            result.reason = "COMO did not produce data_tum_all_frames.txt"
        else:
            try:
                metrics, trajectory_poses, associated, last_timestamp, frozen = (
                    evaluate_all_frames(trajectory_source, args.dataset_dir / "groundtruth.txt")
                )
                result.trajectory_poses = trajectory_poses
                result.associated_poses = associated
                result.last_timestamp = last_timestamp
                # Completeness is about whether tracking produced the sequence,
                # not whether GT happens to cover every input timestamp.  fr2
                # ground truth has gaps, while the saved trajectory is complete.
                result.coverage_ratio = trajectory_poses / len(matched_timestamps)
                if frozen:
                    result.status = "FAIL_POSE_FROZEN"
                    result.reason = "Final 30 poses froze while ground truth kept moving"
                elif result.coverage_ratio < args.minimum_coverage:
                    result.status = "FAIL_INCOMPLETE"
                    result.reason = (
                        f"Coverage {result.coverage_ratio:.3f} is below "
                        f"{args.minimum_coverage:.3f}"
                    )
                elif last_timestamp < (
                    matched_timestamps[-1] - args.completion_tolerance_seconds
                ):
                    result.status = "FAIL_INCOMPLETE"
                    result.reason = (
                        f"Trajectory ended at {last_timestamp:.6f}, before final dataset "
                        f"timestamp {matched_timestamps[-1]:.6f}"
                    )
                else:
                    result.status = "PASS"
                    result.reason = "Completed the full matched RGB-D sequence"
                    result.se3_ate_rmse_m = metrics["rmse"]
                    result.se3_ate_mean_m = metrics["mean"]
                    result.se3_ate_median_m = metrics["median"]
                    result.se3_ate_max_m = metrics["max"]
                    result.se3_ate_std_m = metrics["std"]
                    result.rotation_ape_rmse_deg = metrics["rotation_ape_rmse_deg"]
                    result.rotation_ape_mean_deg = metrics["rotation_ape_mean_deg"]
                    result.rotation_ape_max_deg = metrics["rotation_ape_max_deg"]
                    result.translation_rpe_rmse_m = metrics["translation_rpe_rmse"]
                    result.translation_rpe_max_m = metrics["translation_rpe_max"]
                    result.rotation_rpe_rmse_deg = metrics["rotation_rpe_rmse_deg"]
                    result.rotation_rpe_max_deg = metrics["rotation_rpe_max_deg"]
                    result.allframe_sim3_rmse_m = metrics["legacy_sim3_rmse"]
                    result.allframe_sim3_mean_m = metrics["legacy_sim3_mean"]
                    result.allframe_sim3_scale = metrics["legacy_sim3_scale"]
                    if keyframe_source.is_file():
                        try:
                            legacy, legacy_poses = core.evaluate_legacy_keyframe_sim3(
                                keyframe_source,
                                args.dataset_dir / "groundtruth.txt",
                                matched_timestamps[0],
                                matched_timestamps[-1],
                            )
                            result.keyframe_associated_poses = legacy_poses
                            if legacy is not None:
                                result.keyframe_sim3_rmse_m = legacy[
                                    "legacy_sim3_rmse"
                                ]
                                result.keyframe_sim3_mean_m = legacy[
                                    "legacy_sim3_mean"
                                ]
                                result.keyframe_sim3_scale = legacy[
                                    "legacy_sim3_scale"
                                ]
                        except (FloatingPointError, ValueError, OSError) as error:
                            console.say(
                                "[LEGACY DIAGNOSTIC UNAVAILABLE] "
                                f"label={spec.label} reason={error}"
                            )
                        try:
                            historical = evaluate_historical_evo_ape(
                                args.evo_ape,
                                args.dataset_dir / "groundtruth.txt",
                                keyframe_source,
                            )
                            result.historical_evo_ape_rmse_m = historical["rmse"]
                            result.historical_evo_ape_mean_m = historical["mean"]
                        except (RuntimeError, ValueError, OSError) as error:
                            console.say(
                                "[HISTORICAL ATE UNAVAILABLE] "
                                f"label={spec.label} reason={error}"
                            )
                        try:
                            result.historical_evo_rpe_rmse_m = (
                                evaluate_historical_evo_rpe(
                                    args.evo_rpe,
                                    args.dataset_dir / "groundtruth.txt",
                                    keyframe_source,
                                )
                            )
                        except (RuntimeError, ValueError, OSError) as error:
                            console.say(
                                "[HISTORICAL RPE UNAVAILABLE] "
                                f"label={spec.label} reason={error}"
                            )
            except FloatingPointError as error:
                result.status = "FAIL_INVALID_TRAJECTORY"
                result.reason = str(error)
            except (ValueError, OSError) as error:
                result.status = "ERROR_TRAJECTORY_EVALUATION"
                result.reason = str(error)

    if trajectory_source.is_file():
        saved = artifact_dir / f"replicate_{replicate}.all_frames.tum.txt"
        shutil.copy2(trajectory_source, saved)
        result.trajectory_path = str(saved)
    if keyframe_source.is_file():
        saved = artifact_dir / f"replicate_{replicate}.keyframes.tum.txt"
        shutil.copy2(keyframe_source, saved)
        result.keyframe_trajectory_path = str(saved)

    store.add(result)
    if result.status == "PASS":
        console.say(
            f"[PASS] label={spec.label} channels={spec.display} "
            f"SE3_ATE_RMSE={result.se3_ate_rmse_m * 100:.4f}cm "
            f"SE3_ATE_mean={result.se3_ate_mean_m * 100:.4f}cm "
            f"Trans_RPE_max={result.translation_rpe_max_m * 100:.3f}cm "
            f"Rot_RPE_max={result.rotation_rpe_max_deg:.3f}deg "
            f"coverage={result.trajectory_poses}/{result.expected_matched_frames} "
            f"GT_associated={result.associated_poses} "
            f"historical_evo_ATE_mean="
            f"{core.format_optional_scaled(result.historical_evo_ape_mean_m, 100, '.4f')}"
            f"{'cm' if result.historical_evo_ape_mean_m is not None else ''} "
            f"historical_evo_RPE_RMSE="
            f"{core.format_optional_scaled(result.historical_evo_rpe_rmse_m, 100, '.4f')}"
            f"{'cm' if result.historical_evo_rpe_rmse_m is not None else ''} "
            f"runtime={elapsed:.1f}s"
        )
    else:
        console.say(
            f"[{result.status}] label={spec.label} channels={spec.display} "
            f"frame={result.failure_frame_index} runtime={elapsed:.1f}s "
            f"reason={result.reason}"
        )
    return result


def export_results(store: ResultStore, output_dir: Path) -> None:
    rows = store.rows()
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        columns = [key for key in rows[0].keys() if key != "id"]
        with (output_dir / "all_evaluations.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([row[column] for column in columns])

    passes = [row for row in rows if row["status"] == "PASS"]
    passes.sort(
        key=lambda row: (
            row["historical_evo_ape_mean_m"] is None,
            row["historical_evo_ape_mean_m"]
            if row["historical_evo_ape_mean_m"] is not None
            else math.inf,
            row["se3_ate_rmse_m"],
        )
    )
    with (output_dir / "pass_ranking.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "label",
                "channels",
                "role",
                "replicate",
                "coverage",
                "historical_evo_ape_mean_cm",
                "historical_evo_ape_rmse_cm",
                "historical_evo_rpe_rmse_cm",
                "se3_ate_rmse_cm",
                "se3_ate_mean_cm",
                "se3_ate_max_cm",
                "translation_rpe_rmse_cm",
                "translation_rpe_max_cm",
                "rotation_rpe_rmse_deg",
                "rotation_rpe_max_deg",
                "allframe_sim3_mean_cm",
                "allframe_sim3_scale",
                "keyframe_sim3_mean_cm",
                "keyframe_sim3_scale",
            ]
        )
        for rank, row in enumerate(passes, start=1):
            writer.writerow(
                [
                    rank,
                    row["label"],
                    row["candidate_key"],
                    row["role"],
                    row["replicate"],
                    row["coverage_ratio"],
                    (
                        row["historical_evo_ape_mean_m"] * 100
                        if row["historical_evo_ape_mean_m"] is not None
                        else ""
                    ),
                    (
                        row["historical_evo_ape_rmse_m"] * 100
                        if row["historical_evo_ape_rmse_m"] is not None
                        else ""
                    ),
                    (
                        row["historical_evo_rpe_rmse_m"] * 100
                        if row["historical_evo_rpe_rmse_m"] is not None
                        else ""
                    ),
                    row["se3_ate_rmse_m"] * 100,
                    row["se3_ate_mean_m"] * 100,
                    row["se3_ate_max_m"] * 100,
                    row["translation_rpe_rmse_m"] * 100,
                    row["translation_rpe_max_m"] * 100,
                    row["rotation_rpe_rmse_deg"],
                    row["rotation_rpe_max_deg"],
                    row["allframe_sim3_mean_m"] * 100,
                    row["allframe_sim3_scale"],
                    (
                        row["keyframe_sim3_mean_m"] * 100
                        if row["keyframe_sim3_mean_m"] is not None
                        else ""
                    ),
                    row["keyframe_sim3_scale"],
                ]
            )


def refresh_historical_metrics(
    store: ResultStore, args: argparse.Namespace, console: Console
) -> None:
    refreshed = 0
    unavailable = 0
    for row in store.rows():
        if row["status"] != "PASS" or not row["keyframe_trajectory_path"]:
            continue
        trajectory = Path(row["keyframe_trajectory_path"])
        if not trajectory.is_file():
            console.say(
                f"[HISTORICAL REFRESH SKIP] label={row['label']} missing={trajectory}"
            )
            unavailable += 1
            continue
        try:
            metrics = evaluate_historical_evo_ape(
                args.evo_ape,
                args.dataset_dir / "groundtruth.txt",
                trajectory,
            )
        except (RuntimeError, ValueError, OSError) as error:
            console.say(
                f"[HISTORICAL REFRESH SKIP] label={row['label']} reason={error}"
            )
            unavailable += 1
            continue
        store.update_historical_evo_ape(
            row["label"], row["replicate"], metrics["rmse"], metrics["mean"]
        )
        try:
            rpe_rmse = evaluate_historical_evo_rpe(
                args.evo_rpe,
                args.dataset_dir / "groundtruth.txt",
                trajectory,
            )
        except (RuntimeError, ValueError, OSError) as error:
            console.say(
                f"[HISTORICAL RPE REFRESH SKIP] label={row['label']} reason={error}"
            )
            rpe_rmse = None
        if rpe_rmse is not None:
            store.update_historical_evo_rpe(
                row["label"], row["replicate"], rpe_rmse
            )
        console.say(
            f"[HISTORICAL REFRESH] label={row['label']} "
            f"evo_ATE_mean={metrics['mean'] * 100:.4f}cm "
            f"evo_ATE_RMSE={metrics['rmse'] * 100:.4f}cm "
            f"evo_RPE_RMSE="
            f"{core.format_optional_scaled(rpe_rmse, 100, '.4f')}"
            f"{'cm' if rpe_rmse is not None else ''}"
        )
        refreshed += 1
    console.say(
        f"[HISTORICAL REFRESH COMPLETE] refreshed={refreshed} unavailable={unavailable}"
    )


def validate_inputs(args: argparse.Namespace) -> tuple[list[float], tuple[EvaluationSpec, ...]]:
    required = [
        args.dataset_dir / "matched_rgb.txt",
        args.dataset_dir / "groundtruth.txt",
        args.como_dir / "config/como.yml",
        args.como_dir / "como/como_dataset.py",
        args.python,
        args.evo_ape,
        args.evo_rpe,
        CORE_SCRIPT,
    ]
    if args.candidate_plan is not None:
        required.append(args.candidate_plan)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths: " + ", ".join(map(str, missing)))
    if args.replicates < 1:
        raise ValueError("--replicates must be at least 1")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if args.batch_hours is not None and args.batch_hours <= 0:
        raise ValueError("--batch-hours must be positive")
    if not 0 < args.minimum_coverage <= 1:
        raise ValueError("--minimum-coverage must be in (0,1]")
    timestamps = read_timestamp_index(args.dataset_dir / "matched_rgb.txt")
    available = (
        load_candidate_plan(args.candidate_plan)
        if args.candidate_plan is not None
        else EVALUATION_SPECS
    )
    requested = set(args.only) if args.only is not None else None
    if requested is not None:
        unknown = requested - {spec.label for spec in available}
        if unknown:
            raise ValueError("Unknown --only labels: " + ", ".join(sorted(unknown)))
    selected = tuple(
        spec for spec in available if requested is None or spec.label in requested
    )
    if not selected:
        raise ValueError("No evaluation configurations were selected")
    return timestamps, selected


def main() -> None:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.como_dir = args.como_dir.resolve()
    args.python = args.python.resolve()
    args.evo_ape = args.evo_ape.resolve()
    args.evo_rpe = args.evo_rpe.resolve()
    if args.candidate_plan is not None:
        args.candidate_plan = args.candidate_plan.resolve()
    timestamps, specs = validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    console = Console(args.output_dir / "console.log")
    store = ResultStore(args.output_dir / "evaluations.sqlite3")
    plan = {
        "dataset": str(args.dataset_dir),
        "matched_frames": len(timestamps),
        "first_timestamp": timestamps[0],
        "last_timestamp": timestamps[-1],
        "minimum_coverage": args.minimum_coverage,
        "replicates": args.replicates,
        "candidate_plan": str(args.candidate_plan) if args.candidate_plan else None,
        "candidate_plan_sha256": (
            hashlib.sha256(args.candidate_plan.read_bytes()).hexdigest()
            if args.candidate_plan
            else None
        ),
        "timeout_seconds": args.timeout_seconds,
        "batch_hours": args.batch_hours,
        "configurations": [asdict(spec) for spec in specs],
        "primary_metric": (
            "historical keyframe evo_ape mean: tum GT data_tum.txt --align "
            "--correct_scale"
        ),
        "diagnostic_metrics": [
            "historical keyframe evo_rpe RMSE with --align --correct_scale",
            "all-frame metric-scale SE(3) translation ATE/RPE",
        ],
    }
    (args.output_dir / "evaluation_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )
    console.say("=" * 78)
    console.say("FULL-SEQUENCE CONV1 VALIDATION: TUM dataset")
    console.say("=" * 78)
    console.say(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    console.say(
        f"Dataset: {args.dataset_dir}; matched frames={len(timestamps)}; "
        f"time={timestamps[0]:.6f}--{timestamps[-1]:.6f}"
    )
    console.say(
        "Primary: historical keyframe evo_ape ATE mean using --align "
        "--correct_scale (identical to run_random_channel_search.sh)"
    )
    console.say(
        "Diagnostics: historical keyframe evo_rpe plus all-frame metric-scale "
        "SE(3) ATE/RPE remain available but do not drive the primary ranking"
    )
    console.say(
        f"Completeness gate: coverage >= {args.minimum_coverage:.1%} and final pose "
        f"within {args.completion_tolerance_seconds:.2f}s of sequence end"
    )
    console.say(
        f"Configurations: {len(specs)}; replicates={args.replicates}; "
        f"planned runs={len(specs) * args.replicates}"
    )
    preview_indices = set(range(min(5, len(specs)))) | set(
        range(max(0, len(specs) - 5), len(specs))
    )
    for index, spec in enumerate(specs, start=1):
        if index - 1 in preview_indices:
            console.say(
                f"[PLAN {index:04d}] {spec.label}: channels={spec.display}; {spec.role}"
            )
        elif index == 6:
            console.say(f"[PLAN] ... {len(specs) - 10:,} intermediate candidates ...")
    if args.refresh_historical_metrics:
        refresh_historical_metrics(store, args, console)
    if not args.execute:
        export_results(store, args.output_dir)
        console.say("DRY RUN COMPLETE: add --execute to launch COMO")
        store.close()
        console.close()
        return

    guard = core.ConfigGuard(args.como_dir / "config/como.yml", SHARED_CONFIG_LOCK)
    try:
        run_specs = [
            (spec, replicate)
            for replicate in range(args.replicates)
            for spec in specs
        ]
        batch_started = time.monotonic()
        evaluated_this_batch = 0
        for index, (spec, replicate) in enumerate(run_specs, start=1):
            if (
                args.batch_hours is not None
                and evaluated_this_batch > 0
                and time.monotonic() - batch_started >= args.batch_hours * 3600
            ):
                console.say(
                    f"[BATCH COMPLETE] Reached {args.batch_hours:.2f}h after "
                    f"{evaluated_this_batch:,} new evaluations; stopping cleanly "
                    f"before planned run {index:,}/{len(run_specs):,}"
                )
                break
            result = evaluate_one(
                args,
                console,
                store,
                guard,
                spec,
                replicate,
                index,
                len(run_specs),
                timestamps,
            )
            if result is not None:
                evaluated_this_batch += 1
                completed = store.connection.execute(
                    "SELECT COUNT(*) FROM evaluations"
                ).fetchone()[0]
                status_rows = store.connection.execute(
                    "SELECT status, COUNT(*) FROM evaluations GROUP BY status"
                ).fetchall()
                statuses = ", ".join(
                    f"{row[0]}={row[1]:,}" for row in status_rows
                )
                remaining = max(0, len(run_specs) - completed)
                recent_seconds = time.monotonic() - batch_started
                average_seconds = recent_seconds / evaluated_this_batch
                eta_hours = remaining * average_seconds / 3600
                console.say(
                    f"[PROGRESS] preserved={completed:,}/{len(run_specs):,} "
                    f"remaining={remaining:,} {statuses} "
                    f"batch_avg={average_seconds:.1f}s ETA={eta_hours:.1f}h"
                )
            export_results(store, args.output_dir)
        console.say("[DONE] Full-sequence evaluation finished; CSV summaries exported")
    finally:
        guard.close()
        export_results(store, args.output_dir)
        store.close()
        console.close()


if __name__ == "__main__":
    main()
