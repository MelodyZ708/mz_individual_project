#!/usr/bin/env python3
"""Fail-fast r=0.70 Conv1 search on the 50-frame failure-centred MVS.

The script is deliberately safe by default: without ``--execute`` it only
validates inputs, enumerates candidates, and prints the planned work.  It runs
COMO serially because every evaluation shares one GPU and one upstream config
file.  Results are resumable through SQLite.

Stages
------
regression
    Gray must fail near the known anchor; [5,29,40,52] must complete.
bruteforce
    Exhaust all legal four-channel combinations of the r=0.70 representatives.
swapback
    One-member swap-back for Top-20 plus cluster-coverage contexts, followed by
    factorial swap-back around the current Top-5.
rescue
    Insert r=0.80 representatives removed by r=0.70 into Top-20 contexts.  This
    intentionally permits same-r=0.70-cluster co-selection to audit aggressive
    merges.
repeat
    Give the final Top-20 three total observations (the search observation plus
    two fresh repeats) and report ATE RMSE/mean variability.
all
    Run the stages above in order.

ATE is computed over the scored MVS window only (indices 10--49), with Sim(3)
Umeyama alignment.  Values are stored in metres and printed in centimetres.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import fcntl
import hashlib
import itertools
import json
import math
import os
import queue
import re
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMO_DIR = PROJECT_ROOT / "como"
DEFAULT_DATASET = Path(
    "/home/melody/data/tum/"
    "rgbd_dataset_freiburg1_desk_lightswitch_"
    "mvs_failure_anchor_idx248_brighten_dim_50f"
)
DEFAULT_R070 = (
    PROJECT_ROOT / "channel_selection_results/step_b_correlation_clustering/"
    "threshold_r070/clusters/clusters_conv1.json"
)
DEFAULT_R080 = (
    PROJECT_ROOT / "channel_selection_results/step_b_correlation_clustering/"
    "threshold_r080/clusters/clusters_conv1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "channel_selection_results/step_d_fail_fast_evaluation/r070_bruteforce_v2"
)
DEFAULT_PYTHON = Path("/home/melody/anaconda3/envs/como/bin/python")

KNOWN_GOOD = (5, 29, 40, 52)
TRANSLATION_RPE_MAX_M = 0.06
ROTATION_RPE_MAX_DEG = 5.0
TRACKING_NAN_RE = re.compile(
    r"\[KF aff received\].*(?:\bnan\b|[+-]?\binf\b)", re.IGNORECASE
)
TRACKING_DIAG_NONFINITE_RE = re.compile(
    r"\[TRACK_DIAG\].*(?:non.?finite|pose_nan|invalid pose|"
    r"(?:sigma_r|res_med|res_mean|res_max|delta_norm)=[+-]?(?:nan|inf)\b|"
    r"pose_finite=False)",
    re.IGNORECASE,
)
TIMESTAMP_RE = re.compile(r"\bts=([0-9]+(?:\.[0-9]+)?)")
KNOWN_RUNTIME_MARKERS = (
    "RuntimeError: Caught an unknown exception!",
    "AABB can't be empty",
)
FAIL_STATUSES = {
    "FAIL_TRACKING_NAN",
    "FAIL_TRACKING_DIAGNOSTIC",
    "FAIL_TRACKING_RUNTIME",
    "FAIL_INCOMPLETE",
    "FAIL_POSE_FROZEN",
    "FAIL_INVALID_TRAJECTORY",
    "TIMEOUT",
}
RETRYABLE_BRUTEFORCE_STATUSES = {
    "ERROR_RUNTIME",
    "ERROR_TRAJECTORY_EVALUATION",
    "TIMEOUT",
}


@dataclass(frozen=True)
class Candidate:
    channels: tuple[int, ...] | None
    label: str = ""

    @property
    def key(self) -> str:
        if self.channels is None:
            return "gray"
        return ",".join(str(channel) for channel in sorted(self.channels))

    @property
    def display(self) -> str:
        if self.channels is None:
            return "gray"
        return "[" + ",".join(str(channel) for channel in self.channels) + "]"


@dataclass
class Evaluation:
    stage: str
    candidate_key: str
    channels: tuple[int, ...] | None
    replicate: int
    status: str
    reason: str
    elapsed_seconds: float
    exit_code: int | None
    failure_timestamp: float | None = None
    failure_mvs_index: int | None = None
    last_timestamp: float | None = None
    associated_poses: int | None = None
    full_associated_poses: int | None = None
    ate_rmse_m: float | None = None
    ate_mean_m: float | None = None
    ate_median_m: float | None = None
    ate_max_m: float | None = None
    ate_std_m: float | None = None
    rotation_ape_rmse_deg: float | None = None
    rotation_ape_mean_deg: float | None = None
    rotation_ape_max_deg: float | None = None
    translation_rpe_rmse_m: float | None = None
    translation_rpe_max_m: float | None = None
    rotation_rpe_rmse_deg: float | None = None
    rotation_rpe_max_deg: float | None = None
    legacy_sim3_rmse_m: float | None = None
    legacy_sim3_mean_m: float | None = None
    legacy_sim3_scale: float | None = None
    legacy_keyframe_poses: int | None = None
    full_ate_rmse_m: float | None = None
    full_rotation_ape_rmse_deg: float | None = None
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
    log_tail: str = ""


class Console:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        self.handle.close()

    def say(self, message: str = "") -> None:
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}" if message else ""
        print(line, flush=True)
        self.handle.write(line + "\n")


class ResultStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        # Each candidate takes seconds to evaluate, so the small fsync overhead
        # is worth making every committed result durable across a hard reset.
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage TEXT NOT NULL,
                candidate_key TEXT NOT NULL,
                channels_json TEXT,
                replicate INTEGER NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                elapsed_seconds REAL NOT NULL,
                exit_code INTEGER,
                failure_timestamp REAL,
                failure_mvs_index INTEGER,
                last_timestamp REAL,
                associated_poses INTEGER,
                full_associated_poses INTEGER,
                ate_rmse_m REAL,
                ate_mean_m REAL,
                ate_median_m REAL,
                ate_max_m REAL,
                ate_std_m REAL,
                rotation_ape_rmse_deg REAL,
                rotation_ape_mean_deg REAL,
                rotation_ape_max_deg REAL,
                translation_rpe_rmse_m REAL,
                translation_rpe_max_m REAL,
                rotation_rpe_rmse_deg REAL,
                rotation_rpe_max_deg REAL,
                legacy_sim3_rmse_m REAL,
                legacy_sim3_mean_m REAL,
                legacy_sim3_scale REAL,
                legacy_keyframe_poses INTEGER,
                full_ate_rmse_m REAL,
                full_rotation_ape_rmse_deg REAL,
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
                log_tail TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(stage, candidate_key, replicate)
            )
            """
        )
        existing_columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(evaluations)")
        }
        if "legacy_keyframe_poses" not in existing_columns:
            self.connection.execute(
                "ALTER TABLE evaluations ADD COLUMN legacy_keyframe_poses INTEGER"
            )
        if "photo_mse_nonfinite_count" not in existing_columns:
            self.connection.execute(
                "ALTER TABLE evaluations ADD COLUMN photo_mse_nonfinite_count "
                "INTEGER NOT NULL DEFAULT 0"
            )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def has(self, stage: str, candidate_key: str, replicate: int) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM evaluations WHERE stage=? AND candidate_key=? AND replicate=?",
            (stage, candidate_key, replicate),
        ).fetchone()
        return row is not None

    def get(self, stage: str, candidate_key: str, replicate: int) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM evaluations WHERE stage=? AND candidate_key=? AND replicate=?",
            (stage, candidate_key, replicate),
        ).fetchone()

    def cached_search_result(self, candidate_key: str) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT * FROM evaluations
            WHERE candidate_key=? AND replicate=0
              AND stage IN ('bruteforce','swapback_single','swapback_factorial','rescue')
            ORDER BY id LIMIT 1
            """,
            (candidate_key,),
        ).fetchone()

    def add(self, evaluation: Evaluation) -> None:
        payload = asdict(evaluation)
        payload["channels_json"] = (
            json.dumps(list(evaluation.channels))
            if evaluation.channels is not None
            else None
        )
        payload["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        columns = [
            "stage",
            "candidate_key",
            "channels_json",
            "replicate",
            "status",
            "reason",
            "elapsed_seconds",
            "exit_code",
            "failure_timestamp",
            "failure_mvs_index",
            "last_timestamp",
            "associated_poses",
            "full_associated_poses",
            "ate_rmse_m",
            "ate_mean_m",
            "ate_median_m",
            "ate_max_m",
            "ate_std_m",
            "rotation_ape_rmse_deg",
            "rotation_ape_mean_deg",
            "rotation_ape_max_deg",
            "translation_rpe_rmse_m",
            "translation_rpe_max_m",
            "rotation_rpe_rmse_deg",
            "rotation_rpe_max_deg",
            "legacy_sim3_rmse_m",
            "legacy_sim3_mean_m",
            "legacy_sim3_scale",
            "legacy_keyframe_poses",
            "full_ate_rmse_m",
            "full_rotation_ape_rmse_deg",
            "diagnostic_frames",
            "photo_mse_median",
            "photo_mse_p95",
            "photo_mse_nonfinite_count",
            "valid_ratio_min",
            "valid_ratio_median",
            "h_cond_max",
            "delta_norm_max",
            "crazy_affine_count",
            "trajectory_path",
            "keyframe_trajectory_path",
            "log_path",
            "log_tail",
            "created_at",
        ]
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

    def search_ranking(self) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT * FROM evaluations
            WHERE replicate=0 AND status='PASS'
              AND stage IN ('bruteforce','swapback_single','swapback_factorial','rescue')
            ORDER BY ate_rmse_m ASC, ate_mean_m ASC
            """
        ).fetchall()
        best: dict[str, sqlite3.Row] = {}
        for row in rows:
            if row["candidate_key"] not in best:
                best[row["candidate_key"]] = row
        return [row_to_candidate_record(row) for row in best.values()]

    def brute_ranking(self) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT * FROM evaluations
            WHERE stage='bruteforce' AND replicate=0 AND status='PASS'
            ORDER BY ate_rmse_m ASC, ate_mean_m ASC
            """
        ).fetchall()
        return [row_to_candidate_record(row) for row in rows]

    def retryable_bruteforce_candidates(self) -> list[Candidate]:
        placeholders = ",".join("?" for _ in RETRYABLE_BRUTEFORCE_STATUSES)
        rows = self.connection.execute(
            f"""
            SELECT channels_json FROM evaluations
            WHERE stage='bruteforce' AND replicate=0
              AND status IN ({placeholders})
            ORDER BY id
            """,
            tuple(sorted(RETRYABLE_BRUTEFORCE_STATUSES)),
        ).fetchall()
        return [
            Candidate(tuple(int(channel) for channel in json.loads(row[0])))
            for row in rows
        ]


def row_to_candidate_record(row: sqlite3.Row) -> dict:
    return {
        "candidate_key": row["candidate_key"],
        "channels": tuple(json.loads(row["channels_json"])),
        "ate_rmse_m": row["ate_rmse_m"],
        "ate_mean_m": row["ate_mean_m"],
        "rotation_ape_rmse_deg": row["rotation_ape_rmse_deg"],
        "translation_rpe_rmse_m": row["translation_rpe_rmse_m"],
        "translation_rpe_max_m": row["translation_rpe_max_m"],
        "rotation_rpe_rmse_deg": row["rotation_rpe_rmse_deg"],
        "rotation_rpe_max_deg": row["rotation_rpe_max_deg"],
        "legacy_sim3_mean_m": row["legacy_sim3_mean_m"],
        "legacy_sim3_scale": row["legacy_sim3_scale"],
        "legacy_keyframe_poses": row["legacy_keyframe_poses"],
        "photo_mse_median": row["photo_mse_median"],
        "photo_mse_nonfinite_count": row["photo_mse_nonfinite_count"],
        "valid_ratio_min": row["valid_ratio_min"],
        "stage": row["stage"],
    }


class ConfigGuard:
    """Exclusive lock plus per-run atomic config replacement/restoration."""

    def __init__(self, config_path: Path, lock_path: Path):
        self.config_path = config_path
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_handle = lock_path.open("a+")
        fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX)
        self.original = config_path.read_bytes()

    def close(self) -> None:
        self.restore()
        fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
        self.lock_handle.close()

    def restore(self) -> None:
        if self.config_path.read_bytes() != self.original:
            atomic_write_bytes(self.config_path, self.original)

    def apply(self, candidate: Candidate) -> dict:
        config = yaml.safe_load(self.original)
        config["mapping"]["color"] = "gray"
        tracking = config["tracking"]
        tracking["debug_tracking_diagnostics"] = True
        tracking["debug_tracking_print_every_frame"] = True
        tracking["debug_tracking_save_suspicious"] = False
        if candidate.channels is None:
            tracking["color"] = "gray"
            tracking["cnn_mode"] = "cnn_only"
        else:
            tracking.update(
                {
                    "color": "cnn",
                    "cnn_layer_name": "conv1",
                    "cnn_channels": len(candidate.channels),
                    "cnn_channel_select": ",".join(
                        f"d{channel}" for channel in candidate.channels
                    ),
                    "cnn_layer_full_channels": 64,
                    "cnn_mode": "cnn_only",
                }
            )
        encoded = yaml.safe_dump(
            config, default_flow_style=False, allow_unicode=True, sort_keys=False
        ).encode("utf-8")
        atomic_write_bytes(self.config_path, encoded)
        return config


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fail-fast exhaustive r=0.70 Conv1 channel evaluation.",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "regression",
            "bruteforce",
            "retry-errors",
            "swapback",
            "rescue",
            "repeat",
            "summary",
            "all",
        ),
        default="summary",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch COMO. Without this flag the script is a dry run.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Re-evaluate and replace exact stage/candidate/replicate cache entries.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--r070-clusters", type=Path, default=DEFAULT_R070)
    parser.add_argument("--r080-clusters", type=Path, default=DEFAULT_R080)
    parser.add_argument("--como-dir", type=Path, default=DEFAULT_COMO_DIR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--terminate-grace-seconds", type=float, default=2.0)
    parser.add_argument(
        "--max-stage-hours",
        type=float,
        default=0.0,
        help=(
            "Stop the bruteforce stage cleanly between candidates after this many "
            "wall-clock hours. Zero disables batching."
        ),
    )
    parser.add_argument(
        "--minimum-associated-poses",
        type=int,
        default=40,
        help="Minimum all-frame poses associated inside MVS indices 10--49.",
    )
    parser.add_argument(
        "--diagnostic-failure-streak",
        type=int,
        default=0,
        help=(
            "Optional consecutive-frame fail gate for chol_ok=False or "
            "valid_ratio<0.05. Zero records these diagnostics without failing."
        ),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--factorial-top-k", type=int, default=5)
    parser.add_argument(
        "--total-top-observations",
        type=int,
        default=3,
        help="Total observations per final candidate, including its search run.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit candidates in a stage; useful for a runtime pilot. Zero means all.",
    )
    parser.add_argument(
        "--keep-all-logs",
        action="store_true",
        help="Keep full stdout for every run; otherwise DB keeps a tail and full logs are retained for regression/repeats/errors.",
    )
    parser.add_argument(
        "--keep-all-trajectories",
        action="store_true",
        help="Keep every successful trajectory; otherwise only regression and repeat trajectories are retained.",
    )
    parser.add_argument(
        "--allow-regression-mismatch",
        action="store_true",
        help="Allow --stage all to continue if gray/known-good expectations fail.",
    )
    return parser.parse_args()


def load_cluster_document(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("layer") != "conv1":
        raise ValueError(f"Expected Conv1 cluster JSON: {path}")
    if not document.get("clusters") or not document.get("representative_channels"):
        raise ValueError(f"Incomplete cluster JSON: {path}")
    return document


def cluster_maps(document: dict) -> tuple[dict[int, int], dict[int, list[int]]]:
    member_to_cluster: dict[int, int] = {}
    cluster_to_members: dict[int, list[int]] = {}
    for cluster in document["clusters"]:
        cluster_id = int(cluster["cluster_id"])
        members = [int(channel) for channel in cluster["members"]]
        cluster_to_members[cluster_id] = members
        for channel in members:
            if channel in member_to_cluster:
                raise ValueError(f"Channel {channel} occurs in multiple final clusters")
            member_to_cluster[channel] = cluster_id
    return member_to_cluster, cluster_to_members


def enumerate_legal_combinations(
    representatives: Sequence[int], member_to_cluster: dict[int, int]
) -> list[Candidate]:
    candidates = []
    for channels in itertools.combinations(sorted(representatives), 4):
        cluster_ids = {member_to_cluster[channel] for channel in channels}
        if len(cluster_ids) == 4:
            candidates.append(Candidate(channels))
    return candidates


def read_manifest(dataset_dir: Path) -> list[dict[str, str]]:
    with (dataset_dir / "frame_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["mvs_index"]))
    return rows


def nearest_mvs_index(
    timestamp: float | None, manifest: list[dict[str, str]]
) -> int | None:
    if timestamp is None:
        return None
    return min(
        range(len(manifest)),
        key=lambda index: abs(float(manifest[index]["rgb_timestamp"]) - timestamp),
    )


def timestamp_after(
    timestamp: float | None, manifest: list[dict[str, str]]
) -> float | None:
    """Infer the next input timestamp after the last finite tracking message."""
    index = nearest_mvs_index(timestamp, manifest)
    if index is None:
        return None
    next_index = min(index + 1, len(manifest) - 1)
    return float(manifest[next_index]["rgb_timestamp"])


def read_tum_poses(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = []
    positions = []
    quaternions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 8:
            raise ValueError(f"Invalid TUM trajectory row in {path}: {line}")
        values = [float(value) for value in fields[:8]]
        timestamps.append(values[0])
        positions.append(values[1:4])
        quaternions.append(values[4:8])
    times = np.asarray(timestamps, dtype=float)
    xyz = np.asarray(positions, dtype=float).reshape(-1, 3)
    quat = np.asarray(quaternions, dtype=float).reshape(-1, 4)
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(xyz)) or not np.all(
        np.isfinite(quat)
    ):
        raise FloatingPointError(f"Trajectory contains NaN or Inf: {path}")
    if len(times) > 1:
        differences = np.diff(times)
        if np.any(differences < 0):
            raise ValueError(f"Trajectory timestamps are decreasing: {path}")
        if np.any(differences == 0):
            # Official/derived TUM files can contain two poses at exactly the same
            # timestamp (fr2/desk has one such row).  Association requires unique
            # timestamps, so retain the later row, matching the usual dict-based
            # TUM readers, while continuing to reject genuinely decreasing input.
            keep = np.concatenate((differences != 0, np.asarray([True])))
            times = times[keep]
            xyz = xyz[keep]
            quat = quat[keep]
    return times, xyz, quat


def associate_poses(
    reference_timestamps: np.ndarray,
    reference_positions: np.ndarray,
    reference_quaternions: np.ndarray,
    estimate_timestamps: np.ndarray,
    estimate_positions: np.ndarray,
    estimate_quaternions: np.ndarray,
    maximum_difference: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reference_list = reference_timestamps.tolist()
    matched_reference = []
    matched_estimate = []
    matched_reference_quaternions = []
    matched_estimate_quaternions = []
    used: set[int] = set()
    for estimate_index, timestamp in enumerate(estimate_timestamps):
        insertion = bisect.bisect_left(reference_list, float(timestamp))
        choices = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < len(reference_list)
        ]
        choices.sort(key=lambda index: abs(reference_list[index] - timestamp))
        selected = next(
            (
                index
                for index in choices
                if index not in used
                and abs(reference_list[index] - timestamp) <= maximum_difference
            ),
            None,
        )
        if selected is not None:
            used.add(selected)
            matched_reference.append(reference_positions[selected])
            matched_estimate.append(estimate_positions[estimate_index])
            matched_reference_quaternions.append(reference_quaternions[selected])
            matched_estimate_quaternions.append(estimate_quaternions[estimate_index])
    return (
        np.asarray(matched_reference, dtype=float).reshape(-1, 3),
        np.asarray(matched_estimate, dtype=float).reshape(-1, 3),
        np.asarray(matched_reference_quaternions, dtype=float).reshape(-1, 4),
        np.asarray(matched_estimate_quaternions, dtype=float).reshape(-1, 4),
    )


def align_positions(
    reference: np.ndarray, estimate: np.ndarray, correct_scale: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    if (
        reference.shape != estimate.shape
        or reference.ndim != 2
        or reference.shape[1] != 3
    ):
        raise ValueError("ATE inputs must have matching [N,3] shapes")
    if reference.shape[0] < 3:
        raise ValueError("At least three associated poses are required for alignment")
    reference_mean = reference.mean(axis=0)
    estimate_mean = estimate.mean(axis=0)
    reference_centered = reference - reference_mean
    estimate_centered = estimate - estimate_mean
    covariance = (reference_centered.T @ estimate_centered) / reference.shape[0]
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_transpose) < 0:
        correction[-1, -1] = -1
    rotation = left @ correction @ right_transpose
    variance = float(np.mean(np.sum(estimate_centered**2, axis=1)))
    if variance <= 1e-15:
        raise ValueError("Estimated trajectory has zero spatial variance")
    scale = (
        float(np.sum(singular_values * np.diag(correction)) / variance)
        if correct_scale
        else 1.0
    )
    translation = reference_mean - scale * (rotation @ estimate_mean)
    aligned = (scale * (rotation @ estimate.T)).T + translation
    return aligned, rotation, scale


def quaternion_rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("Trajectory contains a zero-norm quaternion")
    q = quaternions / norms[:, None]
    x, y, z, w = q.T
    matrices = np.empty((len(q), 3, 3), dtype=float)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - z * w)
    matrices[:, 0, 2] = 2 * (x * z + y * w)
    matrices[:, 1, 0] = 2 * (x * y + z * w)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - x * w)
    matrices[:, 2, 0] = 2 * (x * z - y * w)
    matrices[:, 2, 1] = 2 * (y * z + x * w)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def rotation_angles_deg(matrices: np.ndarray) -> np.ndarray:
    cosines = np.clip((np.trace(matrices, axis1=1, axis2=2) - 1.0) / 2.0, -1, 1)
    return np.degrees(np.arccos(cosines))


def trajectory_error_metrics(
    reference_positions: np.ndarray,
    estimate_positions: np.ndarray,
    reference_quaternions: np.ndarray,
    estimate_quaternions: np.ndarray,
) -> dict[str, float]:
    aligned, alignment_rotation, _ = align_positions(
        reference_positions, estimate_positions, correct_scale=False
    )
    errors = np.linalg.norm(aligned - reference_positions, axis=1)
    legacy_aligned, _, legacy_scale = align_positions(
        reference_positions, estimate_positions, correct_scale=True
    )
    legacy_errors = np.linalg.norm(legacy_aligned - reference_positions, axis=1)

    reference_rotations = quaternion_rotation_matrices(reference_quaternions)
    estimate_rotations = quaternion_rotation_matrices(estimate_quaternions)
    aligned_estimate_rotations = alignment_rotation[None, :, :] @ estimate_rotations
    absolute_rotation_errors = rotation_angles_deg(
        np.transpose(reference_rotations, (0, 2, 1)) @ aligned_estimate_rotations
    )

    reference_steps = np.diff(reference_positions, axis=0)
    estimate_steps = np.diff(aligned, axis=0)
    translation_rpe = np.linalg.norm(estimate_steps - reference_steps, axis=1)
    reference_relative_rotations = (
        np.transpose(reference_rotations[:-1], (0, 2, 1))
        @ reference_rotations[1:]
    )
    estimate_relative_rotations = (
        np.transpose(estimate_rotations[:-1], (0, 2, 1))
        @ estimate_rotations[1:]
    )
    rotation_rpe = rotation_angles_deg(
        np.transpose(reference_relative_rotations, (0, 2, 1))
        @ estimate_relative_rotations
    )
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)),
        "std": float(np.std(errors)),
        "rotation_ape_rmse_deg": float(
            np.sqrt(np.mean(absolute_rotation_errors**2))
        ),
        "rotation_ape_mean_deg": float(np.mean(absolute_rotation_errors)),
        "rotation_ape_max_deg": float(np.max(absolute_rotation_errors)),
        "translation_rpe_rmse": float(np.sqrt(np.mean(translation_rpe**2))),
        "translation_rpe_max": float(np.max(translation_rpe)),
        "rotation_rpe_rmse_deg": float(np.sqrt(np.mean(rotation_rpe**2))),
        "rotation_rpe_max_deg": float(np.max(rotation_rpe)),
        "legacy_sim3_rmse": float(np.sqrt(np.mean(legacy_errors**2))),
        "legacy_sim3_mean": float(np.mean(legacy_errors)),
        "legacy_sim3_scale": legacy_scale,
    }


def evaluate_trajectory(
    trajectory_path: Path,
    groundtruth_path: Path,
    scored_start: float,
    scored_end: float,
) -> tuple[dict[str, float], int, int, float, bool]:
    estimate_times, estimate_positions, estimate_quaternions = read_tum_poses(
        trajectory_path
    )
    if len(estimate_times) == 0:
        raise ValueError("Trajectory is empty")
    last_timestamp = float(estimate_times[-1])
    reference_times, reference_positions, reference_quaternions = read_tum_poses(
        groundtruth_path
    )
    full_reference, full_estimate, full_reference_q, full_estimate_q = associate_poses(
        reference_times,
        reference_positions,
        reference_quaternions,
        estimate_times,
        estimate_positions,
        estimate_quaternions,
    )
    if len(full_estimate) < 3:
        raise ValueError(
            f"Only {len(full_estimate)} all-frame poses could be associated to ground truth"
        )
    full_metrics = trajectory_error_metrics(
        full_reference, full_estimate, full_reference_q, full_estimate_q
    )
    window = (estimate_times >= scored_start - 0.02) & (
        estimate_times <= scored_end + 0.02
    )
    scored_times = estimate_times[window]
    scored_positions = estimate_positions[window]
    scored_quaternions = estimate_quaternions[window]
    reference, estimate, reference_q, estimate_q = associate_poses(
        reference_times,
        reference_positions,
        reference_quaternions,
        scored_times,
        scored_positions,
        scored_quaternions,
    )
    metrics = trajectory_error_metrics(reference, estimate, reference_q, estimate_q)
    metrics["full_rmse"] = full_metrics["rmse"]
    metrics["full_rotation_ape_rmse_deg"] = full_metrics[
        "rotation_ape_rmse_deg"
    ]
    frozen = False
    if len(estimate) >= 10:
        estimate_motion = float(
            np.max(np.linalg.norm(estimate[-10:] - estimate[-1], axis=1))
        )
        reference_motion = float(
            np.max(np.linalg.norm(reference[-10:] - reference[-1], axis=1))
        )
        frozen = estimate_motion < 1e-4 and reference_motion > 1e-3
    return metrics, len(estimate), len(full_estimate), last_timestamp, frozen


def evaluate_legacy_keyframe_sim3(
    trajectory_path: Path,
    groundtruth_path: Path,
    scored_start: float,
    scored_end: float,
) -> tuple[dict[str, float] | None, int]:
    estimate_times, estimate_positions, estimate_quaternions = read_tum_poses(
        trajectory_path
    )
    window = (estimate_times >= scored_start - 0.02) & (
        estimate_times <= scored_end + 0.02
    )
    reference_times, reference_positions, reference_quaternions = read_tum_poses(
        groundtruth_path
    )
    reference, estimate, reference_q, estimate_q = associate_poses(
        reference_times,
        reference_positions,
        reference_quaternions,
        estimate_times[window],
        estimate_positions[window],
        estimate_quaternions[window],
    )
    if len(estimate) < 3:
        return None, len(estimate)
    metrics = trajectory_error_metrics(reference, estimate, reference_q, estimate_q)
    return metrics, len(estimate)


def terminate_process_group(
    process: subprocess.Popen[str], grace_seconds: float
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=max(grace_seconds, 1.0))


def parse_tracking_diag(line: str) -> dict[str, float | int | bool | str] | None:
    if "[TRACK_DIAG]" not in line or "frame=" not in line:
        return None
    parsed: dict[str, float | int | bool | str] = {}
    for key, raw_value in re.findall(r"([A-Za-z_]+)=([^\s]+)", line):
        if raw_value in {"True", "False"}:
            parsed[key] = raw_value == "True"
            continue
        try:
            numeric = float(raw_value)
            parsed[key] = int(numeric) if key in {"frame", "level", "iters"} else numeric
        except ValueError:
            parsed[key] = raw_value
    return parsed


def finite_diag_values(
    diagnostics: Sequence[dict[str, float | int | bool | str]], key: str
) -> np.ndarray:
    values = [
        float(item[key])
        for item in diagnostics
        if key in item
        and isinstance(item[key], (int, float))
        and not isinstance(item[key], bool)
        and math.isfinite(float(item[key]))
    ]
    return np.asarray(values, dtype=float)


class SearchRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        console: Console,
        store: ResultStore,
        config_guard: ConfigGuard | None,
        manifest: list[dict[str, str]],
        member_to_cluster: dict[int, int],
    ):
        self.args = args
        self.console = console
        self.store = store
        self.config_guard = config_guard
        self.manifest = manifest
        self.member_to_cluster = member_to_cluster
        metadata = json.loads(
            (args.dataset_dir / "mvs_metadata.json").read_text(encoding="utf-8")
        )
        scored_start_index, scored_end_index = metadata["scored_mvs_indices_inclusive"]
        self.scored_start = float(manifest[scored_start_index]["rgb_timestamp"])
        self.scored_end = float(manifest[scored_end_index]["rgb_timestamp"])
        dimming_markers = [
            marker
            for marker in metadata.get("event_markers", [])
            if marker.get("role") == "clear_dimming_onset"
        ]
        self.minimum_completion_timestamp = float(
            dimming_markers[0]["timestamp"] if dimming_markers else self.scored_end
        )
        self.keyframe_trajectory_source = args.como_dir / "results/data_tum.txt"
        self.trajectory_source = args.como_dir / "results/data_tum_all_frames.txt"
        self.groundtruth = args.dataset_dir / "groundtruth.txt"
        self.command = [
            str(args.python),
            "-u",
            "como/como_dataset.py",
            "--dataset_type=tum",
            f"--dataset_dir={args.dataset_dir}",
        ]
        self.recent_durations: deque[float] = deque(maxlen=200)
        self.consecutive_infrastructure_errors = 0

    def evaluate(
        self,
        candidate: Candidate,
        stage: str,
        replicate: int,
        run_index: int,
        total_runs: int,
        force_keep: bool = False,
    ) -> Evaluation | None:
        if self.store.has(stage, candidate.key, replicate) and not self.args.rerun_existing:
            self.console.say(
                f"[SKIP] stage={stage} channels={candidate.display} replicate={replicate} "
                "already has a recorded result"
            )
            return None
        if not self.args.rerun_existing and replicate == 0 and stage in {
            "bruteforce",
            "swapback_single",
            "swapback_factorial",
            "rescue",
        }:
            cached = self.store.cached_search_result(candidate.key)
            if cached is not None:
                self.console.say(
                    f"[CACHE] stage={stage} channels={candidate.display} reuses "
                    f"{cached['stage']} status={cached['status']}"
                )
                return None
        cluster_text = "n/a"
        if candidate.channels is not None:
            cluster_text = (
                "["
                + ",".join(
                    str(self.member_to_cluster.get(channel, "R"))
                    for channel in candidate.channels
                )
                + "]"
            )
        self.console.say("")
        self.console.say(
            f"[RUN {run_index:,}/{total_runs:,}] stage={stage} replicate={replicate} "
            f"channels={candidate.display} r070_clusters={cluster_text}"
        )
        if not self.args.execute:
            return None
        assert self.config_guard is not None
        config = self.config_guard.apply(candidate)
        config_digest = hashlib.sha256(
            yaml.safe_dump(config, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        self.console.say(
            f"[CONFIG] tracking={config['tracking']['color']} mapping={config['mapping']['color']} "
            f"config_sha256={config_digest} timeout={self.args.timeout_seconds:.1f}s"
        )
        self.trajectory_source.unlink(missing_ok=True)
        self.keyframe_trajectory_source.unlink(missing_ok=True)
        started = time.monotonic()
        lines: list[str] = []
        failure_status: str | None = None
        failure_reason = ""
        failure_timestamp: float | None = None
        last_tracking_timestamp: float | None = None
        tracking_diagnostics: list[dict[str, float | int | bool | str]] = []
        diagnostic_failure_streak = 0
        crazy_affine_count = 0
        process: subprocess.Popen[str] | None = None
        try:
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            environment["COMO_SAVE_ALL_FRAME_TRAJECTORY"] = "1"
            process = subprocess.Popen(
                self.command,
                cwd=self.args.como_dir,
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

            reader_thread = threading.Thread(
                target=read_output,
                name=f"como-output-{candidate.key}",
                daemon=True,
            )
            reader_thread.start()
            eof = False
            while not eof:
                elapsed = time.monotonic() - started
                if elapsed > self.args.timeout_seconds:
                    failure_status = "TIMEOUT"
                    failure_reason = (
                        f"Exceeded {self.args.timeout_seconds:.1f}s wall-clock timeout"
                    )
                    self.console.say(
                        f"[FAIL-FAST] {failure_reason}; terminating process group"
                    )
                    terminate_process_group(process, self.args.terminate_grace_seconds)
                    break
                try:
                    line = output_queue.get(timeout=0.2)
                except queue.Empty:
                    if process.poll() is not None:
                        reader_thread.join(timeout=1.0)
                    continue
                if line is None:
                    eof = True
                    continue
                stripped = line.rstrip("\n")
                lines.append(stripped)
                timestamp_match = TIMESTAMP_RE.search(stripped)
                current_timestamp = (
                    float(timestamp_match.group(1)) if timestamp_match else None
                )
                if "[KF aff received]" in stripped and current_timestamp is not None:
                    last_tracking_timestamp = current_timestamp
                if "Crazy affine detected" in stripped:
                    crazy_affine_count += 1
                diagnostic = parse_tracking_diag(stripped)
                if diagnostic is not None:
                    tracking_diagnostics.append(diagnostic)
                    if current_timestamp is not None:
                        last_tracking_timestamp = current_timestamp
                if TRACKING_NAN_RE.search(
                    stripped
                ) or TRACKING_DIAG_NONFINITE_RE.search(stripped):
                    failure_status = "FAIL_TRACKING_NAN"
                    failure_reason = (
                        "Tracking emitted non-finite affine/pose diagnostics"
                    )
                    failure_timestamp = current_timestamp or timestamp_after(
                        last_tracking_timestamp, self.manifest
                    )
                elif diagnostic is not None:
                    chol_ok = diagnostic.get("chol_ok", True)
                    valid_ratio = float(diagnostic.get("valid_ratio", 1.0))
                    diagnostically_bad = chol_ok is False or valid_ratio < 0.05
                    diagnostic_failure_streak = (
                        diagnostic_failure_streak + 1 if diagnostically_bad else 0
                    )
                    if (
                        self.args.diagnostic_failure_streak > 0
                        and diagnostic_failure_streak
                        >= self.args.diagnostic_failure_streak
                    ):
                        failure_status = "FAIL_TRACKING_DIAGNOSTIC"
                        failure_reason = (
                            f"{diagnostic_failure_streak} consecutive numerically "
                            "degenerate tracking frames "
                            f"(valid_ratio={valid_ratio:.3f}, chol_ok={chol_ok})"
                        )
                        failure_timestamp = current_timestamp or timestamp_after(
                            last_tracking_timestamp, self.manifest
                        )
                elif any(marker in stripped for marker in KNOWN_RUNTIME_MARKERS):
                    failure_status = "FAIL_TRACKING_RUNTIME"
                    failure_reason = (
                        "Open3D/Filament received empty geometry (AABB failure)"
                        if "AABB can't be empty" in stripped
                        else "COMO raised its unknown tracking/runtime exception"
                    )
                    failure_timestamp = current_timestamp or timestamp_after(
                        last_tracking_timestamp, self.manifest
                    )
                if failure_status is not None:
                    failure_index = nearest_mvs_index(failure_timestamp, self.manifest)
                    where = (
                        f"MVS={failure_index} ts={failure_timestamp:.6f}"
                        if failure_timestamp is not None
                        else "timestamp unavailable"
                    )
                    self.console.say(
                        f"[FAIL-FAST] status={failure_status} {where}; "
                        "stopping this run without waiting for later frames"
                    )
                    terminate_process_group(process, self.args.terminate_grace_seconds)
                    break
            if process.poll() is None:
                terminate_process_group(process, self.args.terminate_grace_seconds)
            exit_code = process.returncode
        except BaseException:
            if process is not None:
                terminate_process_group(process, self.args.terminate_grace_seconds)
            raise
        finally:
            self.config_guard.restore()
        elapsed = time.monotonic() - started
        self.recent_durations.append(elapsed)
        log_tail = "\n".join(lines[-80:])
        photo_mse_nonfinite_count = sum(
            1
            for item in tracking_diagnostics
            if "photo_mse" in item
            and isinstance(item["photo_mse"], (int, float))
            and not math.isfinite(float(item["photo_mse"]))
        )
        evaluation = Evaluation(
            stage=stage,
            candidate_key=candidate.key,
            channels=candidate.channels,
            replicate=replicate,
            status=failure_status or "PENDING",
            reason=failure_reason,
            elapsed_seconds=elapsed,
            exit_code=exit_code,
            failure_timestamp=failure_timestamp,
            failure_mvs_index=nearest_mvs_index(failure_timestamp, self.manifest),
            diagnostic_frames=len(tracking_diagnostics),
            photo_mse_nonfinite_count=photo_mse_nonfinite_count,
            crazy_affine_count=crazy_affine_count,
            log_tail=log_tail,
        )
        photo_mse = finite_diag_values(tracking_diagnostics, "photo_mse")
        valid_ratios = finite_diag_values(tracking_diagnostics, "valid_ratio")
        h_conditions = finite_diag_values(tracking_diagnostics, "h_cond")
        delta_norms = finite_diag_values(tracking_diagnostics, "delta_norm")
        if len(photo_mse):
            evaluation.photo_mse_median = float(np.median(photo_mse))
            evaluation.photo_mse_p95 = float(np.percentile(photo_mse, 95))
        if len(valid_ratios):
            evaluation.valid_ratio_min = float(np.min(valid_ratios))
            evaluation.valid_ratio_median = float(np.median(valid_ratios))
        if len(h_conditions):
            evaluation.h_cond_max = float(np.max(h_conditions))
        if len(delta_norms):
            evaluation.delta_norm_max = float(np.max(delta_norms))
        if failure_status is None:
            if exit_code != 0:
                evaluation.status = "ERROR_RUNTIME"
                evaluation.reason = f"COMO exited with code {exit_code} without a recognised tracking-failure signature"
            elif not self.trajectory_source.is_file():
                evaluation.status = "FAIL_INVALID_TRAJECTORY"
                evaluation.reason = (
                    "COMO exited but did not produce results/data_tum_all_frames.txt"
                )
            else:
                try:
                    metrics, associated, full_associated, last_timestamp, frozen = evaluate_trajectory(
                        self.trajectory_source,
                        self.groundtruth,
                        self.scored_start,
                        self.scored_end,
                    )
                    evaluation.associated_poses = associated
                    evaluation.full_associated_poses = full_associated
                    evaluation.last_timestamp = last_timestamp
                    legacy_metrics = None
                    legacy_poses = 0
                    if self.keyframe_trajectory_source.is_file():
                        try:
                            legacy_metrics, legacy_poses = evaluate_legacy_keyframe_sim3(
                                self.keyframe_trajectory_source,
                                self.groundtruth,
                                self.scored_start,
                                self.scored_end,
                            )
                        except (FloatingPointError, ValueError, OSError) as error:
                            self.console.say(
                                "[LEGACY DIAGNOSTIC UNAVAILABLE] "
                                f"channels={candidate.display} reason={error}"
                            )
                    else:
                        self.console.say(
                            "[LEGACY DIAGNOSTIC UNAVAILABLE] "
                            f"channels={candidate.display} reason=keyframe trajectory missing"
                        )
                    evaluation.legacy_keyframe_poses = legacy_poses
                    if frozen:
                        evaluation.status = "FAIL_POSE_FROZEN"
                        evaluation.reason = (
                            "Final estimated poses froze while ground truth kept moving"
                        )
                    elif associated < self.args.minimum_associated_poses:
                        evaluation.status = "FAIL_INVALID_TRAJECTORY"
                        evaluation.reason = (
                            f"Only {associated} all-frame poses associated in the scored "
                            f"window; need at least {self.args.minimum_associated_poses}"
                        )
                    elif last_timestamp < self.minimum_completion_timestamp - 0.02:
                        evaluation.status = "FAIL_INCOMPLETE"
                        evaluation.reason = (
                            "Trajectory ended before the clear-dimming recovery phase: "
                            f"last={last_timestamp:.6f}, required≈{self.minimum_completion_timestamp:.6f}"
                        )
                    else:
                        evaluation.status = "PASS"
                        evaluation.reason = (
                            "Completed scored window with a finite trajectory"
                        )
                        evaluation.ate_rmse_m = metrics["rmse"]
                        evaluation.ate_mean_m = metrics["mean"]
                        evaluation.ate_median_m = metrics["median"]
                        evaluation.ate_max_m = metrics["max"]
                        evaluation.ate_std_m = metrics["std"]
                        evaluation.rotation_ape_rmse_deg = metrics[
                            "rotation_ape_rmse_deg"
                        ]
                        evaluation.rotation_ape_mean_deg = metrics[
                            "rotation_ape_mean_deg"
                        ]
                        evaluation.rotation_ape_max_deg = metrics[
                            "rotation_ape_max_deg"
                        ]
                        evaluation.translation_rpe_rmse_m = metrics[
                            "translation_rpe_rmse"
                        ]
                        evaluation.translation_rpe_max_m = metrics[
                            "translation_rpe_max"
                        ]
                        evaluation.rotation_rpe_rmse_deg = metrics[
                            "rotation_rpe_rmse_deg"
                        ]
                        evaluation.rotation_rpe_max_deg = metrics[
                            "rotation_rpe_max_deg"
                        ]
                        if legacy_metrics is not None:
                            evaluation.legacy_sim3_rmse_m = legacy_metrics[
                                "legacy_sim3_rmse"
                            ]
                            evaluation.legacy_sim3_mean_m = legacy_metrics[
                                "legacy_sim3_mean"
                            ]
                            evaluation.legacy_sim3_scale = legacy_metrics[
                                "legacy_sim3_scale"
                            ]
                        evaluation.full_ate_rmse_m = metrics["full_rmse"]
                        evaluation.full_rotation_ape_rmse_deg = metrics[
                            "full_rotation_ape_rmse_deg"
                        ]
                except FloatingPointError as error:
                    evaluation.status = "FAIL_INVALID_TRAJECTORY"
                    evaluation.reason = str(error)
                except (ValueError, OSError) as error:
                    evaluation.status = "ERROR_TRAJECTORY_EVALUATION"
                    evaluation.reason = str(error)
        keep_full_log = (
            self.args.keep_all_logs
            or force_keep
            or evaluation.status.startswith("ERROR")
        )
        if keep_full_log:
            log_path = self.result_artifact_path(stage, candidate, replicate, ".log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            evaluation.log_path = str(log_path)
        keep_trajectory = (
            self.args.keep_all_trajectories
            or force_keep
            or stage in {"regression", "repeat"}
        )
        if keep_trajectory and self.trajectory_source.is_file():
            trajectory_path = self.result_artifact_path(
                stage, candidate, replicate, ".tum.txt"
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.trajectory_source, trajectory_path)
            evaluation.trajectory_path = str(trajectory_path)
        if keep_trajectory and self.keyframe_trajectory_source.is_file():
            keyframe_path = self.result_artifact_path(
                stage, candidate, replicate, ".keyframes.tum.txt"
            )
            keyframe_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.keyframe_trajectory_source, keyframe_path)
            evaluation.keyframe_trajectory_path = str(keyframe_path)
        self.store.add(evaluation)
        self.print_result(candidate, evaluation)
        if evaluation.status.startswith("ERROR"):
            self.consecutive_infrastructure_errors += 1
        else:
            self.consecutive_infrastructure_errors = 0
        if self.consecutive_infrastructure_errors >= 3:
            raise RuntimeError(
                "Three consecutive infrastructure errors occurred. Search stopped to "
                "avoid misclassifying a broken environment as bad channel combinations."
            )
        return evaluation

    def result_artifact_path(
        self, stage: str, candidate: Candidate, replicate: int, suffix: str
    ) -> Path:
        safe_key = candidate.key.replace(",", "-")
        return (
            self.args.output_dir
            / "artifacts"
            / stage
            / f"channels_{safe_key}__rep{replicate}{suffix}"
        )

    def print_result(self, candidate: Candidate, evaluation: Evaluation) -> None:
        if evaluation.status == "PASS":
            translation_jump = evaluation.translation_rpe_max_m > TRANSLATION_RPE_MAX_M
            rotation_jump = evaluation.rotation_rpe_max_deg > ROTATION_RPE_MAX_DEG
            self.console.say(
                f"[PASS] channels={candidate.display} "
                f"SE3_ATE_RMSE={evaluation.ate_rmse_m * 100:.4f}cm "
                f"SE3_ATE_mean={evaluation.ate_mean_m * 100:.4f}cm "
                f"SE3_ATE_max={evaluation.ate_max_m * 100:.4f}cm "
                f"Rot_APE_RMSE={evaluation.rotation_ape_rmse_deg:.3f}deg "
                f"Trans_RPE_RMSE={evaluation.translation_rpe_rmse_m * 100:.4f}cm "
                f"Rot_RPE_RMSE={evaluation.rotation_rpe_rmse_deg:.3f}deg "
                f"Trans_RPE_max={evaluation.translation_rpe_max_m * 100:.3f}cm "
                f"Rot_RPE_max={evaluation.rotation_rpe_max_deg:.3f}deg "
                f"RPE_safety={'FLAG' if translation_jump or rotation_jump else 'PASS'} "
                f"poses={evaluation.associated_poses}/{evaluation.full_associated_poses} "
                f"runtime={evaluation.elapsed_seconds:.2f}s"
            )
            self.console.say(
                f"[DIAGNOSTICS] channels={candidate.display} "
                f"legacy_keyframe_Sim3_mean="
                f"{format_optional_scaled(evaluation.legacy_sim3_mean_m, 100, '.4f')}"
                f"{'cm' if evaluation.legacy_sim3_mean_m is not None else ''} "
                f"legacy_scale={format_optional(evaluation.legacy_sim3_scale, '.4f')} "
                f"legacy_keyframes={evaluation.legacy_keyframe_poses} "
                f"photo_mse_median={format_optional(evaluation.photo_mse_median, '.6g')} "
                f"photo_mse_p95={format_optional(evaluation.photo_mse_p95, '.6g')} "
                f"photo_mse_nonfinite={evaluation.photo_mse_nonfinite_count} "
                f"valid_ratio_min={format_optional(evaluation.valid_ratio_min, '.3f')} "
                f"h_cond_max={format_optional(evaluation.h_cond_max, '.3e')} "
                f"crazy_affine={evaluation.crazy_affine_count}"
            )
        else:
            location = ""
            if evaluation.failure_mvs_index is not None:
                location = (
                    f" failure_mvs={evaluation.failure_mvs_index}"
                    f" failure_ts={evaluation.failure_timestamp:.6f}"
                )
            self.console.say(
                f"[{evaluation.status}] channels={candidate.display}{location} "
                f"runtime={evaluation.elapsed_seconds:.2f}s reason={evaluation.reason}"
            )

    def print_eta(self, completed: int, total: int, counts: Counter[str]) -> None:
        if not self.recent_durations:
            return
        average = sum(self.recent_durations) / len(self.recent_durations)
        remaining_seconds = max(total - completed, 0) * average
        ranking = self.store.search_ranking()
        best = ranking[:3]
        best_text = (
            "; ".join(
                f"{record['channels']}={record['ate_rmse_m'] * 100:.3f}cm"
                f"{'[JUMP]' if not rpe_safety_flags(record)[2] else ''}"
                for record in best
            )
            or "no PASS result yet"
        )
        safe_best = [record for record in ranking if rpe_safety_flags(record)[2]][:3]
        safe_text = (
            "; ".join(
                f"{record['channels']}={record['ate_rmse_m'] * 100:.3f}cm"
                for record in safe_best
            )
            or "no RPE-safe PASS yet"
        )
        self.console.say(
            f"[PROGRESS] completed={completed:,}/{total:,} "
            f"PASS={counts['PASS']:,} FAIL={sum(counts[s] for s in FAIL_STATUSES):,} "
            f"ERROR={sum(v for k, v in counts.items() if k.startswith('ERROR')):,} "
            f"recent_avg={average:.2f}s ETA={format_duration(remaining_seconds)} "
            f"raw_ATE_top3: {best_text} | safe_ATE_top3: {safe_text}"
        )


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    days, remainder = divmod(int(seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    return f"{hours}h {minutes}m"


def format_optional(value: float | None, spec: str) -> str:
    return format(value, spec) if value is not None else "n/a"


def format_optional_scaled(value: float | None, scale: float, spec: str) -> str:
    return format(value * scale, spec) if value is not None else "n/a"


def run_candidate_stage(
    runner: SearchRunner,
    candidates: Sequence[Candidate],
    stage: str,
    replicate: int = 0,
    force_keep: bool = False,
) -> list[Evaluation]:
    total = len(candidates)
    results = []
    counts: Counter[str] = Counter()
    stage_started = time.monotonic()
    for index, candidate in enumerate(candidates, start=1):
        elapsed = time.monotonic() - stage_started
        if (
            results
            and runner.args.max_stage_hours > 0
            and elapsed >= runner.args.max_stage_hours * 3600
        ):
            runner.console.say(
                f"[BATCH COMPLETE] Reached {runner.args.max_stage_hours:.2f}h "
                f"wall-time limit after {len(results):,} new evaluations; stopping "
                f"cleanly before candidate {index:,}/{total:,}"
            )
            runner.print_eta(index - 1, total, counts)
            export_results(runner.store, runner.args.output_dir)
            break
        evaluation = runner.evaluate(
            candidate, stage, replicate, index, total, force_keep=force_keep
        )
        if evaluation is not None:
            results.append(evaluation)
            counts[evaluation.status] += 1
        if index % 100 == 0 or index == total:
            runner.print_eta(index, total, counts)
            export_results(runner.store, runner.args.output_dir)
    return results


def rpe_safety_flags(record: dict) -> tuple[bool, bool, bool]:
    translation_jump = (
        record["translation_rpe_max_m"] is None
        or record["translation_rpe_max_m"] > TRANSLATION_RPE_MAX_M
    )
    rotation_jump = (
        record["rotation_rpe_max_deg"] is None
        or record["rotation_rpe_max_deg"] > ROTATION_RPE_MAX_DEG
    )
    return translation_jump, rotation_jump, not (translation_jump or rotation_jump)


def pareto_records(records: Sequence[dict]) -> list[dict]:
    """Return the non-dominated front for ATE and translation/rotation RPE RMSE."""
    ordered = sorted(
        records,
        key=lambda record: (
            record["ate_rmse_m"],
            record["translation_rpe_rmse_m"],
            record["rotation_rpe_rmse_deg"],
        ),
    )
    front: list[dict] = []
    rpe_skyline: list[tuple[float, float]] = []
    for record in ordered:
        translation = record["translation_rpe_rmse_m"]
        rotation = record["rotation_rpe_rmse_deg"]
        if any(t <= translation and r <= rotation for t, r in rpe_skyline):
            continue
        front.append(record)
        rpe_skyline = [
            (t, r)
            for t, r in rpe_skyline
            if not (translation <= t and rotation <= r)
        ]
        rpe_skyline.append((translation, rotation))
    return front


def select_multimetric_records(records: Sequence[dict], count: int) -> list[dict]:
    """Select an explainable, RPE-safe mix rather than ATE-only finalists.

    Half the slots emphasize ATE, one quarter add Pareto-front candidates, and
    the remaining slots explicitly cover low translation and low rotation RPE.
    Unfilled slots return to the safe ATE ranking.  Unsafe candidates stay in
    the main ranking CSV but do not enter swap-back/rescue/repeat contexts.
    """
    safe = [record for record in records if rpe_safety_flags(record)[2]]
    safe_by_ate = sorted(
        safe, key=lambda record: (record["ate_rmse_m"], record["ate_mean_m"])
    )
    if not safe_by_ate or count <= 0:
        return []

    ate_quota = max(1, count // 2)
    pareto_quota = max(1, count // 4)
    translation_quota = max(1, (count + 7) // 8)
    rotation_quota = max(1, count - ate_quota - pareto_quota - translation_quota)
    selected: dict[str, dict] = {}

    def add_from(pool: Sequence[dict], quota: int, reason: str) -> None:
        if quota <= 0 or len(selected) >= count:
            return
        added = 0
        for source in pool:
            key = source["candidate_key"]
            if key in selected:
                reasons = selected[key]["selection_reasons"]
                if reason not in reasons:
                    reasons.append(reason)
                continue
            record = dict(source)
            record["selection_reasons"] = [reason]
            selected[key] = record
            added += 1
            if added >= quota or len(selected) >= count:
                break

    add_from(safe_by_ate, ate_quota, "top_ate")
    add_from(pareto_records(safe), pareto_quota, "pareto")
    add_from(
        sorted(
            safe,
            key=lambda record: (
                record["translation_rpe_rmse_m"],
                record["ate_rmse_m"],
            ),
        ),
        translation_quota,
        "low_translation_rpe",
    )
    add_from(
        sorted(
            safe,
            key=lambda record: (
                record["rotation_rpe_rmse_deg"],
                record["ate_rmse_m"],
            ),
        ),
        rotation_quota,
        "low_rotation_rpe",
    )
    add_from(safe_by_ate, count - len(selected), "ate_fill")
    return list(selected.values())[:count]


def candidates_from_records(records: Sequence[dict]) -> list[Candidate]:
    return [Candidate(tuple(record["channels"])) for record in records]


def swapback_single_candidates(
    base_records: Sequence[dict],
    member_to_cluster: dict[int, int],
    cluster_to_members: dict[int, list[int]],
) -> list[Candidate]:
    generated: dict[str, Candidate] = {}
    for record in base_records:
        channels = tuple(record["channels"])
        for position, channel in enumerate(channels):
            cluster_id = member_to_cluster[channel]
            for replacement in cluster_to_members[cluster_id]:
                if replacement == channel or replacement in channels:
                    continue
                proposal = list(channels)
                proposal[position] = replacement
                candidate = Candidate(
                    tuple(sorted(proposal)), "cluster_swapback_single"
                )
                generated[candidate.key] = candidate
    return sorted(generated.values(), key=lambda candidate: candidate.channels or ())


def factorial_swapback_candidates(
    base_records: Sequence[dict],
    member_to_cluster: dict[int, int],
    cluster_to_members: dict[int, list[int]],
) -> list[Candidate]:
    generated: dict[str, Candidate] = {}
    for record in base_records:
        channels = tuple(record["channels"])
        alternatives = [
            cluster_to_members[member_to_cluster[channel]] for channel in channels
        ]
        for proposal in itertools.product(*alternatives):
            if tuple(proposal) == channels or len(set(proposal)) != 4:
                continue
            candidate = Candidate(tuple(sorted(proposal)), "cluster_swapback_factorial")
            generated[candidate.key] = candidate
    return sorted(generated.values(), key=lambda candidate: candidate.channels or ())


def add_cluster_coverage_contexts(
    selected: list[dict],
    brute_ranking: Sequence[dict],
    non_singleton_cluster_ids: set[int],
    member_to_cluster: dict[int, int],
) -> list[dict]:
    output = {record["candidate_key"]: record for record in selected}
    covered = {
        member_to_cluster[channel]
        for record in selected
        for channel in record["channels"]
    }
    for cluster_id in sorted(non_singleton_cluster_ids - covered):
        record = next(
            (
                item
                for item in brute_ranking
                if cluster_id in {member_to_cluster[ch] for ch in item["channels"]}
            ),
            None,
        )
        if record is not None:
            output[record["candidate_key"]] = record
    return list(output.values())


def rescue_candidates(
    base_records: Sequence[dict], removed_r080_representatives: Sequence[int]
) -> list[Candidate]:
    generated: dict[str, Candidate] = {}
    for record in base_records:
        channels = tuple(record["channels"])
        for replacement in removed_r080_representatives:
            if replacement in channels:
                continue
            for position in range(4):
                proposal = list(channels)
                proposal[position] = replacement
                if len(set(proposal)) != 4:
                    continue
                candidate = Candidate(tuple(sorted(proposal)), "r080_rescue")
                generated[candidate.key] = candidate
    return sorted(generated.values(), key=lambda candidate: candidate.channels or ())


def export_results(store: ResultStore, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = store.rows()
    columns = [
        "stage",
        "candidate_key",
        "channels_json",
        "replicate",
        "status",
        "reason",
        "elapsed_seconds",
        "failure_mvs_index",
        "failure_timestamp",
        "last_timestamp",
        "associated_poses",
        "full_associated_poses",
        "ate_rmse_m",
        "ate_mean_m",
        "ate_median_m",
        "ate_max_m",
        "ate_std_m",
        "rotation_ape_rmse_deg",
        "rotation_ape_mean_deg",
        "rotation_ape_max_deg",
        "translation_rpe_rmse_m",
        "translation_rpe_max_m",
        "rotation_rpe_rmse_deg",
        "rotation_rpe_max_deg",
        "legacy_sim3_rmse_m",
        "legacy_sim3_mean_m",
        "legacy_sim3_scale",
        "legacy_keyframe_poses",
        "full_ate_rmse_m",
        "full_rotation_ape_rmse_deg",
        "diagnostic_frames",
        "photo_mse_median",
        "photo_mse_p95",
        "photo_mse_nonfinite_count",
        "valid_ratio_min",
        "valid_ratio_median",
        "h_cond_max",
        "delta_norm_max",
        "crazy_affine_count",
        "trajectory_path",
        "keyframe_trajectory_path",
        "log_path",
        "created_at",
    ]
    with (output_dir / "all_evaluations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            columns
            + ["translation_jump_flag", "rotation_jump_flag", "rpe_safety_pass"]
        )
        for row in rows:
            if row["status"] == "PASS":
                flags = rpe_safety_flags(row_to_candidate_record(row))
            else:
                flags = ("", "", "")
            writer.writerow([row[column] for column in columns] + list(flags))
    ranking = store.search_ranking()
    with (output_dir / "search_ranking.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "channels",
                "source_stage",
                "se3_ate_rmse_cm",
                "se3_ate_mean_cm",
                "rotation_ape_rmse_deg",
                "translation_rpe_rmse_cm",
                "translation_rpe_max_cm",
                "rotation_rpe_rmse_deg",
                "rotation_rpe_max_deg",
                "translation_jump_flag",
                "rotation_jump_flag",
                "rpe_safety_pass",
                "legacy_keyframe_sim3_mean_cm",
                "legacy_keyframe_sim3_scale",
                "legacy_keyframe_poses",
                "photo_mse_median",
                "photo_mse_nonfinite_count",
                "valid_ratio_min",
            ]
        )
        for rank, record in enumerate(ranking, start=1):
            translation_jump, rotation_jump, rpe_safe = rpe_safety_flags(record)
            writer.writerow(
                [
                    rank,
                    record["candidate_key"],
                    record["stage"],
                    record["ate_rmse_m"] * 100,
                    record["ate_mean_m"] * 100,
                    record["rotation_ape_rmse_deg"],
                    record["translation_rpe_rmse_m"] * 100,
                    record["translation_rpe_max_m"] * 100,
                    record["rotation_rpe_rmse_deg"],
                    record["rotation_rpe_max_deg"],
                    translation_jump,
                    rotation_jump,
                    rpe_safe,
                    (
                        record["legacy_sim3_mean_m"] * 100
                        if record["legacy_sim3_mean_m"] is not None
                        else ""
                    ),
                    record["legacy_sim3_scale"],
                    record["legacy_keyframe_poses"],
                    record["photo_mse_median"],
                    record["photo_mse_nonfinite_count"],
                    record["valid_ratio_min"],
                ]
            )
    finalists = select_multimetric_records(ranking, 20)
    with (output_dir / "multimetric_top20.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "selection_order",
                "channels",
                "selection_reasons",
                "source_stage",
                "se3_ate_rmse_cm",
                "se3_ate_mean_cm",
                "translation_rpe_rmse_cm",
                "translation_rpe_max_cm",
                "rotation_rpe_rmse_deg",
                "rotation_rpe_max_deg",
            ]
        )
        for order, record in enumerate(finalists, start=1):
            writer.writerow(
                [
                    order,
                    record["candidate_key"],
                    "+".join(record["selection_reasons"]),
                    record["stage"],
                    record["ate_rmse_m"] * 100,
                    record["ate_mean_m"] * 100,
                    record["translation_rpe_rmse_m"] * 100,
                    record["translation_rpe_max_m"] * 100,
                    record["rotation_rpe_rmse_deg"],
                    record["rotation_rpe_max_deg"],
                ]
            )
    export_repeat_summary(rows, output_dir / "final_repeat_summary.csv")


def export_repeat_summary(rows: Sequence[sqlite3.Row], path: Path) -> None:
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        if row["candidate_key"] != "gray":
            grouped[row["candidate_key"]].append(row)
    records = []
    for key, observations in grouped.items():
        repeat_rows = [
            row
            for row in observations
            if row["stage"] == "repeat"
            or (
                row["replicate"] == 0
                and row["stage"]
                in {"bruteforce", "swapback_single", "swapback_factorial", "rescue"}
            )
        ]
        if not any(row["stage"] == "repeat" for row in repeat_rows):
            continue
        passes = [row for row in repeat_rows if row["status"] == "PASS"]
        rmse = np.asarray([row["ate_rmse_m"] * 100 for row in passes])
        means = np.asarray([row["ate_mean_m"] * 100 for row in passes])
        rotation_ape = np.asarray(
            [row["rotation_ape_rmse_deg"] for row in passes]
        )
        translation_rpe = np.asarray(
            [row["translation_rpe_rmse_m"] * 100 for row in passes]
        )
        rotation_rpe = np.asarray(
            [row["rotation_rpe_rmse_deg"] for row in passes]
        )
        records.append(
            [
                key,
                len(repeat_rows),
                len(passes),
                len(passes) / len(repeat_rows),
                float(rmse.mean()) if len(rmse) else "",
                float(rmse.std()) if len(rmse) else "",
                float(means.mean()) if len(means) else "",
                float(means.std()) if len(means) else "",
                float(rotation_ape.mean()) if len(rotation_ape) else "",
                float(rotation_ape.std()) if len(rotation_ape) else "",
                float(translation_rpe.mean()) if len(translation_rpe) else "",
                float(translation_rpe.std()) if len(translation_rpe) else "",
                float(rotation_rpe.mean()) if len(rotation_rpe) else "",
                float(rotation_rpe.std()) if len(rotation_rpe) else "",
            ]
        )
    records.sort(key=lambda row: (-(row[3]), row[4] if row[4] != "" else math.inf))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "channels",
                "observations",
                "passes",
                "pass_rate",
                "ate_rmse_cm_mean",
                "ate_rmse_cm_std",
                "ate_mean_cm_mean",
                "ate_mean_cm_std",
                "rotation_ape_rmse_deg_mean",
                "rotation_ape_rmse_deg_std",
                "translation_rpe_rmse_cm_mean",
                "translation_rpe_rmse_cm_std",
                "rotation_rpe_rmse_deg_mean",
                "rotation_rpe_rmse_deg_std",
            ]
        )
        writer.writerows(records)


def validate_inputs(args: argparse.Namespace) -> None:
    required = [
        args.dataset_dir / "frame_manifest.csv",
        args.dataset_dir / "mvs_metadata.json",
        args.dataset_dir / "groundtruth.txt",
        args.r070_clusters,
        args.r080_clusters,
        args.como_dir / "config/como.yml",
        args.como_dir / "config/open3d_viz.yml",
        args.como_dir / "como/como_dataset.py",
        args.python,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required inputs are missing:\n- " + "\n- ".join(missing)
        )
    if args.timeout_seconds <= 0 or args.terminate_grace_seconds <= 0:
        raise ValueError("Timeout and termination grace must be positive")
    if args.max_stage_hours < 0:
        raise ValueError("Maximum stage hours cannot be negative")
    if args.max_stage_hours > 0 and args.stage != "bruteforce":
        raise ValueError("--max-stage-hours is supported only with --stage bruteforce")
    if args.top_k < 1 or args.factorial_top_k < 1:
        raise ValueError("Top-K values must be positive")
    if args.total_top_observations < 2:
        raise ValueError("Top candidates need at least two total observations")
    if args.minimum_associated_poses < 3:
        raise ValueError("Minimum associated poses must be at least three")
    if args.diagnostic_failure_streak < 0:
        raise ValueError("Diagnostic failure streak cannot be negative")
    if args.execute and not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "DISPLAY is unset and Xvfb is not available in this environment. "
            "Run from the graphical session or provide a valid DISPLAY."
        )


def print_plan(
    console: Console,
    args: argparse.Namespace,
    r070: dict,
    r080: dict,
    legal_candidates: Sequence[Candidate],
) -> None:
    member_to_cluster, cluster_to_members = cluster_maps(r070)
    non_singletons = [
        members for members in cluster_to_members.values() if len(members) > 1
    ]
    r080_removed = sorted(
        set(r080["representative_channels"]) - set(r070["representative_channels"])
    )
    nominal_hours_7 = len(legal_candidates) * 7 / 3600
    nominal_hours_8 = len(legal_candidates) * 8 / 3600
    console.say("=" * 78)
    console.say("R=0.70 CONV1 FAIL-FAST SEARCH PLAN")
    console.say("=" * 78)
    console.say(
        f"Mode: {'EXECUTE' if args.execute else 'DRY RUN (COMO will not be launched)'}"
    )
    console.say(f"Requested stage: {args.stage}")
    if args.max_stage_hours > 0:
        console.say(
            f"Batch mode: stop cleanly between candidates after "
            f"{args.max_stage_hours:.2f} wall-clock hours"
        )
    console.say(f"Dataset: {args.dataset_dir}")
    console.say(
        "Scoring: all-frame tracking poses in MVS indices 10--49; metric-scale "
        "SE(3) alignment"
    )
    console.say(
        "Primary ranking: translation SE(3) ATE RMSE; rotation APE and "
        "translation/rotation RPE expose jumps"
    )
    console.say(
        "Historical comparison: Sim(3) ATE mean/RMSE and fitted scale are retained "
        "but do not drive ranking"
    )
    console.say(
        f"Trajectory gate: all {args.minimum_associated_poses} expected all-frame poses "
        "must associate in the scored window"
    )
    console.say(
        f"RPE safety marks: translation max > {TRANSLATION_RPE_MAX_M * 100:.1f}cm "
        f"or rotation max > {ROTATION_RPE_MAX_DEG:.1f}deg; unsafe runs remain visible "
        "in the ATE ranking but cannot enter multi-metric finalist contexts"
    )
    console.say(
        "Finalist mix: 50% top safe ATE, 25% safe Pareto front, remaining slots "
        "low translation/rotation RPE, then safe ATE fill"
    )
    console.say(
        f"r=0.70: {len(cluster_to_members)} final clusters, "
        f"{len(r070['representative_channels'])} representatives, "
        f"{len(non_singletons)} non-singleton clusters"
    )
    console.say(
        f"Legal four-channel brute-force candidates: {len(legal_candidates):,} "
        "(same final-cluster co-selection prohibited)"
    )
    console.say(
        f"Nominal brute-force time at 7--8 s/run: "
        f"{nominal_hours_7:.1f}--{nominal_hours_8:.1f} h before fail-fast savings"
    )
    console.say(
        "Immediate fail signatures: non-finite affine/pose/residual/update diagnostics, "
        "empty-AABB/runtime exception, timeout; isolated non-finite photo cost, finite "
        "low-quality diagnostics, and Crazy affine warnings are recorded but do not "
        "fail a run by default"
    )
    console.say(
        f"Post-search: Top-{args.top_k} repeats to "
        f"{args.total_top_observations} total observations; Top-{args.factorial_top_k} "
        "factorial swap-back"
    )
    console.say(f"r=0.80 rescue channels: {r080_removed}")
    console.say(
        "Safety: shared COMO config is exclusively locked and restored after every run; "
        "results are resumable; three consecutive infrastructure errors abort the search"
    )
    if not args.execute:
        console.say("DRY RUN COMPLETE: add --execute only after reviewing this plan.")


def main() -> None:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.r070_clusters = args.r070_clusters.resolve()
    args.r080_clusters = args.r080_clusters.resolve()
    args.como_dir = args.como_dir.resolve()
    args.python = args.python.resolve()
    args.output_dir = args.output_dir.resolve()
    validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    console = Console(args.output_dir / "search_console.log")
    store = ResultStore(args.output_dir / "evaluations.sqlite3")
    config_guard: ConfigGuard | None = None
    try:
        r070 = load_cluster_document(args.r070_clusters)
        r080 = load_cluster_document(args.r080_clusters)
        member_to_cluster, cluster_to_members = cluster_maps(r070)
        representatives = [int(channel) for channel in r070["representative_channels"]]
        legal = enumerate_legal_combinations(representatives, member_to_cluster)
        rng = np.random.default_rng(args.shuffle_seed)
        rng.shuffle(legal)
        if args.limit > 0:
            legal = legal[: args.limit]
        manifest = read_manifest(args.dataset_dir)
        print_plan(console, args, r070, r080, legal)
        if not args.execute:
            export_results(store, args.output_dir)
            return
        config_guard = ConfigGuard(
            args.como_dir / "config/como.yml",
            args.output_dir / ".como_config.lock",
        )
        runner = SearchRunner(
            args, console, store, config_guard, manifest, member_to_cluster
        )

        def regression() -> None:
            console.say(
                "[STAGE] Regression gate: gray should fail; known CNN should pass"
            )
            run_candidate_stage(
                runner,
                [Candidate(None, "gray_baseline")],
                "regression",
                force_keep=True,
            )
            run_candidate_stage(
                runner,
                [Candidate(KNOWN_GOOD, "known_good")],
                "regression",
                force_keep=True,
            )
            gray_result = store.get("regression", "gray", 0)
            cnn_result = store.get("regression", Candidate(KNOWN_GOOD).key, 0)
            if gray_result is not None and cnn_result is not None:
                gray_ok = gray_result["status"] in {
                    "FAIL_TRACKING_NAN",
                    "FAIL_TRACKING_DIAGNOSTIC",
                    "FAIL_TRACKING_RUNTIME",
                }
                cnn_ok = cnn_result["status"] == "PASS"
                console.say(
                    f"[REGRESSION] gray_expected_fail={'OK' if gray_ok else 'MISMATCH'}; "
                    f"known_good_expected_pass={'OK' if cnn_ok else 'MISMATCH'}"
                )
                if not (gray_ok and cnn_ok) and not args.allow_regression_mismatch:
                    raise RuntimeError(
                        "Regression gate failed. Use --allow-regression-mismatch only after "
                        "examining retained logs and confirming the changed behaviour."
                    )

        def bruteforce() -> None:
            console.say(
                f"[STAGE] Exhaustive r=0.70 brute force: {len(legal):,} legal combinations"
            )
            run_candidate_stage(runner, legal, "bruteforce")

        def retry_errors() -> None:
            candidates = store.retryable_bruteforce_candidates()
            console.say(
                "[STAGE] Targeted brute-force error retry: "
                f"{len(candidates):,} candidates with statuses "
                f"{sorted(RETRYABLE_BRUTEFORCE_STATUSES)}"
            )
            if not candidates:
                console.say("[RETRY] No retryable brute-force records remain")
                return
            previous_rerun_existing = args.rerun_existing
            args.rerun_existing = True
            try:
                run_candidate_stage(
                    runner,
                    candidates,
                    "bruteforce",
                    force_keep=True,
                )
            finally:
                args.rerun_existing = previous_rerun_existing
            remaining = store.retryable_bruteforce_candidates()
            console.say(
                f"[RETRY] Completed targeted pass; retryable records remaining: "
                f"{len(remaining):,}"
            )

        def swapback() -> None:
            brute_ranking = store.brute_ranking()
            if not brute_ranking:
                raise RuntimeError(
                    "Swap-back requires PASS results from the brute-force stage"
                )
            initial = select_multimetric_records(brute_ranking, args.top_k)
            console.say(
                f"[SELECTION] Multi-metric swap-back contexts: {len(initial)} "
                f"RPE-safe candidates (ATE/Pareto/low-RPE quotas)"
            )
            non_singleton_ids = {
                cluster_id
                for cluster_id, members in cluster_to_members.items()
                if len(members) > 1
            }
            contexts = add_cluster_coverage_contexts(
                initial, brute_ranking, non_singleton_ids, member_to_cluster
            )
            single_candidates = swapback_single_candidates(
                contexts, member_to_cluster, cluster_to_members
            )
            console.say(
                f"[STAGE] Single-member swap-back: {len(contexts)} contexts, "
                f"{len(single_candidates):,} unique proposals"
            )
            run_candidate_stage(runner, single_candidates, "swapback_single")
            current_top = select_multimetric_records(
                runner.store.search_ranking(), args.factorial_top_k
            )
            factorial = factorial_swapback_candidates(
                current_top, member_to_cluster, cluster_to_members
            )
            console.say(
                f"[STAGE] Factorial swap-back around Top-{args.factorial_top_k}: "
                f"{len(factorial):,} unique proposals"
            )
            run_candidate_stage(runner, factorial, "swapback_factorial")

        def rescue() -> None:
            ranking = runner.store.search_ranking()
            if not ranking:
                raise RuntimeError("r=0.80 rescue requires prior PASS search results")
            removed = sorted(
                set(r080["representative_channels"])
                - set(r070["representative_channels"])
            )
            rescue_contexts = select_multimetric_records(ranking, args.top_k)
            proposals = rescue_candidates(rescue_contexts, removed)
            console.say(
                f"[STAGE] r=0.80 rescue: removed representatives={removed}; "
                f"{len(proposals):,} unique insertion proposals. Same-r070-cluster "
                "co-selection is intentionally allowed in this audit."
            )
            run_candidate_stage(runner, proposals, "rescue")

        def repeat() -> None:
            ranking = runner.store.search_ranking()
            if not ranking:
                raise RuntimeError("Final repeats require prior PASS search results")
            finalist_records = select_multimetric_records(ranking, args.top_k)
            candidates = candidates_from_records(finalist_records)
            fresh_repeats = args.total_top_observations - 1
            console.say(
                f"[STAGE] Final repeat validation: Top-{len(candidates)}, "
                f"{fresh_repeats} fresh repeats each, retaining trajectories/logs"
            )
            for index, record in enumerate(finalist_records, start=1):
                console.say(
                    f"[FINALIST {index:02d}] channels={record['channels']} "
                    f"reasons={'+'.join(record['selection_reasons'])} "
                    f"ATE_RMSE={record['ate_rmse_m'] * 100:.3f}cm "
                    f"Trans_RPE_max={record['translation_rpe_max_m'] * 100:.3f}cm "
                    f"Rot_RPE_max={record['rotation_rpe_max_deg']:.3f}deg"
                )
            for replicate in range(1, fresh_repeats + 1):
                run_candidate_stage(
                    runner,
                    candidates,
                    "repeat",
                    replicate=replicate,
                    force_keep=True,
                )

        if args.stage == "regression":
            regression()
        elif args.stage == "bruteforce":
            bruteforce()
        elif args.stage == "retry-errors":
            retry_errors()
        elif args.stage == "swapback":
            swapback()
        elif args.stage == "rescue":
            rescue()
        elif args.stage == "repeat":
            repeat()
        elif args.stage == "summary":
            export_results(store, args.output_dir)
        elif args.stage == "all":
            regression()
            bruteforce()
            retry_errors()
            swapback()
            rescue()
            repeat()
        export_results(store, args.output_dir)
        console.say("[DONE] Requested stages finished; CSV summaries exported")
    finally:
        if config_guard is not None:
            config_guard.close()
        export_results(store, args.output_dir)
        store.close()
        console.close()


if __name__ == "__main__":
    main()
