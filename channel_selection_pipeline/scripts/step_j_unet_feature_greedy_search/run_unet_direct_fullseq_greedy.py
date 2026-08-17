#!/usr/bin/env python3
"""Direct 1--6 channel UNet encoder greedy search on the full lightswitch sequence.

The implementation reuses Step-E process control, immediate failure detection,
trajectory-completeness gates and the historical primary metric: keyframe
evo_ape TUM alignment with scale correction, ranked by ATE mean.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import itertools
import json
import math
import random
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
COMO_DIR = PROJECT_ROOT / "como"
FULL_EVALUATOR_PATH = (
    PROJECT_ROOT
    / "channel_selection_pipeline/scripts/step_e_full_sequence_evaluation/"
    "run_full_sequence_evaluation.py"
)
DEFAULT_DATASET = Path(
    "/home/melody/data/tum/rgbd_dataset_freiburg1_desk_lightswitch"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "channel_selection_results/step_j_unet_direct_fullseq_greedy"
)
DEFAULT_PYTHON = Path("/home/melody/anaconda3/envs/como/bin/python")
DEFAULT_EVO_APE = Path("/home/melody/anaconda3/envs/como/bin/evo_ape")
DEFAULT_EVO_RPE = Path("/home/melody/anaconda3/envs/como/bin/evo_rpe")
SHARED_CONFIG_LOCK = (
    PROJECT_ROOT
    / "channel_selection_results/step_d_fail_fast_evaluation/"
    "r070_bruteforce_v2/.como_config.lock"
)

NUM_CHANNELS = 32
MAX_CHANNELS = 6
RANDOM_SEED = 20260814
PROTOCOL = "unet_enc1_direct_full_sequence_greedy_v1"
UNET_ENC_LEVEL = 1
ENCODER_LABEL = "enc1"
ALL_CHANNEL_ANCHOR_LABEL = "unet_all32"
BQS_REFERENCE_CANDIDATES: tuple[tuple[str, tuple[int, ...], str, str], ...] = (
    (
        "bqs_historical_top4",
        (4, 9, 10, 15),
        "historical BQS-greedy path Top-4 prefix, re-evaluated here",
        "B4_historical_bqs_prefix",
    ),
    (
        "bqs_historical_top5",
        (4, 9, 10, 15, 30),
        "historical BQS-greedy Top-5, re-evaluated here",
        "B5_historical_bqs_top5",
    ),
)


def all_channels_display() -> str:
    return f"all{NUM_CHANNELS}"


def forward_extension_budget(seed_cardinality: int) -> int:
    """Upper bound before cache reuse for one greedy path through K=max."""

    return sum(
        NUM_CHANNELS - cardinality + 1
        for cardinality in range(seed_cardinality + 1, MAX_CHANNELS + 1)
    )


def theoretical_budget(starts: int, seed_cardinality: int) -> dict[str, int]:
    """Protocol-level upper bounds, excluding final repeat observations."""

    extensions = forward_extension_budget(seed_cardinality)
    direct = NUM_CHANNELS + starts * extensions
    if seed_cardinality == 2:
        direct += math.comb(NUM_CHANNELS, 2)
    random = starts * extensions
    swap = MAX_CHANNELS * (NUM_CHANNELS - MAX_CHANNELS)
    return {
        "direct": direct,
        "random": random,
        "swap_max": swap,
        "pre_repeat_total_max": direct + random + swap,
    }


def load_full_evaluator():
    spec = importlib.util.spec_from_file_location(
        "unet_full_sequence_evaluator", FULL_EVALUATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Step-E evaluator: {FULL_EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


full = load_full_evaluator()
core = full.core


@dataclass(frozen=True)
class SearchCandidate:
    label: str
    channels: tuple[int, ...] | None
    role: str

    @property
    def candidate_key(self) -> str:
        return candidate_key(self.channels)

    @property
    def display(self) -> str:
        if self.channels is None:
            return "gray"
        if self.channels == tuple(range(NUM_CHANNELS)):
            return all_channels_display()
        return "[" + ",".join(f"d{channel}" for channel in self.channels) + "]"

    def to_evaluator_spec(self):
        return full.EvaluationSpec(self.label, self.channels, self.role)


class BatchLimitReached(RuntimeError):
    """A deliberate, clean stop between evaluator invocations."""


class UNetConfigGuard(core.ConfigGuard):
    """Apply UNet tracking settings while retaining gray mapping and sensor depth."""

    def apply(self, candidate: core.Candidate) -> dict:
        config = yaml.safe_load(self.original)
        config["mapping"]["color"] = "gray"
        tracking = config["tracking"]
        tracking["debug_tracking_diagnostics"] = True
        tracking["debug_tracking_print_every_frame"] = True
        tracking["debug_tracking_save_suspicious"] = False
        if candidate.channels is None:
            tracking["color"] = "gray"
        else:
            channels = tuple(sorted(int(channel) for channel in candidate.channels))
            if not channels or channels[0] < 0 or channels[-1] >= NUM_CHANNELS:
                raise ValueError(
                    f"UNet {ENCODER_LABEL} channels must be unique indices in "
                    f"0--{NUM_CHANNELS - 1}: {channels}"
                )
            if len(set(channels)) != len(channels):
                raise ValueError(f"Duplicate UNet channels: {channels}")
            tracking["color"] = "unet"
            tracking["unet_enc_level"] = UNET_ENC_LEVEL
            tracking["unet_channel_select"] = (
                "all"
                if channels == tuple(range(NUM_CHANNELS))
                else ",".join(f"d{channel}" for channel in channels)
            )
        encoded = yaml.safe_dump(
            config, default_flow_style=False, allow_unicode=True, sort_keys=False
        ).encode("utf-8")
        core.atomic_write_bytes(self.config_path, encoded)
        return config


class SearchStore:
    """Protocol state and stage provenance alongside the established evaluator DB."""

    def __init__(self, path: Path):
        self.results = full.ResultStore(path)
        self.connection = self.results.connection
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS stage_candidates (
                stage TEXT NOT NULL,
                label TEXT NOT NULL,
                replicate INTEGER NOT NULL,
                candidate_key TEXT NOT NULL,
                channels_json TEXT,
                source_label TEXT,
                role TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(stage, label, replicate)
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS search_state (
                name TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.results.close()

    def get_state(self, name: str, default: Any = None) -> Any:
        row = self.connection.execute(
            "SELECT value_json FROM search_state WHERE name=?", (name,)
        ).fetchone()
        return default if row is None else json.loads(row[0])

    def set_state(self, name: str, value: Any) -> None:
        self.connection.execute(
            """
            INSERT INTO search_state(name, value_json, updated_at)
            VALUES(?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
              value_json=excluded.value_json,
              updated_at=excluded.updated_at
            """,
            (
                name,
                json.dumps(value, indent=2, sort_keys=True),
                dt.datetime.now(dt.timezone.utc).isoformat(),
            ),
        )
        self.connection.commit()

    def register(
        self,
        stage: str,
        candidate: SearchCandidate,
        replicate: int,
        source_label: str | None,
        metadata: dict[str, Any],
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO stage_candidates(
                stage, label, replicate, candidate_key, channels_json,
                source_label, role, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stage,
                candidate.label,
                replicate,
                candidate.candidate_key,
                json.dumps(candidate.channels) if candidate.channels is not None else None,
                source_label,
                candidate.role,
                json.dumps(metadata, sort_keys=True),
                dt.datetime.now(dt.timezone.utc).isoformat(),
            ),
        )
        self.connection.commit()

    def evaluation_for_key(self, key: str, replicate: int = 0):
        return self.connection.execute(
            """
            SELECT * FROM evaluations
            WHERE candidate_key=? AND replicate=?
            ORDER BY id
            LIMIT 1
            """,
            (key, replicate),
        ).fetchone()

    def evaluations_for_key(self, key: str):
        return self.connection.execute(
            "SELECT * FROM evaluations WHERE candidate_key=? ORDER BY replicate, id",
            (key,),
        ).fetchall()

    def direct_keys_for_cardinality(self, cardinality: int) -> set[str]:
        rows = self.connection.execute(
            """
            SELECT DISTINCT candidate_key, channels_json FROM stage_candidates
            WHERE stage='direct_greedy' AND replicate=0
            """
        ).fetchall()
        return {
            row["candidate_key"]
            for row in rows
            if (channels := parse_channels_json(row["channels_json"])) is not None
            and len(channels) == cardinality
        }

    def all_evaluation_keys(self) -> set[str]:
        rows = self.connection.execute("SELECT DISTINCT candidate_key FROM evaluations").fetchall()
        return {str(row[0]) for row in rows}

    def export_registry(self, output_dir: Path) -> None:
        rows = self.connection.execute(
            "SELECT * FROM stage_candidates ORDER BY stage, label, replicate"
        ).fetchall()
        if not rows:
            return
        with (output_dir / "candidate_registry.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            columns = list(rows[0].keys())
            writer.writerow(columns)
            for row in rows:
                writer.writerow([row[column] for column in columns])


def candidate_key(channels: tuple[int, ...] | None) -> str:
    return "gray" if channels is None else ",".join(str(item) for item in channels)


def parse_channels_json(value: str | None) -> tuple[int, ...] | None:
    return None if value is None else tuple(sorted(int(item) for item in json.loads(value)))


def key_to_channels(key: str) -> tuple[int, ...] | None:
    return None if key == "gray" else tuple(int(item) for item in key.split(","))


def channel_tag(channels: Sequence[int] | None) -> str:
    return "gray" if channels is None else "_".join(f"d{channel:02d}" for channel in channels)


def format_channels(channels: tuple[int, ...] | None) -> str:
    if channels is None:
        return "gray"
    if channels == tuple(range(NUM_CHANNELS)):
        return all_channels_display()
    return "[" + ",".join(f"d{channel}" for channel in channels) + "]"


def primary_metric(row) -> float:
    if row is None or row["status"] != "PASS":
        return math.inf
    value = row["historical_evo_ape_mean_m"]
    return float(value) if value is not None and math.isfinite(float(value)) else math.inf


def format_metric(value: float) -> str:
    return f"{value * 100:.4f}cm" if math.isfinite(value) else "n/a"


def anchor_candidates() -> tuple[SearchCandidate, ...]:
    anchors: list[SearchCandidate] = [
        SearchCandidate("gray_current", None, "current photometric gray control"),
        SearchCandidate(
            ALL_CHANNEL_ANCHOR_LABEL,
            tuple(range(NUM_CHANNELS)),
            f"unselected UNet {ENCODER_LABEL} {all_channels_display()}-channel control",
        ),
    ]
    anchors.extend(
        SearchCandidate(label, channels, role)
        for label, channels, role, _ in BQS_REFERENCE_CANDIDATES
    )
    return tuple(anchors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            f"Search 1--6 UNet {ENCODER_LABEL} channels using full-sequence "
            "fr1/desk_lightswitch ATE only."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("all", "anchors", "greedy", "random", "swap", "repeats", "export"),
        default="all",
        help="Stage to run. all follows the frozen protocol in order.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch COMO. Without this flag only validation is performed.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--como-dir", type=Path, default=COMO_DIR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--evo-ape", type=Path, default=DEFAULT_EVO_APE)
    parser.add_argument("--evo-rpe", type=Path, default=DEFAULT_EVO_RPE)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--starts",
        choices=("auto", "2", "3", "4"),
        default="auto",
        help="Singleton-ATE greedy seeds; auto uses anchor median runtime.",
    )
    parser.add_argument(
        "--batch-hours",
        type=float,
        default=36.0,
        help="Cleanly stop between candidates after this wall time; use 0 to disable.",
    )
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--minimum-coverage", type=float, default=0.90)
    parser.add_argument("--completion-tolerance-seconds", type=float, default=0.10)
    parser.add_argument("--terminate-grace-seconds", type=float, default=3.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[float]:
    required = (
        args.dataset_dir / "matched_rgb.txt",
        args.dataset_dir / "groundtruth.txt",
        args.como_dir / "config/como.yml",
        args.como_dir / "como/como_dataset.py",
        args.python,
        args.evo_ape,
        args.evo_rpe,
        FULL_EVALUATOR_PATH,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if args.batch_hours < 0:
        raise ValueError("--batch-hours must be zero or positive")
    if not 0 < args.minimum_coverage <= 1:
        raise ValueError("--minimum-coverage must be in (0, 1]")
    config = yaml.safe_load((args.como_dir / "config/como.yml").read_text())
    if not isinstance(config.get("tracking"), dict) or not isinstance(
        config.get("mapping"), dict
    ):
        raise ValueError("COMO config must contain tracking and mapping dictionaries")
    return full.read_timestamp_index(args.dataset_dir / "matched_rgb.txt")


def write_protocol(args: argparse.Namespace, timestamps: Sequence[float]) -> None:
    payload = {
        "protocol": PROTOCOL,
        "dataset_dir": str(args.dataset_dir),
        "matched_frames": len(timestamps),
        "first_timestamp": timestamps[0],
        "last_timestamp": timestamps[-1],
        "feature_space": {
            "network": "UNet encoder",
            "enc_level": UNET_ENC_LEVEL,
            "available_channels": NUM_CHANNELS,
            "channel_names": [f"d{index}" for index in range(NUM_CHANNELS)],
        },
        "tracking": {"color": "unet", "unet_enc_level": UNET_ENC_LEVEL},
        "mapping": {"color": "gray", "use_sensor_depth": True},
        "primary_metric": (
            "keyframe evo_ape tum groundtruth.txt data_tum.txt with alignment "
            "and scale correction, ranked by translation ATE mean"
        ),
        "max_channels": MAX_CHANNELS,
        "timeout_seconds": args.timeout_seconds,
        "random_seed": args.random_seed,
        "auto_start_rule": {"t50_le_60_s": 4, "t50_le_100_s": 3, "otherwise": 2},
        "theoretical_evaluation_budgets": {
            f"starts_{starts}": theoretical_budget(starts, 1)
            for starts in (2, 3, 4)
        },
        "all_singletons_fail_fallback": {
            "method": (
                f"exhaustively evaluate all C({NUM_CHANNELS},2)="
                f"{math.comb(NUM_CHANNELS, 2)} pairs, use the best "
                "PASS pairs as greedy seeds, then extend through K=6"
            ),
            "random_control_note": (
                "K=2 is exhaustive, so no disjoint random pairs remain; retain "
                "equal-budget random controls at K=3--6"
            ),
            "theoretical_pre_repeat_budget": {
                f"starts_{starts}": theoretical_budget(starts, 2)
                for starts in (2, 3, 4)
            },
        },
        "anchors": [asdict(candidate) for candidate in anchor_candidates()],
    }
    path = args.output_dir / "candidate_plan.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("protocol") != PROTOCOL:
            raise RuntimeError(
                "Output directory contains a different protocol; choose a new "
                "directory rather than mixing experiments."
            )
        for key in ("dataset_dir", "random_seed"):
            if existing.get(key) != payload[key]:
                raise RuntimeError(f"Output directory has a different {key}.")
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class Controller:
    def __init__(
        self,
        args: argparse.Namespace,
        store: SearchStore,
        console,
        timestamps: Sequence[float],
    ):
        self.args = args
        self.store = store
        self.console = console
        self.timestamps = timestamps
        self.guard: UNetConfigGuard | None = None
        self.started = time.monotonic()
        self.stage_new_runs = 0
        self.stage_run_hint = 1

    def set_stage_progress(self, total_hint: int) -> None:
        self.stage_new_runs = 0
        self.stage_run_hint = max(1, total_hint)

    def check_batch_limit(self) -> None:
        if self.args.batch_hours <= 0:
            return
        elapsed = time.monotonic() - self.started
        if self.stage_new_runs > 0 and elapsed >= self.args.batch_hours * 3600:
            raise BatchLimitReached(
                f"Reached {self.args.batch_hours:.2f}h after {self.stage_new_runs:,} "
                "new evaluator invocations; stopping before the next candidate."
            )

    def stage_banner(self, title: str) -> None:
        self.console.say("")
        self.console.say("[STAGE] " + title)

    def ensure(
        self,
        stage: str,
        candidate: SearchCandidate,
        *,
        replicate: int = 0,
        metadata: dict[str, Any] | None = None,
    ):
        """Use a matching saved row, or run exactly one full-sequence evaluation."""

        metadata = metadata or {}
        existing = self.store.evaluation_for_key(candidate.candidate_key, replicate)
        if existing is not None:
            self.store.register(
                stage,
                candidate,
                replicate,
                str(existing["label"]),
                {**metadata, "reused_existing_evaluation": True},
            )
            self.console.say(
                f"[CACHE] stage={stage} channels={candidate.display} "
                f"replicate={replicate} source={existing['label']} "
                f"status={existing['status']} primary_ATE={format_metric(primary_metric(existing))}"
            )
            return existing

        if not self.args.execute:
            self.console.say(
                f"[DRY RUN] stage={stage} would evaluate {candidate.label} "
                f"channels={candidate.display} replicate={replicate}"
            )
            return None

        self.check_batch_limit()
        if self.guard is None:
            raise RuntimeError("Evaluator config guard is not open")
        self.stage_new_runs += 1
        full.evaluate_one(
            self.args,
            self.console,
            self.store.results,
            self.guard,
            candidate.to_evaluator_spec(),
            replicate,
            self.stage_new_runs,
            self.stage_run_hint,
            self.timestamps,
        )
        row = self.store.evaluation_for_key(candidate.candidate_key, replicate)
        if row is None:
            raise RuntimeError(
                f"Step-E evaluator returned without saving {candidate.candidate_key}"
            )
        self.store.register(
            stage,
            candidate,
            replicate,
            str(row["label"]),
            {**metadata, "reused_existing_evaluation": False},
        )
        full.export_results(self.store.results, self.args.output_dir)
        self.store.export_registry(self.args.output_dir)
        return row

    def anchors(self) -> None:
        self.stage_banner("Anchor controls: one observation each, never used as greedy seeds")
        anchors = anchor_candidates()
        self.set_stage_progress(len(anchors))
        for candidate in anchors:
            self.ensure(
                "anchors",
                candidate,
                metadata={"protocol_role": "control_or_historical_anchor"},
            )
        if self.args.execute:
            self.store.set_state("anchors_completed", True)

    def effective_starts(self) -> tuple[int, float]:
        existing = self.store.get_state("greedy_setup")
        if existing is not None:
            return int(existing["effective_starts"]), float(existing["anchor_t50_seconds"])

        durations = []
        for candidate in anchor_candidates():
            row = self.store.evaluation_for_key(candidate.candidate_key)
            if row is not None and row["status"] == "PASS":
                durations.append(float(row["elapsed_seconds"]))
        if not durations:
            raise RuntimeError(
                "No successful anchors available. Resolve anchor failures before greedy."
            )
        t50 = float(np.median(durations))
        if self.args.starts == "auto":
            starts = 4 if t50 <= 60 else (3 if t50 <= 100 else 2)
            rule = "auto: t50<=60:4; 60<t50<=100:3; t50>100:2"
        else:
            starts = int(self.args.starts)
            rule = f"manual --starts {starts}"
        self.store.set_state(
            "greedy_setup",
            {
                "protocol": PROTOCOL,
                "effective_starts": starts,
                "anchor_t50_seconds": t50,
                "selection_rule": rule,
            },
        )
        self.console.say(
            f"[BUDGET] anchor t50={t50:.1f}s -> {starts} starts; normal "
            f"direct={theoretical_budget(starts, 1)['direct']}, "
            f"random={theoretical_budget(starts, 1)['random']}, "
            f"swap<={theoretical_budget(starts, 1)['swap_max']}. "
            "If fewer than the planned number of singleton seeds PASS, a K=2 "
            "sweep augments the missing starts."
        )
        return starts, t50

    def best_key(self, keys: Iterable[str]):
        ranked = []
        for key in sorted(set(keys)):
            row = self.store.evaluation_for_key(key)
            metric = primary_metric(row)
            if math.isfinite(metric):
                ranked.append((metric, key, row))
        if not ranked:
            return None, None
        _, key, row = min(ranked, key=lambda item: (item[0], item[1]))
        return key, row

    def greedy(self) -> None:
        if not self.store.get_state("anchors_completed"):
            raise RuntimeError("Run --stage anchors first, or use --stage all.")
        self.stage_banner("Full singleton sweep and forward greedy paths through K=6")
        self.set_stage_progress(NUM_CHANNELS)
        singleton_rows: dict[int, Any] = {}
        for channel in range(NUM_CHANNELS):
            candidate = SearchCandidate(
                f"g_singleton_d{channel:02d}",
                (channel,),
                "direct full-sequence singleton ATE sweep",
            )
            singleton_rows[channel] = self.ensure(
                "direct_greedy",
                candidate,
                metadata={"phase": "singleton", "cardinality": 1, "channel": channel},
            )
        if not self.args.execute:
            return

        starts, _ = self.effective_starts()
        ranked = sorted(
            (
                (primary_metric(row), channel)
                for channel, row in singleton_rows.items()
                if math.isfinite(primary_metric(row))
            ),
            key=lambda item: (item[0], item[1]),
        )
        singleton_seeds = [(channel,) for _, channel in ranked[:starts]]
        pair_rows: dict[tuple[int, int], Any] = {}
        ranked_pairs: list[tuple[float, tuple[int, int]]] = []

        def exhaustive_pair_sweep(reason: str) -> None:
            """Record every K=2 outcome once; existing rows are reused by key."""

            title = (
                "Fallback: all singleton runs failed; exhaustive two-channel seed sweep"
                if not singleton_seeds
                else (
                    "Sparse-singleton augmentation: exhaustive two-channel seed sweep "
                    "for the missing greedy starts"
                )
            )
            self.stage_banner(title)
            self.set_stage_progress(NUM_CHANNELS * (NUM_CHANNELS - 1) // 2)
            for left, right in itertools.combinations(range(NUM_CHANNELS), 2):
                pair = (left, right)
                candidate = SearchCandidate(
                    f"g_pair_seed_sweep_{channel_tag(pair)}",
                    pair,
                    "exhaustive two-channel full-sequence seed sweep "
                    f"({reason})",
                )
                pair_rows[pair] = self.ensure(
                    "direct_greedy",
                    candidate,
                    metadata={
                        "phase": "pair_seed_sweep",
                        "cardinality": 2,
                        "sweep_reason": reason,
                    },
                )
            ranked_pairs.extend(
                sorted(
                    (
                        (primary_metric(row), pair)
                        for pair, row in pair_rows.items()
                        if math.isfinite(primary_metric(row))
                    ),
                    key=lambda item: (item[0], item[1]),
                )
            )

        pair_seed_sets: list[tuple[int, int]] = []
        reserved_singleton_pair_keys: set[str] = set()
        if len(singleton_seeds) >= starts:
            seed_sets: list[tuple[int, ...]] = singleton_seeds
            seed_source = "best PASS singleton ATE"
            seed_policy = "singleton_only"
        elif not singleton_seeds:
            exhaustive_pair_sweep("all_singletons_failed")
            if not ranked_pairs:
                raise RuntimeError(
                    f"All {math.comb(NUM_CHANNELS, 2)} two-channel fallback "
                    "combinations also failed; no valid direct full-sequence "
                    "greedy path can be started."
                )
            pair_seed_sets = [pair for _, pair in ranked_pairs[:starts]]
            seed_sets = pair_seed_sets
            seed_source = "best PASS pair ATE after all-singleton-failure fallback"
            seed_policy = "pair_only_after_all_singleton_failure"
            self.console.say(
                f"[PAIR FALLBACK] {len(ranked_pairs)}/{math.comb(NUM_CHANNELS, 2)} "
                "pairs PASS; selected "
                + ", ".join(format_channels(pair) for pair in pair_seed_sets)
            )
        else:
            missing_starts = starts - len(singleton_seeds)
            exhaustive_pair_sweep(
                f"only_{len(singleton_seeds)}_singleton_passes_for_{starts}_planned_starts"
            )
            if ranked_pairs:
                # The singleton path itself selects its strongest adjacent K=2
                # candidate. Reserve that one so a pair-start path cannot become
                # an exact duplicate of the singleton route at K=2.
                for singleton in singleton_seeds:
                    incident_keys = [
                        candidate_key(pair)
                        for _, pair in ranked_pairs
                        if singleton[0] in pair
                    ]
                    best_incident_key, _ = self.best_key(incident_keys)
                    if best_incident_key is not None:
                        reserved_singleton_pair_keys.add(best_incident_key)
                pair_seed_sets = [
                    pair
                    for _, pair in ranked_pairs
                    if candidate_key(pair) not in reserved_singleton_pair_keys
                ][:missing_starts]
                # If every passing pair was reserved, retain the strongest ones
                # rather than discarding valid starts.
                if len(pair_seed_sets) < missing_starts:
                    selected = set(pair_seed_sets)
                    pair_seed_sets.extend(
                        pair
                        for _, pair in ranked_pairs
                        if pair not in selected
                    )
                    pair_seed_sets = pair_seed_sets[:missing_starts]
            seed_sets = [*singleton_seeds, *pair_seed_sets]
            seed_source = (
                "PASS singleton seeds plus best non-duplicate PASS pair seeds "
                "after sparse-singleton augmentation"
            )
            seed_policy = "mixed_singleton_and_pair_augmentation"
            self.console.say(
                f"[SPARSE SINGLETON] {len(singleton_seeds)} PASS singleton seed(s); "
                f"pair sweep found {len(ranked_pairs)} PASS pair(s). "
                "Retained singleton seeds="
                + ", ".join(format_channels(seed) for seed in singleton_seeds)
                + "; added pair seeds="
                + (
                    ", ".join(format_channels(seed) for seed in pair_seed_sets)
                    if pair_seed_sets
                    else "none"
                )
                + "."
            )
            if len(seed_sets) < starts:
                self.console.say(
                    f"[SEED WARNING] Only {len(seed_sets)}/{starts} distinct "
                    "PASS seed paths are available after pair augmentation."
                )
        setup = self.store.get_state("greedy_setup")
        existing_seeds = setup.get("seeds")
        encoded_seeds = [list(seed_set) for seed_set in seed_sets]
        if existing_seeds is not None and existing_seeds != encoded_seeds:
            can_upgrade_sparse_singleton_setup = (
                not self.store.get_state("greedy_completed")
                and setup.get("seed_policy") is None
                and all(len(seed) == 1 for seed in existing_seeds)
                and encoded_seeds[: len(existing_seeds)] == existing_seeds
                and seed_policy == "mixed_singleton_and_pair_augmentation"
            )
            if not can_upgrade_sparse_singleton_setup:
                raise RuntimeError(
                    "Saved greedy seeds conflict with the current seed policy. "
                    "Do not silently change a resumable search."
                )
            setup["pre_augmentation_seeds"] = existing_seeds
            self.console.say(
                "[SEED UPGRADE] Preserving saved singleton path(s) and appending "
                "pair-start paths for sparse-singleton coverage."
            )
        if existing_seeds is None or existing_seeds != encoded_seeds:
            setup["seeds"] = encoded_seeds
            setup["seed_cardinalities"] = [len(seed_set) for seed_set in seed_sets]
            setup["seed_cardinality"] = (
                len(seed_sets[0])
                if len({len(seed_set) for seed_set in seed_sets}) == 1
                else None
            )
            setup["seed_source"] = seed_source
            setup["seed_policy"] = seed_policy
            setup["singleton_ranking"] = [
                {"channel": channel, "historical_ate_mean_cm": score * 100}
                for score, channel in ranked
            ]
            setup["pair_seed_sweep"] = {
                "ran": bool(pair_rows),
                "pass_count": len(ranked_pairs),
                "selected_pair_seeds": [list(pair) for pair in pair_seed_sets],
                "reserved_singleton_pair_keys": sorted(reserved_singleton_pair_keys),
            }
            self.store.set_state("greedy_setup", setup)
        (self.args.output_dir / "seed_selection_plan.json").write_text(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "effective_starts": starts,
                    "seed_policy": seed_policy,
                    "seed_source": seed_source,
                    "singleton_passes": [
                        {"channel": channel, "historical_ate_mean_cm": score * 100}
                        for score, channel in ranked
                    ],
                    "selected_seeds": [list(seed) for seed in seed_sets],
                    "selected_pair_seeds": [list(pair) for pair in pair_seed_sets],
                    "reserved_singleton_pair_keys": sorted(reserved_singleton_pair_keys),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.console.say(
            f"[SEEDS] source={seed_source}: "
            + ", ".join(format_channels(seed_set) for seed_set in seed_sets)
        )

        self.set_stage_progress(
            sum(forward_extension_budget(len(seed_set)) for seed_set in seed_sets)
        )
        paths = self.store.get_state("greedy_paths", {})
        for seed_index, seed_set in enumerate(seed_sets, start=1):
            seed_cardinality = len(seed_set)
            path_key = f"seed_{seed_index:02d}_{channel_tag(seed_set)}"
            path = paths.get(
                path_key,
                {
                    "seed_index": seed_index,
                    "seed_channels": list(seed_set),
                    "seed_cardinality": seed_cardinality,
                    "steps": {str(seed_cardinality): list(seed_set)},
                    "status": "active",
                },
            )
            if (
                tuple(int(item) for item in path["seed_channels"]) != seed_set
                or int(path["seed_cardinality"]) != seed_cardinality
            ):
                raise RuntimeError(
                    f"Saved path {path_key} does not match its immutable seed."
                )
            if path.get("status") != "active":
                continue
            current = tuple(int(item) for item in path["steps"][str(seed_cardinality)])
            for cardinality in range(seed_cardinality + 1, MAX_CHANNELS + 1):
                stored = path["steps"].get(str(cardinality))
                if stored is None and str(cardinality) in path["steps"]:
                    path["status"] = "terminated_no_pass_extension"
                    break
                if stored is not None:
                    current = tuple(int(item) for item in stored)
                    continue

                trial_keys: list[str] = []
                for added in range(NUM_CHANNELS):
                    if added in current:
                        continue
                    trial = tuple(sorted((*current, added)))
                    candidate = SearchCandidate(
                        (
                            f"g_s{seed_index:02d}_k{cardinality:02d}_"
                            f"from_{channel_tag(current)}_add_d{added:02d}"
                        ),
                        trial,
                        "direct full-sequence ATE forward-greedy trial",
                    )
                    row = self.ensure(
                        "direct_greedy",
                        candidate,
                        metadata={
                            "phase": "forward_extension",
                            "seed_index": seed_index,
                            "seed_channels": list(seed_set),
                            "parent_channels": list(current),
                            "added_channel": added,
                            "cardinality": cardinality,
                        },
                    )
                    if row is not None:
                        trial_keys.append(candidate.candidate_key)
                winner_key, winner_row = self.best_key(trial_keys)
                if winner_key is None:
                    path["steps"][str(cardinality)] = None
                    path["status"] = "terminated_no_pass_extension"
                    self.console.say(
                        f"[GREEDY TERMINAL] seed={format_channels(seed_set)}; no PASS trial at "
                        f"K={cardinality}."
                    )
                    paths[path_key] = path
                    self.store.set_state("greedy_paths", paths)
                    break
                current = key_to_channels(winner_key)
                assert current is not None
                path["steps"][str(cardinality)] = list(current)
                path["last_primary_ate_mean_cm"] = primary_metric(winner_row) * 100
                paths[path_key] = path
                self.store.set_state("greedy_paths", paths)
                self.console.say(
                    f"[GREEDY CHOICE] seed={format_channels(seed_set)} K={cardinality} "
                    f"channels={format_channels(current)} primary_ATE="
                    f"{primary_metric(winner_row) * 100:.4f}cm"
                )
            paths[path_key] = path
            self.store.set_state("greedy_paths", paths)
        self.store.set_state("greedy_completed", True)
        self.write_direct_path_csv()

    def greedy_endpoints(self) -> dict[int, str]:
        paths = self.store.get_state("greedy_paths", {})
        endpoint_keys: dict[int, list[str]] = {
            cardinality: [] for cardinality in range(1, MAX_CHANNELS + 1)
        }
        for path in paths.values():
            for raw_k, raw_channels in path.get("steps", {}).items():
                if raw_channels is None:
                    continue
                cardinality = int(raw_k)
                channels = tuple(int(item) for item in raw_channels)
                if len(channels) == cardinality:
                    endpoint_keys[cardinality].append(candidate_key(channels))
        result: dict[int, str] = {}
        for cardinality, keys in endpoint_keys.items():
            key, _ = self.best_key(keys)
            if key is not None:
                result[cardinality] = key
        return result

    def write_direct_path_csv(self) -> None:
        paths = self.store.get_state("greedy_paths", {})
        with (self.args.output_dir / "direct_greedy_path.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "seed_index",
                    "seed_channels",
                    "cardinality",
                    "channels",
                    "path_status",
                    "single_run_primary_ate_mean_cm",
                    "evaluation_label",
                ]
            )
            for path in sorted(paths.values(), key=lambda item: item["seed_index"]):
                for raw_k, raw_channels in sorted(
                    path.get("steps", {}).items(), key=lambda item: int(item[0])
                ):
                    channels = (
                        tuple(int(item) for item in raw_channels)
                        if raw_channels is not None
                        else None
                    )
                    row = (
                        self.store.evaluation_for_key(candidate_key(channels))
                        if channels is not None
                        else None
                    )
                    metric = primary_metric(row)
                    writer.writerow(
                        [
                            path["seed_index"],
                            json.dumps(path["seed_channels"]),
                            raw_k,
                            json.dumps(channels) if channels is not None else "",
                            path.get("status", "active"),
                            metric * 100 if math.isfinite(metric) else "",
                            row["label"] if row is not None else "",
                        ]
                    )

    def random_control(self) -> None:
        if not self.store.get_state("greedy_completed"):
            raise RuntimeError("Run --stage greedy first, or use --stage all.")
        self.stage_banner("Equal-budget random control, sampled separately at K=2--6")
        plan = self.load_or_create_random_plan()
        self.set_stage_progress(len(plan["candidates"]))
        for entry in plan["candidates"]:
            channels = tuple(int(item) for item in entry["channels"])
            candidate = SearchCandidate(
                entry["label"],
                channels,
                "budget-matched cardinality-stratified random full-sequence control",
            )
            self.ensure(
                "random",
                candidate,
                metadata={
                    "phase": "random_control",
                    "cardinality": len(channels),
                    "random_seed": self.args.random_seed,
                    "random_rank_within_cardinality": entry["rank_within_cardinality"],
                },
            )
        if self.args.execute:
            self.store.set_state("random_completed", True)

    def load_or_create_random_plan(self) -> dict[str, Any]:
        path = self.args.output_dir / "random_budget_matched_plan.json"
        if path.is_file():
            plan = json.loads(path.read_text(encoding="utf-8"))
            if plan.get("protocol") != PROTOCOL:
                raise RuntimeError(f"Random-plan protocol mismatch: {path}")
            if int(plan.get("random_seed")) != self.args.random_seed:
                raise RuntimeError(
                    "Random plan seed differs from --random-seed. Use a new output "
                    "directory rather than changing a resumable experiment."
                )
            return plan

        counts = {
            cardinality: len(self.store.direct_keys_for_cardinality(cardinality))
            for cardinality in range(2, MAX_CHANNELS + 1)
        }
        if not any(counts.values()):
            raise RuntimeError("No direct-greedy extension candidates are recorded.")
        excluded: dict[int, set[tuple[int, ...]]] = {
            cardinality: set() for cardinality in range(2, MAX_CHANNELS + 1)
        }
        for key in self.store.all_evaluation_keys():
            channels = key_to_channels(key)
            if channels is not None and len(channels) in excluded:
                excluded[len(channels)].add(channels)

        requested_counts: dict[int, int] = {}
        available_counts: dict[int, int] = {}
        selected_counts: dict[int, int] = {}
        candidates: list[dict[str, Any]] = []
        for cardinality in range(2, MAX_CHANNELS + 1):
            rng = random.Random(self.args.random_seed + 1009 * cardinality)
            selected: set[tuple[int, ...]] = set()
            requested = counts[cardinality]
            available = math.comb(NUM_CHANNELS, cardinality) - len(
                excluded[cardinality]
            )
            target = min(requested, available)
            requested_counts[cardinality] = requested
            available_counts[cardinality] = available
            selected_counts[cardinality] = target
            if target < requested:
                self.console.say(
                    f"[RANDOM EXHAUSTED] K={cardinality}: direct search already "
                    f"occupies {requested} candidates, but only {available} disjoint "
                    "random candidates remain. The random control uses all remaining "
                    "candidates; K=2 is exact rather than randomly compared when "
                    "the fallback pair sweep exhausted it."
                )
            while len(selected) < target:
                trial = tuple(sorted(rng.sample(range(NUM_CHANNELS), cardinality)))
                if trial not in selected and trial not in excluded[cardinality]:
                    selected.add(trial)
            for rank, channels in enumerate(sorted(selected), start=1):
                candidates.append(
                    {
                        "label": (
                            f"r_k{cardinality:02d}_{rank:04d}_{channel_tag(channels)}"
                        ),
                        "channels": list(channels),
                        "rank_within_cardinality": rank,
                    }
                )
        plan = {
            "protocol": PROTOCOL,
            "random_seed": self.args.random_seed,
            "sampling": (
                "Uniform without replacement within each cardinality. Excludes "
                "already evaluated combinations at that cardinality."
            ),
            "direct_greedy_counts_by_cardinality": {
                str(cardinality): counts[cardinality]
                for cardinality in range(2, MAX_CHANNELS + 1)
            },
            "disjoint_random_capacity_by_cardinality": {
                str(cardinality): available_counts[cardinality]
                for cardinality in range(2, MAX_CHANNELS + 1)
            },
            "selected_random_counts_by_cardinality": {
                str(cardinality): selected_counts[cardinality]
                for cardinality in range(2, MAX_CHANNELS + 1)
            },
            "candidates": candidates,
        }
        path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        self.console.say(
            "[RANDOM PLAN] "
            + ", ".join(
                f"K{k}={selected_counts[k]}/{counts[k]}"
                for k in range(2, MAX_CHANNELS + 1)
            )
            + f"; total={len(candidates)}; seed={self.args.random_seed}"
        )
        return plan

    def random_endpoints(self) -> dict[int, str]:
        path = self.args.output_dir / "random_budget_matched_plan.json"
        if not path.is_file():
            return {}
        plan = json.loads(path.read_text(encoding="utf-8"))
        keys_by_k: dict[int, list[str]] = {
            cardinality: [] for cardinality in range(2, MAX_CHANNELS + 1)
        }
        for entry in plan["candidates"]:
            channels = tuple(int(item) for item in entry["channels"])
            keys_by_k[len(channels)].append(candidate_key(channels))
        endpoints: dict[int, str] = {}
        for cardinality, keys in keys_by_k.items():
            key, _ = self.best_key(keys)
            if key is not None:
                endpoints[cardinality] = key
        return endpoints

    def provisional_gstar(self) -> tuple[str, Any, int]:
        endpoints = self.greedy_endpoints()
        key, row = self.best_key(endpoints.values())
        if key is None or row is None:
            raise RuntimeError("No PASS greedy endpoint exists for swap audit.")
        channels = key_to_channels(key)
        assert channels is not None
        return key, row, len(channels)

    def swap(self) -> None:
        if not self.store.get_state("random_completed"):
            raise RuntimeError("Run --stage random first, or use --stage all.")
        self.stage_banner("One-channel swap audit around provisional best greedy endpoint")
        gstar_key, gstar_row, cardinality = self.provisional_gstar()
        path = self.args.output_dir / "one_swap_plan.json"
        if path.is_file():
            plan = json.loads(path.read_text(encoding="utf-8"))
            if plan.get("centre_candidate_key") != gstar_key:
                raise RuntimeError(
                    "Saved swap plan has a different greedy centre; do not overwrite "
                    "a resumable local audit."
                )
        else:
            centre = key_to_channels(gstar_key)
            assert centre is not None
            neighbours = [
                tuple(sorted((*[item for item in centre if item != removed], added)))
                for removed in centre
                for added in range(NUM_CHANNELS)
                if added not in centre
            ]
            expected = len(centre) * (NUM_CHANNELS - len(centre))
            if len(neighbours) != expected or len(set(neighbours)) != expected:
                raise AssertionError("Unexpected one-swap neighbourhood")
            plan = {
                "protocol": PROTOCOL,
                "centre_candidate_key": gstar_key,
                "centre_channels": list(centre),
                "centre_cardinality": cardinality,
                "centre_single_run_primary_ate_mean_cm": primary_metric(gstar_row) * 100,
                "candidate_count": expected,
                "candidates": [
                    {
                        "label": f"swap_k{cardinality:02d}_{rank:04d}_{channel_tag(channels)}",
                        "channels": list(channels),
                    }
                    for rank, channels in enumerate(neighbours, start=1)
                ],
            }
            path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        self.console.say(
            f"[SWAP PLAN] centre={format_channels(key_to_channels(gstar_key))}; "
            f"K={cardinality}; neighbours={plan['candidate_count']}."
        )
        self.set_stage_progress(int(plan["candidate_count"]))
        for entry in plan["candidates"]:
            channels = tuple(int(item) for item in entry["channels"])
            candidate = SearchCandidate(
                entry["label"],
                channels,
                "one-channel swap audit around provisional greedy endpoint",
            )
            self.ensure(
                "swap",
                candidate,
                metadata={
                    "phase": "one_swap_audit",
                    "centre_candidate_key": gstar_key,
                    "cardinality": len(channels),
                },
            )
        if self.args.execute:
            all_keys = [gstar_key] + [
                candidate_key(tuple(int(item) for item in entry["channels"]))
                for entry in plan["candidates"]
            ]
            lstar_key, lstar_row = self.best_key(all_keys)
            self.store.set_state(
                "swap_completed",
                {
                    "centre_gstar_single_key": gstar_key,
                    "centre_cardinality": cardinality,
                    "lstar_single_key": lstar_key,
                    "lstar_single_primary_ate_mean_cm": (
                        primary_metric(lstar_row) * 100 if lstar_row is not None else None
                    ),
                },
            )

    def final_candidate_set(self) -> list[dict[str, Any]]:
        """Unique configurations that must have three observations."""

        references: list[tuple[str, str]] = [
            ("gray_current", "gray"),
            (ALL_CHANNEL_ANCHOR_LABEL, candidate_key(tuple(range(NUM_CHANNELS)))),
        ]
        references.extend(
            (tag, candidate_key(channels))
            for _, channels, _, tag in BQS_REFERENCE_CANDIDATES
        )
        references.extend(
            (f"G{cardinality}", key)
            for cardinality, key in sorted(self.greedy_endpoints().items())
        )
        random_endpoints = self.random_endpoints()
        if 4 in random_endpoints:
            references.append(("R4", random_endpoints[4]))
        rstar_key, _ = self.best_key(random_endpoints.values())
        if rstar_key is not None:
            references.append(("Rstar_single", rstar_key))
        swap_state = self.store.get_state("swap_completed", {})
        if swap_state.get("lstar_single_key") is not None:
            references.append(("Lstar_single", str(swap_state["lstar_single_key"])))

        grouped: dict[str, list[str]] = {}
        for tag, key in references:
            grouped.setdefault(key, []).append(tag)
        return [
            {"candidate_key": key, "tags": tags, "tag": "+".join(tags)}
            for key, tags in grouped.items()
        ]

    def write_final_candidate_plan(self, candidates: list[dict[str, Any]]) -> None:
        payload = {
            "protocol": PROTOCOL,
            "selection_rule": (
                "Repeat each G1--G6 endpoint, R4, Rstar, Lstar and every "
                "anchor to three total observations. Choose Gstar after repeats."
            ),
            "candidates": candidates,
        }
        (self.args.output_dir / "final_repeat_candidate_plan.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def repeats(self) -> None:
        if not self.store.get_state("swap_completed"):
            raise RuntimeError("Run --stage swap first, or use --stage all.")
        self.stage_banner("Final selected configurations: repeat to 3 total observations")
        candidates = self.final_candidate_set()
        self.write_final_candidate_plan(candidates)
        self.set_stage_progress(len(candidates) * 2)
        for entry in candidates:
            key = entry["candidate_key"]
            base_row = self.store.evaluation_for_key(key, 0)
            if base_row is None:
                self.console.say(
                    f"[FINAL SKIP] {entry['tag']}: initial evaluation is unavailable."
                )
                continue
            candidate = SearchCandidate(
                str(base_row["label"]),
                key_to_channels(key),
                "final repeat validation for " + ", ".join(entry["tags"]),
            )
            for replicate in (1, 2):
                self.ensure(
                    "final_repeats",
                    candidate,
                    replicate=replicate,
                    metadata={
                        "phase": "final_repeat",
                        "tags": entry["tags"],
                        "target_total_observations": 3,
                    },
                )
        if self.args.execute:
            self.store.set_state("repeats_completed", True)
        self.write_final_summary(candidates)

    def write_final_summary(self, candidates: list[dict[str, Any]] | None = None) -> None:
        if candidates is None:
            try:
                candidates = self.final_candidate_set()
            except RuntimeError:
                return
        report_rows: list[dict[str, Any]] = []
        for entry in candidates:
            rows = self.store.evaluations_for_key(entry["candidate_key"])
            values = [
                primary_metric(row) * 100
                for row in rows
                if math.isfinite(primary_metric(row))
            ]
            passes = [row for row in rows if row["status"] == "PASS"]
            report_rows.append(
                {
                    "tags": "+".join(entry["tags"]),
                    "candidate_key": entry["candidate_key"],
                    "channels": format_channels(key_to_channels(entry["candidate_key"])),
                    "observations_saved": len(rows),
                    "pass_count": len(passes),
                    "historical_ate_mean_cm": float(np.mean(values)) if values else None,
                    "historical_ate_std_cm": (
                        float(np.std(values, ddof=1))
                        if len(values) > 1
                        else (0.0 if len(values) == 1 else None)
                    ),
                    "historical_ate_median_cm": float(np.median(values)) if values else None,
                    "historical_ate_min_cm": float(np.min(values)) if values else None,
                    "historical_ate_max_cm": float(np.max(values)) if values else None,
                    "allframe_se3_ate_rmse_mean_cm": mean_column(
                        passes, "se3_ate_rmse_m", 100
                    ),
                    "historical_rpe_rmse_mean_cm": mean_column(
                        passes, "historical_evo_rpe_rmse_m", 100
                    ),
                    "runtime_mean_seconds": mean_column(passes, "elapsed_seconds", 1),
                    "labels": ",".join(sorted({str(row["label"]) for row in rows})),
                }
            )
        report_rows.sort(
            key=lambda row: (
                row["historical_ate_mean_cm"] is None,
                row["historical_ate_mean_cm"]
                if row["historical_ate_mean_cm"] is not None
                else math.inf,
            )
        )
        if report_rows:
            with (self.args.output_dir / "final_repeats_summary.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
                writer.writeheader()
                writer.writerows(report_rows)

        endpoints = self.greedy_endpoints()
        rows_by_key = {row["candidate_key"]: row for row in report_rows}
        repeat_valid_g = [
            (rows_by_key[key]["historical_ate_mean_cm"], cardinality, rows_by_key[key])
            for cardinality, key in endpoints.items()
            if key in rows_by_key
            and rows_by_key[key]["pass_count"] == 3
            and rows_by_key[key]["historical_ate_mean_cm"] is not None
        ]
        chosen = min(repeat_valid_g, default=None)
        g4_row = rows_by_key.get(endpoints[4]) if 4 in endpoints else None
        recommendation = {
            "protocol": PROTOCOL,
            "primary_metric": "historical keyframe evo_ape ATE mean",
            "adaptive_gstar_after_three_repeats": (
                {
                    "cardinality": chosen[1],
                    "candidate_key": chosen[2]["candidate_key"],
                    "channels": chosen[2]["channels"],
                    "historical_ate_mean_cm": chosen[0],
                    "historical_ate_std_cm": chosen[2]["historical_ate_std_cm"],
                }
                if chosen is not None
                else None
            ),
            "fixed_g4_after_three_repeats": (
                {
                    "candidate_key": g4_row["candidate_key"],
                    "channels": g4_row["channels"],
                    "historical_ate_mean_cm": g4_row["historical_ate_mean_cm"],
                    "historical_ate_std_cm": g4_row["historical_ate_std_cm"],
                }
                if g4_row is not None and g4_row["pass_count"] == 3
                else None
            ),
            "candidates": report_rows,
        }
        (self.args.output_dir / "recommendation.json").write_text(
            json.dumps(recommendation, indent=2), encoding="utf-8"
        )
        lines = [
            "# UNet direct full-sequence greedy search",
            "",
            "Primary metric: keyframe evo_ape ATE mean after alignment and scale correction.",
            "All new candidates were ranked only by full-sequence tracking, not MVS/BQS.",
            "",
            "| Tags | Channels | PASS | ATE mean cm | ATE std cm |",
            "|---|---|---:|---:|---:|",
        ]
        for row in report_rows:
            mean = (
                f"{row['historical_ate_mean_cm']:.4f}"
                if row["historical_ate_mean_cm"] is not None
                else ""
            )
            std = (
                f"{row['historical_ate_std_cm']:.4f}"
                if row["historical_ate_std_cm"] is not None
                else ""
            )
            lines.append(
                f"| {row['tags']} | {row['channels']} | "
                f"{row['pass_count']}/{row['observations_saved']} | {mean} | {std} |"
            )
        (self.args.output_dir / "summary.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def export(self) -> None:
        full.export_results(self.store.results, self.args.output_dir)
        self.store.export_registry(self.args.output_dir)
        self.write_direct_path_csv()
        self.write_final_summary()


def mean_column(rows: Sequence[Any], column: str, scale: float) -> float | None:
    values = [
        float(row[column]) * scale
        for row in rows
        if row[column] is not None and math.isfinite(float(row[column]))
    ]
    return float(np.mean(values)) if values else None


def print_dry_run(console) -> None:
    console.say("[DRY RUN] No COMO process is launched without --execute.")
    console.say(
        f"[DRY RUN] Anchors: gray, UNet-{all_channels_display()}, "
        + ", ".join(label for label, _, _, _ in BQS_REFERENCE_CANDIDATES)
        + "."
    )
    console.say(
        f"[DRY RUN] Greedy: all {NUM_CHANNELS} singletons, then every selected path extends "
        "through K=6 even after a temporary ATE degradation."
    )
    console.say(
        "[DRY RUN] Auto pre-repeat upper bounds: "
        + "; ".join(
            f"{starts} starts={theoretical_budget(starts, 1)['pre_repeat_total_max']}"
            for starts in (4, 3, 2)
        )
        + "."
    )
    console.say(
        f"[DRY RUN] If fewer than the planned singleton seeds PASS: exhaustively "
        f"score {math.comb(NUM_CHANNELS, 2)} pairs; retain singleton paths and "
        "fill missing starts with non-duplicate pair paths. K=2 random control is "
        "skipped because that space has already been exhausted."
    )
    console.say(
        "[DRY RUN] Final output keeps G1--G6 and explicit G4; it chooses adaptive "
        "Gstar from three-repeat ATE means."
    )


def main() -> None:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.como_dir = args.como_dir.resolve()
    args.python = args.python.resolve()
    args.evo_ape = args.evo_ape.resolve()
    args.evo_rpe = args.evo_rpe.resolve()
    timestamps = validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_protocol(args, timestamps)

    console = full.Console(args.output_dir / "console.log")
    store = SearchStore(args.output_dir / "evaluations.sqlite3")
    controller = Controller(args, store, console, timestamps)
    try:
        console.say("=" * 78)
        console.say(
            f"UNET {ENCODER_LABEL.upper()} DIRECT FULL-SEQUENCE ATE GREEDY SEARCH"
        )
        console.say("=" * 78)
        console.say(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
        console.say(f"Requested stage: {args.stage}")
        console.say(
            f"Dataset: {args.dataset_dir}; matched frames={len(timestamps)}; "
            f"time={timestamps[0]:.6f}--{timestamps[-1]:.6f}"
        )
        console.say(
            "Primary rank: historical keyframe evo_ape ATE mean with alignment and "
            "scale correction, identical to run_random_channel_search.sh."
        )
        console.say(
            f"Tracking uses UNet {ENCODER_LABEL}; mapping stays gray with sensor depth. "
            "No MVS, BQS or convergence proxy ranks a new candidate."
        )
        console.say(
            f"Timeout={args.timeout_seconds:.1f}s; batch limit="
            f"{'disabled' if args.batch_hours == 0 else f'{args.batch_hours:.2f}h'}."
        )
        if not args.execute:
            print_dry_run(console)
            controller.export()
            return

        controller.guard = UNetConfigGuard(
            args.como_dir / "config/como.yml", SHARED_CONFIG_LOCK
        )
        stages = (
            ("anchors", controller.anchors),
            ("greedy", controller.greedy),
            ("random", controller.random_control),
            ("swap", controller.swap),
            ("repeats", controller.repeats),
            ("export", controller.export),
        )
        requested = [name for name, _ in stages] if args.stage == "all" else [args.stage]
        for name, action in stages:
            if name in requested:
                action()
        console.say("[DONE] Requested stages finished; authoritative SQLite is saved.")
    except BatchLimitReached as error:
        console.say(f"[BATCH COMPLETE] {error}")
        console.say("[NEXT] Re-run the same command; saved rows will be reused.")
    finally:
        if controller.guard is not None:
            controller.guard.close()
        controller.export()
        store.close()
        console.close()


if __name__ == "__main__":
    main()
