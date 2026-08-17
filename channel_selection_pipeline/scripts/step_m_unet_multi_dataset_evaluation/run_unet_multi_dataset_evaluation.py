#!/usr/bin/env python3
"""Resumable full-sequence U-Net Enc0/Enc1 evaluation on TUM datasets.

This is deliberately separate from the ResNet Step-F evaluator.  It reuses its
trajectory completeness gates, failure detection, historical ATE calculation
and SQLite schema, but accepts variable-sized U-Net selections at Enc0 or
Enc1.  The mapping side remains gray with sensor depth; only tracking receives
the selected U-Net encoder activation channels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
FULL_EVALUATOR_PATH = (
    PROJECT_ROOT
    / "channel_selection_pipeline/scripts/step_e_full_sequence_evaluation/"
    "run_full_sequence_evaluation.py"
)
DEFAULT_DATASET = Path("/home/melody/data/tum/rgbd_dataset_freiburg1_desk_lightswitch")
DEFAULT_OUTPUT = PROJECT_ROOT / "channel_selection_results/step_m_unet_multi_dataset_evaluation"
DEFAULT_PLAN = SCRIPT_DIR / "unet_candidate_plan.json"
DEFAULT_PYTHON = Path("/home/melody/anaconda3/envs/como/bin/python")
DEFAULT_EVO_APE = Path("/home/melody/anaconda3/envs/como/bin/evo_ape")
DEFAULT_EVO_RPE = Path("/home/melody/anaconda3/envs/como/bin/evo_rpe")


def load_full_evaluator():
    spec = importlib.util.spec_from_file_location("unet_multi_dataset_full_evaluator", FULL_EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import full-sequence evaluator: {FULL_EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


full = load_full_evaluator()
core = full.core


@dataclass(frozen=True)
class UNetEvaluationSpec:
    label: str
    channels: tuple[int, ...]
    enc_level: int
    role: str

    @property
    def total_channels(self) -> int:
        return 16 * (2**self.enc_level)

    @property
    def candidate_key(self) -> str:
        if self.channels == tuple(range(self.total_channels)):
            return f"enc{self.enc_level}:all"
        return f"enc{self.enc_level}:" + ",".join(str(channel) for channel in self.channels)

    @property
    def display(self) -> str:
        if self.channels == tuple(range(self.total_channels)):
            return f"Enc{self.enc_level} all{self.total_channels}"
        return f"Enc{self.enc_level} [" + ",".join(f"d{channel}" for channel in self.channels) + "]"


class UNetConfigGuard(core.ConfigGuard):
    """Apply U-Net tracking settings while preserving gray/sensor-depth mapping."""

    def __init__(self, config_path: Path, lock_path: Path, specs: Sequence[UNetEvaluationSpec]):
        super().__init__(config_path, lock_path)
        self.spec_by_label = {spec.label: spec for spec in specs}

    def apply(self, candidate: core.Candidate) -> dict:
        spec = self.spec_by_label.get(candidate.label)
        if spec is None:
            raise KeyError(f"No U-Net specification registered for {candidate.label}")
        config = yaml.safe_load(self.original)
        config["mapping"]["color"] = "gray"
        tracking = config["tracking"]
        tracking["debug_tracking_diagnostics"] = True
        tracking["debug_tracking_print_every_frame"] = True
        tracking["debug_tracking_save_suspicious"] = False
        tracking["color"] = "unet"
        tracking["cnn_mode"] = "cnn_only"
        tracking["unet_enc_level"] = spec.enc_level
        tracking["unet_channel_select"] = (
            "all"
            if spec.channels == tuple(range(spec.total_channels))
            else ",".join(f"d{channel}" for channel in spec.channels)
        )
        encoded = yaml.safe_dump(
            config, default_flow_style=False, allow_unicode=True, sort_keys=False
        ).encode("utf-8")
        core.atomic_write_bytes(self.config_path, encoded)
        return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate selected U-Net Enc0/Enc1 feature configurations on a full TUM sequence.",
    )
    parser.add_argument("--execute", action="store_true", help="Launch COMO runs; otherwise validate only.")
    parser.add_argument("--rerun-existing", action="store_true", help="Replace already stored candidate/replicate rows.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--como-dir", type=Path, default=PROJECT_ROOT / "como")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--evo-ape", type=Path, default=DEFAULT_EVO_APE)
    parser.add_argument("--evo-rpe", type=Path, default=DEFAULT_EVO_RPE)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=500.0)
    parser.add_argument("--minimum-coverage", type=float, default=0.90)
    parser.add_argument("--completion-tolerance-seconds", type=float, default=0.10)
    parser.add_argument("--terminate-grace-seconds", type=float, default=3.0)
    parser.add_argument(
        "--exclude-candidate-key",
        action="append",
        default=[],
        help="Candidate key to omit for this dataset; may be supplied more than once.",
    )
    parser.add_argument(
        "--exclusion-reason",
        default="",
        help="Human-readable reason recorded for dataset-specific safety exclusions.",
    )
    parser.add_argument("--validate-only", action="store_true", help="Validate plans and inputs without creating outputs.")
    return parser.parse_args()


def load_candidate_plan(path: Path) -> tuple[UNetEvaluationSpec, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "unet_enc0_enc1_multi_dataset_candidate_plan_v1":
        raise ValueError("Unexpected U-Net candidate-plan protocol")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("Candidate plan has no candidates")
    expected_count = payload.get("selection", {}).get("selected_count")
    if expected_count is not None and int(expected_count) != len(raw_candidates):
        raise ValueError("Candidate-plan selected_count does not match candidates")

    specs: list[UNetEvaluationSpec] = []
    labels: set[str] = set()
    keys: set[str] = set()
    for position, item in enumerate(raw_candidates, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Candidate {position} is not an object")
        label = str(item.get("label", "")).strip()
        level = item.get("enc_level")
        raw_channels = item.get("channels")
        if not label or label in labels:
            raise ValueError(f"Candidate {position} has a missing/duplicate label: {label!r}")
        if level not in (0, 1):
            raise ValueError(f"Candidate {label} must specify Enc0 or Enc1")
        if not isinstance(raw_channels, list) or not raw_channels:
            raise ValueError(f"Candidate {label} needs a non-empty channels list")
        channels = tuple(sorted(int(value) for value in raw_channels))
        total = 16 * (2**int(level))
        if len(set(channels)) != len(channels) or channels[0] < 0 or channels[-1] >= total:
            raise ValueError(f"Candidate {label} has invalid Enc{level} channels: {channels}")
        spec = UNetEvaluationSpec(
            label=label,
            channels=channels,
            enc_level=int(level),
            role=str(item.get("role", "U-Net multi-dataset evaluation")),
        )
        declared_key = item.get("candidate_key")
        if declared_key != spec.candidate_key:
            raise ValueError(
                f"Candidate {label} declares {declared_key!r}, expected {spec.candidate_key!r}"
            )
        if spec.candidate_key in keys:
            raise ValueError(f"Duplicate U-Net candidate key: {spec.candidate_key}")
        labels.add(label)
        keys.add(spec.candidate_key)
        specs.append(spec)
    return tuple(specs)


def validate_inputs(args: argparse.Namespace) -> tuple[list[float], tuple[UNetEvaluationSpec, ...], frozenset[str]]:
    required = [
        args.dataset_dir / "matched_rgb.txt",
        args.dataset_dir / "matched_depth.txt",
        args.dataset_dir / "groundtruth.txt",
        args.como_dir / "config/como.yml",
        args.como_dir / "como/como_dataset.py",
        args.python,
        args.evo_ape,
        args.evo_rpe,
        args.candidate_plan,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths: " + ", ".join(map(str, missing)))
    if args.replicates != 1:
        raise ValueError("This frozen multi-dataset protocol requires exactly one replicate")
    if args.timeout_seconds != 500.0:
        raise ValueError("This frozen multi-dataset protocol requires a 500-second timeout")
    if not 0.0 < args.minimum_coverage <= 1.0:
        raise ValueError("minimum coverage must be in (0,1]")
    timestamps = full.read_timestamp_index(args.dataset_dir / "matched_rgb.txt")
    specs = load_candidate_plan(args.candidate_plan)
    candidate_keys = {spec.candidate_key for spec in specs}
    excluded_keys = frozenset(str(key).strip() for key in args.exclude_candidate_key if str(key).strip())
    unknown_keys = sorted(excluded_keys - candidate_keys)
    if unknown_keys:
        raise ValueError(f"Unknown excluded candidate keys: {unknown_keys}")
    if excluded_keys and not args.exclusion_reason.strip():
        raise ValueError("A dataset-specific candidate exclusion requires --exclusion-reason")
    if len(excluded_keys) == len(specs):
        raise ValueError("Every candidate was excluded; at least one configuration must remain evaluable")
    return timestamps, specs, excluded_keys


def export_summary(
    store: full.ResultStore,
    args: argparse.Namespace,
    specs: Sequence[UNetEvaluationSpec],
    excluded_keys: frozenset[str],
) -> None:
    rows = store.rows()
    status_counts = Counter(str(row["status"]) for row in rows)
    lines = [
        "# U-Net multi-dataset full-sequence evaluation",
        "",
        f"- Dataset: `{args.dataset_dir}`",
        f"- Matched frames: {full.read_timestamp_index(args.dataset_dir / 'matched_rgb.txt').__len__()}",
        "- Tracking: selected U-Net Enc0/Enc1 channels; mapping: gray with sensor depth.",
        "- Primary metric: keyframe `evo_ape --align --correct_scale` translation ATE mean.",
        "- Diagnostics: historical keyframe RPE plus all-frame metric-scale SE(3) ATE/RPE and coverage.",
        f"- Timeout per configuration: {args.timeout_seconds:.0f} seconds.",
        f"- Dataset-specific safety exclusions: {len(excluded_keys)}.",
        f"- Persisted rows: {len(rows)}/{len(specs) - len(excluded_keys)} active configurations; status counts: {dict(sorted(status_counts.items()))}",
        "",
        "| Label | Encoder level | Channels | Status | Historical ATE mean (cm) | Coverage | Runtime (s) |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    by_label = {str(row["label"]): row for row in rows}
    for spec in specs:
        row = by_label.get(spec.label)
        if row is None:
            status = "SKIPPED_BY_SAFETY" if spec.candidate_key in excluded_keys else "NOT_RUN"
            lines.append(f"| {spec.label} | {spec.enc_level} | {spec.display} | {status} |  |  |  |")
            continue
        ate = row["historical_evo_ape_mean_m"]
        ate_text = "" if ate is None else f"{float(ate) * 100:.4f}"
        coverage = row["coverage_ratio"]
        coverage_text = "" if coverage is None else f"{float(coverage):.4f}"
        lines.append(
            f"| {spec.label} | {spec.enc_level} | {spec.display} | {row['status']} | "
            f"{ate_text} | {coverage_text} | "
            f"{float(row['elapsed_seconds']):.1f} |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for attribute in ("dataset_dir", "output_dir", "candidate_plan", "como_dir", "python", "evo_ape", "evo_rpe"):
        setattr(args, attribute, getattr(args, attribute).resolve())
    timestamps, specs, excluded_keys = validate_inputs(args)
    active_specs = tuple(spec for spec in specs if spec.candidate_key not in excluded_keys)
    if args.validate_only:
        print(
            "[VALID] "
            f"dataset={args.dataset_dir} frames={len(timestamps)} active_candidates={len(active_specs)} "
            f"replicates={args.replicates} timeout={args.timeout_seconds:.0f}s"
        )
        for spec in specs:
            if spec.candidate_key in excluded_keys:
                print(f"[VALID] {spec.label}: SKIPPED_BY_SAFETY; reason={args.exclusion_reason}")
            else:
                print(f"[VALID] {spec.label}: {spec.display}; role={spec.role}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "protocol": "unet_enc0_enc1_multi_dataset_full_sequence_v1",
        "dataset": str(args.dataset_dir),
        "matched_frames": len(timestamps),
        "replicates": args.replicates,
        "timeout_seconds": args.timeout_seconds,
        "candidate_plan": str(args.candidate_plan),
        "candidate_plan_sha256": hashlib.sha256(args.candidate_plan.read_bytes()).hexdigest(),
        "primary_metric": "historical keyframe evo_ape mean: TUM GT + data_tum.txt --align --correct_scale",
        "diagnostic_metrics": [
            "historical keyframe evo_rpe RMSE with --align --correct_scale",
            "all-frame metric-scale SE(3) translation ATE/RPE",
        ],
        "configurations": [asdict(spec) for spec in specs],
        "active_candidate_keys": [spec.candidate_key for spec in active_specs],
        "excluded_candidate_keys": sorted(excluded_keys),
        "exclusion_reason": args.exclusion_reason.strip(),
    }
    (args.output_dir / "evaluation_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )
    console = full.Console(args.output_dir / "console.log")
    store = full.ResultStore(args.output_dir / "evaluations.sqlite3")
    guard: UNetConfigGuard | None = None
    try:
        console.say("=" * 78)
        console.say("FULL-SEQUENCE U-NET ENC0/ENC1 MULTI-DATASET VALIDATION")
        console.say("=" * 78)
        console.say(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
        console.say(f"Dataset: {args.dataset_dir}; matched frames={len(timestamps)}")
        console.say("Tracking: U-Net post-LeakyReLU features; mapping: gray; sensor depth unchanged")
        console.say("Primary: historical keyframe evo_ape ATE mean using --align --correct_scale")
        console.say("Diagnostics: historical RPE and all-frame metric-scale SE(3) ATE/RPE")
        console.say(
            f"Configurations={len(active_specs)} active / {len(specs)} listed; replicates=1; timeout={args.timeout_seconds:.0f}s; "
            f"coverage>={args.minimum_coverage:.1%}"
        )
        for spec in specs:
            if spec.candidate_key in excluded_keys:
                console.say(f"[SKIPPED_BY_SAFETY] {spec.label}: {args.exclusion_reason}")
            else:
                console.say(f"[PLAN] {spec.label}: {spec.display}; {spec.role}")
        if not args.execute:
            full.export_results(store, args.output_dir)
            export_summary(store, args, specs, excluded_keys)
            console.say("DRY RUN COMPLETE: add --execute to launch/resume COMO")
            return

        guard = UNetConfigGuard(
            args.como_dir / "config/como.yml", full.SHARED_CONFIG_LOCK, active_specs
        )
        for index, spec in enumerate(active_specs, start=1):
            full.evaluate_one(
                args,
                console,
                store,
                guard,
                spec,
                0,
                index,
                len(active_specs),
                timestamps,
            )
            full.export_results(store, args.output_dir)
            export_summary(store, args, specs, excluded_keys)
        console.say("DONE: all requested U-Net configurations were processed")
    finally:
        if guard is not None:
            guard.close()
        full.export_results(store, args.output_dir)
        export_summary(store, args, specs, excluded_keys)
        store.close()
        console.close()


if __name__ == "__main__":
    main()
