#!/usr/bin/env python3
"""Resumable C2F validation of the strongest ResNet and U-Net subsets.

The previous channel searches were single-layer experiments.  This runner tests
whether the best shallow/fine and deep/coarse subsets complement one another
in the existing three-level C2F tracking architecture:

* Variant A: coarse at L0/L1, fine at L2.
* Variant B: coarse at L0, fine at L1/L2.

Each architecture receives a frozen 6 x 6 subset grid, evaluated once per
cell.  SQLite is the source of truth: a later invocation skips every stored
label/replicate pair, including saved failures, unless --rerun-existing is
explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
FULL_EVALUATOR_PATH = (
    PROJECT_ROOT
    / "channel_selection_pipeline/scripts/step_e_full_sequence_evaluation/"
    "run_full_sequence_evaluation.py"
)
DEFAULT_DATASET = Path("/home/melody/data/tum/rgbd_dataset_freiburg1_desk_lightswitch")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "channel_selection_results/step_p_c2f_best_channels_evaluation"
DEFAULT_PYTHON = Path("/home/melody/anaconda3/envs/como/bin/python")
DEFAULT_EVO_APE = Path("/home/melody/anaconda3/envs/como/bin/evo_ape")
DEFAULT_EVO_RPE = Path("/home/melody/anaconda3/envs/como/bin/evo_rpe")


def load_full_evaluator():
    spec = importlib.util.spec_from_file_location("c2f_full_sequence_evaluator", FULL_EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import full-sequence evaluator: {FULL_EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


full = load_full_evaluator()
core = full.core


@dataclass(frozen=True)
class BranchSelection:
    rank: int
    channels: tuple[int, ...]
    source_ate_mean_cm: float
    tag: str

    @property
    def display(self) -> str:
        return "[" + ",".join(f"d{channel}" for channel in self.channels) + "]"


@dataclass(frozen=True)
class C2FEvaluationSpec:
    architecture: str
    variant: str
    fine: BranchSelection
    coarse: BranchSelection
    fine_layer: str
    coarse_layer: str

    @property
    def label(self) -> str:
        return (
            f"{self.architecture}_c2f_{self.variant.lower()}_"
            f"fine{self.fine.rank:02d}_coarse{self.coarse.rank:02d}"
        )

    @property
    def candidate_key(self) -> str:
        fine = ",".join(str(channel) for channel in self.fine.channels)
        coarse = ",".join(str(channel) for channel in self.coarse.channels)
        return f"{self.architecture}|{self.variant}|fine:{fine}|coarse:{coarse}"

    @property
    def channels(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "variant": self.variant,
            "fine_layer": self.fine_layer,
            "fine": list(self.fine.channels),
            "coarse_layer": self.coarse_layer,
            "coarse": list(self.coarse.channels),
        }

    @property
    def display(self) -> str:
        return (
            f"C2F-{self.variant}; fine {self.fine_layer}{self.fine.display}; "
            f"coarse {self.coarse_layer}{self.coarse.display}"
        )

    @property
    def role(self) -> str:
        return (
            f"fine rank {self.fine.rank} ({self.fine.tag}, source {self.fine.source_ate_mean_cm:.4f}cm) + "
            f"coarse rank {self.coarse.rank} ({self.coarse.tag}, source {self.coarse.source_ate_mean_cm:.4f}cm)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate a frozen 6x6 C2F promising-subset grid on the complete fr1/desk_lightswitch sequence.",
    )
    parser.add_argument("--architecture", choices=("resnet", "unet"), required=True)
    parser.add_argument("--execute", action="store_true", help="Launch/resume COMO; otherwise perform validation only.")
    parser.add_argument("--rerun-existing", action="store_true", help="Overwrite already-saved label/replicate rows.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidate-plan", type=Path)
    parser.add_argument("--como-dir", type=Path, default=PROJECT_ROOT / "como")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--evo-ape", type=Path, default=DEFAULT_EVO_APE)
    parser.add_argument("--evo-rpe", type=Path, default=DEFAULT_EVO_RPE)
    parser.add_argument("--timeout-seconds", type=float, default=500.0)
    parser.add_argument("--minimum-coverage", type=float, default=0.90)
    parser.add_argument("--completion-tolerance-seconds", type=float, default=0.10)
    parser.add_argument("--terminate-grace-seconds", type=float, default=3.0)
    parser.add_argument(
        "--only-variant",
        action="append",
        choices=("A", "B"),
        default=[],
        help="Optional valid variant filter.  May be supplied more than once.",
    )
    parser.add_argument(
        "--only-label",
        action="append",
        default=[],
        help="Optional exact C2F label filter for a targeted retry/debug run.",
    )
    return parser.parse_args()


def default_plan_path(architecture: str) -> Path:
    return SCRIPT_DIR / f"{architecture}_c2f_candidate_plan.json"


def channel_limit(architecture: str, branch: str) -> int:
    limits = {
        "resnet": {"fine": 64, "coarse": 128},
        "unet": {"fine": 16, "coarse": 32},
    }
    return limits[architecture][branch]


def load_branch(payload: dict[str, Any], architecture: str, branch: str) -> tuple[str, tuple[BranchSelection, ...]]:
    branch_doc = payload.get(f"{branch}_branch")
    if not isinstance(branch_doc, dict):
        raise ValueError(f"Candidate plan lacks {branch}_branch")
    layer = str(branch_doc.get("layer", "")).strip()
    total = branch_doc.get("total_channels")
    expected_total = channel_limit(architecture, branch)
    if not layer or int(total) != expected_total:
        raise ValueError(
            f"{architecture} {branch} branch must declare its expected total channel count {expected_total}"
        )
    raw = branch_doc.get("candidates")
    if not isinstance(raw, list) or len(raw) != 6:
        raise ValueError(f"{architecture} {branch} branch must contain exactly six candidates")

    selections: list[BranchSelection] = []
    seen_channels: set[tuple[int, ...]] = set()
    for expected_rank, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or int(item.get("rank", -1)) != expected_rank:
            raise ValueError(f"{architecture} {branch} candidate ranks must be the ordered values 1--6")
        raw_channels = item.get("channels")
        if not isinstance(raw_channels, list) or not raw_channels:
            raise ValueError(f"{architecture} {branch} rank {expected_rank} has no channels")
        channels = tuple(sorted(int(value) for value in raw_channels))
        if len(channels) != len(set(channels)) or channels[0] < 0 or channels[-1] >= expected_total:
            raise ValueError(
                f"{architecture} {branch} rank {expected_rank} has invalid channel indices: {channels}"
            )
        if channels in seen_channels:
            raise ValueError(f"Duplicate {architecture} {branch} candidate: {channels}")
        seen_channels.add(channels)
        selections.append(
            BranchSelection(
                rank=expected_rank,
                channels=channels,
                source_ate_mean_cm=float(item["source_ate_mean_cm"]),
                tag=str(item.get("tag", "source candidate")),
            )
        )
    return layer, tuple(selections)


def load_specs(args: argparse.Namespace) -> tuple[dict[str, Any], tuple[C2FEvaluationSpec, ...]]:
    plan_path = args.candidate_plan or default_plan_path(args.architecture)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "c2f_promising_channel_grid_v1":
        raise ValueError("Unexpected C2F candidate-plan protocol")
    if payload.get("architecture") != args.architecture:
        raise ValueError(
            f"Candidate plan architecture {payload.get('architecture')!r} does not match --architecture {args.architecture!r}"
        )
    if payload.get("variants") != ["A", "B"]:
        raise ValueError("The frozen protocol must contain exactly the two valid C2F variants A and B")
    if int(payload.get("replicates", -1)) != 1 or float(payload.get("timeout_seconds", -1)) != 500.0:
        raise ValueError("The frozen C2F plan requires one replicate and a 500-second timeout")
    fine_layer, fine_candidates = load_branch(payload, args.architecture, "fine")
    coarse_layer, coarse_candidates = load_branch(payload, args.architecture, "coarse")
    specs = tuple(
        C2FEvaluationSpec(args.architecture, variant, fine, coarse, fine_layer, coarse_layer)
        for variant in ("A", "B")
        for fine in fine_candidates
        for coarse in coarse_candidates
    )
    if len(specs) != int(payload.get("nominal_evaluations", -1)) or len(specs) != 72:
        raise ValueError("C2F plan must expand to exactly 72 labels (2 variants x 6 fine x 6 coarse)")
    if len({spec.label for spec in specs}) != len(specs):
        raise ValueError("C2F labels are not unique")
    variants = set(args.only_variant)
    labels = set(args.only_label)
    if labels - {spec.label for spec in specs}:
        raise ValueError("Unknown --only-label values: " + ", ".join(sorted(labels - {spec.label for spec in specs})))
    filtered = tuple(
        spec
        for spec in specs
        if (not variants or spec.variant in variants) and (not labels or spec.label in labels)
    )
    if not filtered:
        raise ValueError("No C2F configurations remain after filtering")
    payload["_resolved_plan_path"] = str(plan_path.resolve())
    return payload, filtered


class C2FConfigGuard(core.ConfigGuard):
    """Write only tracking-side C2F settings; preserve gray sensor-depth mapping."""

    def __init__(self, config_path: Path, lock_path: Path, specs: Sequence[C2FEvaluationSpec]):
        super().__init__(config_path, lock_path)
        self.by_label = {spec.label: spec for spec in specs}

    def apply(self, candidate: core.Candidate) -> dict[str, Any]:
        spec = self.by_label.get(candidate.label)
        if spec is None:
            raise KeyError(f"No C2F specification registered for {candidate.label}")
        config = yaml.safe_load(self.original)
        config["mapping"]["color"] = "gray"
        # Freeze the established evaluation condition rather than inheriting a
        # possibly edited interactive config: mapping receives sensor/GT depth.
        config["mapping"]["use_sensor_depth"] = True
        tracking = config["tracking"]
        tracking["debug_tracking_diagnostics"] = True
        tracking["debug_tracking_print_every_frame"] = True
        tracking["debug_tracking_save_suspicious"] = False
        # Avoid a stale legacy value disagreeing with the explicit A/B version.
        tracking.pop("cnn_c2f_fine_levels", None)

        if spec.architecture == "resnet":
            tracking.update(
                color="cnn_c2f",
                cnn_c2f_version=spec.variant,
                cnn_mode="cnn_only",
                cnn_layer_coarse="layer2",
                cnn_channels_coarse=len(spec.coarse.channels),
                cnn_channel_select_coarse=",".join(f"d{channel}" for channel in spec.coarse.channels),
                cnn_layer_full_channels_coarse=128,
                cnn_layer_fine="conv1",
                cnn_channels_fine=len(spec.fine.channels),
                cnn_channel_select_fine=",".join(f"d{channel}" for channel in spec.fine.channels),
                cnn_layer_full_channels_fine=64,
            )
        elif spec.architecture == "unet":
            tracking.update(
                color="unet_c2f",
                cnn_c2f_version=spec.variant,
                cnn_mode="cnn_only",
                unet_enc_level_coarse=1,
                unet_channel_select_coarse=",".join(f"d{channel}" for channel in spec.coarse.channels),
                unet_enc_level_fine=0,
                unet_channel_select_fine=",".join(f"d{channel}" for channel in spec.fine.channels),
            )
        else:
            raise ValueError(f"Unsupported C2F architecture: {spec.architecture}")
        encoded = yaml.safe_dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False).encode("utf-8")
        core.atomic_write_bytes(self.config_path, encoded)
        return config


def validate_inputs(args: argparse.Namespace, plan: dict[str, Any], specs: Sequence[C2FEvaluationSpec]) -> list[float]:
    required = (
        args.dataset_dir / "matched_rgb.txt",
        args.dataset_dir / "matched_depth.txt",
        args.dataset_dir / "groundtruth.txt",
        args.como_dir / "config/como.yml",
        args.como_dir / "como/como_dataset.py",
        args.como_dir / "como/odom/Tracking.py",
        args.como_dir / "como/odom/sequential/ComoSeq.py",
        args.python,
        args.evo_ape,
        args.evo_rpe,
        Path(plan["_resolved_plan_path"]),
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required paths: " + ", ".join(map(str, missing)))
    if args.timeout_seconds != 500.0:
        raise ValueError("This protocol fixes --timeout-seconds at 500 seconds")
    if not 0.0 < args.minimum_coverage <= 1.0:
        raise ValueError("--minimum-coverage must lie in (0, 1]")
    tracking_source = (args.como_dir / "como/odom/Tracking.py").read_text(encoding="utf-8")
    required_mode = "cnn_c2f" if args.architecture == "resnet" else "unet_c2f"
    if required_mode not in tracking_source:
        raise RuntimeError(f"Current Tracking.py does not contain the required {required_mode} implementation")
    if args.architecture == "unet" and "UNetC2FFeatureExtractor" not in tracking_source:
        raise RuntimeError("Current Tracking.py does not import/use UNetC2FFeatureExtractor")
    timestamps = full.read_timestamp_index(args.dataset_dir / "matched_rgb.txt")
    if not timestamps:
        raise ValueError("matched_rgb.txt has no timestamps")
    if not specs:
        raise ValueError("No C2F candidate specifications were supplied")
    return timestamps


def write_summary(output_dir: Path, store: full.ResultStore, args: argparse.Namespace, specs: Sequence[C2FEvaluationSpec]) -> None:
    rows = store.rows()
    statuses = Counter(str(row["status"]) for row in rows)
    row_by_label = {str(row["label"]): row for row in rows}
    lines = [
        f"# {args.architecture.upper()} C2F promising-channel grid",
        "",
        f"- Dataset: `{args.dataset_dir}`",
        "- Mapping: gray with sensor depth; tracking only receives mixed C2F features.",
        "- C2F-A: coarse at L0/L1, fine at L2.  C2F-B: coarse at L0, fine at L1/L2.",
        "- Primary ranking metric: historical keyframe `evo_ape --align --correct_scale` translation ATE mean.",
        "- Diagnostics: historical RPE, all-frame metric-scale SE(3) ATE/RPE, coverage, and tracking diagnostics.",
        f"- Persisted rows: {len(rows)}/{len(specs)}; status counts: {dict(sorted(statuses.items()))}",
        "",
        "| Label | Variant | Fine subset | Coarse subset | Status | ATE mean (cm) | Coverage | Runtime (s) |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for spec in specs:
        row = row_by_label.get(spec.label)
        status = "NOT_RUN" if row is None else str(row["status"])
        ate = "" if row is None or row["historical_evo_ape_mean_m"] is None else f"{float(row['historical_evo_ape_mean_m']) * 100:.4f}"
        coverage = "" if row is None or row["coverage_ratio"] is None else f"{float(row['coverage_ratio']):.4f}"
        runtime = "" if row is None else f"{float(row['elapsed_seconds']):.1f}"
        lines.append(
            f"| {spec.label} | {spec.variant} | {spec.fine.display} | {spec.coarse.display} | {status} | {ate} | {coverage} | {runtime} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for name in ("dataset_dir", "como_dir", "python", "evo_ape", "evo_rpe"):
        setattr(args, name, getattr(args, name).resolve())
    if args.candidate_plan is not None:
        args.candidate_plan = args.candidate_plan.resolve()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / f"{args.architecture}_fr1_desk_lightswitch"
    args.output_dir = args.output_dir.resolve()

    plan, specs = load_specs(args)
    timestamps = validate_inputs(args, plan, specs)
    planned_labels = {spec.label for spec in specs}
    print("=" * 78)
    print(f"{args.architecture.upper()} C2F PROMISING-CHANNEL GRID")
    print("=" * 78)
    print(f"Mode: {'EXECUTE/RESUME' if args.execute else 'VALIDATE ONLY'}")
    print(f"Dataset: {args.dataset_dir} ({len(timestamps)} matched frames)")
    print(f"Grid: {len(specs)} configurations = {len(set(spec.variant for spec in specs))} variant(s) x 6 fine x 6 coarse")
    print("Variants: A=L0/L1 coarse + L2 fine; B=L0 coarse + L1/L2 fine")
    print("Mapping: gray + sensor depth; only tracking is altered")
    print("Primary ranking: keyframe evo_ape ATE mean (--align --correct_scale)")
    print(f"Timeout: {args.timeout_seconds:.0f}s; completeness coverage >= {args.minimum_coverage:.0%}")
    for spec in specs[:4]:
        print(f"[PLAN] {spec.label}: {spec.display}")
    if len(specs) > 8:
        print(f"[PLAN] ... {len(specs) - 8} intermediate configurations ...")
    for spec in specs[-4:]:
        print(f"[PLAN] {spec.label}: {spec.display}")

    if not args.execute:
        print("[VALID] No COMO process, SQLite row, trajectory, or config edit was created.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_plan = {
        "protocol": "c2f_promising_channel_grid_execution_v1",
        "architecture": args.architecture,
        "dataset": str(args.dataset_dir),
        "matched_frames": len(timestamps),
        "candidate_plan": plan["_resolved_plan_path"],
        "candidate_plan_sha256": hashlib.sha256(Path(plan["_resolved_plan_path"]).read_bytes()).hexdigest(),
        "variants": sorted({spec.variant for spec in specs}),
        "replicates": 1,
        "timeout_seconds": args.timeout_seconds,
        "minimum_coverage": args.minimum_coverage,
        "primary_metric": "historical keyframe evo_ape mean: --align --correct_scale",
        "diagnostic_metrics": ["historical evo_rpe", "all-frame metric-scale SE(3) ATE/RPE", "coverage", "tracking numerical diagnostics"],
        "configurations": [asdict(spec) for spec in specs],
    }
    (args.output_dir / "evaluation_plan.json").write_text(json.dumps(frozen_plan, indent=2), encoding="utf-8")
    console = full.Console(args.output_dir / "console.log")
    store = full.ResultStore(args.output_dir / "evaluations.sqlite3")
    guard: C2FConfigGuard | None = None
    try:
        console.say("=" * 78)
        console.say(f"{args.architecture.upper()} C2F PROMISING-CHANNEL GRID")
        console.say("=" * 78)
        console.say(f"Dataset: {args.dataset_dir}; matched frames={len(timestamps)}")
        console.say("C2F-A=L0/L1 coarse + L2 fine; C2F-B=L0 coarse + L1/L2 fine")
        console.say("Mapping remains gray with sensor depth. Tracking is the only changed component.")
        console.say(f"Configurations={len(specs)}; one replicate; timeout={args.timeout_seconds:.0f}s")
        guard = C2FConfigGuard(args.como_dir / "config/como.yml", full.SHARED_CONFIG_LOCK, specs)
        batch_started = time.monotonic()
        new_evaluations = 0
        for index, spec in enumerate(specs, start=1):
            result = full.evaluate_one(args, console, store, guard, spec, 0, index, len(specs), timestamps)
            if result is not None:
                new_evaluations += 1
                completed = sum(1 for row in store.rows() if str(row["label"]) in planned_labels)
                status_rows = store.connection.execute(
                    "SELECT status, COUNT(*) FROM evaluations WHERE replicate=0 GROUP BY status"
                ).fetchall()
                status_text = ", ".join(f"{row[0]}={row[1]}" for row in status_rows)
                elapsed = time.monotonic() - batch_started
                average = elapsed / new_evaluations
                remaining = len(specs) - completed
                console.say(
                    f"[PROGRESS] preserved={completed}/{len(specs)} remaining={remaining} {status_text} "
                    f"batch_avg={average:.1f}s ETA={remaining * average / 3600:.2f}h"
                )
            full.export_results(store, args.output_dir)
            write_summary(args.output_dir, store, args, specs)
        console.say("[DONE] C2F grid complete; SQLite, CSV ranking, artifacts, and summary were exported")
    finally:
        if guard is not None:
            guard.close()
        full.export_results(store, args.output_dir)
        write_summary(args.output_dir, store, args, specs)
        store.close()
        console.close()


if __name__ == "__main__":
    main()
