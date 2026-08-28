#!/usr/bin/env python3
"""Run/resume parent-comparative C2F evaluation on one complete TUM sequence.

The associated launcher invokes this evaluator once for every sequence in the
frozen 3-family × 3-lighting-condition plan.  A C2F row is deliberately kept
beside its direct fine/coarse parents, so the aggregation step can report a
within-sequence C2F gain or loss rather than making a claim from a raw ATE
average across different sequences.
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
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "channel_selection_results/step_q_c2f_multi_dataset_evaluation"
DEFAULT_PYTHON = Path("/home/melody/anaconda3/envs/como/bin/python")
DEFAULT_EVO_APE = Path("/home/melody/anaconda3/envs/como/bin/evo_ape")
DEFAULT_EVO_RPE = Path("/home/melody/anaconda3/envs/como/bin/evo_rpe")


def load_full_evaluator():
    spec = importlib.util.spec_from_file_location("c2f_parent_full_sequence_evaluator", FULL_EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import full-sequence evaluator: {FULL_EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


full = load_full_evaluator()
core = full.core


@dataclass(frozen=True)
class EvaluationSpec:
    label: str
    candidate_key: str
    architecture: str
    mode: str
    role: str
    source_ate_mean_cm: float | None
    direct_layer: str | None = None
    direct_channels: tuple[int, ...] = ()
    variant: str | None = None
    fine_layer: str | None = None
    fine_channels: tuple[int, ...] = ()
    coarse_layer: str | None = None
    coarse_channels: tuple[int, ...] = ()
    parent_labels: tuple[str, ...] = ()

    @property
    def channels(self) -> tuple[int, ...] | None:
        """A compact provenance field persisted by the shared evaluator.

        The actual C2F routing is defined by the explicitly named fine/coarse
        fields above; this flattened tuple is only written to `channels_json`
        by the inherited SQLite schema.
        """
        if self.mode == "gray":
            return None
        if self.mode == "direct":
            return self.direct_channels
        return self.fine_channels + self.coarse_channels

    @staticmethod
    def _format(layer: str, channels: Sequence[int]) -> str:
        return f"{layer}[" + ",".join(f"d{value}" for value in channels) + "]"

    @property
    def display(self) -> str:
        if self.mode == "gray":
            return "gray"
        if self.mode == "direct":
            assert self.direct_layer is not None
            return "direct " + self._format(self.direct_layer, self.direct_channels)
        assert self.variant is not None and self.fine_layer is not None and self.coarse_layer is not None
        return (
            f"C2F-{self.variant}; fine {self._format(self.fine_layer, self.fine_channels)}; "
            f"coarse {self._format(self.coarse_layer, self.coarse_channels)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--architecture", choices=("resnet", "unet"), required=True)
    parser.add_argument("--execute", action="store_true", help="Launch/resume COMO; otherwise validate only.")
    parser.add_argument("--rerun-existing", action="store_true", help="Replace saved label/replicate rows.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
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
    return parser.parse_args()


def default_plan_path(architecture: str) -> Path:
    return SCRIPT_DIR / f"{architecture}_c2f_parent_comparison_plan.json"


def allowed_layers(architecture: str) -> dict[str, int]:
    return {
        "resnet": {"conv1": 64, "layer2": 128},
        "unet": {"enc0": 16, "enc1": 32},
    }[architecture]


def checked_channels(raw: Any, layer: str, architecture: str, context: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{context} needs a non-empty channel list")
    channels = tuple(sorted(int(value) for value in raw))
    limit = allowed_layers(architecture).get(layer)
    if limit is None:
        raise ValueError(f"{context} uses unsupported {architecture} layer {layer!r}")
    if len(channels) != len(set(channels)) or channels[0] < 0 or channels[-1] >= limit:
        raise ValueError(f"{context} has invalid {layer} channels {channels}; expected unique values in 0--{limit - 1}")
    return channels


def load_specs(path: Path, architecture: str) -> tuple[dict[str, Any], tuple[EvaluationSpec, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "c2f_parent_comparative_candidate_plan_v1":
        raise ValueError("Unexpected C2F parent-comparison candidate-plan protocol")
    if payload.get("architecture") != architecture:
        raise ValueError(f"Candidate plan is for {payload.get('architecture')!r}, not {architecture!r}")
    raw_candidates = payload.get("candidates")
    selected = payload.get("selection", {}).get("selected_count")
    if not isinstance(raw_candidates, list) or not raw_candidates or selected != len(raw_candidates):
        raise ValueError("Candidate-plan selection count does not match candidate list")
    if payload.get("selection", {}).get("replicates_per_dataset") != 1:
        raise ValueError("This protocol permits exactly one replicate per dataset")

    specs: list[EvaluationSpec] = []
    labels: set[str] = set()
    keys: set[str] = set()
    gray_count = 0
    for number, item in enumerate(raw_candidates, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Candidate {number} is not a JSON object")
        label = str(item.get("label", "")).strip()
        key = str(item.get("candidate_key", "")).strip()
        mode = str(item.get("mode", "")).strip()
        role = str(item.get("role", "")).strip()
        if not label or label in labels or not key or key in keys or not role:
            raise ValueError(f"Candidate {number} has an empty/duplicate label, key, or role")
        source_raw = item.get("source_fr1_lightswitch_ate_mean_cm")
        source = None if source_raw is None else float(source_raw)
        if mode == "gray":
            if key != "gray" or source is not None:
                raise ValueError("The gray control must use candidate_key='gray' and no source ATE")
            gray_count += 1
            spec = EvaluationSpec(label, key, architecture, mode, role, source)
        elif mode == "direct":
            layer = str(item.get("layer", "")).strip()
            channels = checked_channels(item.get("channels"), layer, architecture, f"{label} direct")
            if source is None:
                raise ValueError(f"{label} direct configuration needs source_fr1_lightswitch_ate_mean_cm")
            spec = EvaluationSpec(label, key, architecture, mode, role, source, direct_layer=layer, direct_channels=channels)
        elif mode == "c2f":
            variant = str(item.get("variant", "")).strip()
            fine = item.get("fine")
            coarse = item.get("coarse")
            parents = item.get("parent_labels")
            if variant not in {"A", "B"} or not isinstance(fine, dict) or not isinstance(coarse, dict):
                raise ValueError(f"{label} must declare valid variant A/B and fine/coarse objects")
            if not isinstance(parents, list) or len(parents) != 2 or len(set(parents)) != 2:
                raise ValueError(f"{label} must declare exactly two different direct parent labels")
            fine_layer = str(fine.get("layer", "")).strip()
            coarse_layer = str(coarse.get("layer", "")).strip()
            fine_channels = checked_channels(fine.get("channels"), fine_layer, architecture, f"{label} fine")
            coarse_channels = checked_channels(coarse.get("channels"), coarse_layer, architecture, f"{label} coarse")
            expected = ("conv1", "layer2") if architecture == "resnet" else ("enc0", "enc1")
            if (fine_layer, coarse_layer) != expected:
                raise ValueError(f"{label} must use fine/coarse layers {expected}, not {(fine_layer, coarse_layer)}")
            if source is None:
                raise ValueError(f"{label} C2F configuration needs source_fr1_lightswitch_ate_mean_cm")
            spec = EvaluationSpec(
                label, key, architecture, mode, role, source,
                variant=variant, fine_layer=fine_layer, fine_channels=fine_channels,
                coarse_layer=coarse_layer, coarse_channels=coarse_channels,
                parent_labels=tuple(str(parent).strip() for parent in parents),
            )
        else:
            raise ValueError(f"{label} has unsupported mode {mode!r}")
        labels.add(label)
        keys.add(key)
        specs.append(spec)
    if gray_count != 1:
        raise ValueError("Plan must contain exactly one gray baseline")
    by_label = {spec.label: spec for spec in specs}
    for spec in specs:
        if spec.mode != "c2f":
            continue
        parents = tuple(by_label.get(label) for label in spec.parent_labels)
        if None in parents or any(parent.mode != "direct" for parent in parents):
            raise ValueError(f"{spec.label} parent_labels must name two direct configurations")
        parent_layers = {parent.direct_layer for parent in parents if parent is not None}
        if parent_layers != {spec.fine_layer, spec.coarse_layer}:
            raise ValueError(f"{spec.label} parents do not match its fine/coarse branches")
    payload["_resolved_plan_path"] = str(path.resolve())
    return payload, tuple(specs)


class ParentComparisonConfigGuard(core.ConfigGuard):
    """Apply direct or C2F tracking while freezing gray sensor-depth mapping."""

    def __init__(self, config_path: Path, lock_path: Path, specs: Sequence[EvaluationSpec]):
        super().__init__(config_path, lock_path)
        self.by_label = {spec.label: spec for spec in specs}

    def apply(self, candidate: core.Candidate) -> dict[str, Any]:
        spec = self.by_label.get(candidate.label)
        if spec is None:
            raise KeyError(f"No parent-comparison specification registered for {candidate.label}")
        config = yaml.safe_load(self.original)
        config["mapping"]["color"] = "gray"
        config["mapping"]["use_sensor_depth"] = True
        tracking = config["tracking"]
        tracking["debug_tracking_diagnostics"] = True
        tracking["debug_tracking_print_every_frame"] = True
        tracking["debug_tracking_save_suspicious"] = False
        tracking.pop("cnn_c2f_fine_levels", None)

        if spec.mode == "gray":
            tracking.update(color="gray", cnn_mode="cnn_only")
        elif spec.mode == "direct" and spec.architecture == "resnet":
            assert spec.direct_layer is not None
            tracking.update(
                color="cnn",
                cnn_layer_name=spec.direct_layer,
                cnn_channels=len(spec.direct_channels),
                cnn_channel_select=",".join(f"d{value}" for value in spec.direct_channels),
                cnn_layer_full_channels=allowed_layers("resnet")[spec.direct_layer],
                cnn_mode="cnn_only",
            )
        elif spec.mode == "direct" and spec.architecture == "unet":
            assert spec.direct_layer is not None
            tracking.update(
                color="unet",
                cnn_mode="cnn_only",
                unet_enc_level=0 if spec.direct_layer == "enc0" else 1,
                unet_channel_select=",".join(f"d{value}" for value in spec.direct_channels),
            )
        elif spec.mode == "c2f" and spec.architecture == "resnet":
            tracking.update(
                color="cnn_c2f",
                cnn_c2f_version=spec.variant,
                cnn_mode="cnn_only",
                cnn_layer_coarse="layer2",
                cnn_channels_coarse=len(spec.coarse_channels),
                cnn_channel_select_coarse=",".join(f"d{value}" for value in spec.coarse_channels),
                cnn_layer_full_channels_coarse=128,
                cnn_layer_fine="conv1",
                cnn_channels_fine=len(spec.fine_channels),
                cnn_channel_select_fine=",".join(f"d{value}" for value in spec.fine_channels),
                cnn_layer_full_channels_fine=64,
            )
        elif spec.mode == "c2f" and spec.architecture == "unet":
            tracking.update(
                color="unet_c2f",
                cnn_c2f_version=spec.variant,
                cnn_mode="cnn_only",
                unet_enc_level_coarse=1,
                unet_channel_select_coarse=",".join(f"d{value}" for value in spec.coarse_channels),
                unet_enc_level_fine=0,
                unet_channel_select_fine=",".join(f"d{value}" for value in spec.fine_channels),
            )
        else:
            raise ValueError(f"Unsupported architecture/mode: {spec.architecture}/{spec.mode}")
        encoded = yaml.safe_dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False).encode("utf-8")
        core.atomic_write_bytes(self.config_path, encoded)
        return config


def validate_inputs(args: argparse.Namespace, plan: dict[str, Any], specs: Sequence[EvaluationSpec]) -> list[float]:
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
        raise ValueError("This frozen protocol fixes --timeout-seconds at 500 seconds")
    if not 0.0 < args.minimum_coverage <= 1.0:
        raise ValueError("--minimum-coverage must lie in (0, 1]")
    tracking_source = (args.como_dir / "como/odom/Tracking.py").read_text(encoding="utf-8")
    required_mode = "cnn_c2f" if args.architecture == "resnet" else "unet_c2f"
    if required_mode not in tracking_source:
        raise RuntimeError(f"Current Tracking.py does not contain required C2F mode {required_mode}")
    if args.architecture == "unet" and "UNetC2FFeatureExtractor" not in tracking_source:
        raise RuntimeError("Current Tracking.py does not use UNetC2FFeatureExtractor")
    timestamps = full.read_timestamp_index(args.dataset_dir / "matched_rgb.txt")
    if not timestamps or not specs:
        raise ValueError("Dataset has no matched timestamps or candidate plan has no specifications")
    return timestamps


def write_summary(output_dir: Path, store: full.ResultStore, args: argparse.Namespace, specs: Sequence[EvaluationSpec], frame_count: int) -> None:
    rows = store.rows()
    by_label = {str(row["label"]): row for row in rows}
    status_counts = Counter(str(row["status"]) for row in rows)
    lines = [
        f"# {args.architecture.upper()} C2F parent-comparison evaluation",
        "",
        f"- Dataset: `{args.dataset_dir}` ({frame_count} matched RGB-D frames).",
        "- Mapping: gray + sensor depth.  Only tracking features are altered.",
        "- Direct parents and C2F cells are run on the same sequence; aggregation computes C2F deltas only within each dataset.",
        "- C2F-A: coarse L0/L1 + fine L2. C2F-B: coarse L0 + fine L1/L2.",
        "- Primary metric: historical keyframe `evo_ape --align --correct_scale` translation ATE mean.",
        f"- Persisted rows: {len(rows)}/{len(specs)}; statuses: {dict(sorted(status_counts.items()))}.",
        "",
        "| Label | Mode | Configuration | Status | ATE mean (cm) | Coverage | Runtime (s) |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for spec in specs:
        row = by_label.get(spec.label)
        status = "NOT_RUN" if row is None else str(row["status"])
        ate = "" if row is None or row["historical_evo_ape_mean_m"] is None else f"{float(row['historical_evo_ape_mean_m']) * 100:.4f}"
        coverage = "" if row is None or row["coverage_ratio"] is None else f"{float(row['coverage_ratio']):.4f}"
        runtime = "" if row is None else f"{float(row['elapsed_seconds']):.1f}"
        lines.append(f"| {spec.label} | {spec.mode} | {spec.display} | {status} | {ate} | {coverage} | {runtime} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for name in ("dataset_dir", "como_dir", "python", "evo_ape", "evo_rpe"):
        setattr(args, name, getattr(args, name).resolve())
    if args.candidate_plan is None:
        args.candidate_plan = default_plan_path(args.architecture)
    args.candidate_plan = args.candidate_plan.resolve()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / args.architecture / "single_dataset"
    args.output_dir = args.output_dir.resolve()

    plan, specs = load_specs(args.candidate_plan, args.architecture)
    timestamps = validate_inputs(args, plan, specs)
    c2f_count = sum(spec.mode == "c2f" for spec in specs)
    direct_count = sum(spec.mode == "direct" for spec in specs)
    print("=" * 88)
    print(f"{args.architecture.upper()} C2F PARENT-COMPARISON FULL-SEQUENCE EVALUATION")
    print("=" * 88)
    print(f"Mode: {'EXECUTE/RESUME' if args.execute else 'VALIDATE ONLY'}")
    print(f"Dataset: {args.dataset_dir} ({len(timestamps)} matched frames)")
    print(f"Configurations: {len(specs)} = 1 gray + {direct_count} direct parents + {c2f_count} C2F comparisons")
    print("Mapping: gray + sensor depth; primary ATE: keyframe evo_ape --align --correct_scale")
    print("C2F reports will compare each C2F row with both direct parents within this same dataset.")
    print(f"Timeout: {args.timeout_seconds:.0f}s; coverage gate: {args.minimum_coverage:.0%}")
    for spec in specs:
        parent_text = "" if not spec.parent_labels else f"; parents={list(spec.parent_labels)}"
        print(f"[PLAN] {spec.label}: {spec.display}{parent_text}")
    if not args.execute:
        print("[VALID] No COMO process, SQLite row, trajectory, or config edit was created.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    execution_plan = {
        "protocol": "c2f_parent_comparative_full_sequence_execution_v1",
        "architecture": args.architecture,
        "dataset": str(args.dataset_dir),
        "matched_frames": len(timestamps),
        "candidate_plan": plan["_resolved_plan_path"],
        "candidate_plan_sha256": hashlib.sha256(args.candidate_plan.read_bytes()).hexdigest(),
        "replicates": 1,
        "timeout_seconds": args.timeout_seconds,
        "minimum_coverage": args.minimum_coverage,
        "primary_metric": "historical keyframe evo_ape mean: --align --correct_scale",
        "diagnostic_metrics": ["historical evo_rpe", "all-frame metric-scale SE(3) ATE/RPE", "coverage", "tracking diagnostics"],
        "configurations": [asdict(spec) for spec in specs],
    }
    (args.output_dir / "evaluation_plan.json").write_text(json.dumps(execution_plan, indent=2), encoding="utf-8")
    console = full.Console(args.output_dir / "console.log")
    store = full.ResultStore(args.output_dir / "evaluations.sqlite3")
    guard: ParentComparisonConfigGuard | None = None
    try:
        console.say("=" * 88)
        console.say(f"{args.architecture.upper()} C2F PARENT-COMPARISON FULL-SEQUENCE EVALUATION")
        console.say("=" * 88)
        console.say(f"Dataset: {args.dataset_dir}; matched frames={len(timestamps)}")
        console.say("Mapping stays gray + sensor depth; only tracking configuration changes.")
        console.say(f"Configurations={len(specs)}; direct parents={direct_count}; C2F comparisons={c2f_count}; timeout={args.timeout_seconds:.0f}s")
        guard = ParentComparisonConfigGuard(args.como_dir / "config/como.yml", full.SHARED_CONFIG_LOCK, specs)
        started = time.monotonic()
        new_rows = 0
        for index, spec in enumerate(specs, start=1):
            outcome = full.evaluate_one(args, console, store, guard, spec, 0, index, len(specs), timestamps)
            if outcome is not None:
                new_rows += 1
                completed = sum(1 for row in store.rows() if str(row["label"]) in {item.label for item in specs})
                elapsed = time.monotonic() - started
                average = elapsed / new_rows
                status_rows = store.connection.execute(
                    "SELECT status, COUNT(*) FROM evaluations WHERE replicate=0 GROUP BY status"
                ).fetchall()
                status_text = ", ".join(f"{row[0]}={row[1]}" for row in status_rows)
                console.say(
                    f"[PROGRESS] preserved={completed}/{len(specs)} remaining={len(specs) - completed} "
                    f"{status_text} batch_avg={average:.1f}s ETA={(len(specs) - completed) * average / 3600:.2f}h"
                )
            full.export_results(store, args.output_dir)
            write_summary(args.output_dir, store, args, specs, len(timestamps))
        console.say("[DONE] Parent-comparison evaluation complete; SQLite, CSV, artifacts, and summary were exported")
    finally:
        if guard is not None:
            guard.close()
        full.export_results(store, args.output_dir)
        write_summary(args.output_dir, store, args, specs, len(timestamps))
        store.close()
        console.close()


if __name__ == "__main__":
    main()
