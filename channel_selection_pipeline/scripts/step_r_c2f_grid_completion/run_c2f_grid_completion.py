#!/usr/bin/env python3
"""Run only the missing cells of a reduced complete C2F grid on one sequence.

Step-Q already evaluated four C2F pairs and five direct parents per
architecture over nine sequences.  This runner derives the requested larger
grid from the frozen Step-P top-six candidate plans, verifies those Step-Q rows
exist, and evaluates *only* the non-overlapping cells.  It never copies,
overwrites, or reruns the earlier observations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
STEP_Q_EVALUATOR = (
    PROJECT_ROOT / "channel_selection_pipeline/scripts/step_q_c2f_multi_dataset_evaluation/"
    "run_c2f_parent_comparison.py"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "channel_selection_results/step_r_c2f_grid_completion"
DEFAULT_PYTHON = Path("/home/melody/anaconda3/envs/como/bin/python")
DEFAULT_EVO_APE = Path("/home/melody/anaconda3/envs/como/bin/evo_ape")
DEFAULT_EVO_RPE = Path("/home/melody/anaconda3/envs/como/bin/evo_rpe")


def load_step_q_module():
    spec = importlib.util.spec_from_file_location("c2f_grid_completion_step_q", STEP_Q_EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Step-Q evaluator: {STEP_Q_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


q = load_step_q_module()
full = q.full
core = q.core


@dataclass(frozen=True)
class Branch:
    rank: int
    layer: str
    channels: tuple[int, ...]
    source_ate_mean_cm: float


@dataclass(frozen=True)
class GridItem:
    label: str
    kind: str
    spec: q.EvaluationSpec
    fine_rank: int | None = None
    coarse_rank: int | None = None
    variant: str | None = None
    reused_step_q_label: str | None = None

    @property
    def is_reused(self) -> bool:
        return self.reused_step_q_label is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    parser.add_argument("--architecture", choices=("resnet", "unet"), required=True)
    parser.add_argument("--execute", action="store_true", help="Launch/resume only missing COMO evaluations.")
    parser.add_argument("--rerun-existing", action="store_true", help="Rerun cells already saved in this Step-R output only.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--completion-plan", type=Path)
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
    return SCRIPT_DIR / f"{architecture}_c2f_grid_completion_plan.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def channels(raw: Any, layer: str, architecture: str, context: str) -> tuple[int, ...]:
    limits = {"resnet": {"conv1": 64, "layer2": 128}, "unet": {"enc0": 16, "enc1": 32}}[architecture]
    if not isinstance(raw, list) or not raw or layer not in limits:
        raise ValueError(f"Invalid channels/layer for {context}")
    values = tuple(sorted(int(value) for value in raw))
    if len(values) != len(set(values)) or values[0] < 0 or values[-1] >= limits[layer]:
        raise ValueError(f"Invalid {layer} channel indices for {context}: {values}")
    return values


def resolve_step_q_direct_map(step_q_plan: dict[str, Any], architecture: str) -> dict[tuple[str, tuple[int, ...]], str]:
    result: dict[tuple[str, tuple[int, ...]], str] = {}
    for item in step_q_plan.get("candidates", []):
        if item.get("mode") != "direct":
            continue
        layer = str(item.get("layer", ""))
        key = (layer, channels(item.get("channels"), layer, architecture, str(item.get("label"))))
        if key in result:
            raise ValueError(f"Duplicate Step-Q direct parent: {key}")
        result[key] = str(item["label"])
    return result


def resolve_step_q_c2f_map(step_q_plan: dict[str, Any], architecture: str) -> dict[tuple[str, str, tuple[int, ...], str, tuple[int, ...]], str]:
    result: dict[tuple[str, str, tuple[int, ...], str, tuple[int, ...]], str] = {}
    for item in step_q_plan.get("candidates", []):
        if item.get("mode") != "c2f":
            continue
        fine, coarse = item.get("fine"), item.get("coarse")
        if not isinstance(fine, dict) or not isinstance(coarse, dict):
            raise ValueError(f"Invalid Step-Q C2F candidate: {item.get('label')}")
        fine_layer, coarse_layer = str(fine.get("layer", "")), str(coarse.get("layer", ""))
        key = (
            str(item.get("variant", "")), fine_layer, channels(fine.get("channels"), fine_layer, architecture, str(item.get("label"))),
            coarse_layer, channels(coarse.get("channels"), coarse_layer, architecture, str(item.get("label"))),
        )
        if key in result:
            raise ValueError(f"Duplicate Step-Q C2F candidate: {key}")
        result[key] = str(item["label"])
    return result


def load_grid(args: argparse.Namespace) -> tuple[dict[str, Any], list[GridItem], Path]:
    plan_path = args.completion_plan or default_plan_path(args.architecture)
    plan_path = plan_path.resolve()
    plan = read_json(plan_path)
    if plan.get("protocol") != "c2f_full_grid_completion_v1" or plan.get("architecture") != args.architecture:
        raise ValueError("Unexpected completion plan or architecture")
    if plan.get("variants") != ["A", "B"] or plan.get("replicates") != 1 or float(plan.get("timeout_seconds", -1)) != 500.0:
        raise ValueError("The frozen grid protocol requires A/B, one replicate, and 500 seconds")
    grid_plan = read_json((SCRIPT_DIR / str(plan["source_grid_plan"])).resolve())
    if grid_plan.get("protocol") != "c2f_promising_channel_grid_v1" or grid_plan.get("architecture") != args.architecture:
        raise ValueError("Unexpected Step-P source grid")
    q_plan = read_json((SCRIPT_DIR / str(plan["reuse_step_q_plan"])).resolve())
    if q_plan.get("protocol") != "c2f_parent_comparative_candidate_plan_v1" or q_plan.get("architecture") != args.architecture:
        raise ValueError("Unexpected Step-Q reuse plan")

    fine_doc, coarse_doc = grid_plan["fine_branch"], grid_plan["coarse_branch"]
    fine_count, coarse_count = int(plan["fine_top_count"]), int(plan["coarse_top_count"])
    raw_fine, raw_coarse = fine_doc.get("candidates", []), coarse_doc.get("candidates", [])
    if fine_count > len(raw_fine) or coarse_count > len(raw_coarse):
        raise ValueError("Requested top-count exceeds the Step-P frozen six-candidate list")
    fine = [
        Branch(int(item["rank"]), str(fine_doc["layer"]), channels(item["channels"], str(fine_doc["layer"]), args.architecture, "fine"), float(item["source_ate_mean_cm"]))
        for item in raw_fine[:fine_count]
    ]
    coarse = [
        Branch(int(item["rank"]), str(coarse_doc["layer"]), channels(item["channels"], str(coarse_doc["layer"]), args.architecture, "coarse"), float(item["source_ate_mean_cm"]))
        for item in raw_coarse[:coarse_count]
    ]
    if [item.rank for item in fine] != list(range(1, fine_count + 1)) or [item.rank for item in coarse] != list(range(1, coarse_count + 1)):
        raise ValueError("Step-P branch ranks must start at one and remain consecutive")
    direct_reuse = resolve_step_q_direct_map(q_plan, args.architecture)
    c2f_reuse = resolve_step_q_c2f_map(q_plan, args.architecture)
    items: list[GridItem] = []
    for branch in fine:
        reused = direct_reuse.get((branch.layer, branch.channels))
        label = f"{args.architecture}_direct_fine{branch.rank:02d}"
        spec = q.EvaluationSpec(
            label=label, candidate_key=f"{args.architecture}:direct:{branch.layer}:" + ",".join(map(str, branch.channels)),
            architecture=args.architecture, mode="direct", role=f"Step-R direct fine rank {branch.rank}",
            source_ate_mean_cm=branch.source_ate_mean_cm, direct_layer=branch.layer, direct_channels=branch.channels,
        )
        items.append(GridItem(label, "direct_fine", spec, fine_rank=branch.rank, reused_step_q_label=reused))
    for branch in coarse:
        reused = direct_reuse.get((branch.layer, branch.channels))
        label = f"{args.architecture}_direct_coarse{branch.rank:02d}"
        spec = q.EvaluationSpec(
            label=label, candidate_key=f"{args.architecture}:direct:{branch.layer}:" + ",".join(map(str, branch.channels)),
            architecture=args.architecture, mode="direct", role=f"Step-R direct coarse rank {branch.rank}",
            source_ate_mean_cm=branch.source_ate_mean_cm, direct_layer=branch.layer, direct_channels=branch.channels,
        )
        items.append(GridItem(label, "direct_coarse", spec, coarse_rank=branch.rank, reused_step_q_label=reused))
    for variant in ("A", "B"):
        for fine_branch in fine:
            for coarse_branch in coarse:
                reuse_key = (variant, fine_branch.layer, fine_branch.channels, coarse_branch.layer, coarse_branch.channels)
                reused = c2f_reuse.get(reuse_key)
                label = f"{args.architecture}_c2f_{variant.lower()}_fine{fine_branch.rank:02d}_coarse{coarse_branch.rank:02d}"
                spec = q.EvaluationSpec(
                    label=label,
                    candidate_key=(f"{args.architecture}:c2f:{variant}:fine" + ",".join(map(str, fine_branch.channels)) + ":coarse" + ",".join(map(str, coarse_branch.channels))),
                    architecture=args.architecture, mode="c2f", role=f"Step-R full-grid C2F-{variant}; F{fine_branch.rank} × C{coarse_branch.rank}",
                    source_ate_mean_cm=None, variant=variant, fine_layer=fine_branch.layer, fine_channels=fine_branch.channels,
                    coarse_layer=coarse_branch.layer, coarse_channels=coarse_branch.channels,
                )
                items.append(GridItem(label, "c2f", spec, fine_branch.rank, coarse_branch.rank, variant, reused))

    expected = plan["expected"]
    c2f_items = [item for item in items if item.kind == "c2f"]
    direct_items = [item for item in items if item.kind != "c2f"]
    if len(c2f_items) != int(expected["full_c2f_pairs"]) or len(direct_items) != int(expected["full_direct_parents"]):
        raise ValueError("Expanded grid count does not match frozen plan")
    if sum(item.is_reused for item in c2f_items) != int(expected["reused_step_q_c2f_pairs"]):
        raise ValueError("Unexpected number of C2F cells reusable from Step-Q")
    if sum(item.is_reused for item in direct_items) != int(expected["reused_step_q_direct_parents"]):
        raise ValueError("Unexpected number of direct parents reusable from Step-Q")
    new_items = [item for item in items if not item.is_reused]
    if len(new_items) != int(expected["new_evaluations_per_dataset"]):
        raise ValueError("New evaluation count does not match frozen plan")
    reuse_output = (PROJECT_ROOT / str(plan["reuse_step_q_output"])).resolve()
    return plan, items, reuse_output


def validate_reused_rows(reuse_output: Path, dataset_key: str, items: Sequence[GridItem]) -> None:
    database = reuse_output / "per_dataset" / dataset_key / "evaluations.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(f"Step-Q reuse database is missing: {database}")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError(f"Step-Q SQLite integrity check failed: {database}")
        labels = {str(row[0]) for row in connection.execute("SELECT label FROM evaluations WHERE replicate=0")}
    finally:
        connection.close()
    missing = [item.reused_step_q_label for item in items if item.is_reused and item.reused_step_q_label not in labels]
    if missing:
        raise ValueError(f"Step-Q rows needed for reuse are missing in {dataset_key}: {sorted(set(missing))}")


def validate_inputs(args: argparse.Namespace, plan: dict[str, Any], items: Sequence[GridItem], reuse_output: Path) -> tuple[list[float], str]:
    required = (
        args.dataset_dir / "matched_rgb.txt", args.dataset_dir / "matched_depth.txt", args.dataset_dir / "groundtruth.txt",
        args.como_dir / "config/como.yml", args.como_dir / "como/como_dataset.py", args.como_dir / "como/odom/Tracking.py",
        args.python, args.evo_ape, args.evo_rpe,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required paths: " + ", ".join(map(str, missing)))
    if args.timeout_seconds != 500.0 or not 0.0 < args.minimum_coverage <= 1.0:
        raise ValueError("Completion protocol requires 500 seconds and a valid coverage gate")
    tracking_source = (args.como_dir / "como/odom/Tracking.py").read_text(encoding="utf-8")
    required_mode = "cnn_c2f" if args.architecture == "resnet" else "unet_c2f"
    if required_mode not in tracking_source:
        raise RuntimeError(f"Current Tracking.py lacks required mode {required_mode}")
    timestamps = full.read_timestamp_index(args.dataset_dir / "matched_rgb.txt")
    if not timestamps:
        raise ValueError("No matched RGB timestamps")
    # Resolve the key via absolute dataset path rather than trusting caller order.
    datasets = json.loads((PROJECT_ROOT / "channel_selection_pipeline/scripts/step_q_c2f_multi_dataset_evaluation/c2f_multi_dataset_plan.json").read_text(encoding="utf-8"))["datasets"]
    root = Path("/home/melody/data/tum")
    key = next((item["key"] for item in datasets if (root / item["directory_name"]).resolve() == args.dataset_dir), None)
    if key is None:
        raise ValueError(f"Dataset is not part of the frozen 3×3 plan: {args.dataset_dir}")
    validate_reused_rows(reuse_output, key, items)
    return timestamps, str(key)


def write_summary(output_dir: Path, store: full.ResultStore, args: argparse.Namespace, items: Sequence[GridItem], dataset_key: str, frame_count: int) -> None:
    rows = store.rows()
    statuses = Counter(str(row["status"]) for row in rows)
    new_items = [item for item in items if not item.is_reused]
    lines = [
        f"# {args.architecture.upper()} Step-R C2F grid completion ({dataset_key})",
        "",
        f"- Matched frames: {frame_count}; mapping remains gray + sensor depth.",
        "- This database contains only missing Step-R cells. Existing Step-Q rows are intentionally reused externally and never copied or rerun.",
        f"- Full selected grid: {sum(item.kind == 'c2f' for item in items)} C2F pairs + {sum(item.kind != 'c2f' for item in items)} direct parents.",
        f"- Reused Step-Q rows: {sum(item.is_reused for item in items)}; new Step-R rows: {len(new_items)}.",
        f"- Persisted new rows: {len(rows)}/{len(new_items)}; statuses: {dict(sorted(statuses.items()))}.",
        "",
        "| Standard label | Kind | Source | Status | ATE mean (cm) |",
        "|---|---|---|---|---:|",
    ]
    by_label = {str(row["label"]): row for row in rows}
    for item in items:
        if item.is_reused:
            lines.append(f"| {item.label} | {item.kind} | Step-Q `{item.reused_step_q_label}` | REUSED |  |")
            continue
        row = by_label.get(item.label)
        status = "NOT_RUN" if row is None else str(row["status"])
        ate = "" if row is None or row["historical_evo_ape_mean_m"] is None else f"{float(row['historical_evo_ape_mean_m']) * 100:.4f}"
        lines.append(f"| {item.label} | {item.kind} | Step-R | {status} | {ate} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for attribute in ("dataset_dir", "como_dir", "python", "evo_ape", "evo_rpe"):
        setattr(args, attribute, getattr(args, attribute).resolve())
    if args.completion_plan is not None:
        args.completion_plan = args.completion_plan.resolve()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / args.architecture / "single_dataset"
    args.output_dir = args.output_dir.resolve()
    plan, items, reuse_output = load_grid(args)
    timestamps, dataset_key = validate_inputs(args, plan, items, reuse_output)
    new_items = [item for item in items if not item.is_reused]
    print("=" * 92)
    print(f"{args.architecture.upper()} STEP-R COMPLETE C2F GRID: MISSING CELLS ONLY")
    print("=" * 92)
    print(f"Mode: {'EXECUTE/RESUME' if args.execute else 'VALIDATE ONLY'}")
    print(f"Dataset: {dataset_key} = {args.dataset_dir} ({len(timestamps)} matched frames)")
    print(f"Full selected grid: {sum(item.kind == 'c2f' for item in items)} C2F pairs + {sum(item.kind != 'c2f' for item in items)} direct parents")
    print(f"Reuse: {sum(item.is_reused for item in items)} completed Step-Q rows; new work: {len(new_items)} cells")
    print("No gray baseline is included. Mapping is gray + sensor depth; only tracking is changed.")
    print("Primary metric: historical keyframe evo_ape ATE mean (--align --correct_scale)")
    print(f"Timeout: {args.timeout_seconds:.0f}s; coverage gate: {args.minimum_coverage:.0%}")
    for item in [item for item in items if item.is_reused]:
        print(f"[REUSE] {item.label} <= Step-Q {item.reused_step_q_label}")
    print(f"[PLAN] {len(new_items)} missing cells will be run/resumed in this Step-R database")
    if not args.execute:
        print("[VALID] No COMO process, SQLite row, trajectory, or config edit was created.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    execution = {
        "protocol": "c2f_full_grid_completion_execution_v1", "architecture": args.architecture,
        "dataset_key": dataset_key, "dataset": str(args.dataset_dir), "matched_frames": len(timestamps),
        "completion_plan": str((args.completion_plan or default_plan_path(args.architecture)).resolve()),
        "completion_plan_sha256": hashlib.sha256((args.completion_plan or default_plan_path(args.architecture)).read_bytes()).hexdigest(),
        "step_q_reuse_output": str(reuse_output), "timeout_seconds": args.timeout_seconds, "minimum_coverage": args.minimum_coverage,
        "full_grid": [asdict(item) for item in items], "new_step_r_labels": [item.label for item in new_items],
    }
    (args.output_dir / "evaluation_plan.json").write_text(json.dumps(execution, indent=2), encoding="utf-8")
    console = full.Console(args.output_dir / "console.log")
    store = full.ResultStore(args.output_dir / "evaluations.sqlite3")
    guard: q.ParentComparisonConfigGuard | None = None
    try:
        console.say("=" * 92)
        console.say(f"{args.architecture.upper()} STEP-R COMPLETE C2F GRID: MISSING CELLS ONLY")
        console.say(f"Dataset={dataset_key}; reuse Step-Q={sum(item.is_reused for item in items)}; new Step-R={len(new_items)}")
        guard = q.ParentComparisonConfigGuard(args.como_dir / "config/como.yml", full.SHARED_CONFIG_LOCK, [item.spec for item in new_items])
        started = time.monotonic()
        new_count = 0
        for index, item in enumerate(new_items, start=1):
            outcome = full.evaluate_one(args, console, store, guard, item.spec, 0, index, len(new_items), timestamps)
            if outcome is not None:
                new_count += 1
                complete = sum(1 for row in store.rows() if str(row["label"]) in {item.label for item in new_items})
                elapsed = time.monotonic() - started
                average = elapsed / new_count
                statuses = store.connection.execute("SELECT status, COUNT(*) FROM evaluations WHERE replicate=0 GROUP BY status").fetchall()
                console.say(f"[PROGRESS] preserved={complete}/{len(new_items)} remaining={len(new_items)-complete} " + ", ".join(f"{row[0]}={row[1]}" for row in statuses) + f" batch_avg={average:.1f}s ETA={(len(new_items)-complete)*average/3600:.2f}h")
            full.export_results(store, args.output_dir)
            write_summary(args.output_dir, store, args, items, dataset_key, len(timestamps))
        console.say("[DONE] Missing Step-R cells completed; Step-Q rows remain externally reused without duplication")
    finally:
        if guard is not None:
            guard.close()
        full.export_results(store, args.output_dir)
        write_summary(args.output_dir, store, args, items, dataset_key, len(timestamps))
        store.close()
        console.close()


if __name__ == "__main__":
    main()
