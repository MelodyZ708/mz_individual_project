#!/usr/bin/env python3
"""Direct full-sequence greedy search for ResNet Conv1 (project ``conv0``).

This is deliberately an adapter around the already validated UNet Enc1 search
engine.  It keeps the same full-sequence, multi-start, pair-rescue, matched
random-control, local swap and three-repeat protocol while changing only the
feature extractor settings and the available channel universe (64 Conv1
post-ReLU channels).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
STEP_J_RUNNER = (
    PROJECT_ROOT
    / "channel_selection_pipeline/scripts/step_j_unet_feature_greedy_search/"
    "run_unet_direct_fullseq_greedy.py"
)
DEFAULT_DATASET = Path("/home/melody/data/tum/rgbd_dataset_freiburg1_desk_lightswitch")
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "channel_selection_results/step_n_resnet_conv1_direct_fullseq_greedy"
)
DEFAULT_PYTHON = Path("/home/melody/anaconda3/envs/como/bin/python")
DEFAULT_EVO_APE = Path("/home/melody/anaconda3/envs/como/bin/evo_ape")
DEFAULT_EVO_RPE = Path("/home/melody/anaconda3/envs/como/bin/evo_rpe")
SHARED_CONFIG_LOCK = (
    PROJECT_ROOT
    / "channel_selection_results/step_d_fail_fast_evaluation/"
    "r070_bruteforce_v2/.como_config.lock"
)

NUM_CHANNELS = 64
MAX_CHANNELS = 6
RANDOM_SEED = 20260817
PROTOCOL = "resnet_conv1_direct_full_sequence_greedy_v1"
HISTORICAL_ANCHORS: tuple[tuple[str, tuple[int, ...], str, str], ...] = (
    (
        "historical_cnn_baseline_ch_5_29_40_52",
        (5, 29, 40, 52),
        "historical four-channel CNN baseline, re-evaluated only as an anchor",
        "H4_historical_cnn_baseline",
    ),
    (
        "historical_correlation_search_ch_5_6_24_29",
        (5, 6, 24, 29),
        "historical correlation-clustering search result, re-evaluated only as an anchor",
        "H4_historical_correlation_search",
    ),
)


def load_engine():
    spec = importlib.util.spec_from_file_location("resnet_conv1_greedy_engine", STEP_J_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load common greedy engine: {STEP_J_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = load_engine()
full = engine.full
core = engine.core


def configure_shared_engine() -> None:
    """Make the imported engine operate over the ResNet Conv1 feature space."""

    engine.NUM_CHANNELS = NUM_CHANNELS
    engine.MAX_CHANNELS = MAX_CHANNELS
    engine.RANDOM_SEED = RANDOM_SEED
    engine.PROTOCOL = PROTOCOL
    engine.ENCODER_LABEL = "resnet_conv1"
    engine.ALL_CHANNEL_ANCHOR_LABEL = "resnet_conv1_all64"
    # The engine uses this tuple solely as historical anchors and final-report tags.
    engine.BQS_REFERENCE_CANDIDATES = HISTORICAL_ANCHORS

    def resnet_anchor_candidates():
        anchors = [
            engine.SearchCandidate("gray_current", None, "current photometric gray control"),
            engine.SearchCandidate(
                engine.ALL_CHANNEL_ANCHOR_LABEL,
                tuple(range(NUM_CHANNELS)),
                "unselected ResNet Conv1 all64-channel control",
            ),
        ]
        anchors.extend(
            engine.SearchCandidate(label, channels, role)
            for label, channels, role, _ in HISTORICAL_ANCHORS
        )
        return tuple(anchors)

    engine.anchor_candidates = resnet_anchor_candidates


configure_shared_engine()


class ResNetConv1ConfigGuard(core.ConfigGuard):
    """Apply post-ReLU Conv1 direct-channel tracking, with gray depth mapping."""

    def apply(self, candidate: core.Candidate) -> dict:
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
            channels = tuple(sorted(int(channel) for channel in candidate.channels))
            if not channels or channels[0] < 0 or channels[-1] >= NUM_CHANNELS:
                raise ValueError(
                    f"ResNet Conv1 channels must be unique indices in 0--{NUM_CHANNELS - 1}: "
                    f"{channels}"
                )
            if len(set(channels)) != len(channels):
                raise ValueError(f"Duplicate ResNet Conv1 channels: {channels}")
            tracking.update(
                color="cnn",
                cnn_layer_name="conv1",
                cnn_channels=len(channels),
                cnn_channel_select=",".join(f"d{channel}" for channel in channels),
                cnn_layer_full_channels=NUM_CHANNELS,
                cnn_mode="cnn_only",
            )
        encoded = yaml.safe_dump(
            config, default_flow_style=False, allow_unicode=True, sort_keys=False
        ).encode("utf-8")
        core.atomic_write_bytes(self.config_path, encoded)
        return config


class ResNetController(engine.Controller):
    """The common controller with ResNet-specific report heading."""

    def write_final_summary(self, candidates=None) -> None:
        super().write_final_summary(candidates)
        summary_path = self.args.output_dir / "summary.md"
        if summary_path.is_file():
            text = summary_path.read_text(encoding="utf-8")
            text = text.replace(
                "# UNet direct full-sequence greedy search",
                "# ResNet Conv1 direct full-sequence greedy search",
                1,
            )
            text = text.replace(
                "All new candidates were ranked only by full-sequence tracking, not MVS/BQS.",
                "All new candidates were ranked only by full-sequence tracking; correlation clustering is retained only as a historical anchor.",
                1,
            )
            summary_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Search 1--6 ResNet Conv1 post-ReLU channels using the direct full-sequence "
            "fr1/desk_lightswitch greedy protocol."
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
    parser.add_argument("--como-dir", type=Path, default=PROJECT_ROOT / "como")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--evo-ape", type=Path, default=DEFAULT_EVO_APE)
    parser.add_argument("--evo-rpe", type=Path, default=DEFAULT_EVO_RPE)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--starts",
        choices=("auto", "2", "3", "4"),
        default="auto",
        help="Full-sequence singleton-ATE greedy seeds; auto uses anchor median runtime.",
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


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_protocol(args: argparse.Namespace, timestamps: Sequence[float]) -> None:
    payload = {
        "protocol": PROTOCOL,
        "shared_algorithm_source": str(STEP_J_RUNNER),
        "shared_algorithm_sha256": source_sha256(STEP_J_RUNNER),
        "dataset_dir": str(args.dataset_dir),
        "matched_frames": len(timestamps),
        "first_timestamp": timestamps[0],
        "last_timestamp": timestamps[-1],
        "feature_space": {
            "network": "ImageNet ResNet-18",
            "layer": "conv1",
            "activation_position": "post-ReLU",
            "available_channels": NUM_CHANNELS,
            "channel_names": [f"d{index}" for index in range(NUM_CHANNELS)],
        },
        "tracking": {
            "color": "cnn",
            "cnn_layer_name": "conv1",
            "cnn_mode": "cnn_only",
        },
        "mapping": {"color": "gray", "use_sensor_depth": True},
        "primary_metric": (
            "keyframe evo_ape TUM groundtruth.txt data_tum.txt with alignment and "
            "scale correction, ranked by translation ATE mean"
        ),
        "max_channels": MAX_CHANNELS,
        "timeout_seconds": args.timeout_seconds,
        "random_seed": args.random_seed,
        "auto_start_rule": {"t50_le_60_s": 4, "t50_le_100_s": 3, "otherwise": 2},
        "theoretical_evaluation_budgets": {
            f"starts_{starts}": engine.theoretical_budget(starts, 1)
            for starts in (2, 3, 4)
        },
        "all_singletons_fail_fallback": {
            "method": (
                f"exhaustively evaluate all C({NUM_CHANNELS},2)={math.comb(NUM_CHANNELS, 2)} "
                "pairs, use the best PASS pairs as greedy seeds, then extend through K=6"
            ),
            "random_control_note": (
                "K=2 is exhaustive, so no disjoint random pairs remain; retain equal-budget "
                "random controls at K=3--6"
            ),
            "theoretical_pre_repeat_budget": {
                f"starts_{starts}": engine.theoretical_budget(starts, 2)
                for starts in (2, 3, 4)
            },
        },
        "historical_anchors_not_used_as_seeds_or_ranking": [
            dataclasses.asdict(candidate) for candidate in engine.anchor_candidates()
        ],
    }
    path = args.output_dir / "candidate_plan.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("protocol") != PROTOCOL:
            raise RuntimeError(
                "Output directory contains a different protocol; choose a new directory "
                "rather than mixing experiments."
            )
        for key in ("dataset_dir", "random_seed", "shared_algorithm_sha256"):
            if existing.get(key) != payload[key]:
                raise RuntimeError(f"Output directory has a different {key}.")
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_dry_run(console) -> None:
    console.say("[DRY RUN] No COMO process is launched without --execute.")
    console.say(
        "[DRY RUN] Anchors: gray, ResNet Conv1 all64, "
        + ", ".join(label for label, _, _, _ in HISTORICAL_ANCHORS)
        + "."
    )
    console.say(
        f"[DRY RUN] Greedy: all {NUM_CHANNELS} singleton channels, then every selected "
        "path extends through K=6 even after a temporary ATE degradation."
    )
    console.say(
        "[DRY RUN] Auto pre-repeat upper bounds: "
        + "; ".join(
            f"{starts} starts={engine.theoretical_budget(starts, 1)['pre_repeat_total_max']}"
            for starts in (4, 3, 2)
        )
        + "."
    )
    console.say(
        f"[DRY RUN] If fewer than the planned singleton seeds PASS: exhaustively score "
        f"{math.comb(NUM_CHANNELS, 2)} pairs; retain singleton paths and fill missing starts "
        "with non-duplicate pair paths. K=2 random control is skipped because that space "
        "has already been exhausted."
    )
    console.say(
        "[DRY RUN] Final output keeps G1--G6 and explicit G4; it chooses adaptive Gstar "
        "from three-repeat ATE means."
    )


def main() -> None:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.como_dir = args.como_dir.resolve()
    args.python = args.python.resolve()
    args.evo_ape = args.evo_ape.resolve()
    args.evo_rpe = args.evo_rpe.resolve()
    timestamps = engine.validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_protocol(args, timestamps)

    console = full.Console(args.output_dir / "console.log")
    store = engine.SearchStore(args.output_dir / "evaluations.sqlite3")
    controller = ResNetController(args, store, console, timestamps)
    try:
        console.say("=" * 78)
        console.say("RESNET CONV1 (PROJECT CONV0) DIRECT FULL-SEQUENCE ATE GREEDY SEARCH")
        console.say("=" * 78)
        console.say(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
        console.say(f"Requested stage: {args.stage}")
        console.say(
            f"Dataset: {args.dataset_dir}; matched frames={len(timestamps)}; "
            f"time={timestamps[0]:.6f}--{timestamps[-1]:.6f}"
        )
        console.say(
            "Primary rank: historical keyframe evo_ape ATE mean with alignment and scale "
            "correction, identical to run_random_channel_search.sh."
        )
        console.say(
            "Tracking uses post-ReLU ResNet Conv1 channels; mapping stays gray with sensor "
            "depth. Correlation clustering is historical context only, never a seed or rank."
        )
        console.say(
            f"Timeout={args.timeout_seconds:.1f}s; batch limit="
            f"{'disabled' if args.batch_hours == 0 else f'{args.batch_hours:.2f}h'}."
        )
        if not args.execute:
            print_dry_run(console)
            controller.export()
            return

        controller.guard = ResNetConv1ConfigGuard(
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
    except engine.BatchLimitReached as error:
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
