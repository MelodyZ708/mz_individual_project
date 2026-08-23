#!/usr/bin/env python3
"""Direct full-sequence greedy search for 128 post-ReLU ResNet-18 Layer2 channels.

The search engine, stages, score and SQLite resume semantics are shared with the
completed UNet Enc1 and ResNet Conv1 greedy searches.  This adapter changes the
feature layer and channel universe only; it also has a documented timing
fallback so that an all-failing anchor set cannot prevent the pair-rescue path.
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

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
STEP_N_RUNNER = (
    PROJECT_ROOT
    / "channel_selection_pipeline/scripts/step_n_resnet_conv1_feature_greedy_search/"
    "run_resnet_conv1_direct_fullseq_greedy.py"
)
DEFAULT_DATASET = Path("/home/melody/data/tum/rgbd_dataset_freiburg1_desk_lightswitch")
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "channel_selection_results/step_o_resnet_layer2_direct_fullseq_greedy"
)
DEFAULT_PYTHON = Path("/home/melody/anaconda3/envs/como/bin/python")
DEFAULT_EVO_APE = Path("/home/melody/anaconda3/envs/como/bin/evo_ape")
DEFAULT_EVO_RPE = Path("/home/melody/anaconda3/envs/como/bin/evo_rpe")
SHARED_CONFIG_LOCK = (
    PROJECT_ROOT
    / "channel_selection_results/step_d_fail_fast_evaluation/"
    "r070_bruteforce_v2/.como_config.lock"
)

NUM_CHANNELS = 128
MAX_CHANNELS = 6
RANDOM_SEED = 20260819
PROTOCOL = "resnet_layer2_direct_full_sequence_greedy_v1"


def load_conv1_adapter():
    spec = importlib.util.spec_from_file_location("resnet_layer2_greedy_adapter", STEP_N_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ResNet greedy adapter: {STEP_N_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_conv1_adapter()
engine = runner.engine
full = runner.full
core = runner.core


def configure_shared_engine() -> None:
    """Retarget the common UNet greedy engine to ResNet Layer2."""

    runner.NUM_CHANNELS = NUM_CHANNELS
    runner.MAX_CHANNELS = MAX_CHANNELS
    runner.RANDOM_SEED = RANDOM_SEED
    runner.PROTOCOL = PROTOCOL
    engine.NUM_CHANNELS = NUM_CHANNELS
    engine.MAX_CHANNELS = MAX_CHANNELS
    engine.RANDOM_SEED = RANDOM_SEED
    engine.PROTOCOL = PROTOCOL
    engine.ENCODER_LABEL = "resnet_layer2"
    engine.ALL_CHANNEL_ANCHOR_LABEL = "resnet_layer2_all128"
    # There is no historical full-sequence Layer2 subset.  In particular, the
    # old Conv1 correlation result must not be copied here as an invalid anchor.
    engine.BQS_REFERENCE_CANDIDATES = ()

    def layer2_anchor_candidates():
        return (
            engine.SearchCandidate("gray_current", None, "current photometric gray control"),
            engine.SearchCandidate(
                engine.ALL_CHANNEL_ANCHOR_LABEL,
                tuple(range(NUM_CHANNELS)),
                "unselected ResNet Layer2 all128-channel control",
            ),
        )

    engine.anchor_candidates = layer2_anchor_candidates


configure_shared_engine()


class ResNetLayer2ConfigGuard(core.ConfigGuard):
    """Apply direct 128-channel Layer2 tracking while retaining gray mapping."""

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
                    f"ResNet Layer2 channels must be unique indices in 0--{NUM_CHANNELS - 1}: "
                    f"{channels}"
                )
            if len(set(channels)) != len(channels):
                raise ValueError(f"Duplicate ResNet Layer2 channels: {channels}")
            tracking.update(
                color="cnn",
                cnn_layer_name="layer2",
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


class ResNetLayer2Controller(runner.ResNetController):
    """Common controller plus a safe timing fallback for a new untested layer."""

    def effective_starts(self) -> tuple[int, float]:
        existing = self.store.get_state("greedy_setup")
        if existing is not None:
            raw_t50 = existing.get("anchor_t50_seconds")
            return int(existing["effective_starts"]), (
                float(raw_t50) if raw_t50 is not None else math.nan
            )

        durations: list[float] = []
        timing_source = "successful Layer2 anchor control"
        for candidate in engine.anchor_candidates():
            row = self.store.evaluation_for_key(candidate.candidate_key)
            if row is not None and row["status"] == "PASS":
                durations.append(float(row["elapsed_seconds"]))

        # The base engine normally obtains timing only from anchors.  Layer2 has
        # no historical subset control.  After the singleton sweep has run, use
        # successful singleton timing if necessary; it changes no ranking.
        if not durations:
            timing_source = "successful Layer2 singleton evaluation"
            for channel in range(NUM_CHANNELS):
                row = self.store.evaluation_for_key(engine.candidate_key((channel,)))
                if row is not None and row["status"] == "PASS":
                    durations.append(float(row["elapsed_seconds"]))

        if durations:
            t50: float | None = float(np.median(durations))
            if self.args.starts == "auto":
                starts = 4 if t50 <= 60 else (3 if t50 <= 100 else 2)
                rule = f"auto from {timing_source}: t50<=60:4; 60<t50<=100:3; t50>100:2"
            else:
                starts = int(self.args.starts)
                rule = f"manual --starts {starts}; timing from {timing_source}"
        else:
            # Required when gray, all128 and every singleton fail.  The greedy
            # method must still be able to run exhaustive pair rescue; use the
            # maximum coverage setting rather than silently abandoning it.
            t50 = None
            starts = 4 if self.args.starts == "auto" else int(self.args.starts)
            rule = (
                "no PASS anchor or singleton; auto provisionally selects 4 pair starts "
                "after exhaustive pair rescue"
                if self.args.starts == "auto"
                else f"manual --starts {starts}; no PASS anchor/singleton timing"
            )
        self.store.set_state(
            "greedy_setup",
            {
                "protocol": PROTOCOL,
                "effective_starts": starts,
                "anchor_t50_seconds": t50,
                "selection_rule": rule,
                "timing_source": timing_source if durations else "no PASS timing source",
            },
        )
        self.console.say(
            f"[BUDGET] timing="
            f"{'n/a' if t50 is None else f't50={t50:.1f}s from {timing_source}'} -> "
            f"{starts} starts; normal direct={engine.theoretical_budget(starts, 1)['direct']}, "
            f"random={engine.theoretical_budget(starts, 1)['random']}, "
            f"swap<={engine.theoretical_budget(starts, 1)['swap_max']}. "
            "If fewer than the planned singleton seeds PASS, a K=2 sweep augments "
            "the missing starts."
        )
        return starts, math.nan if t50 is None else t50

    def write_final_summary(self, candidates=None) -> None:
        super().write_final_summary(candidates)
        summary_path = self.args.output_dir / "summary.md"
        if summary_path.is_file():
            text = summary_path.read_text(encoding="utf-8")
            text = text.replace(
                "# ResNet Conv1 direct full-sequence greedy search",
                "# ResNet Layer2 direct full-sequence greedy search",
                1,
            )
            text = text.replace(
                "All new candidates were ranked only by full-sequence tracking; correlation clustering is retained only as a historical anchor.",
                "All new candidates were ranked only by full-sequence tracking; Layer2 correlation outputs are not used as a seed, representative constraint or ranking signal.",
                1,
            )
            summary_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Search 1--6 ResNet-18 Layer2 post-ReLU channels using the direct full-sequence "
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
        help="Full-sequence greedy seeds; auto selects 2--4 paths from observed timing.",
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_protocol(args: argparse.Namespace, timestamps: Sequence[float]) -> None:
    payload = {
        "protocol": PROTOCOL,
        "adapter_source": str(STEP_N_RUNNER),
        "adapter_sha256": sha256(STEP_N_RUNNER),
        "shared_engine_source": str(runner.STEP_J_RUNNER),
        "shared_engine_sha256": sha256(runner.STEP_J_RUNNER),
        "dataset_dir": str(args.dataset_dir),
        "matched_frames": len(timestamps),
        "first_timestamp": timestamps[0],
        "last_timestamp": timestamps[-1],
        "feature_space": {
            "network": "ImageNet ResNet-18",
            "layer": "layer2",
            "activation_position": "post-ReLU",
            "native_resolution": "H/8 x W/8 before COMO's established x8 tracking upsample",
            "available_channels": NUM_CHANNELS,
            "channel_names": [f"d{index}" for index in range(NUM_CHANNELS)],
        },
        "tracking": {
            "color": "cnn",
            "cnn_layer_name": "layer2",
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
        "anchor_policy": (
            "gray and all128 only; no prior Layer2 subset exists. If both controls and all "
            "singletons fail, use the documented provisional four-start pair-rescue path."
        ),
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
        "anchors": [dataclasses.asdict(candidate) for candidate in engine.anchor_candidates()],
    }
    path = args.output_dir / "candidate_plan.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("protocol") != PROTOCOL:
            raise RuntimeError(
                "Output directory contains a different protocol; choose a new directory "
                "rather than mixing experiments."
            )
        for key in (
            "dataset_dir",
            "random_seed",
            "adapter_sha256",
            "shared_engine_sha256",
        ):
            if existing.get(key) != payload[key]:
                raise RuntimeError(f"Output directory has a different {key}.")
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_dry_run(console) -> None:
    console.say("[DRY RUN] No COMO process is launched without --execute.")
    console.say("[DRY RUN] Anchors: gray and ResNet Layer2 all128 only.")
    console.say(
        f"[DRY RUN] Greedy: all {NUM_CHANNELS} singleton channels, then every selected "
        "path extends through K=6 even after a temporary ATE degradation."
    )
    console.say(
        "[DRY RUN] Normal pre-repeat upper bounds: "
        + "; ".join(
            f"{starts} starts={engine.theoretical_budget(starts, 1)['pre_repeat_total_max']}"
            for starts in (4, 3, 2)
        )
        + "."
    )
    console.say(
        f"[DRY RUN] If singleton seeds are missing: exhaustively score all "
        f"{math.comb(NUM_CHANNELS, 2)} pairs. Pair-rescue pre-repeat bounds: "
        + "; ".join(
            f"{starts} starts={engine.theoretical_budget(starts, 2)['pre_repeat_total_max']}"
            for starts in (4, 3, 2)
        )
        + "."
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
    controller = ResNetLayer2Controller(args, store, console, timestamps)
    try:
        console.say("=" * 78)
        console.say("RESNET LAYER2 DIRECT FULL-SEQUENCE ATE GREEDY SEARCH")
        console.say("=" * 78)
        console.say(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
        console.say(f"Requested stage: {args.stage}")
        console.say(
            f"Dataset: {args.dataset_dir}; matched frames={len(timestamps)}; "
            f"time={timestamps[0]:.6f}--{timestamps[-1]:.6f}"
        )
        console.say(
            "Primary rank: historical keyframe evo_ape ATE mean with alignment and scale "
            "correction, identical to the completed Conv1 and UNet greedy searches."
        )
        console.say(
            "Tracking uses post-ReLU ResNet Layer2 channels (128 total); mapping stays gray "
            "with sensor depth. Correlation clustering is not a seed or ranking constraint."
        )
        console.say(
            f"Timeout={args.timeout_seconds:.1f}s; batch limit="
            f"{'disabled' if args.batch_hours == 0 else f'{args.batch_hours:.2f}h'}."
        )
        if not args.execute:
            print_dry_run(console)
            controller.export()
            return

        controller.guard = ResNetLayer2ConfigGuard(
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
