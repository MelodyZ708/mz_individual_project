#!/usr/bin/env python3
"""Run the Step-J direct full-sequence protocol on UNet encoder level 0.

This wrapper deliberately reuses the tested search engine from the enc1
experiment.  It changes only the feature-space constants and the BQS reference
controls: enc0 has 16 full-resolution channels rather than 32 H/2 channels.
All process control, failure handling, ATE scoring, fallback, random-control,
one-swap and repeat logic therefore remains identical to the enc1 protocol.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
ENGINE_PATH = (
    PROJECT_ROOT
    / "channel_selection_pipeline/scripts/step_j_unet_feature_greedy_search/"
    "run_unet_direct_fullseq_greedy.py"
)


def load_engine():
    spec = importlib.util.spec_from_file_location("unet_direct_greedy_engine", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import shared UNet greedy engine: {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = load_engine()
engine.DEFAULT_OUTPUT = (
    PROJECT_ROOT / "channel_selection_results/step_k_unet_enc0_direct_fullseq_greedy"
)
engine.NUM_CHANNELS = 16
engine.UNET_ENC_LEVEL = 0
engine.ENCODER_LABEL = "enc0"
engine.ALL_CHANNEL_ANCHOR_LABEL = "unet_enc0_all16"
engine.PROTOCOL = "unet_enc0_direct_full_sequence_greedy_v1"
engine.BQS_REFERENCE_CANDIDATES = (
    (
        "bqs_historical_top3",
        (0, 10, 15),
        "0621 BQS-ranked enc0 Top-3 [d15,d10,d0], re-evaluated on this protocol",
        "B3_historical_bqs_top3",
    ),
    (
        "bqs_historical_top5",
        (0, 3, 10, 14, 15),
        "0621 BQS-ranked enc0 Top-5 [d15,d10,d0,d14,d3], direct ATE control",
        "B5_historical_bqs_top5",
    ),
)


if __name__ == "__main__":
    engine.main()
