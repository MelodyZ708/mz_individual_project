# Step K: UNet enc0 direct full-sequence greedy search

This is an independent, resumable control experiment for the 16 full-resolution
channels of the COMO U-Net encoder base output (`enc_level: 0`). It uses the
same full-sequence ATE-first protocol as Step J (enc1); only the feature level,
candidate space and BQS reference controls differ.

- Dataset: `rgbd_dataset_freiburg1_desk_lightswitch` (all 573 matched frames).
- Tracking: `color: unet`, `unet_enc_level: 0`, selected `d0`--`d15`.
- Mapping: fixed `color: gray` with sensor depth, unchanged from Step J.
- Primary rank: historical keyframe `evo_ape tum` ATE mean after alignment and
  scale correction, matching `run_random_channel_search.sh`.
- Timeout: 300 s per run; failed NaN/runtime trajectories stop immediately.
- Results: `channel_selection_results/step_k_unet_enc0_direct_fullseq_greedy/`.

## Protocol

1. Evaluate gray, enc0-all16, and the 0621 BQS-ranked enc0 Top-3
   `[d0,d10,d15]` / Top-5 `[d0,d3,d10,d14,d15]` controls.
2. Evaluate all 16 singleton channels. If fewer than the planned multi-start
   seeds pass, exhaustively evaluate all `C(16,2)=120` pairs. Retain every
   passing singleton path and fill the missing starts with the best pair seeds
   that do not duplicate the singleton route at K=2. If all singletons fail,
   the best pairs provide every seed.
3. Run the same multi-start forward greedy expansion through K=6, including a
   step even when the previous step temporarily worsened ATE.
4. Run a matched, cardinality-stratified random control; then test every
   one-channel replacement around the provisional greedy optimum.
5. Repeat selected anchors, all greedy endpoints, random references and the
   local optimum to three total observations. Keep the explicit K=4 result.

At the expected four starts, a pure singleton or pure-pair policy has a
596-evaluation pre-repeat upper bound. The active enc0 run has one passing
singleton, so its retained-singleton + three-pair augmentation policy has a
611-evaluation pre-repeat upper bound, plus at most 30 anchor/repeat
observations (about 641 total). At approximately 50 s per run this is about
8--10 hours; completed rows are reused after a restart.

## Run

```bash
cd /home/melody/code/individual_project

./channel_selection_pipeline/scripts/step_k_unet_enc0_feature_greedy_search/run_unet_enc0_direct_fullseq_greedy.sh \
  --stage all \
  --execute
```

The launcher checks that Intel Turbo is disabled, retains the same safe CPU
affinity as Step J, snapshots the current COMO config, and backs up the
independent SQLite database before every execute launch.
