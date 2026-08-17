# Step J: UNet enc1 direct full-sequence greedy search

This is a new, independent experiment. It searches the 32 channels of UNet
encoder level 1 on the complete rgbd_dataset_freiburg1_desk_lightswitch
sequence. It never constructs or scores an MVS proxy.

## Fixed protocol

- Tracking: color: unet and unet_enc_level: 1.
- Mapping remains color: gray and keeps use_sensor_depth: true.
- Every candidate uses the full 573-frame matched sequence.
- Main metric: the same historical keyframe evo_ape ATE mean as
  run_random_channel_search.sh, with alignment and scale correction.
- Per-run timeout: 300 seconds.
- A non-finite tracking diagnostic, known runtime error, timeout, incomplete
  trajectory, or frozen trajectory is a failed run.

The search records all cardinalities 1--6. It does **not** stop adding channels
after a temporary metric degradation: later additions can create a useful
interaction. The final adaptive choice is selected only after three total runs
for each greedy endpoint. The K=4 greedy endpoint is separately retained as
the controlled comparison with the ResNet four-channel result.

## Stages

1. anchors: gray, UNet-all32, and the historical BQS Top-4/Top-5 controls.
2. greedy: all 32 singleton runs, followed by 2--6 channel forward greedy
   paths from the best 2--4 singleton seeds. Anchor median runtime chooses 4
   seeds at 60s or faster, 3 at 60--100s, and 2 above 100s.
3. random: an equal number of uniform random candidates at each K=2--6,
   using the frozen seed 20260814.
4. swap: all one-channel replacements around the best single-run greedy
   endpoint, at most 156 neighbours.
5. repeats: brings G1--G6, G4, random controls, swap result, and anchors to
   three observations when applicable.

Every stage is resumable. A saved row is reused by canonical channel set and
replicate, including when two greedy paths merge.

If **all 32 singletons fail**, the script automatically performs a full
two-channel rescue sweep: every C(32,2)=496 pair is evaluated on the same
full sequence, the best PASS pairs become the greedy seeds, and those paths
continue through K=6. This is still direct ATE search, not a proxy. Because
the K=2 space is then exhaustive, no disjoint random pair exists; the random
control automatically records K=2 as exhaustive and remains budget-matched at
K=3--6.

## Commands

From /home/melody/code/individual_project, inspect the protocol without
starting COMO:

    ./channel_selection_pipeline/scripts/step_j_unet_feature_greedy_search/run_unet_direct_fullseq_greedy.sh

Run the complete search:

    ./channel_selection_pipeline/scripts/step_j_unet_feature_greedy_search/run_unet_direct_fullseq_greedy.sh --stage all --execute

The standard per-invocation limit is 36 hours. If it stops at that clean
boundary, rerun the same command and it will continue from SQLite. To disable
that overall limit intentionally:

    ./channel_selection_pipeline/scripts/step_j_unet_feature_greedy_search/run_unet_direct_fullseq_greedy.sh --stage all --execute --batch-hours 0

Stages can also be run separately; for example:

    ./channel_selection_pipeline/scripts/step_j_unet_feature_greedy_search/run_unet_direct_fullseq_greedy.sh --stage anchors --execute

## Result directory

channel_selection_results/step_j_unet_direct_fullseq_greedy/ contains:

- evaluations.sqlite3: authoritative resumable evaluator records;
- candidate_plan.json: frozen high-level protocol and budget rule;
- candidate_registry.csv: every stage request, including cache reuse;
- direct_greedy_path.csv: all selected G1--G6 paths;
- random_budget_matched_plan.json: reproducible stratified random sample;
- one_swap_plan.json: frozen local-neighbour audit;
- final_repeat_candidate_plan.json, final_repeats_summary.csv, and
  recommendation.json: repeat validation and final adaptive/G4 result;
- all_evaluations.csv, pass_ranking.csv, artifacts/, and console.log:
  raw Step-E-compatible evidence.
