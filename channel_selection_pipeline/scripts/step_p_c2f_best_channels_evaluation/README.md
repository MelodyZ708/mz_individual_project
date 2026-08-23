# C2F best-channel grid: fr1/desk_lightswitch

This stage re-enables coarse-to-fine tracking after independently finding the
best single-layer subsets.  It is deliberately a **pairing experiment**, not a
new channel search: each branch receives six subsets already supported by
three repeated PASS measurements from its own direct full-sequence greedy run.

## C2F variants

The TUM configuration uses a three-level tracking pyramid (`L0` lowest,
`L2` highest resolution).  There are exactly two valid variants:

| Variant | Coarse representation | Fine representation |
|---|---|---|
| A | L0, L1 | L2 |
| B | L0 | L1, L2 |

Variant C would place no level on the coarse branch, so the tracker rejects it
as an invalid C2F configuration.  This matches the 0713 C2F report.

## Frozen grids

Each architecture has `2 variants × 6 fine subsets × 6 coarse subsets = 72`
single-run cells:

- **ResNet:** Conv1 is fine; Layer2 is coarse.
- **U-Net:** Enc0 is fine; Enc1 is coarse.

The exact selections and their source ATE values are frozen in the corresponding
JSON plan.  The grid is a full cross-product, rather than pairing rank 1 with
rank 1 only, because independently low single-layer ATE does not establish
which coarse/fine subsets are complementary.

For U-Net, `unet_c2f` is a new tracking-only mode.  It derives Enc0 and Enc1
from one shared U-Net encoder pass for each frame, then assigns their selected
channels to the same C2F-A/B pyramid switching rule as ResNet.  It does **not**
alter mapping: mapping remains grayscale and uses sensor/ground-truth depth.

## Evaluation protocol

- Dataset: `rgbd_dataset_freiburg1_desk_lightswitch`, all matched RGB-D frames.
- One run per cell; a 500-second timeout; a full-trajectory coverage gate of
  90% plus final-timestamp completeness.
- Primary ranking: historical keyframe translation ATE mean from
  `evo_ape tum --align --correct_scale`, matching the earlier full-sequence
  channel-search reports.
- Diagnostics retained for every run: historical RPE, all-frame metric-scale
  SE(3) ATE/RPE, coverage, tracking diagnostics, raw log, and both trajectories.
- SQLite uses `synchronous=FULL`.  It is authoritative and each stored
  label/replicate pair is skipped on a later launch, including saved failures.

## Launch

```bash
cd /home/melody/code/individual_project

# Non-mutating structural validation first.
./channel_selection_pipeline/scripts/step_p_c2f_best_channels_evaluation/run_c2f_best_channel_grid.sh resnet
./channel_selection_pipeline/scripts/step_p_c2f_best_channels_evaluation/run_c2f_best_channel_grid.sh unet

# Then execute each architecture serially.  Re-running resumes automatically.
./channel_selection_pipeline/scripts/step_p_c2f_best_channels_evaluation/run_c2f_best_channel_grid.sh resnet --execute
./channel_selection_pipeline/scripts/step_p_c2f_best_channels_evaluation/run_c2f_best_channel_grid.sh unet --execute
```

The launcher prevents execution if Intel Turbo is enabled, verifies CUDA and
CPU affinity, snapshots the current COMO config and prior SQLite database, and
restores the config on normal exit or interruption.  Do not run the ResNet and
U-Net commands at the same time because both safely lock and temporarily edit
the same `como/config/como.yml`.

## Time estimate

Direct full-sequence greedy runs took roughly 43–49 seconds per configuration.
The C2F runs have two feature branches, so budget **about 1.5–2.5 hours per
72-cell architecture** (roughly 3–5 hours serially), excluding rare
five-minute timeouts.  The console log updates an empirical ETA after every
newly evaluated cell.

## Outputs

Results are kept separately:

- `channel_selection_results/step_p_c2f_best_channels_evaluation/resnet_fr1_desk_lightswitch/`
- `channel_selection_results/step_p_c2f_best_channels_evaluation/unet_fr1_desk_lightswitch/`

Each has `evaluations.sqlite3`, `all_evaluations.csv`, `pass_ranking.csv`,
`summary.md`, `console.log`, an immutable execution plan, per-cell artifacts,
and timestamped pre-launch backups.
