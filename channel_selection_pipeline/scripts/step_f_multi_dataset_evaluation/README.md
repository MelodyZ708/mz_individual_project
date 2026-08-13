# Step F — Top-7 multi-dataset evaluation

This stage evaluates the six `fr1/desk_lightswitch` winners and the rank-7
historical baseline once on each of eight new sequences.  The source-selection
sequence `fr1/desk_lightswitch` is deliberately not rerun or included in the
Step-F aggregate generalisation score.

## Frozen datasets

| Key | Directory | Matched frames |
|---|---|---:|
| `fr1_desk_clean` | `rgbd_dataset_freiburg1_desk` | 573 |
| `fr1_desk_flashlight` | `rgbd_dataset_freiburg1_desk_flashlight` | 573 |
| `fr2_desk_clean` | `rgbd_dataset_freiburg2_desk` | 2,893 |
| `fr2_desk_flashlight` | `rgbd_dataset_freiburg2_desk_flashlight` | 2,893 |
| `fr2_desk_lightswitch` | `rgbd_dataset_freiburg2_desk_lightswitch` | 2,893 |
| `fr3_office_clean` | `rgbd_dataset_freiburg3_long_office_household` | 2,488 |
| `fr3_office_flashlight` | `rgbd_dataset_freiburg3_long_office_household_flashlight` | 2,488 |
| `fr3_office_lightswitch` | `rgbd_dataset_freiburg3_long_office_household_lightswitch` | 2,488 |

This is 8 datasets × 7 configurations = 56 runs.  Every run has a 500-second
timeout.  The complete metric set is retained: historical keyframe `evo_ape`
ATE mean/RMSE and `evo_rpe` RMSE, all-frame metric-scale SE(3) ATE,
translation/rotation RPE, trajectory coverage/completion,
photometric/numerical diagnostics and failure location.

## Commands

Validate all plans and paths without starting COMO:

```bash
cd /home/melody/code/individual_project

./channel_selection_pipeline/scripts/step_f_multi_dataset_evaluation/run_top7_multi_dataset_evaluation.sh
```

Run or resume all evaluations:

```bash
./channel_selection_pipeline/scripts/step_f_multi_dataset_evaluation/run_top7_multi_dataset_evaluation.sh \
  --execute
```

Rebuild aggregate tables from saved databases without running COMO:

```bash
./channel_selection_pipeline/scripts/step_f_multi_dataset_evaluation/run_top7_multi_dataset_evaluation.sh \
  --aggregate-only
```

The `--execute` command is resumable.  Each dataset has an independent SQLite
database, and saved candidate rows are skipped after interruption.

## Results

All outputs are isolated under:

```text
channel_selection_results/step_f_multi_dataset_evaluation/
```

Important files:

- `per_dataset/<key>/evaluations.sqlite3`: authoritative per-dataset results;
- `per_dataset/<key>/artifacts/`: logs and all-frame/keyframe trajectories;
- `all_runs_raw.csv`: every original evaluator field plus dataset metadata;
- `dataset_scorecard.csv`: all 56 planned cells in readable units;
- `candidate_aggregate_summary.csv`: failure-aware cross-dataset summary;
- `condition_robustness.csv`: flashlight/lightswitch relative to matching clean;
- `aggregate_summary.md`: current progress and aggregate ranking;
- `plots/`: ATE, baseline-relative improvement and rank heatmaps.

Each successful run records the same historical keyframe metrics as
`run_ate_multi_seq_local.sh`: `evo_ape` ATE mean/RMSE and `evo_rpe` RMSE with
`--align --correct_scale`.  The richer all-frame SE(3) ATE/RPE, coverage and
numerical diagnostics are retained alongside them.

Absolute ATE is not averaged directly across sequences.  Aggregate ordering
first maximises the number of PASS datasets, then uses matched-dataset
ATE/baseline ratios and within-dataset ranks.  This avoids allowing a long or
high-error sequence to dominate solely through scale.
