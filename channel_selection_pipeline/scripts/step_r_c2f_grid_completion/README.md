# Step-R: reduced complete C2F grid across nine sequences

This stage expands the earlier focused Step-Q comparison into a systematic
pairing experiment, while preserving every Step-Q observation unchanged.

## Requested grids

| Architecture | Fine subsets | Coarse subsets | C2F pairs | Direct parents |
|---|---:|---:|---:|---:|
| U-Net | top 5 Enc0 | top 4 Enc1 | `5 × 4 × A/B = 40` | 5 + 4 = 9 |
| ResNet | top 6 Conv1 | top 5 Layer2 | `6 × 5 × A/B = 60` | 6 + 5 = 11 |

The full selected grid is evaluated over fr1/fr2/fr3 × clean/lightswitch/
flashlight (nine complete sequences).  Gray is deliberately omitted.

## Reuse, not rerun

Step-Q already contains four C2F cells and five direct parents per
architecture on all nine datasets.  Those rows are checked in read-only mode
and reused in the aggregate; they are never copied into Step-R or rerun.

| Architecture | Step-Q C2F reused | Step-Q directs reused | New C2F | New directs | New evaluations |
|---|---:|---:|---:|---:|---:|
| U-Net | 4 | 5 | 36 | 4 | 360 |
| ResNet | 4 | 5 | 56 | 6 | 558 |

Thus Step-R runs **918** new evaluations.  Using the observed timings from
Step-Q, the expected serial duration is about **30.4 hours**; reserve
**32–35 hours** for process start-up and any timeout outliers.

## Launch

```bash
cd /home/melody/code/individual_project

# Non-mutating validation (checks all Step-Q rows needed for reuse).
./channel_selection_pipeline/scripts/step_r_c2f_grid_completion/run_c2f_grid_completion.sh unet
./channel_selection_pipeline/scripts/step_r_c2f_grid_completion/run_c2f_grid_completion.sh resnet

# Execute serially. Re-running resumes automatically.
./channel_selection_pipeline/scripts/step_r_c2f_grid_completion/run_c2f_grid_completion.sh unet --execute
./channel_selection_pipeline/scripts/step_r_c2f_grid_completion/run_c2f_grid_completion.sh resnet --execute
```

Each architecture and dataset has a separate Step-R SQLite database.  A
stored PASS or failure is skipped on rerun.  The shell launcher snapshots only
the Step-R databases, retains the established CUDA/CPU-affinity/Intel-Turbo
checks, and restores `como/config/como.yml` after any exit.

`merged_pairwise_comparison.csv` is the final evidence file: it combines the
read-only Step-Q observations with new Step-R cells and compares every C2F
pair with its two direct parents on the same sequence.
