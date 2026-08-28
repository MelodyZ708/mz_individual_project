# Multi-sequence C2F parent comparison

This stage answers a specific question: **when does C2F help relative to the
direct single-branch configurations from which it was built?**  It is not a
new channel search.

## Scope

- Nine full sequences: fr1/fr2/fr3 × clean/lightswitch/flashlight.
- One run per configuration per sequence, with a five-minute timeout and the
  established 90% trajectory-completeness gate.
- Mapping remains grayscale with sensor/GT depth.  Tracking alone receives the
  selected direct ResNet/U-Net features or the C2F feature pair.
- Primary metric remains historical keyframe translation ATE mean from
  `evo_ape tum --align --correct_scale`.

The candidate plans are deliberately compact and parent-complete:

- **U-Net:** 10 configurations × 9 datasets = 90 cells.
- **ResNet:** 10 configurations × 9 datasets = 90 cells.

Each plan includes:

1. gray baseline;
2. every direct fine/coarse parent required by a focused C2F case;
3. a C2F positive-synergy case;
4. the best C2F result on fr1/desk_lightswitch; and
5. a C2F case that is worse than an already high-ranked direct parent.

This makes the final table evidence for both the benefit and the limitation of
C2F, rather than cherry-picking only gains.

## C2F variants

| Variant | Coarse representation | Fine representation |
|---|---|---|
| A | L0, L1 | L2 |
| B | L0 | L1, L2 |

## Launch

```bash
cd /home/melody/code/individual_project

# Non-mutating validation first.
./channel_selection_pipeline/scripts/step_q_c2f_multi_dataset_evaluation/run_c2f_multi_dataset_evaluation.sh unet
./channel_selection_pipeline/scripts/step_q_c2f_multi_dataset_evaluation/run_c2f_multi_dataset_evaluation.sh resnet

# Execute each architecture serially. Re-running automatically resumes.
./channel_selection_pipeline/scripts/step_q_c2f_multi_dataset_evaluation/run_c2f_multi_dataset_evaluation.sh unet --execute
./channel_selection_pipeline/scripts/step_q_c2f_multi_dataset_evaluation/run_c2f_multi_dataset_evaluation.sh resnet --execute
```

The launcher has the same CUDA/CPU-affinity/Intel-Turbo checks as the earlier
evaluators, snapshots the shared COMO configuration and every existing SQLite
database, restores the config even on interruption, and locks the full C2F
run so U-Net and ResNet cannot be launched concurrently.

## Resumption and output

Each architecture and each dataset has a separate SQLite database. A saved
label/replicate row is skipped on the next launch, whether that row is a PASS
or a recorded failure. Interrupted runs therefore resume at the first missing
cell without overwriting prior evidence.

The two most important files are:

- `c2f_pairwise_comparison.csv`: each C2F configuration beside both direct
  parents on the same dataset, including absolute and percent deltas.
- `c2f_effect_summary.csv`: number of comparable sequences, number of wins
  against the fine parent and against the better direct parent, plus median
  per-sequence deltas.

`ate_mean_matrix.csv` is the wide presentation-oriented table. Raw values are
not averaged across fr1/fr2/fr3, since their trajectory scale and difficulty
differ; all C2F conclusions come from within-dataset parent comparisons.
