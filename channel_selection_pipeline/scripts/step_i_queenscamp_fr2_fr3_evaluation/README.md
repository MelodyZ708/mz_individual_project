# fr2/fr3 QueensCAMP degradation robustness

This protocol evaluates the frozen Top-7 Conv1 configurations plus the gray baseline on seven QueensCAMP degradations of `fr2/desk` and seven of `fr3/long_office_household`.

- 14 datasets × 8 configurations × 3 independent repetitions = **336 runs**.
- Each run has a **500 s** timeout.
- The primary metric is the historical comparable keyframe `evo_ape` ATE mean using `--align --correct_scale`; all-frame SE(3)/RPE values remain diagnostics.
- Results are resumable per `(dataset, configuration, replicate)` in SQLite and written under `channel_selection_results/step_i_queenscamp_fr2_fr3_evaluation/three_repeats/`.
- The aggregator reports per-dataset and per-family tables. It never pools raw ATE across fr2 and fr3; cross-degradation comparisons use the per-dataset ATE ratio to `[5,29,40,52]`.

Validate only (no COMO process):

```bash
cd /home/melody/code/individual_project
./channel_selection_pipeline/scripts/step_i_queenscamp_fr2_fr3_evaluation/run_queenscamp_fr2_fr3_3x_evaluation.sh
```

Run or resume:

```bash
cd /home/melody/code/individual_project
./channel_selection_pipeline/scripts/step_i_queenscamp_fr2_fr3_evaluation/run_queenscamp_fr2_fr3_3x_evaluation.sh --execute
```

Rebuild reports without a tracking run:

```bash
./channel_selection_pipeline/scripts/step_i_queenscamp_fr2_fr3_evaluation/run_queenscamp_fr2_fr3_3x_evaluation.sh --aggregate-only
```
