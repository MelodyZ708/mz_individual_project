# QueensCAMP degradation robustness evaluation

This experiment evaluates the six selected channel combinations, the historical
four-channel CNN baseline, and the grayscale control on seven QueensCAMP-style
degradations. Each dataset/configuration cell is repeated three times.

Protocol: 7 datasets × 8 configurations × 3 repeats = 168 runs. The primary
accuracy metric is the historical keyframe `evo_ape` ATE mean with `--align
--correct_scale`; full-frame SE(3) ATE/RPE, keyframe RPE, coverage and failure
locations remain diagnostics. Means and standard deviations use PASS runs only,
while every summary also reports PASS count out of three.

Validate without launching COMO:

```bash
cd /home/melody/code/individual_project
./channel_selection_pipeline/scripts/step_h_queenscamp_degradation_evaluation/run_queenscamp_3x_evaluation.sh
```

Run or resume:

```bash
cd /home/melody/code/individual_project
./channel_selection_pipeline/scripts/step_h_queenscamp_degradation_evaluation/run_queenscamp_3x_evaluation.sh --execute
```

Rebuild only the aggregate tables and plots:

```bash
./channel_selection_pipeline/scripts/step_h_queenscamp_degradation_evaluation/run_queenscamp_3x_evaluation.sh --aggregate-only
```

Results are isolated under
`channel_selection_results/step_h_queenscamp_degradation_evaluation/three_repeats`.
The execute command is resumable: saved `(configuration, replicate)` rows are
skipped, and existing databases are backed up before each launch.
