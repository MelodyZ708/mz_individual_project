# Top-7 independent repeat validation

This package independently reruns the six channel combinations that beat the
historical baseline in the second-stage full-sequence search, plus the baseline
itself.  It then compares the new ATE values with the original run, visualises
the four Conv1 post-ReLU maps, and maps every selected channel back to the final
global `r=0.70` correlation cluster.

The seven frozen combinations are:

1. `[5,6,24,29]`
2. `[1,26,30,40]`
3. `[15,17,52,59]`
4. `[1,5,24,29]`
5. `[5,6,15,35]`
6. `[6,10,34,41]`
7. baseline `[5,29,40,52]`

The repeat uses the complete 573-frame `fr1/desk_lightswitch` sequence and a
300-second timeout.  Its primary metric is exactly the historical metric used
by `run_random_channel_search.sh`: keyframe
`evo_ape tum ... --align --correct_scale` translation ATE mean.

Feature-map frames are the paired samples at original indices 246, 250 and 254,
representing before, peak and after around the principal turn-on event.  The
feature files are the existing native-resolution ResNet-18 Conv1 post-ReLU
archives used for correlation clustering.  Per-combination overview figures
show all four lightswitch maps over the three frames.  Detailed figures also
show the matched clean map and absolute clean/light difference.

From the project root, inspect the frozen run plan:

```bash
./channel_selection_pipeline/scripts/step_e_full_sequence_evaluation/top7_repeat_validation/run_top7_repeat_validation.sh
```

Launch the seven new runs and all analyses:

```bash
./channel_selection_pipeline/scripts/step_e_full_sequence_evaluation/top7_repeat_validation/run_top7_repeat_validation.sh \
  --execute
```

The run is resumable: repeating the same command skips saved candidates and
rebuilds the derived analysis.  Results are stored at:

```text
channel_selection_results/step_e_full_sequence_evaluation/
  top7_repeat_feature_cluster_analysis/
```

Read `summary.md` first.  The authoritative new results remain in
`evaluations.sqlite3`; `repeat_comparison.csv` contains the original value, new
value, delta and two-run mean for each combination.
