# Full-sequence evaluation

This stage validates the final Conv1 shortlist on the complete
`rgbd_dataset_freiburg1_desk_lightswitch` sequence (573 matched RGB-D
timestamps). It is intentionally separate from the 50-frame MVS search.

## Candidate set

The nine configurations cover:

- gray and historical `[5,29,40,52]` controls;
- the first two RPE-safe MVS ATE candidates;
- the balanced MVS candidate;
- the r=0.80 rescue candidate;
- moderate-, translation-, and rotation-jump stability candidates.

The exact labels, channels, and roles are written to `evaluation_plan.json`.

## Metrics

Primary ranking (identical to `run_random_channel_search.sh`):

- use COMO's keyframe trajectory `results/data_tum.txt`;
- run `evo_ape tum groundtruth.txt data_tum.txt --align --correct_scale`;
- rank by the reported translation ATE **mean**.

Additional diagnostics:

- SE(3) ATE mean/median/max;
- rotation APE;
- translation and rotation RPE RMSE/max;
- trajectory coverage and completion;
- all-frame Sim(3) metrics;
- all-frame and independently calculated keyframe Sim(3) diagnostics;
- photometric and numerical tracking diagnostics.

A run is not a valid PASS unless at least 90% of the 573 matched timestamps
associate to ground truth and the trajectory reaches the end of the sequence.
Non-finite tracking diagnostics and recognised runtime failures stop that run
immediately. A failed control does not prevent later candidates from running.
The default wall-clock timeout is 300 seconds (5 minutes) per configuration.

## Commands

From `/home/melody/code/individual_project`, inspect the plan without launching
COMO:

```bash
./channel_selection_pipeline/scripts/step_e_full_sequence_evaluation/run_full_sequence_evaluation.sh
```

Run all nine configurations once:

```bash
./channel_selection_pipeline/scripts/step_e_full_sequence_evaluation/run_full_sequence_evaluation.sh \
  --execute
```

The evaluation is resumable. Re-running the same command skips saved
label/replicate pairs. To run a small control gate first:

```bash
./channel_selection_pipeline/scripts/step_e_full_sequence_evaluation/run_full_sequence_evaluation.sh \
  --only gray_baseline known_cnn_baseline \
  --execute
```

Then run the complete command; the two controls will be skipped.

Recompute the exact historical `evo_ape` metric and ranking from trajectories
that are already saved, without rerunning COMO:

```bash
./channel_selection_pipeline/scripts/step_e_full_sequence_evaluation/run_full_sequence_evaluation.sh \
  --refresh-historical-metrics
```

## Safety and outputs

The launcher requires Turbo to be disabled, excludes logical CPUs
`2,3,10,11`, checks the current boot for MCE/GPU events, verifies GPU access,
and snapshots an existing SQLite database before launch. The evaluator holds
the same COMO configuration lock used by the MVS search and restores the
original configuration after every run.

Default output directory:

```text
channel_selection_results/step_e_full_sequence_evaluation/fr1_desk_lightswitch/
```

Important files:

- `evaluations.sqlite3`: authoritative resumable results;
- `all_evaluations.csv`: all PASS/failure records;
- `pass_ranking.csv`: historical `evo_ape` ATE-mean ranking, with all-frame
  SE(3) and RPE diagnostics retained in later columns;
- `evaluation_plan.json`: frozen protocol and candidate list;
- `console.log`: explanatory run log;
- `artifacts/<label>/`: raw logs and all-frame/keyframe trajectories.

## Second round: 3,713 MVS-qualified combinations

The frozen second-round population applies all four agreed gates to the
authoritative first-round `bruteforce`, replicate-0 rows:

- status is `PASS` with all 40 scored MVS poses;
- MVS SE(3) ATE RMSE is no more than 2% above `[5,29,40,52]`, giving a
  15.1091 cm cutoff;
- translation RPE max is at most 6 cm;
- rotation RPE max is at most 5 degrees.

This produces exactly 3,713 combinations, including the known baseline. The
candidate population is frozen in `second_round_baseline_plus2_rpe_safe/`
under the results directory as both JSON and CSV. To run the complete second
round without an overall wall-time limit:

```bash
cd /home/melody/code/individual_project

./channel_selection_pipeline/scripts/step_e_full_sequence_evaluation/run_second_round_3713.sh \
  --execute
```

Completed database rows are skipped, so after a manual stop or unexpected
interruption the same command resumes from the first unevaluated combination.
Every invocation verifies the frozen plan, backs up the database and COMO
configuration, and enforces the 300-second per-run timeout. There is no overall
batch wall-time limit.
