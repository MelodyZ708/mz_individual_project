# r=0.70 Conv1 fail-fast brute-force evaluation

Main script:

```text
run_r070_bruteforce.py
```

It evaluates all 55,554 legal four-channel combinations from the final,
bootstrap-refined r=0.70 clusters. A combination is legal only when its four
representatives come from four different final clusters.

The script does **not** launch COMO unless `--execute` is explicitly present.

## Recommended run order

Run commands from `/home/melody/code/individual_project`.

### 1. Inspect the plan (no tracking run)

```bash
/home/melody/anaconda3/envs/como/bin/python \
  channel_selection_pipeline/scripts/step_d_fail_fast_evaluation/run_r070_bruteforce.py \
  --stage summary
```

### 2. Run the regression gate only

```bash
/home/melody/anaconda3/envs/como/bin/python \
  channel_selection_pipeline/scripts/step_d_fail_fast_evaluation/run_r070_bruteforce.py \
  --stage regression \
  --execute
```

Expected behaviour:

- gray: fail-fast near MVS 13--14;
- CNN `[5,29,40,52]`: complete, export an all-frame trajectory, and report
  SE(3) translation/rotation metrics plus tracking diagnostics.

Do not start the exhaustive stage if this gate reports `MISMATCH`.

### 3. Measure a 200-combination runtime pilot

```bash
/home/melody/anaconda3/envs/como/bin/python \
  channel_selection_pipeline/scripts/step_d_fail_fast_evaluation/run_r070_bruteforce.py \
  --stage bruteforce \
  --limit 200 \
  --execute
```

The final progress line gives the measured PASS/FAIL split, recent mean runtime,
and ETA. These 200 results remain cached and are skipped during the full run.

### 4. Run the complete pipeline

```bash
/home/melody/anaconda3/envs/como/bin/python \
  channel_selection_pipeline/scripts/step_d_fail_fast_evaluation/run_r070_bruteforce.py \
  --stage all \
  --execute
```

`all` performs:

1. regression gate;
2. exhaustive r=0.70 search;
3. Top-20 plus cluster-coverage single-member swap-back;
4. Top-5 factorial swap-back;
5. r=0.80 rescue insertions for `[12,19,22,28,42,44,57,58]`;
6. two fresh repeats of the final Top-20, giving three total observations.

Every stage is resumable. Re-running the command skips recorded results.
SQLite uses `synchronous=FULL`, so a completed per-candidate commit is made
durable before the next candidate starts, including across a hard reset.

## Fail-fast interpretation

The current run stops immediately when one of these is observed:

- `[KF aff received]` contains `nan` or `inf`;
- tracking diagnostics contain a non-finite pose, affine, residual scale, or
  pose update;
- the known empty-AABB / `RuntimeError: Caught an unknown exception!` signature;
- wall-clock timeout;
- post-run trajectory is missing, non-finite, frozen, or ends before the clear
  dimming phase.

`Crazy affine detected` alone is a warning and does not fail a candidate.
Finite `chol_ok=False` and low-valid-ratio diagnostics are also recorded rather
than failed by default, because a later frame can recover. An optional persistent
gate can be enabled explicitly with `--diagnostic-failure-streak N`.
An isolated non-finite `photo_mse` is counted but does not fail a run when the
pose, residuals, update, and Cholesky result remain valid; this can occur when
the robust residual scale approaches zero.

Recognised tracking failures are ranked as failed candidates, not assigned a
finite ATE. Three consecutive infrastructure errors stop the whole search so a
broken environment cannot silently mark thousands of combinations as bad.

## Trajectory and accuracy definitions

- warm-up MVS 0--9 is excluded;
- MVS 10--49 is scored;
- `data_tum_all_frames.txt` contains every pose delivered by Tracking and is the
  primary evaluation trajectory; the historical `data_tum.txt` keyframe file is
  retained for compatibility;
- all 40 expected all-frame poses must associate to ground truth in the scored
  window;
- because this is RGB-D, translation ATE uses metric-scale SE(3) alignment and
  does not fit a free scale;
- translation SE(3) ATE RMSE is the primary rank;
- rotation APE, translation RPE, and rotation RPE expose rotational and
  single-step jumps that translation ATE alone can miss;
- the historical keyframe-trajectory Sim(3) mean/RMSE and fitted scale are
  computed separately and retained only for comparison with older searches.

Every successful run also reports the median and 95th-percentile tracking
photometric MSE, minimum valid projection ratio, maximum Hessian condition
number, and number of crazy-affine warnings. Raw photometric error is a
diagnostic rather than the primary rank because different CNN channels need not
have identical activation scales.

## RPE safety and final multi-metric selection

The main `search_ranking.csv` remains an auditable all-frame SE(3) ATE ranking.
Each PASS row is additionally marked when translation RPE max exceeds 6 cm or
rotation RPE max exceeds 5 degrees. Marked candidates stay visible in the main
ranking but cannot seed swap-back, rescue, or final repeats.

`multimetric_top20.csv` explains the actual finalist set. Its 20 slots use:

- 50% top RPE-safe ATE;
- 25% RPE-safe ATE/translation-RPE/rotation-RPE Pareto candidates;
- the remaining slots for especially low translation or rotation RPE;
- safe ATE order to fill any unused slots.

The same selector supplies the Top-20 swap-back contexts, Top-5 factorial
contexts, r=0.80 rescue contexts, and final repeat candidates.

All terminal ATE values are printed in centimetres. SQLite stores metres.

## Compatibility with full-sequence evaluation

The evaluator enables the extra diagnostics and all-frame export only for its
own child COMO process. It does this through temporary tracking configuration
keys and `COMO_SAVE_ALL_FRAME_TRAJECTORY=1`; the original config is restored
after every candidate. Ordinary full-sequence commands retain the historical
behaviour and continue to write the same keyframe-only `results/data_tum.txt`,
so existing `evo_ape --align --correct_scale` workflows are unaffected.

## Results

Default output directory:

```text
channel_selection_results/step_d_fail_fast_evaluation/r070_bruteforce_v2/
```

Important files:

- `evaluations.sqlite3`: authoritative resumable database;
- `all_evaluations.csv`: every run and failure reason;
- `search_ranking.csv`: best single search observation per unique combination;
- `multimetric_top20.csv`: RPE-safe final candidates and explicit selection
  reasons;
- `final_repeat_summary.csv`: Top-20 pass rate and ATE mean/std across repeats;
- `search_console.log`: explanatory master log;
- `artifacts/`: retained regression/repeat trajectories and diagnostic logs.

To avoid tens of thousands of tiny files, brute-force trajectories and full
logs are not retained by default. Their metrics and the final 80 log lines are
stored in SQLite. Add `--keep-all-logs` or `--keep-all-trajectories` only when
the extra storage is genuinely needed.

## Safety and interruption

- Run only one copy on a GPU.
- A graphical `DISPLAY` must be available because this COMO entry point creates
  the Open3D window. Xvfb is not installed on the current machine.
- The script locks `como/config/como.yml`, writes each candidate atomically, and
  restores the exact original bytes after every run.
- It removes only the known stale `como/results/data_tum.txt` and
  `como/results/data_tum_all_frames.txt` immediately before a new evaluation.
- `Ctrl+C` terminates the active COMO process group and restores the config.
- Do not manually edit `como.yml` while the search owns the lock.

After an abnormal machine shutdown, use the recovery launcher below. It checks
and snapshots the SQLite database (including committed WAL rows), repairs a
trailing NUL-only master-log fragment, restores the pre-search COMO settings,
and resumes only missing brute-force candidates without `--rerun-existing`.
The launcher runs a seven-hour batch by default, then exits cleanly so RAS
errors can be checked before the next batch:

```bash
channel_selection_pipeline/scripts/step_d_fail_fast_evaluation/\
resume_r070_bruteforce_after_crash.sh
```

After each batch, run `sudo ras-mc-ctl --errors`. If all categories remain
clear, launch the same script for the next seven-hour batch. The batch duration
can be overridden, for example with
`BATCH_HOURS=5 ./.../resume_r070_bruteforce_after_crash.sh`.

Useful options:

```bash
...run_r070_bruteforce.py --help
```

The default timeout is 120 seconds per candidate. Change it only after the
runtime pilot demonstrates that a different limit is justified.
