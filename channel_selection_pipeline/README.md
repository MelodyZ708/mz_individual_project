# Channel Selection Pipeline

This directory contains only the scripts for the channel-selection project.
Generated data, plots, and reports belong in the separate
`channel_selection_results/` directory.

## Layout

- `scripts/step_0_data_preparation/`
  - paired-frame sampling
  - native-resolution post-ReLU feature extraction
- `scripts/step_a_functional_profiling/`
  - functional descriptors, soft labels, and confirmed-dead review
- `scripts/step_b_correlation_clustering/`
  - robust channel correlations, HCA, bootstrap stability, and representatives

Later Step C--G directories can be added when those stages are implemented.

## Current command

Run from any working directory:

```bash
/home/melody/anaconda3/envs/como/bin/python \
  /home/melody/code/individual_project/channel_selection_pipeline/scripts/step_0_data_preparation/sample_paired_frames.py
```

The default output is written to:

```text
channel_selection_results/step_0_data_preparation/paired_frames/
```

Extract the paired features:

```bash
/home/melody/anaconda3/envs/como/bin/python \
  /home/melody/code/individual_project/channel_selection_pipeline/scripts/step_0_data_preparation/extract_feature_maps.py
```

Features are stored as one compressed NPZ file per timestamp under:

```text
channel_selection_results/step_0_data_preparation/features_post_relu/
```

Export all feature channels as PNG contact sheets:

```bash
/home/melody/anaconda3/envs/como/bin/python \
  /home/melody/code/individual_project/channel_selection_pipeline/scripts/step_0_data_preparation/export_feature_maps_png.py
```

All generated contact sheets are stored flat in one dedicated directory. The
sample ID and timestamp are included in each filename.

Run Step A functional profiling:

```bash
/home/melody/anaconda3/envs/como/bin/python \
  /home/melody/code/individual_project/channel_selection_pipeline/scripts/step_a_functional_profiling/functional_profiling.py
```

Step A is currently retained as an optional diagnostic only. Step B does not
read its profiles or dead-channel candidate file.

Run Step B robust correlation clustering:

```bash
/home/melody/anaconda3/envs/como/bin/python \
  /home/melody/code/individual_project/channel_selection_pipeline/scripts/step_b_correlation_clustering/correlation_clustering.py
```

The primary partition uses average-linkage HCA with robust `|r| >= 0.90`.
Numerical matrices are written as compressed NPZ and CSV files. Every 2-D NPZ
matrix also has a PNG visualisation under `npz_matrix_png/`.

The current threshold comparison additionally uses:

```bash
/home/melody/anaconda3/envs/como/bin/python \
  channel_selection_pipeline/scripts/step_b_correlation_clustering/correlation_clustering.py \
  --correlation-threshold 0.80 \
  --output-dir channel_selection_results/step_b_correlation_clustering/threshold_r080

/home/melody/anaconda3/envs/como/bin/python \
  channel_selection_pipeline/scripts/step_b_correlation_clustering/correlation_clustering.py \
  --correlation-threshold 0.75 \
  --output-dir channel_selection_results/step_b_correlation_clustering/threshold_r075
```

Multi-channel HCA clusters are accepted only after bootstrap consensus. If a
raw cluster contains a pair whose co-clustering probability is below 0.80, it
is refined with complete-linkage consensus clustering rather than accepted as
one redundancy group.

Build the four diverse 30-frame MVS clips:

```bash
/home/melody/anaconda3/envs/como/bin/python \
  /home/melody/code/individual_project/channel_selection_pipeline/scripts/step_d_mvs_construction/build_diverse_mvs_sequences.py
```

The clips are selected from adjacent-frame luminance changes in
`fr1/desk_lightswitch`. They cover two positive and two negative transitions
at different temporal/viewpoint locations. Every clip is a standalone
TUM-format sequence with 10 warm-up frames, 20 scored frames, and a complete
labelled MP4 preview. Generated datasets are stored under
`/home/melody/data/tum/` rather than the results directory.

Build the 50-frame challenging C extension, which contains both the strong
brightening and its subsequent dimming:

```bash
/home/melody/anaconda3/envs/como/bin/python \
  /home/melody/code/individual_project/channel_selection_pipeline/scripts/step_d_mvs_construction/build_diverse_mvs_sequences.py \
  --clips C50
```

This sequence uses source indices 368--417, with MVS frames 0--9 as warm-up
and frames 10--49 as the scored window.
