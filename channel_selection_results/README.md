# Channel Selection Results

All generated outputs from `channel_selection_pipeline/` are stored here,
separated by pipeline stage.

- `step_0_data_preparation/`: paired timestamps, per-frame NPZ feature
  archives, human-readable manifests/statistics, and data-quality plots
  - `features_post_relu_png/`: all PNG contact sheets in one flat folder;
    filenames include sample, timestamp, layer, and condition
- `step_a_functional_profiling/`: optional per-channel descriptors and
  diagnostic plots; currently not consumed by Step B
- `step_b_correlation_clustering/`: correlation matrices, clusters,
  representatives, dendrograms, stability diagnostics, and PNG visual copies
  of the compressed NPZ matrices
  - root outputs: conservative `|r| >= 0.90` baseline
  - `threshold_r080/`: `|r| >= 0.80` comparison
  - `threshold_r075/`: `|r| >= 0.75` comparison

Step D MVS datasets are intentionally stored with the other TUM datasets under
`/home/melody/data/tum/`. Their reproducible builders live in
`channel_selection_pipeline/scripts/step_d_mvs_construction/`.
