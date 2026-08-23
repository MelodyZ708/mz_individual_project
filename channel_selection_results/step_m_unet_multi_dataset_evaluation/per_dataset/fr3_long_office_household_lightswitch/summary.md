# U-Net multi-dataset full-sequence evaluation

- Dataset: `/home/melody/data/tum/rgbd_dataset_freiburg3_long_office_household_lightswitch`
- Matched frames: 2488
- Tracking: selected U-Net Enc0/Enc1 channels; mapping: gray with sensor depth.
- Primary metric: keyframe `evo_ape --align --correct_scale` translation ATE mean.
- Diagnostics: historical keyframe RPE plus all-frame metric-scale SE(3) ATE/RPE and coverage.
- Timeout per configuration: 500 seconds.
- Dataset-specific safety exclusions: 2.
- Persisted rows: 11/11 active configurations; status counts: {'FAIL_TRACKING_NAN': 1, 'PASS': 10}

| Label | Encoder level | Channels | Status | Historical ATE mean (cm) | Coverage | Runtime (s) |
|---|---:|---|---|---:|---:|---:|
| enc0_k01_single_d03 | 0 | Enc0 [d3] | FAIL_TRACKING_NAN |  |  | 55.7 |
| enc0_k02_d02_d14 | 0 | Enc0 [d2,d14] | PASS | 35.1009 | 0.9992 | 130.4 |
| enc0_global_rank02_k03_d03_d07_d12 | 0 | Enc0 [d3,d7,d12] | PASS | 13.8427 | 0.9992 | 124.4 |
| enc0_global_rank03_k04_d02_d03_d12_d14 | 0 | Enc0 [d2,d3,d12,d14] | PASS | 24.1515 | 0.9992 | 126.2 |
| enc0_global_rank01_k06_d02_d03_d07_d12_d13_d14 | 0 | Enc0 [d2,d3,d7,d12,d13,d14] | PASS | 12.9517 | 0.9992 | 126.1 |
| enc0_all16 | 0 | Enc0 all16 | SKIPPED_BY_SAFETY |  |  |  |
| enc0_bqs_top5_d00_d03_d10_d14_d15 | 0 | Enc0 [d0,d3,d10,d14,d15] | PASS | 17.9243 | 0.9992 | 127.0 |
| enc1_k02_d00_d05 | 1 | Enc1 [d0,d5] | PASS | 29.5339 | 0.9992 | 136.1 |
| enc1_global_rank03_k04_d00_d05_d18_d30 | 1 | Enc1 [d0,d5,d18,d30] | PASS | 17.1518 | 0.9992 | 143.6 |
| enc1_global_rank01_k06_d05_d06_d17_d18_d28_d30 | 1 | Enc1 [d5,d6,d17,d18,d28,d30] | PASS | 15.4466 | 0.9992 | 154.1 |
| enc1_global_rank02_k06_d00_d05_d06_d17_d18_d30 | 1 | Enc1 [d0,d5,d6,d17,d18,d30] | PASS | 11.3039 | 0.9992 | 149.1 |
| enc1_all32 | 1 | Enc1 all32 | SKIPPED_BY_SAFETY |  |  |  |
| enc1_bqs_top5_d04_d09_d10_d15_d30 | 1 | Enc1 [d4,d9,d10,d15,d30] | PASS | 26.6312 | 0.9992 | 155.1 |
