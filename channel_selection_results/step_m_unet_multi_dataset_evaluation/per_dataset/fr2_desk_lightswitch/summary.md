# U-Net multi-dataset full-sequence evaluation

- Dataset: `/home/melody/data/tum/rgbd_dataset_freiburg2_desk_lightswitch`
- Matched frames: 2893
- Tracking: selected U-Net Enc0/Enc1 channels; mapping: gray with sensor depth.
- Primary metric: keyframe `evo_ape --align --correct_scale` translation ATE mean.
- Diagnostics: historical keyframe RPE plus all-frame metric-scale SE(3) ATE/RPE and coverage.
- Timeout per configuration: 500 seconds.
- Dataset-specific safety exclusions: 0.
- Persisted rows: 13/13 active configurations; status counts: {'FAIL_TRACKING_NAN': 2, 'PASS': 11}

| Label | Encoder level | Channels | Status | Historical ATE mean (cm) | Coverage | Runtime (s) |
|---|---:|---|---|---:|---:|---:|
| enc0_k01_single_d03 | 0 | Enc0 [d3] | FAIL_TRACKING_NAN |  |  | 159.2 |
| enc0_k02_d02_d14 | 0 | Enc0 [d2,d14] | FAIL_TRACKING_NAN |  |  | 94.1 |
| enc0_global_rank02_k03_d03_d07_d12 | 0 | Enc0 [d3,d7,d12] | PASS | 6.8461 | 0.9993 | 160.5 |
| enc0_global_rank03_k04_d02_d03_d12_d14 | 0 | Enc0 [d2,d3,d12,d14] | PASS | 5.1419 | 0.9993 | 158.4 |
| enc0_global_rank01_k06_d02_d03_d07_d12_d13_d14 | 0 | Enc0 [d2,d3,d7,d12,d13,d14] | PASS | 7.0608 | 0.9993 | 158.6 |
| enc0_all16 | 0 | Enc0 all16 | PASS | 6.6972 | 0.9993 | 159.7 |
| enc0_bqs_top5_d00_d03_d10_d14_d15 | 0 | Enc0 [d0,d3,d10,d14,d15] | PASS | 7.1524 | 0.9993 | 158.0 |
| enc1_k02_d00_d05 | 1 | Enc1 [d0,d5] | PASS | 6.0824 | 0.9993 | 174.9 |
| enc1_global_rank03_k04_d00_d05_d18_d30 | 1 | Enc1 [d0,d5,d18,d30] | PASS | 6.3292 | 0.9993 | 180.5 |
| enc1_global_rank01_k06_d05_d06_d17_d18_d28_d30 | 1 | Enc1 [d5,d6,d17,d18,d28,d30] | PASS | 6.0189 | 0.9993 | 191.2 |
| enc1_global_rank02_k06_d00_d05_d06_d17_d18_d30 | 1 | Enc1 [d0,d5,d6,d17,d18,d30] | PASS | 5.3374 | 0.9993 | 185.6 |
| enc1_all32 | 1 | Enc1 all32 | PASS | 5.2438 | 0.9993 | 202.7 |
| enc1_bqs_top5_d04_d09_d10_d15_d30 | 1 | Enc1 [d4,d9,d10,d15,d30] | PASS | 10.0516 | 0.9993 | 192.6 |
