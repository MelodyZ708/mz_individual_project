# U-Net multi-dataset full-sequence evaluation

- Dataset: `/home/melody/data/tum/rgbd_dataset_freiburg1_desk_flashlight`
- Matched frames: 573
- Tracking: selected U-Net Enc0/Enc1 channels; mapping: gray with sensor depth.
- Primary metric: keyframe `evo_ape --align --correct_scale` translation ATE mean.
- Diagnostics: historical keyframe RPE plus all-frame metric-scale SE(3) ATE/RPE and coverage.
- Timeout per configuration: 500 seconds.
- Dataset-specific safety exclusions: 0.
- Persisted rows: 13/13 active configurations; status counts: {'FAIL_TRACKING_NAN': 1, 'PASS': 12}

| Label | Encoder level | Channels | Status | Historical ATE mean (cm) | Coverage | Runtime (s) |
|---|---:|---|---|---:|---:|---:|
| enc0_k01_single_d03 | 0 | Enc0 [d3] | FAIL_TRACKING_NAN |  |  | 16.6 |
| enc0_k02_d02_d14 | 0 | Enc0 [d2,d14] | PASS | 17.5672 | 0.9965 | 45.9 |
| enc0_global_rank02_k03_d03_d07_d12 | 0 | Enc0 [d3,d7,d12] | PASS | 8.1473 | 0.9965 | 44.3 |
| enc0_global_rank03_k04_d02_d03_d12_d14 | 0 | Enc0 [d2,d3,d12,d14] | PASS | 5.9763 | 0.9965 | 44.7 |
| enc0_global_rank01_k06_d02_d03_d07_d12_d13_d14 | 0 | Enc0 [d2,d3,d7,d12,d13,d14] | PASS | 9.7913 | 0.9965 | 45.0 |
| enc0_all16 | 0 | Enc0 all16 | PASS | 9.2578 | 0.9965 | 45.2 |
| enc0_bqs_top5_d00_d03_d10_d14_d15 | 0 | Enc0 [d0,d3,d10,d14,d15] | PASS | 9.1861 | 0.9965 | 44.7 |
| enc1_k02_d00_d05 | 1 | Enc1 [d0,d5] | PASS | 8.8094 | 0.9965 | 48.1 |
| enc1_global_rank03_k04_d00_d05_d18_d30 | 1 | Enc1 [d0,d5,d18,d30] | PASS | 13.9928 | 0.9965 | 49.1 |
| enc1_global_rank01_k06_d05_d06_d17_d18_d28_d30 | 1 | Enc1 [d5,d6,d17,d18,d28,d30] | PASS | 10.0602 | 0.9965 | 51.8 |
| enc1_global_rank02_k06_d00_d05_d06_d17_d18_d30 | 1 | Enc1 [d0,d5,d6,d17,d18,d30] | PASS | 17.6194 | 0.9965 | 50.6 |
| enc1_all32 | 1 | Enc1 all32 | PASS | 14.7334 | 0.9965 | 54.2 |
| enc1_bqs_top5_d04_d09_d10_d15_d30 | 1 | Enc1 [d4,d9,d10,d15,d30] | PASS | 16.3408 | 0.9965 | 52.6 |
