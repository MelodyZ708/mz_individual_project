# U-Net multi-dataset full-sequence evaluation

- Dataset: `/home/melody/data/tum/rgbd_dataset_freiburg3_long_office_household`
- Matched frames: 2488
- Tracking: selected U-Net Enc0/Enc1 channels; mapping: gray with sensor depth.
- Primary metric: keyframe `evo_ape --align --correct_scale` translation ATE mean.
- Diagnostics: historical keyframe RPE plus all-frame metric-scale SE(3) ATE/RPE and coverage.
- Timeout per configuration: 500 seconds.
- Dataset-specific safety exclusions: 0.
- Persisted rows: 13/13 active configurations; status counts: {'PASS': 13}

| Label | Encoder level | Channels | Status | Historical ATE mean (cm) | Coverage | Runtime (s) |
|---|---:|---|---|---:|---:|---:|
| enc0_k01_single_d03 | 0 | Enc0 [d3] | PASS | 14.2340 | 0.9992 | 132.4 |
| enc0_k02_d02_d14 | 0 | Enc0 [d2,d14] | PASS | 11.4093 | 0.9992 | 126.8 |
| enc0_global_rank02_k03_d03_d07_d12 | 0 | Enc0 [d3,d7,d12] | PASS | 10.4039 | 0.9992 | 124.7 |
| enc0_global_rank03_k04_d02_d03_d12_d14 | 0 | Enc0 [d2,d3,d12,d14] | PASS | 10.7810 | 0.9992 | 124.5 |
| enc0_global_rank01_k06_d02_d03_d07_d12_d13_d14 | 0 | Enc0 [d2,d3,d7,d12,d13,d14] | PASS | 10.2805 | 0.9992 | 123.3 |
| enc0_all16 | 0 | Enc0 all16 | PASS | 10.8092 | 0.9992 | 126.0 |
| enc0_bqs_top5_d00_d03_d10_d14_d15 | 0 | Enc0 [d0,d3,d10,d14,d15] | PASS | 12.6889 | 0.9992 | 123.3 |
| enc1_k02_d00_d05 | 1 | Enc1 [d0,d5] | PASS | 13.8740 | 0.9992 | 133.0 |
| enc1_global_rank03_k04_d00_d05_d18_d30 | 1 | Enc1 [d0,d5,d18,d30] | PASS | 15.1333 | 0.9992 | 140.4 |
| enc1_global_rank01_k06_d05_d06_d17_d18_d28_d30 | 1 | Enc1 [d5,d6,d17,d18,d28,d30] | PASS | 15.1487 | 0.9992 | 150.3 |
| enc1_global_rank02_k06_d00_d05_d06_d17_d18_d30 | 1 | Enc1 [d0,d5,d6,d17,d18,d30] | PASS | 12.7167 | 0.9992 | 145.2 |
| enc1_all32 | 1 | Enc1 all32 | PASS | 14.4282 | 0.9992 | 160.4 |
| enc1_bqs_top5_d04_d09_d10_d15_d30 | 1 | Enc1 [d4,d9,d10,d15,d30] | PASS | 17.4450 | 0.9992 | 154.1 |
