# U-Net multi-dataset full-sequence evaluation

- Dataset: `/home/melody/data/tum/rgbd_dataset_freiburg2_desk_flashlight`
- Matched frames: 2893
- Tracking: selected U-Net Enc0/Enc1 channels; mapping: gray with sensor depth.
- Primary metric: keyframe `evo_ape --align --correct_scale` translation ATE mean.
- Diagnostics: historical keyframe RPE plus all-frame metric-scale SE(3) ATE/RPE and coverage.
- Timeout per configuration: 500 seconds.
- Dataset-specific safety exclusions: 0.
- Persisted rows: 13/13 active configurations; status counts: {'PASS': 13}

| Label | Encoder level | Channels | Status | Historical ATE mean (cm) | Coverage | Runtime (s) |
|---|---:|---|---|---:|---:|---:|
| enc0_k01_single_d03 | 0 | Enc0 [d3] | PASS | 3.7944 | 0.9993 | 168.7 |
| enc0_k02_d02_d14 | 0 | Enc0 [d2,d14] | PASS | 3.9813 | 0.9993 | 161.1 |
| enc0_global_rank02_k03_d03_d07_d12 | 0 | Enc0 [d3,d7,d12] | PASS | 3.0736 | 0.9993 | 156.3 |
| enc0_global_rank03_k04_d02_d03_d12_d14 | 0 | Enc0 [d2,d3,d12,d14] | PASS | 3.5523 | 0.9993 | 156.5 |
| enc0_global_rank01_k06_d02_d03_d07_d12_d13_d14 | 0 | Enc0 [d2,d3,d7,d12,d13,d14] | PASS | 3.5598 | 0.9993 | 156.0 |
| enc0_all16 | 0 | Enc0 all16 | PASS | 4.1993 | 0.9993 | 158.1 |
| enc0_bqs_top5_d00_d03_d10_d14_d15 | 0 | Enc0 [d0,d3,d10,d14,d15] | PASS | 4.3304 | 0.9993 | 155.1 |
| enc1_k02_d00_d05 | 1 | Enc1 [d0,d5] | PASS | 6.0668 | 0.9993 | 170.9 |
| enc1_global_rank03_k04_d00_d05_d18_d30 | 1 | Enc1 [d0,d5,d18,d30] | PASS | 6.4378 | 0.9993 | 174.3 |
| enc1_global_rank01_k06_d05_d06_d17_d18_d28_d30 | 1 | Enc1 [d5,d6,d17,d18,d28,d30] | PASS | 4.9727 | 0.9993 | 185.7 |
| enc1_global_rank02_k06_d00_d05_d06_d17_d18_d30 | 1 | Enc1 [d0,d5,d6,d17,d18,d30] | PASS | 4.9033 | 0.9993 | 178.7 |
| enc1_all32 | 1 | Enc1 all32 | PASS | 4.5766 | 0.9993 | 199.0 |
| enc1_bqs_top5_d04_d09_d10_d15_d30 | 1 | Enc1 [d4,d9,d10,d15,d30] | PASS | 5.8323 | 0.9993 | 185.5 |
