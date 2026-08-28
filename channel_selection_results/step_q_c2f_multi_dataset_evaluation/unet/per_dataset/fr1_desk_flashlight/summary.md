# UNET C2F parent-comparison evaluation

- Dataset: `/home/melody/data/tum/rgbd_dataset_freiburg1_desk_flashlight` (573 matched RGB-D frames).
- Mapping: gray + sensor depth.  Only tracking features are altered.
- Direct parents and C2F cells are run on the same sequence; aggregation computes C2F deltas only within each dataset.
- C2F-A: coarse L0/L1 + fine L2. C2F-B: coarse L0 + fine L1/L2.
- Primary metric: historical keyframe `evo_ape --align --correct_scale` translation ATE mean.
- Persisted rows: 10/10; statuses: {'PASS': 10}.

| Label | Mode | Configuration | Status | ATE mean (cm) | Coverage | Runtime (s) |
|---|---|---|---|---:|---:|---:|
| gray_baseline | gray | gray | PASS | 9.3834 | 0.9965 | 44.4 |
| unet_enc1_direct_c4_d00_d05_d06_d18_d30 | direct | direct enc1[d0,d5,d6,d18,d30] | PASS | 9.2241 | 0.9965 | 50.1 |
| unet_enc1_direct_c2_d00_d05_d06_d17_d18_d30 | direct | direct enc1[d0,d5,d6,d17,d18,d30] | PASS | 17.6194 | 0.9965 | 50.9 |
| unet_enc0_direct_f1_d02_d03_d07_d12_d13_d14 | direct | direct enc0[d2,d3,d7,d12,d13,d14] | PASS | 9.7913 | 0.9965 | 45.1 |
| unet_enc0_direct_f2_d03_d07_d12 | direct | direct enc0[d3,d7,d12] | PASS | 8.1473 | 0.9965 | 44.3 |
| unet_enc0_direct_f5_d02_d03_d07_d12_d13 | direct | direct enc0[d2,d3,d7,d12,d13] | PASS | 10.5885 | 0.9965 | 44.9 |
| unet_c2f_a_f1_c4_negative_high_parent | c2f | C2F-A; fine enc0[d2,d3,d7,d12,d13,d14]; coarse enc1[d0,d5,d6,d18,d30] | PASS | 8.0548 | 0.9965 | 48.7 |
| unet_c2f_a_f2_c4_global_best | c2f | C2F-A; fine enc0[d3,d7,d12]; coarse enc1[d0,d5,d6,d18,d30] | PASS | 7.9202 | 0.9965 | 48.9 |
| unet_c2f_a_f5_c4_positive_synergy | c2f | C2F-A; fine enc0[d2,d3,d7,d12,d13]; coarse enc1[d0,d5,d6,d18,d30] | PASS | 8.0831 | 0.9965 | 48.8 |
| unet_c2f_b_f2_c2_variant_contrast | c2f | C2F-B; fine enc0[d3,d7,d12]; coarse enc1[d0,d5,d6,d17,d18,d30] | PASS | 8.3586 | 0.9965 | 47.0 |
