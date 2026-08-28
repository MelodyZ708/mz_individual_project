# UNET C2F parent-comparison evaluation

- Dataset: `/home/melody/data/tum/rgbd_dataset_freiburg1_desk` (573 matched RGB-D frames).
- Mapping: gray + sensor depth.  Only tracking features are altered.
- Direct parents and C2F cells are run on the same sequence; aggregation computes C2F deltas only within each dataset.
- C2F-A: coarse L0/L1 + fine L2. C2F-B: coarse L0 + fine L1/L2.
- Primary metric: historical keyframe `evo_ape --align --correct_scale` translation ATE mean.
- Persisted rows: 10/10; statuses: {'PASS': 10}.

| Label | Mode | Configuration | Status | ATE mean (cm) | Coverage | Runtime (s) |
|---|---|---|---|---:|---:|---:|
| gray_baseline | gray | gray | PASS | 7.3847 | 0.9965 | 44.2 |
| unet_enc1_direct_c4_d00_d05_d06_d18_d30 | direct | direct enc1[d0,d5,d6,d18,d30] | PASS | 8.1456 | 0.9965 | 49.9 |
| unet_enc1_direct_c2_d00_d05_d06_d17_d18_d30 | direct | direct enc1[d0,d5,d6,d17,d18,d30] | PASS | 13.6948 | 0.9965 | 50.1 |
| unet_enc0_direct_f1_d02_d03_d07_d12_d13_d14 | direct | direct enc0[d2,d3,d7,d12,d13,d14] | PASS | 9.8457 | 0.9965 | 44.5 |
| unet_enc0_direct_f2_d03_d07_d12 | direct | direct enc0[d3,d7,d12] | PASS | 11.4312 | 0.9965 | 44.0 |
| unet_enc0_direct_f5_d02_d03_d07_d12_d13 | direct | direct enc0[d2,d3,d7,d12,d13] | PASS | 7.3887 | 0.9965 | 44.5 |
| unet_c2f_a_f1_c4_negative_high_parent | c2f | C2F-A; fine enc0[d2,d3,d7,d12,d13,d14]; coarse enc1[d0,d5,d6,d18,d30] | PASS | 6.4992 | 0.9965 | 48.1 |
| unet_c2f_a_f2_c4_global_best | c2f | C2F-A; fine enc0[d3,d7,d12]; coarse enc1[d0,d5,d6,d18,d30] | PASS | 6.8586 | 0.9965 | 48.3 |
| unet_c2f_a_f5_c4_positive_synergy | c2f | C2F-A; fine enc0[d2,d3,d7,d12,d13]; coarse enc1[d0,d5,d6,d18,d30] | PASS | 7.2362 | 0.9965 | 48.6 |
| unet_c2f_b_f2_c2_variant_contrast | c2f | C2F-B; fine enc0[d3,d7,d12]; coarse enc1[d0,d5,d6,d17,d18,d30] | PASS | 8.3136 | 0.9965 | 46.7 |
