# RESNET C2F parent-comparison evaluation

- Dataset: `/home/melody/data/tum/rgbd_dataset_freiburg3_long_office_household_flashlight` (2488 matched RGB-D frames).
- Mapping: gray + sensor depth.  Only tracking features are altered.
- Direct parents and C2F cells are run on the same sequence; aggregation computes C2F deltas only within each dataset.
- C2F-A: coarse L0/L1 + fine L2. C2F-B: coarse L0 + fine L1/L2.
- Primary metric: historical keyframe `evo_ape --align --correct_scale` translation ATE mean.
- Persisted rows: 10/10; statuses: {'PASS': 10}.

| Label | Mode | Configuration | Status | ATE mean (cm) | Coverage | Runtime (s) |
|---|---|---|---|---:|---:|---:|
| gray_baseline | gray | gray | PASS | 10.5942 | 0.9992 | 126.5 |
| resnet_layer2_direct_c4_d67_d108_d121 | direct | direct layer2[d67,d108,d121] | PASS | 67.5551 | 0.9992 | 139.1 |
| resnet_layer2_direct_c5_d108_d121 | direct | direct layer2[d108,d121] | PASS | 60.4680 | 0.9992 | 141.4 |
| resnet_conv1_direct_f1_d15_d20_d26_d34 | direct | direct conv1[d15,d20,d26,d34] | PASS | 11.6302 | 0.9992 | 134.1 |
| resnet_conv1_direct_f2_d23_d24_d26_d51_d63 | direct | direct conv1[d23,d24,d26,d51,d63] | PASS | 11.8778 | 0.9992 | 130.8 |
| resnet_conv1_direct_f6_d29_d33_d52 | direct | direct conv1[d29,d33,d52] | PASS | 16.1544 | 0.9992 | 127.6 |
| resnet_c2f_b_f1_c4_negative_high_parent | c2f | C2F-B; fine conv1[d15,d20,d26,d34]; coarse layer2[d67,d108,d121] | PASS | 12.4190 | 0.9992 | 150.5 |
| resnet_c2f_a_f2_c4_variant_negative | c2f | C2F-A; fine conv1[d23,d24,d26,d51,d63]; coarse layer2[d67,d108,d121] | PASS | 22.6986 | 0.9992 | 153.8 |
| resnet_c2f_b_f2_c4_global_best | c2f | C2F-B; fine conv1[d23,d24,d26,d51,d63]; coarse layer2[d67,d108,d121] | PASS | 12.3885 | 0.9992 | 147.5 |
| resnet_c2f_a_f6_c5_positive_synergy | c2f | C2F-A; fine conv1[d29,d33,d52]; coarse layer2[d108,d121] | PASS | 19.1299 | 0.9992 | 151.8 |
