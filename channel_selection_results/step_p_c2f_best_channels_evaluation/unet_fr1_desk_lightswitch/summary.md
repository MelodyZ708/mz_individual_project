# UNET C2F promising-channel grid

- Dataset: `/home/melody/data/tum/rgbd_dataset_freiburg1_desk_lightswitch`
- Mapping: gray with sensor depth; tracking only receives mixed C2F features.
- C2F-A: coarse at L0/L1, fine at L2.  C2F-B: coarse at L0, fine at L1/L2.
- Primary ranking metric: historical keyframe `evo_ape --align --correct_scale` translation ATE mean.
- Diagnostics: historical RPE, all-frame metric-scale SE(3) ATE/RPE, coverage, and tracking diagnostics.
- Persisted rows: 72/72; status counts: {'PASS': 72}

| Label | Variant | Fine subset | Coarse subset | Status | ATE mean (cm) | Coverage | Runtime (s) |
|---|---|---|---|---|---:|---:|---:|
| unet_c2f_a_fine01_coarse01 | A | [d2,d3,d7,d12,d13,d14] | [d5,d6,d17,d18,d28,d30] | PASS | 7.0912 | 0.9965 | 50.6 |
| unet_c2f_a_fine01_coarse02 | A | [d2,d3,d7,d12,d13,d14] | [d0,d5,d6,d17,d18,d30] | PASS | 7.7116 | 0.9965 | 49.2 |
| unet_c2f_a_fine01_coarse03 | A | [d2,d3,d7,d12,d13,d14] | [d0,d5,d18,d30] | PASS | 8.7433 | 0.9965 | 49.1 |
| unet_c2f_a_fine01_coarse04 | A | [d2,d3,d7,d12,d13,d14] | [d0,d5,d6,d18,d30] | PASS | 6.5324 | 0.9965 | 48.7 |
| unet_c2f_a_fine01_coarse05 | A | [d2,d3,d7,d12,d13,d14] | [d0,d5] | PASS | 8.4903 | 0.9965 | 47.4 |
| unet_c2f_a_fine01_coarse06 | A | [d2,d3,d7,d12,d13,d14] | [d0,d4,d10,d22,d26] | PASS | 10.6676 | 0.9965 | 50.6 |
| unet_c2f_a_fine02_coarse01 | A | [d3,d7,d12] | [d5,d6,d17,d18,d28,d30] | PASS | 7.0387 | 0.9965 | 51.2 |
| unet_c2f_a_fine02_coarse02 | A | [d3,d7,d12] | [d0,d5,d6,d17,d18,d30] | PASS | 7.2166 | 0.9965 | 50.4 |
| unet_c2f_a_fine02_coarse03 | A | [d3,d7,d12] | [d0,d5,d18,d30] | PASS | 8.6616 | 0.9965 | 49.0 |
| unet_c2f_a_fine02_coarse04 | A | [d3,d7,d12] | [d0,d5,d6,d18,d30] | PASS | 5.8765 | 0.9965 | 50.0 |
| unet_c2f_a_fine02_coarse05 | A | [d3,d7,d12] | [d0,d5] | PASS | 13.4914 | 0.9965 | 48.3 |
| unet_c2f_a_fine02_coarse06 | A | [d3,d7,d12] | [d0,d4,d10,d22,d26] | PASS | 10.0023 | 0.9965 | 51.6 |
| unet_c2f_a_fine03_coarse01 | A | [d2,d3,d12,d14] | [d5,d6,d17,d18,d28,d30] | PASS | 8.0483 | 0.9965 | 51.0 |
| unet_c2f_a_fine03_coarse02 | A | [d2,d3,d12,d14] | [d0,d5,d6,d17,d18,d30] | PASS | 6.7074 | 0.9965 | 50.3 |
| unet_c2f_a_fine03_coarse03 | A | [d2,d3,d12,d14] | [d0,d5,d18,d30] | PASS | 9.3100 | 0.9965 | 48.9 |
| unet_c2f_a_fine03_coarse04 | A | [d2,d3,d12,d14] | [d0,d5,d6,d18,d30] | PASS | 6.3974 | 0.9965 | 49.9 |
| unet_c2f_a_fine03_coarse05 | A | [d2,d3,d12,d14] | [d0,d5] | PASS | 10.6914 | 0.9965 | 48.1 |
| unet_c2f_a_fine03_coarse06 | A | [d2,d3,d12,d14] | [d0,d4,d10,d22,d26] | PASS | 9.2852 | 0.9965 | 51.7 |
| unet_c2f_a_fine04_coarse01 | A | [d2,d4,d6] | [d5,d6,d17,d18,d28,d30] | PASS | 8.2706 | 0.9965 | 51.6 |
| unet_c2f_a_fine04_coarse02 | A | [d2,d4,d6] | [d0,d5,d6,d17,d18,d30] | PASS | 7.0455 | 0.9965 | 50.8 |
| unet_c2f_a_fine04_coarse03 | A | [d2,d4,d6] | [d0,d5,d18,d30] | PASS | 12.6357 | 0.9965 | 49.9 |
| unet_c2f_a_fine04_coarse04 | A | [d2,d4,d6] | [d0,d5,d6,d18,d30] | PASS | 7.2947 | 0.9965 | 50.7 |
| unet_c2f_a_fine04_coarse05 | A | [d2,d4,d6] | [d0,d5] | PASS | 8.3240 | 0.9965 | 48.6 |
| unet_c2f_a_fine04_coarse06 | A | [d2,d4,d6] | [d0,d4,d10,d22,d26] | PASS | 8.6218 | 0.9965 | 51.8 |
| unet_c2f_a_fine05_coarse01 | A | [d2,d3,d7,d12,d13] | [d5,d6,d17,d18,d28,d30] | PASS | 6.9654 | 0.9965 | 51.4 |
| unet_c2f_a_fine05_coarse02 | A | [d2,d3,d7,d12,d13] | [d0,d5,d6,d17,d18,d30] | PASS | 8.7757 | 0.9965 | 50.3 |
| unet_c2f_a_fine05_coarse03 | A | [d2,d3,d7,d12,d13] | [d0,d5,d18,d30] | PASS | 8.9114 | 0.9965 | 49.3 |
| unet_c2f_a_fine05_coarse04 | A | [d2,d3,d7,d12,d13] | [d0,d5,d6,d18,d30] | PASS | 6.1874 | 0.9965 | 49.7 |
| unet_c2f_a_fine05_coarse05 | A | [d2,d3,d7,d12,d13] | [d0,d5] | PASS | 10.6598 | 0.9965 | 48.3 |
| unet_c2f_a_fine05_coarse06 | A | [d2,d3,d7,d12,d13] | [d0,d4,d10,d22,d26] | PASS | 9.0471 | 0.9965 | 51.7 |
| unet_c2f_a_fine06_coarse01 | A | [d0,d2,d3,d11] | [d5,d6,d17,d18,d28,d30] | PASS | 7.2770 | 0.9965 | 51.6 |
| unet_c2f_a_fine06_coarse02 | A | [d0,d2,d3,d11] | [d0,d5,d6,d17,d18,d30] | PASS | 7.8475 | 0.9965 | 50.3 |
| unet_c2f_a_fine06_coarse03 | A | [d0,d2,d3,d11] | [d0,d5,d18,d30] | PASS | 11.1561 | 0.9965 | 49.4 |
| unet_c2f_a_fine06_coarse04 | A | [d0,d2,d3,d11] | [d0,d5,d6,d18,d30] | PASS | 7.3243 | 0.9965 | 50.4 |
| unet_c2f_a_fine06_coarse05 | A | [d0,d2,d3,d11] | [d0,d5] | PASS | 7.2618 | 0.9965 | 48.4 |
| unet_c2f_a_fine06_coarse06 | A | [d0,d2,d3,d11] | [d0,d4,d10,d22,d26] | PASS | 10.9678 | 0.9965 | 52.0 |
| unet_c2f_b_fine01_coarse01 | B | [d2,d3,d7,d12,d13,d14] | [d5,d6,d17,d18,d28,d30] | PASS | 9.2086 | 0.9965 | 49.0 |
| unet_c2f_b_fine01_coarse02 | B | [d2,d3,d7,d12,d13,d14] | [d0,d5,d6,d17,d18,d30] | PASS | 8.5553 | 0.9965 | 48.3 |
| unet_c2f_b_fine01_coarse03 | B | [d2,d3,d7,d12,d13,d14] | [d0,d5,d18,d30] | PASS | 7.9626 | 0.9965 | 47.6 |
| unet_c2f_b_fine01_coarse04 | B | [d2,d3,d7,d12,d13,d14] | [d0,d5,d6,d18,d30] | PASS | 8.6600 | 0.9965 | 48.3 |
| unet_c2f_b_fine01_coarse05 | B | [d2,d3,d7,d12,d13,d14] | [d0,d5] | PASS | 12.4520 | 0.9965 | 47.1 |
| unet_c2f_b_fine01_coarse06 | B | [d2,d3,d7,d12,d13,d14] | [d0,d4,d10,d22,d26] | PASS | 10.6976 | 0.9965 | 49.0 |
| unet_c2f_b_fine02_coarse01 | B | [d3,d7,d12] | [d5,d6,d17,d18,d28,d30] | PASS | 6.7999 | 0.9965 | 49.2 |
| unet_c2f_b_fine02_coarse02 | B | [d3,d7,d12] | [d0,d5,d6,d17,d18,d30] | PASS | 6.3454 | 0.9965 | 48.1 |
| unet_c2f_b_fine02_coarse03 | B | [d3,d7,d12] | [d0,d5,d18,d30] | PASS | 8.5506 | 0.9965 | 47.7 |
| unet_c2f_b_fine02_coarse04 | B | [d3,d7,d12] | [d0,d5,d6,d18,d30] | PASS | 6.6030 | 0.9965 | 47.9 |
| unet_c2f_b_fine02_coarse05 | B | [d3,d7,d12] | [d0,d5] | PASS | 7.3265 | 0.9965 | 47.5 |
| unet_c2f_b_fine02_coarse06 | B | [d3,d7,d12] | [d0,d4,d10,d22,d26] | PASS | 8.5334 | 0.9965 | 49.0 |
| unet_c2f_b_fine03_coarse01 | B | [d2,d3,d12,d14] | [d5,d6,d17,d18,d28,d30] | PASS | 8.4477 | 0.9965 | 48.9 |
| unet_c2f_b_fine03_coarse02 | B | [d2,d3,d12,d14] | [d0,d5,d6,d17,d18,d30] | PASS | 6.5437 | 0.9965 | 47.9 |
| unet_c2f_b_fine03_coarse03 | B | [d2,d3,d12,d14] | [d0,d5,d18,d30] | PASS | 8.6384 | 0.9965 | 48.2 |
| unet_c2f_b_fine03_coarse04 | B | [d2,d3,d12,d14] | [d0,d5,d6,d18,d30] | PASS | 7.2297 | 0.9965 | 48.1 |
| unet_c2f_b_fine03_coarse05 | B | [d2,d3,d12,d14] | [d0,d5] | PASS | 9.6330 | 0.9965 | 47.0 |
| unet_c2f_b_fine03_coarse06 | B | [d2,d3,d12,d14] | [d0,d4,d10,d22,d26] | PASS | 8.7855 | 0.9965 | 49.2 |
| unet_c2f_b_fine04_coarse01 | B | [d2,d4,d6] | [d5,d6,d17,d18,d28,d30] | PASS | 10.1612 | 0.9965 | 49.4 |
| unet_c2f_b_fine04_coarse02 | B | [d2,d4,d6] | [d0,d5,d6,d17,d18,d30] | PASS | 12.1087 | 0.9965 | 48.4 |
| unet_c2f_b_fine04_coarse03 | B | [d2,d4,d6] | [d0,d5,d18,d30] | PASS | 9.6102 | 0.9965 | 48.1 |
| unet_c2f_b_fine04_coarse04 | B | [d2,d4,d6] | [d0,d5,d6,d18,d30] | PASS | 10.3979 | 0.9965 | 47.9 |
| unet_c2f_b_fine04_coarse05 | B | [d2,d4,d6] | [d0,d5] | PASS | 10.7253 | 0.9965 | 47.0 |
| unet_c2f_b_fine04_coarse06 | B | [d2,d4,d6] | [d0,d4,d10,d22,d26] | PASS | 11.1586 | 0.9965 | 48.9 |
| unet_c2f_b_fine05_coarse01 | B | [d2,d3,d7,d12,d13] | [d5,d6,d17,d18,d28,d30] | PASS | 10.2823 | 0.9965 | 49.5 |
| unet_c2f_b_fine05_coarse02 | B | [d2,d3,d7,d12,d13] | [d0,d5,d6,d17,d18,d30] | PASS | 8.3290 | 0.9965 | 48.4 |
| unet_c2f_b_fine05_coarse03 | B | [d2,d3,d7,d12,d13] | [d0,d5,d18,d30] | PASS | 7.7622 | 0.9965 | 47.0 |
| unet_c2f_b_fine05_coarse04 | B | [d2,d3,d7,d12,d13] | [d0,d5,d6,d18,d30] | PASS | 6.4618 | 0.9965 | 48.4 |
| unet_c2f_b_fine05_coarse05 | B | [d2,d3,d7,d12,d13] | [d0,d5] | PASS | 11.0397 | 0.9965 | 47.6 |
| unet_c2f_b_fine05_coarse06 | B | [d2,d3,d7,d12,d13] | [d0,d4,d10,d22,d26] | PASS | 11.4585 | 0.9965 | 49.8 |
| unet_c2f_b_fine06_coarse01 | B | [d0,d2,d3,d11] | [d5,d6,d17,d18,d28,d30] | PASS | 9.9931 | 0.9965 | 48.9 |
| unet_c2f_b_fine06_coarse02 | B | [d0,d2,d3,d11] | [d0,d5,d6,d17,d18,d30] | PASS | 9.2888 | 0.9965 | 47.9 |
| unet_c2f_b_fine06_coarse03 | B | [d0,d2,d3,d11] | [d0,d5,d18,d30] | PASS | 12.2574 | 0.9965 | 47.7 |
| unet_c2f_b_fine06_coarse04 | B | [d0,d2,d3,d11] | [d0,d5,d6,d18,d30] | PASS | 11.8510 | 0.9965 | 48.1 |
| unet_c2f_b_fine06_coarse05 | B | [d0,d2,d3,d11] | [d0,d5] | PASS | 10.3019 | 0.9965 | 47.3 |
| unet_c2f_b_fine06_coarse06 | B | [d0,d2,d3,d11] | [d0,d4,d10,d22,d26] | PASS | 9.0170 | 0.9965 | 49.3 |
