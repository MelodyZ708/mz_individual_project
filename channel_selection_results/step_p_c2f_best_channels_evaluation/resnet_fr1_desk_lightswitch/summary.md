# RESNET C2F promising-channel grid

- Dataset: `/home/melody/data/tum/rgbd_dataset_freiburg1_desk_lightswitch`
- Mapping: gray with sensor depth; tracking only receives mixed C2F features.
- C2F-A: coarse at L0/L1, fine at L2.  C2F-B: coarse at L0, fine at L1/L2.
- Primary ranking metric: historical keyframe `evo_ape --align --correct_scale` translation ATE mean.
- Diagnostics: historical RPE, all-frame metric-scale SE(3) ATE/RPE, coverage, and tracking diagnostics.
- Persisted rows: 72/72; status counts: {'FAIL_TRACKING_NAN': 6, 'PASS': 66}

| Label | Variant | Fine subset | Coarse subset | Status | ATE mean (cm) | Coverage | Runtime (s) |
|---|---|---|---|---|---:|---:|---:|
| resnet_c2f_a_fine01_coarse01 | A | [d15,d20,d26,d34] | [d41,d60,d67,d108,d121] | PASS | 14.2175 | 0.9965 | 50.4 |
| resnet_c2f_a_fine01_coarse02 | A | [d15,d20,d26,d34] | [d60,d67,d108,d121] | PASS | 15.5747 | 0.9965 | 51.2 |
| resnet_c2f_a_fine01_coarse03 | A | [d15,d20,d26,d34] | [d41,d60,d67,d95,d108,d121] | PASS | 20.7859 | 0.9965 | 58.9 |
| resnet_c2f_a_fine01_coarse04 | A | [d15,d20,d26,d34] | [d67,d108,d121] | PASS | 12.0726 | 0.9965 | 51.3 |
| resnet_c2f_a_fine01_coarse05 | A | [d15,d20,d26,d34] | [d108,d121] | PASS | 13.2954 | 0.9965 | 52.5 |
| resnet_c2f_a_fine01_coarse06 | A | [d15,d20,d26,d34] | [d37,d74,d119,d121] | PASS | 14.8764 | 0.9965 | 50.8 |
| resnet_c2f_a_fine02_coarse01 | A | [d23,d24,d26,d51,d63] | [d41,d60,d67,d108,d121] | PASS | 13.1599 | 0.9965 | 52.8 |
| resnet_c2f_a_fine02_coarse02 | A | [d23,d24,d26,d51,d63] | [d60,d67,d108,d121] | PASS | 20.1132 | 0.9965 | 53.0 |
| resnet_c2f_a_fine02_coarse03 | A | [d23,d24,d26,d51,d63] | [d41,d60,d67,d95,d108,d121] | PASS | 16.3543 | 0.9965 | 53.0 |
| resnet_c2f_a_fine02_coarse04 | A | [d23,d24,d26,d51,d63] | [d67,d108,d121] | PASS | 10.5465 | 0.9965 | 52.6 |
| resnet_c2f_a_fine02_coarse05 | A | [d23,d24,d26,d51,d63] | [d108,d121] | PASS | 12.0326 | 0.9965 | 54.1 |
| resnet_c2f_a_fine02_coarse06 | A | [d23,d24,d26,d51,d63] | [d37,d74,d119,d121] | PASS | 13.7035 | 0.9965 | 51.6 |
| resnet_c2f_a_fine03_coarse01 | A | [d15,d20,d26,d34,d45,d53] | [d41,d60,d67,d108,d121] | PASS | 14.7217 | 0.9965 | 51.1 |
| resnet_c2f_a_fine03_coarse02 | A | [d15,d20,d26,d34,d45,d53] | [d60,d67,d108,d121] | PASS | 16.1560 | 0.9965 | 51.0 |
| resnet_c2f_a_fine03_coarse03 | A | [d15,d20,d26,d34,d45,d53] | [d41,d60,d67,d95,d108,d121] | PASS | 12.9810 | 0.9965 | 51.4 |
| resnet_c2f_a_fine03_coarse04 | A | [d15,d20,d26,d34,d45,d53] | [d67,d108,d121] | PASS | 12.1419 | 0.9965 | 50.8 |
| resnet_c2f_a_fine03_coarse05 | A | [d15,d20,d26,d34,d45,d53] | [d108,d121] | PASS | 12.1811 | 0.9965 | 52.2 |
| resnet_c2f_a_fine03_coarse06 | A | [d15,d20,d26,d34,d45,d53] | [d37,d74,d119,d121] | PASS | 13.4050 | 0.9965 | 49.6 |
| resnet_c2f_a_fine04_coarse01 | A | [d33,d52] | [d41,d60,d67,d108,d121] | PASS | 15.2782 | 0.9965 | 50.7 |
| resnet_c2f_a_fine04_coarse02 | A | [d33,d52] | [d60,d67,d108,d121] | FAIL_TRACKING_NAN |  |  | 43.6 |
| resnet_c2f_a_fine04_coarse03 | A | [d33,d52] | [d41,d60,d67,d95,d108,d121] | PASS | 15.5919 | 0.9965 | 51.0 |
| resnet_c2f_a_fine04_coarse04 | A | [d33,d52] | [d67,d108,d121] | PASS | 13.5264 | 0.9965 | 50.7 |
| resnet_c2f_a_fine04_coarse05 | A | [d33,d52] | [d108,d121] | PASS | 14.1020 | 0.9965 | 52.1 |
| resnet_c2f_a_fine04_coarse06 | A | [d33,d52] | [d37,d74,d119,d121] | PASS | 15.6449 | 0.9965 | 49.2 |
| resnet_c2f_a_fine05_coarse01 | A | [d5,d6,d24,d29] | [d41,d60,d67,d108,d121] | FAIL_TRACKING_NAN |  |  | 44.4 |
| resnet_c2f_a_fine05_coarse02 | A | [d5,d6,d24,d29] | [d60,d67,d108,d121] | FAIL_TRACKING_NAN |  |  | 44.3 |
| resnet_c2f_a_fine05_coarse03 | A | [d5,d6,d24,d29] | [d41,d60,d67,d95,d108,d121] | PASS | 11.4420 | 0.9965 | 51.7 |
| resnet_c2f_a_fine05_coarse04 | A | [d5,d6,d24,d29] | [d67,d108,d121] | PASS | 10.6933 | 0.9965 | 51.8 |
| resnet_c2f_a_fine05_coarse05 | A | [d5,d6,d24,d29] | [d108,d121] | PASS | 12.4945 | 0.9965 | 53.0 |
| resnet_c2f_a_fine05_coarse06 | A | [d5,d6,d24,d29] | [d37,d74,d119,d121] | PASS | 20.3158 | 0.9965 | 49.8 |
| resnet_c2f_a_fine06_coarse01 | A | [d29,d33,d52] | [d41,d60,d67,d108,d121] | FAIL_TRACKING_NAN |  |  | 43.9 |
| resnet_c2f_a_fine06_coarse02 | A | [d29,d33,d52] | [d60,d67,d108,d121] | PASS | 12.8748 | 0.9965 | 50.7 |
| resnet_c2f_a_fine06_coarse03 | A | [d29,d33,d52] | [d41,d60,d67,d95,d108,d121] | PASS | 12.1517 | 0.9965 | 51.2 |
| resnet_c2f_a_fine06_coarse04 | A | [d29,d33,d52] | [d67,d108,d121] | PASS | 11.5763 | 0.9965 | 50.7 |
| resnet_c2f_a_fine06_coarse05 | A | [d29,d33,d52] | [d108,d121] | PASS | 9.8490 | 0.9965 | 52.7 |
| resnet_c2f_a_fine06_coarse06 | A | [d29,d33,d52] | [d37,d74,d119,d121] | PASS | 13.5216 | 0.9965 | 49.5 |
| resnet_c2f_b_fine01_coarse01 | B | [d15,d20,d26,d34] | [d41,d60,d67,d108,d121] | PASS | 13.4174 | 0.9965 | 49.9 |
| resnet_c2f_b_fine01_coarse02 | B | [d15,d20,d26,d34] | [d60,d67,d108,d121] | PASS | 15.8085 | 0.9965 | 50.3 |
| resnet_c2f_b_fine01_coarse03 | B | [d15,d20,d26,d34] | [d41,d60,d67,d95,d108,d121] | PASS | 17.4822 | 0.9965 | 50.3 |
| resnet_c2f_b_fine01_coarse04 | B | [d15,d20,d26,d34] | [d67,d108,d121] | PASS | 11.0315 | 0.9965 | 49.9 |
| resnet_c2f_b_fine01_coarse05 | B | [d15,d20,d26,d34] | [d108,d121] | PASS | 10.1203 | 0.9965 | 50.8 |
| resnet_c2f_b_fine01_coarse06 | B | [d15,d20,d26,d34] | [d37,d74,d119,d121] | PASS | 15.1664 | 0.9965 | 49.2 |
| resnet_c2f_b_fine02_coarse01 | B | [d23,d24,d26,d51,d63] | [d41,d60,d67,d108,d121] | PASS | 10.5499 | 0.9965 | 52.1 |
| resnet_c2f_b_fine02_coarse02 | B | [d23,d24,d26,d51,d63] | [d60,d67,d108,d121] | PASS | 14.8506 | 0.9965 | 52.5 |
| resnet_c2f_b_fine02_coarse03 | B | [d23,d24,d26,d51,d63] | [d41,d60,d67,d95,d108,d121] | PASS | 14.7839 | 0.9965 | 51.9 |
| resnet_c2f_b_fine02_coarse04 | B | [d23,d24,d26,d51,d63] | [d67,d108,d121] | PASS | 9.3293 | 0.9965 | 52.0 |
| resnet_c2f_b_fine02_coarse05 | B | [d23,d24,d26,d51,d63] | [d108,d121] | PASS | 10.2787 | 0.9965 | 53.6 |
| resnet_c2f_b_fine02_coarse06 | B | [d23,d24,d26,d51,d63] | [d37,d74,d119,d121] | PASS | 14.7150 | 0.9965 | 51.2 |
| resnet_c2f_b_fine03_coarse01 | B | [d15,d20,d26,d34,d45,d53] | [d41,d60,d67,d108,d121] | PASS | 14.0698 | 0.9965 | 48.3 |
| resnet_c2f_b_fine03_coarse02 | B | [d15,d20,d26,d34,d45,d53] | [d60,d67,d108,d121] | PASS | 12.9242 | 0.9965 | 48.8 |
| resnet_c2f_b_fine03_coarse03 | B | [d15,d20,d26,d34,d45,d53] | [d41,d60,d67,d95,d108,d121] | PASS | 12.3868 | 0.9965 | 48.8 |
| resnet_c2f_b_fine03_coarse04 | B | [d15,d20,d26,d34,d45,d53] | [d67,d108,d121] | PASS | 11.4852 | 0.9965 | 48.5 |
| resnet_c2f_b_fine03_coarse05 | B | [d15,d20,d26,d34,d45,d53] | [d108,d121] | PASS | 9.8569 | 0.9965 | 49.3 |
| resnet_c2f_b_fine03_coarse06 | B | [d15,d20,d26,d34,d45,d53] | [d37,d74,d119,d121] | PASS | 13.4379 | 0.9965 | 47.7 |
| resnet_c2f_b_fine04_coarse01 | B | [d33,d52] | [d41,d60,d67,d108,d121] | PASS | 14.7243 | 0.9965 | 48.3 |
| resnet_c2f_b_fine04_coarse02 | B | [d33,d52] | [d60,d67,d108,d121] | PASS | 13.4065 | 0.9965 | 48.5 |
| resnet_c2f_b_fine04_coarse03 | B | [d33,d52] | [d41,d60,d67,d95,d108,d121] | PASS | 16.0543 | 0.9965 | 48.5 |
| resnet_c2f_b_fine04_coarse04 | B | [d33,d52] | [d67,d108,d121] | PASS | 12.1327 | 0.9965 | 48.3 |
| resnet_c2f_b_fine04_coarse05 | B | [d33,d52] | [d108,d121] | PASS | 14.6281 | 0.9965 | 48.8 |
| resnet_c2f_b_fine04_coarse06 | B | [d33,d52] | [d37,d74,d119,d121] | PASS | 13.2059 | 0.9965 | 46.8 |
| resnet_c2f_b_fine05_coarse01 | B | [d5,d6,d24,d29] | [d41,d60,d67,d108,d121] | FAIL_TRACKING_NAN |  |  | 42.4 |
| resnet_c2f_b_fine05_coarse02 | B | [d5,d6,d24,d29] | [d60,d67,d108,d121] | FAIL_TRACKING_NAN |  |  | 42.0 |
| resnet_c2f_b_fine05_coarse03 | B | [d5,d6,d24,d29] | [d41,d60,d67,d95,d108,d121] | PASS | 15.5751 | 0.9965 | 48.1 |
| resnet_c2f_b_fine05_coarse04 | B | [d5,d6,d24,d29] | [d67,d108,d121] | PASS | 10.7274 | 0.9965 | 48.9 |
| resnet_c2f_b_fine05_coarse05 | B | [d5,d6,d24,d29] | [d108,d121] | PASS | 11.1759 | 0.9965 | 49.1 |
| resnet_c2f_b_fine05_coarse06 | B | [d5,d6,d24,d29] | [d37,d74,d119,d121] | PASS | 18.8598 | 0.9965 | 47.9 |
| resnet_c2f_b_fine06_coarse01 | B | [d29,d33,d52] | [d41,d60,d67,d108,d121] | PASS | 12.9679 | 0.9965 | 47.5 |
| resnet_c2f_b_fine06_coarse02 | B | [d29,d33,d52] | [d60,d67,d108,d121] | PASS | 14.9426 | 0.9965 | 48.0 |
| resnet_c2f_b_fine06_coarse03 | B | [d29,d33,d52] | [d41,d60,d67,d95,d108,d121] | PASS | 12.3707 | 0.9965 | 48.2 |
| resnet_c2f_b_fine06_coarse04 | B | [d29,d33,d52] | [d67,d108,d121] | PASS | 11.7041 | 0.9965 | 47.6 |
| resnet_c2f_b_fine06_coarse05 | B | [d29,d33,d52] | [d108,d121] | PASS | 14.0583 | 0.9965 | 48.3 |
| resnet_c2f_b_fine06_coarse06 | B | [d29,d33,d52] | [d37,d74,d119,d121] | PASS | 15.0692 | 0.9965 | 47.3 |
