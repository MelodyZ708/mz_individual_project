# RESNET Step-R C2F grid completion (fr3_long_office_household_lightswitch)

- Matched frames: 2488; mapping remains gray + sensor depth.
- This database contains only missing Step-R cells. Existing Step-Q rows are intentionally reused externally and never copied or rerun.
- Full selected grid: 60 C2F pairs + 11 direct parents.
- Reused Step-Q rows: 9; new Step-R rows: 62.
- Persisted new rows: 62/62; statuses: {'FAIL_TRACKING_NAN': 15, 'PASS': 47}.

| Standard label | Kind | Source | Status | ATE mean (cm) |
|---|---|---|---|---:|
| resnet_direct_fine01 | direct_fine | Step-Q `resnet_conv1_direct_f1_d15_d20_d26_d34` | REUSED |  |
| resnet_direct_fine02 | direct_fine | Step-Q `resnet_conv1_direct_f2_d23_d24_d26_d51_d63` | REUSED |  |
| resnet_direct_fine03 | direct_fine | Step-R | PASS | 31.5962 |
| resnet_direct_fine04 | direct_fine | Step-R | PASS | 33.9327 |
| resnet_direct_fine05 | direct_fine | Step-R | PASS | 33.8942 |
| resnet_direct_fine06 | direct_fine | Step-Q `resnet_conv1_direct_f6_d29_d33_d52` | REUSED |  |
| resnet_direct_coarse01 | direct_coarse | Step-R | PASS | 82.1848 |
| resnet_direct_coarse02 | direct_coarse | Step-R | PASS | 83.7617 |
| resnet_direct_coarse03 | direct_coarse | Step-R | PASS | 78.5534 |
| resnet_direct_coarse04 | direct_coarse | Step-Q `resnet_layer2_direct_c4_d67_d108_d121` | REUSED |  |
| resnet_direct_coarse05 | direct_coarse | Step-Q `resnet_layer2_direct_c5_d108_d121` | REUSED |  |
| resnet_c2f_a_fine01_coarse01 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine01_coarse02 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine01_coarse03 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine01_coarse04 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine01_coarse05 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine02_coarse01 | c2f | Step-R | PASS | 26.9530 |
| resnet_c2f_a_fine02_coarse02 | c2f | Step-R | PASS | 26.0900 |
| resnet_c2f_a_fine02_coarse03 | c2f | Step-R | PASS | 36.5170 |
| resnet_c2f_a_fine02_coarse04 | c2f | Step-Q `resnet_c2f_a_f2_c4_variant_negative` | REUSED |  |
| resnet_c2f_a_fine02_coarse05 | c2f | Step-R | PASS | 29.0029 |
| resnet_c2f_a_fine03_coarse01 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine03_coarse02 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine03_coarse03 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine03_coarse04 | c2f | Step-R | PASS | 26.5501 |
| resnet_c2f_a_fine03_coarse05 | c2f | Step-R | PASS | 24.2391 |
| resnet_c2f_a_fine04_coarse01 | c2f | Step-R | PASS | 24.7320 |
| resnet_c2f_a_fine04_coarse02 | c2f | Step-R | PASS | 25.4489 |
| resnet_c2f_a_fine04_coarse03 | c2f | Step-R | PASS | 25.0642 |
| resnet_c2f_a_fine04_coarse04 | c2f | Step-R | PASS | 23.7876 |
| resnet_c2f_a_fine04_coarse05 | c2f | Step-R | PASS | 21.7467 |
| resnet_c2f_a_fine05_coarse01 | c2f | Step-R | PASS | 32.7662 |
| resnet_c2f_a_fine05_coarse02 | c2f | Step-R | PASS | 30.6707 |
| resnet_c2f_a_fine05_coarse03 | c2f | Step-R | PASS | 32.4503 |
| resnet_c2f_a_fine05_coarse04 | c2f | Step-R | PASS | 33.5448 |
| resnet_c2f_a_fine05_coarse05 | c2f | Step-R | PASS | 29.7821 |
| resnet_c2f_a_fine06_coarse01 | c2f | Step-R | PASS | 23.9105 |
| resnet_c2f_a_fine06_coarse02 | c2f | Step-R | PASS | 21.9253 |
| resnet_c2f_a_fine06_coarse03 | c2f | Step-R | PASS | 21.5824 |
| resnet_c2f_a_fine06_coarse04 | c2f | Step-R | PASS | 20.5651 |
| resnet_c2f_a_fine06_coarse05 | c2f | Step-Q `resnet_c2f_a_f6_c5_positive_synergy` | REUSED |  |
| resnet_c2f_b_fine01_coarse01 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_b_fine01_coarse02 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_b_fine01_coarse03 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_b_fine01_coarse04 | c2f | Step-Q `resnet_c2f_b_f1_c4_negative_high_parent` | REUSED |  |
| resnet_c2f_b_fine01_coarse05 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_b_fine02_coarse01 | c2f | Step-R | PASS | 15.1174 |
| resnet_c2f_b_fine02_coarse02 | c2f | Step-R | PASS | 12.8886 |
| resnet_c2f_b_fine02_coarse03 | c2f | Step-R | PASS | 15.4458 |
| resnet_c2f_b_fine02_coarse04 | c2f | Step-Q `resnet_c2f_b_f2_c4_global_best` | REUSED |  |
| resnet_c2f_b_fine02_coarse05 | c2f | Step-R | PASS | 15.4635 |
| resnet_c2f_b_fine03_coarse01 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_b_fine03_coarse02 | c2f | Step-R | PASS | 20.4287 |
| resnet_c2f_b_fine03_coarse03 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_b_fine03_coarse04 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_b_fine03_coarse05 | c2f | Step-R | PASS | 21.0874 |
| resnet_c2f_b_fine04_coarse01 | c2f | Step-R | PASS | 23.1318 |
| resnet_c2f_b_fine04_coarse02 | c2f | Step-R | PASS | 22.6542 |
| resnet_c2f_b_fine04_coarse03 | c2f | Step-R | PASS | 22.2669 |
| resnet_c2f_b_fine04_coarse04 | c2f | Step-R | PASS | 21.8688 |
| resnet_c2f_b_fine04_coarse05 | c2f | Step-R | PASS | 20.1644 |
| resnet_c2f_b_fine05_coarse01 | c2f | Step-R | PASS | 21.8764 |
| resnet_c2f_b_fine05_coarse02 | c2f | Step-R | PASS | 21.4868 |
| resnet_c2f_b_fine05_coarse03 | c2f | Step-R | PASS | 22.8089 |
| resnet_c2f_b_fine05_coarse04 | c2f | Step-R | PASS | 22.3511 |
| resnet_c2f_b_fine05_coarse05 | c2f | Step-R | PASS | 20.4038 |
| resnet_c2f_b_fine06_coarse01 | c2f | Step-R | PASS | 21.8357 |
| resnet_c2f_b_fine06_coarse02 | c2f | Step-R | PASS | 21.2514 |
| resnet_c2f_b_fine06_coarse03 | c2f | Step-R | PASS | 21.3937 |
| resnet_c2f_b_fine06_coarse04 | c2f | Step-R | PASS | 16.6001 |
| resnet_c2f_b_fine06_coarse05 | c2f | Step-R | PASS | 16.0662 |
