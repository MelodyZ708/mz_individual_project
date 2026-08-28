# RESNET Step-R C2F grid completion (fr2_desk_lightswitch)

- Matched frames: 2893; mapping remains gray + sensor depth.
- This database contains only missing Step-R cells. Existing Step-Q rows are intentionally reused externally and never copied or rerun.
- Full selected grid: 60 C2F pairs + 11 direct parents.
- Reused Step-Q rows: 9; new Step-R rows: 62.
- Persisted new rows: 62/62; statuses: {'FAIL_TRACKING_NAN': 19, 'PASS': 43}.

| Standard label | Kind | Source | Status | ATE mean (cm) |
|---|---|---|---|---:|
| resnet_direct_fine01 | direct_fine | Step-Q `resnet_conv1_direct_f1_d15_d20_d26_d34` | REUSED |  |
| resnet_direct_fine02 | direct_fine | Step-Q `resnet_conv1_direct_f2_d23_d24_d26_d51_d63` | REUSED |  |
| resnet_direct_fine03 | direct_fine | Step-R | PASS | 5.7481 |
| resnet_direct_fine04 | direct_fine | Step-R | PASS | 10.5973 |
| resnet_direct_fine05 | direct_fine | Step-R | PASS | 7.9762 |
| resnet_direct_fine06 | direct_fine | Step-Q `resnet_conv1_direct_f6_d29_d33_d52` | REUSED |  |
| resnet_direct_coarse01 | direct_coarse | Step-R | PASS | 18.0133 |
| resnet_direct_coarse02 | direct_coarse | Step-R | PASS | 18.2266 |
| resnet_direct_coarse03 | direct_coarse | Step-R | PASS | 17.4085 |
| resnet_direct_coarse04 | direct_coarse | Step-Q `resnet_layer2_direct_c4_d67_d108_d121` | REUSED |  |
| resnet_direct_coarse05 | direct_coarse | Step-Q `resnet_layer2_direct_c5_d108_d121` | REUSED |  |
| resnet_c2f_a_fine01_coarse01 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine01_coarse02 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine01_coarse03 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine01_coarse04 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine01_coarse05 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine02_coarse01 | c2f | Step-R | PASS | 7.9729 |
| resnet_c2f_a_fine02_coarse02 | c2f | Step-R | PASS | 7.5680 |
| resnet_c2f_a_fine02_coarse03 | c2f | Step-R | PASS | 7.7378 |
| resnet_c2f_a_fine02_coarse04 | c2f | Step-Q `resnet_c2f_a_f2_c4_variant_negative` | REUSED |  |
| resnet_c2f_a_fine02_coarse05 | c2f | Step-R | PASS | 7.6444 |
| resnet_c2f_a_fine03_coarse01 | c2f | Step-R | PASS | 6.3162 |
| resnet_c2f_a_fine03_coarse02 | c2f | Step-R | PASS | 6.3721 |
| resnet_c2f_a_fine03_coarse03 | c2f | Step-R | PASS | 6.3774 |
| resnet_c2f_a_fine03_coarse04 | c2f | Step-R | PASS | 6.1549 |
| resnet_c2f_a_fine03_coarse05 | c2f | Step-R | PASS | 6.1680 |
| resnet_c2f_a_fine04_coarse01 | c2f | Step-R | PASS | 8.2339 |
| resnet_c2f_a_fine04_coarse02 | c2f | Step-R | PASS | 8.2838 |
| resnet_c2f_a_fine04_coarse03 | c2f | Step-R | PASS | 8.2388 |
| resnet_c2f_a_fine04_coarse04 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine04_coarse05 | c2f | Step-R | PASS | 7.9920 |
| resnet_c2f_a_fine05_coarse01 | c2f | Step-R | PASS | 8.2880 |
| resnet_c2f_a_fine05_coarse02 | c2f | Step-R | PASS | 8.4551 |
| resnet_c2f_a_fine05_coarse03 | c2f | Step-R | PASS | 8.4490 |
| resnet_c2f_a_fine05_coarse04 | c2f | Step-R | PASS | 8.0179 |
| resnet_c2f_a_fine05_coarse05 | c2f | Step-R | PASS | 7.7634 |
| resnet_c2f_a_fine06_coarse01 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine06_coarse02 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine06_coarse03 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine06_coarse04 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine06_coarse05 | c2f | Step-Q `resnet_c2f_a_f6_c5_positive_synergy` | REUSED |  |
| resnet_c2f_b_fine01_coarse01 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_b_fine01_coarse02 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_b_fine01_coarse03 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_b_fine01_coarse04 | c2f | Step-Q `resnet_c2f_b_f1_c4_negative_high_parent` | REUSED |  |
| resnet_c2f_b_fine01_coarse05 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_b_fine02_coarse01 | c2f | Step-R | PASS | 6.8361 |
| resnet_c2f_b_fine02_coarse02 | c2f | Step-R | PASS | 7.0117 |
| resnet_c2f_b_fine02_coarse03 | c2f | Step-R | PASS | 6.9850 |
| resnet_c2f_b_fine02_coarse04 | c2f | Step-Q `resnet_c2f_b_f2_c4_global_best` | REUSED |  |
| resnet_c2f_b_fine02_coarse05 | c2f | Step-R | PASS | 6.8548 |
| resnet_c2f_b_fine03_coarse01 | c2f | Step-R | PASS | 5.4463 |
| resnet_c2f_b_fine03_coarse02 | c2f | Step-R | PASS | 5.4583 |
| resnet_c2f_b_fine03_coarse03 | c2f | Step-R | PASS | 5.4346 |
| resnet_c2f_b_fine03_coarse04 | c2f | Step-R | PASS | 5.3847 |
| resnet_c2f_b_fine03_coarse05 | c2f | Step-R | PASS | 5.4761 |
| resnet_c2f_b_fine04_coarse01 | c2f | Step-R | PASS | 8.2813 |
| resnet_c2f_b_fine04_coarse02 | c2f | Step-R | PASS | 8.3345 |
| resnet_c2f_b_fine04_coarse03 | c2f | Step-R | PASS | 8.7099 |
| resnet_c2f_b_fine04_coarse04 | c2f | Step-R | PASS | 8.3265 |
| resnet_c2f_b_fine04_coarse05 | c2f | Step-R | PASS | 8.1554 |
| resnet_c2f_b_fine05_coarse01 | c2f | Step-R | PASS | 7.3954 |
| resnet_c2f_b_fine05_coarse02 | c2f | Step-R | PASS | 7.3286 |
| resnet_c2f_b_fine05_coarse03 | c2f | Step-R | PASS | 7.3159 |
| resnet_c2f_b_fine05_coarse04 | c2f | Step-R | PASS | 6.9079 |
| resnet_c2f_b_fine05_coarse05 | c2f | Step-R | PASS | 7.0142 |
| resnet_c2f_b_fine06_coarse01 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_b_fine06_coarse02 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_b_fine06_coarse03 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_b_fine06_coarse04 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_b_fine06_coarse05 | c2f | Step-R | FAIL_TRACKING_NAN |  |
