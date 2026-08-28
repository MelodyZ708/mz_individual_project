# RESNET Step-R C2F grid completion (fr1_desk_lightswitch)

- Matched frames: 573; mapping remains gray + sensor depth.
- This database contains only missing Step-R cells. Existing Step-Q rows are intentionally reused externally and never copied or rerun.
- Full selected grid: 60 C2F pairs + 11 direct parents.
- Reused Step-Q rows: 9; new Step-R rows: 62.
- Persisted new rows: 62/62; statuses: {'FAIL_TRACKING_NAN': 6, 'PASS': 56}.

| Standard label | Kind | Source | Status | ATE mean (cm) |
|---|---|---|---|---:|
| resnet_direct_fine01 | direct_fine | Step-Q `resnet_conv1_direct_f1_d15_d20_d26_d34` | REUSED |  |
| resnet_direct_fine02 | direct_fine | Step-Q `resnet_conv1_direct_f2_d23_d24_d26_d51_d63` | REUSED |  |
| resnet_direct_fine03 | direct_fine | Step-R | PASS | 11.6218 |
| resnet_direct_fine04 | direct_fine | Step-R | PASS | 13.9810 |
| resnet_direct_fine05 | direct_fine | Step-R | PASS | 14.0623 |
| resnet_direct_fine06 | direct_fine | Step-Q `resnet_conv1_direct_f6_d29_d33_d52` | REUSED |  |
| resnet_direct_coarse01 | direct_coarse | Step-R | PASS | 12.6100 |
| resnet_direct_coarse02 | direct_coarse | Step-R | PASS | 12.6714 |
| resnet_direct_coarse03 | direct_coarse | Step-R | PASS | 12.7413 |
| resnet_direct_coarse04 | direct_coarse | Step-Q `resnet_layer2_direct_c4_d67_d108_d121` | REUSED |  |
| resnet_direct_coarse05 | direct_coarse | Step-Q `resnet_layer2_direct_c5_d108_d121` | REUSED |  |
| resnet_c2f_a_fine01_coarse01 | c2f | Step-R | PASS | 14.2175 |
| resnet_c2f_a_fine01_coarse02 | c2f | Step-R | PASS | 15.5747 |
| resnet_c2f_a_fine01_coarse03 | c2f | Step-R | PASS | 20.7859 |
| resnet_c2f_a_fine01_coarse04 | c2f | Step-R | PASS | 12.0726 |
| resnet_c2f_a_fine01_coarse05 | c2f | Step-R | PASS | 13.2954 |
| resnet_c2f_a_fine02_coarse01 | c2f | Step-R | PASS | 13.1599 |
| resnet_c2f_a_fine02_coarse02 | c2f | Step-R | PASS | 20.1132 |
| resnet_c2f_a_fine02_coarse03 | c2f | Step-R | PASS | 16.3543 |
| resnet_c2f_a_fine02_coarse04 | c2f | Step-Q `resnet_c2f_a_f2_c4_variant_negative` | REUSED |  |
| resnet_c2f_a_fine02_coarse05 | c2f | Step-R | PASS | 12.0326 |
| resnet_c2f_a_fine03_coarse01 | c2f | Step-R | PASS | 14.7217 |
| resnet_c2f_a_fine03_coarse02 | c2f | Step-R | PASS | 16.1560 |
| resnet_c2f_a_fine03_coarse03 | c2f | Step-R | PASS | 12.9810 |
| resnet_c2f_a_fine03_coarse04 | c2f | Step-R | PASS | 12.1419 |
| resnet_c2f_a_fine03_coarse05 | c2f | Step-R | PASS | 12.1811 |
| resnet_c2f_a_fine04_coarse01 | c2f | Step-R | PASS | 15.2782 |
| resnet_c2f_a_fine04_coarse02 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine04_coarse03 | c2f | Step-R | PASS | 15.5919 |
| resnet_c2f_a_fine04_coarse04 | c2f | Step-R | PASS | 13.5264 |
| resnet_c2f_a_fine04_coarse05 | c2f | Step-R | PASS | 14.1020 |
| resnet_c2f_a_fine05_coarse01 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine05_coarse02 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine05_coarse03 | c2f | Step-R | PASS | 11.4420 |
| resnet_c2f_a_fine05_coarse04 | c2f | Step-R | PASS | 10.6933 |
| resnet_c2f_a_fine05_coarse05 | c2f | Step-R | PASS | 12.4945 |
| resnet_c2f_a_fine06_coarse01 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_a_fine06_coarse02 | c2f | Step-R | PASS | 12.8748 |
| resnet_c2f_a_fine06_coarse03 | c2f | Step-R | PASS | 12.1517 |
| resnet_c2f_a_fine06_coarse04 | c2f | Step-R | PASS | 11.5763 |
| resnet_c2f_a_fine06_coarse05 | c2f | Step-Q `resnet_c2f_a_f6_c5_positive_synergy` | REUSED |  |
| resnet_c2f_b_fine01_coarse01 | c2f | Step-R | PASS | 13.4174 |
| resnet_c2f_b_fine01_coarse02 | c2f | Step-R | PASS | 15.8085 |
| resnet_c2f_b_fine01_coarse03 | c2f | Step-R | PASS | 17.4822 |
| resnet_c2f_b_fine01_coarse04 | c2f | Step-Q `resnet_c2f_b_f1_c4_negative_high_parent` | REUSED |  |
| resnet_c2f_b_fine01_coarse05 | c2f | Step-R | PASS | 10.1203 |
| resnet_c2f_b_fine02_coarse01 | c2f | Step-R | PASS | 10.5499 |
| resnet_c2f_b_fine02_coarse02 | c2f | Step-R | PASS | 14.8506 |
| resnet_c2f_b_fine02_coarse03 | c2f | Step-R | PASS | 14.7839 |
| resnet_c2f_b_fine02_coarse04 | c2f | Step-Q `resnet_c2f_b_f2_c4_global_best` | REUSED |  |
| resnet_c2f_b_fine02_coarse05 | c2f | Step-R | PASS | 10.2787 |
| resnet_c2f_b_fine03_coarse01 | c2f | Step-R | PASS | 14.0698 |
| resnet_c2f_b_fine03_coarse02 | c2f | Step-R | PASS | 12.9242 |
| resnet_c2f_b_fine03_coarse03 | c2f | Step-R | PASS | 12.3868 |
| resnet_c2f_b_fine03_coarse04 | c2f | Step-R | PASS | 11.4852 |
| resnet_c2f_b_fine03_coarse05 | c2f | Step-R | PASS | 9.8569 |
| resnet_c2f_b_fine04_coarse01 | c2f | Step-R | PASS | 14.7243 |
| resnet_c2f_b_fine04_coarse02 | c2f | Step-R | PASS | 13.4065 |
| resnet_c2f_b_fine04_coarse03 | c2f | Step-R | PASS | 16.0543 |
| resnet_c2f_b_fine04_coarse04 | c2f | Step-R | PASS | 12.1327 |
| resnet_c2f_b_fine04_coarse05 | c2f | Step-R | PASS | 14.6281 |
| resnet_c2f_b_fine05_coarse01 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_b_fine05_coarse02 | c2f | Step-R | FAIL_TRACKING_NAN |  |
| resnet_c2f_b_fine05_coarse03 | c2f | Step-R | PASS | 15.5751 |
| resnet_c2f_b_fine05_coarse04 | c2f | Step-R | PASS | 10.7274 |
| resnet_c2f_b_fine05_coarse05 | c2f | Step-R | PASS | 11.1759 |
| resnet_c2f_b_fine06_coarse01 | c2f | Step-R | PASS | 12.9679 |
| resnet_c2f_b_fine06_coarse02 | c2f | Step-R | PASS | 14.9426 |
| resnet_c2f_b_fine06_coarse03 | c2f | Step-R | PASS | 12.3707 |
| resnet_c2f_b_fine06_coarse04 | c2f | Step-R | PASS | 11.7041 |
| resnet_c2f_b_fine06_coarse05 | c2f | Step-R | PASS | 14.0583 |
