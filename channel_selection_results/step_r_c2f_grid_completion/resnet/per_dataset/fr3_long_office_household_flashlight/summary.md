# RESNET Step-R C2F grid completion (fr3_long_office_household_flashlight)

- Matched frames: 2488; mapping remains gray + sensor depth.
- This database contains only missing Step-R cells. Existing Step-Q rows are intentionally reused externally and never copied or rerun.
- Full selected grid: 60 C2F pairs + 11 direct parents.
- Reused Step-Q rows: 9; new Step-R rows: 62.
- Persisted new rows: 62/62; statuses: {'PASS': 62}.

| Standard label | Kind | Source | Status | ATE mean (cm) |
|---|---|---|---|---:|
| resnet_direct_fine01 | direct_fine | Step-Q `resnet_conv1_direct_f1_d15_d20_d26_d34` | REUSED |  |
| resnet_direct_fine02 | direct_fine | Step-Q `resnet_conv1_direct_f2_d23_d24_d26_d51_d63` | REUSED |  |
| resnet_direct_fine03 | direct_fine | Step-R | PASS | 13.4051 |
| resnet_direct_fine04 | direct_fine | Step-R | PASS | 19.9852 |
| resnet_direct_fine05 | direct_fine | Step-R | PASS | 16.3337 |
| resnet_direct_fine06 | direct_fine | Step-Q `resnet_conv1_direct_f6_d29_d33_d52` | REUSED |  |
| resnet_direct_coarse01 | direct_coarse | Step-R | PASS | 72.6413 |
| resnet_direct_coarse02 | direct_coarse | Step-R | PASS | 71.6030 |
| resnet_direct_coarse03 | direct_coarse | Step-R | PASS | 74.8198 |
| resnet_direct_coarse04 | direct_coarse | Step-Q `resnet_layer2_direct_c4_d67_d108_d121` | REUSED |  |
| resnet_direct_coarse05 | direct_coarse | Step-Q `resnet_layer2_direct_c5_d108_d121` | REUSED |  |
| resnet_c2f_a_fine01_coarse01 | c2f | Step-R | PASS | 21.5154 |
| resnet_c2f_a_fine01_coarse02 | c2f | Step-R | PASS | 19.3387 |
| resnet_c2f_a_fine01_coarse03 | c2f | Step-R | PASS | 20.7624 |
| resnet_c2f_a_fine01_coarse04 | c2f | Step-R | PASS | 20.0082 |
| resnet_c2f_a_fine01_coarse05 | c2f | Step-R | PASS | 18.7797 |
| resnet_c2f_a_fine02_coarse01 | c2f | Step-R | PASS | 23.7323 |
| resnet_c2f_a_fine02_coarse02 | c2f | Step-R | PASS | 25.1816 |
| resnet_c2f_a_fine02_coarse03 | c2f | Step-R | PASS | 22.7505 |
| resnet_c2f_a_fine02_coarse04 | c2f | Step-Q `resnet_c2f_a_f2_c4_variant_negative` | REUSED |  |
| resnet_c2f_a_fine02_coarse05 | c2f | Step-R | PASS | 20.6017 |
| resnet_c2f_a_fine03_coarse01 | c2f | Step-R | PASS | 20.4595 |
| resnet_c2f_a_fine03_coarse02 | c2f | Step-R | PASS | 19.9998 |
| resnet_c2f_a_fine03_coarse03 | c2f | Step-R | PASS | 22.1124 |
| resnet_c2f_a_fine03_coarse04 | c2f | Step-R | PASS | 20.8045 |
| resnet_c2f_a_fine03_coarse05 | c2f | Step-R | PASS | 19.0065 |
| resnet_c2f_a_fine04_coarse01 | c2f | Step-R | PASS | 23.2325 |
| resnet_c2f_a_fine04_coarse02 | c2f | Step-R | PASS | 22.7225 |
| resnet_c2f_a_fine04_coarse03 | c2f | Step-R | PASS | 22.9188 |
| resnet_c2f_a_fine04_coarse04 | c2f | Step-R | PASS | 22.4791 |
| resnet_c2f_a_fine04_coarse05 | c2f | Step-R | PASS | 22.5630 |
| resnet_c2f_a_fine05_coarse01 | c2f | Step-R | PASS | 31.2592 |
| resnet_c2f_a_fine05_coarse02 | c2f | Step-R | PASS | 28.8644 |
| resnet_c2f_a_fine05_coarse03 | c2f | Step-R | PASS | 28.4870 |
| resnet_c2f_a_fine05_coarse04 | c2f | Step-R | PASS | 27.9965 |
| resnet_c2f_a_fine05_coarse05 | c2f | Step-R | PASS | 26.9427 |
| resnet_c2f_a_fine06_coarse01 | c2f | Step-R | PASS | 19.5231 |
| resnet_c2f_a_fine06_coarse02 | c2f | Step-R | PASS | 18.4537 |
| resnet_c2f_a_fine06_coarse03 | c2f | Step-R | PASS | 19.9423 |
| resnet_c2f_a_fine06_coarse04 | c2f | Step-R | PASS | 18.7187 |
| resnet_c2f_a_fine06_coarse05 | c2f | Step-Q `resnet_c2f_a_f6_c5_positive_synergy` | REUSED |  |
| resnet_c2f_b_fine01_coarse01 | c2f | Step-R | PASS | 12.7450 |
| resnet_c2f_b_fine01_coarse02 | c2f | Step-R | PASS | 12.4170 |
| resnet_c2f_b_fine01_coarse03 | c2f | Step-R | PASS | 12.2045 |
| resnet_c2f_b_fine01_coarse04 | c2f | Step-Q `resnet_c2f_b_f1_c4_negative_high_parent` | REUSED |  |
| resnet_c2f_b_fine01_coarse05 | c2f | Step-R | PASS | 12.1362 |
| resnet_c2f_b_fine02_coarse01 | c2f | Step-R | PASS | 12.9011 |
| resnet_c2f_b_fine02_coarse02 | c2f | Step-R | PASS | 13.4000 |
| resnet_c2f_b_fine02_coarse03 | c2f | Step-R | PASS | 13.1316 |
| resnet_c2f_b_fine02_coarse04 | c2f | Step-Q `resnet_c2f_b_f2_c4_global_best` | REUSED |  |
| resnet_c2f_b_fine02_coarse05 | c2f | Step-R | PASS | 12.6618 |
| resnet_c2f_b_fine03_coarse01 | c2f | Step-R | PASS | 13.9288 |
| resnet_c2f_b_fine03_coarse02 | c2f | Step-R | PASS | 13.7223 |
| resnet_c2f_b_fine03_coarse03 | c2f | Step-R | PASS | 13.9591 |
| resnet_c2f_b_fine03_coarse04 | c2f | Step-R | PASS | 13.8669 |
| resnet_c2f_b_fine03_coarse05 | c2f | Step-R | PASS | 13.9437 |
| resnet_c2f_b_fine04_coarse01 | c2f | Step-R | PASS | 20.2267 |
| resnet_c2f_b_fine04_coarse02 | c2f | Step-R | PASS | 20.4286 |
| resnet_c2f_b_fine04_coarse03 | c2f | Step-R | PASS | 20.5724 |
| resnet_c2f_b_fine04_coarse04 | c2f | Step-R | PASS | 20.8503 |
| resnet_c2f_b_fine04_coarse05 | c2f | Step-R | PASS | 21.0661 |
| resnet_c2f_b_fine05_coarse01 | c2f | Step-R | PASS | 19.0555 |
| resnet_c2f_b_fine05_coarse02 | c2f | Step-R | PASS | 17.9180 |
| resnet_c2f_b_fine05_coarse03 | c2f | Step-R | PASS | 18.4123 |
| resnet_c2f_b_fine05_coarse04 | c2f | Step-R | PASS | 18.8593 |
| resnet_c2f_b_fine05_coarse05 | c2f | Step-R | PASS | 18.0914 |
| resnet_c2f_b_fine06_coarse01 | c2f | Step-R | PASS | 16.2790 |
| resnet_c2f_b_fine06_coarse02 | c2f | Step-R | PASS | 16.2559 |
| resnet_c2f_b_fine06_coarse03 | c2f | Step-R | PASS | 16.1393 |
| resnet_c2f_b_fine06_coarse04 | c2f | Step-R | PASS | 16.4556 |
| resnet_c2f_b_fine06_coarse05 | c2f | Step-R | PASS | 16.5942 |
