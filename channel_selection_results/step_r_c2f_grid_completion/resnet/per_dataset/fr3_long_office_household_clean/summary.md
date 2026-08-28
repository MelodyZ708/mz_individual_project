# RESNET Step-R C2F grid completion (fr3_long_office_household_clean)

- Matched frames: 2488; mapping remains gray + sensor depth.
- This database contains only missing Step-R cells. Existing Step-Q rows are intentionally reused externally and never copied or rerun.
- Full selected grid: 60 C2F pairs + 11 direct parents.
- Reused Step-Q rows: 9; new Step-R rows: 62.
- Persisted new rows: 62/62; statuses: {'PASS': 62}.

| Standard label | Kind | Source | Status | ATE mean (cm) |
|---|---|---|---|---:|
| resnet_direct_fine01 | direct_fine | Step-Q `resnet_conv1_direct_f1_d15_d20_d26_d34` | REUSED |  |
| resnet_direct_fine02 | direct_fine | Step-Q `resnet_conv1_direct_f2_d23_d24_d26_d51_d63` | REUSED |  |
| resnet_direct_fine03 | direct_fine | Step-R | PASS | 13.7991 |
| resnet_direct_fine04 | direct_fine | Step-R | PASS | 20.2349 |
| resnet_direct_fine05 | direct_fine | Step-R | PASS | 16.2062 |
| resnet_direct_fine06 | direct_fine | Step-Q `resnet_conv1_direct_f6_d29_d33_d52` | REUSED |  |
| resnet_direct_coarse01 | direct_coarse | Step-R | PASS | 72.8974 |
| resnet_direct_coarse02 | direct_coarse | Step-R | PASS | 73.8457 |
| resnet_direct_coarse03 | direct_coarse | Step-R | PASS | 73.7711 |
| resnet_direct_coarse04 | direct_coarse | Step-Q `resnet_layer2_direct_c4_d67_d108_d121` | REUSED |  |
| resnet_direct_coarse05 | direct_coarse | Step-Q `resnet_layer2_direct_c5_d108_d121` | REUSED |  |
| resnet_c2f_a_fine01_coarse01 | c2f | Step-R | PASS | 19.5868 |
| resnet_c2f_a_fine01_coarse02 | c2f | Step-R | PASS | 19.3683 |
| resnet_c2f_a_fine01_coarse03 | c2f | Step-R | PASS | 20.0744 |
| resnet_c2f_a_fine01_coarse04 | c2f | Step-R | PASS | 19.5063 |
| resnet_c2f_a_fine01_coarse05 | c2f | Step-R | PASS | 18.9996 |
| resnet_c2f_a_fine02_coarse01 | c2f | Step-R | PASS | 22.9912 |
| resnet_c2f_a_fine02_coarse02 | c2f | Step-R | PASS | 22.9239 |
| resnet_c2f_a_fine02_coarse03 | c2f | Step-R | PASS | 25.0649 |
| resnet_c2f_a_fine02_coarse04 | c2f | Step-Q `resnet_c2f_a_f2_c4_variant_negative` | REUSED |  |
| resnet_c2f_a_fine02_coarse05 | c2f | Step-R | PASS | 21.0194 |
| resnet_c2f_a_fine03_coarse01 | c2f | Step-R | PASS | 21.1246 |
| resnet_c2f_a_fine03_coarse02 | c2f | Step-R | PASS | 20.2280 |
| resnet_c2f_a_fine03_coarse03 | c2f | Step-R | PASS | 22.2695 |
| resnet_c2f_a_fine03_coarse04 | c2f | Step-R | PASS | 20.3777 |
| resnet_c2f_a_fine03_coarse05 | c2f | Step-R | PASS | 18.9445 |
| resnet_c2f_a_fine04_coarse01 | c2f | Step-R | PASS | 23.2753 |
| resnet_c2f_a_fine04_coarse02 | c2f | Step-R | PASS | 21.9525 |
| resnet_c2f_a_fine04_coarse03 | c2f | Step-R | PASS | 22.8384 |
| resnet_c2f_a_fine04_coarse04 | c2f | Step-R | PASS | 22.8183 |
| resnet_c2f_a_fine04_coarse05 | c2f | Step-R | PASS | 22.2569 |
| resnet_c2f_a_fine05_coarse01 | c2f | Step-R | PASS | 27.9317 |
| resnet_c2f_a_fine05_coarse02 | c2f | Step-R | PASS | 28.9135 |
| resnet_c2f_a_fine05_coarse03 | c2f | Step-R | PASS | 27.9675 |
| resnet_c2f_a_fine05_coarse04 | c2f | Step-R | PASS | 27.8229 |
| resnet_c2f_a_fine05_coarse05 | c2f | Step-R | PASS | 27.8606 |
| resnet_c2f_a_fine06_coarse01 | c2f | Step-R | PASS | 18.4390 |
| resnet_c2f_a_fine06_coarse02 | c2f | Step-R | PASS | 18.3585 |
| resnet_c2f_a_fine06_coarse03 | c2f | Step-R | PASS | 19.3174 |
| resnet_c2f_a_fine06_coarse04 | c2f | Step-R | PASS | 18.8925 |
| resnet_c2f_a_fine06_coarse05 | c2f | Step-Q `resnet_c2f_a_f6_c5_positive_synergy` | REUSED |  |
| resnet_c2f_b_fine01_coarse01 | c2f | Step-R | PASS | 12.6202 |
| resnet_c2f_b_fine01_coarse02 | c2f | Step-R | PASS | 12.6639 |
| resnet_c2f_b_fine01_coarse03 | c2f | Step-R | PASS | 12.7798 |
| resnet_c2f_b_fine01_coarse04 | c2f | Step-Q `resnet_c2f_b_f1_c4_negative_high_parent` | REUSED |  |
| resnet_c2f_b_fine01_coarse05 | c2f | Step-R | PASS | 12.7364 |
| resnet_c2f_b_fine02_coarse01 | c2f | Step-R | PASS | 12.7367 |
| resnet_c2f_b_fine02_coarse02 | c2f | Step-R | PASS | 11.9332 |
| resnet_c2f_b_fine02_coarse03 | c2f | Step-R | PASS | 12.4707 |
| resnet_c2f_b_fine02_coarse04 | c2f | Step-Q `resnet_c2f_b_f2_c4_global_best` | REUSED |  |
| resnet_c2f_b_fine02_coarse05 | c2f | Step-R | PASS | 12.7557 |
| resnet_c2f_b_fine03_coarse01 | c2f | Step-R | PASS | 14.0791 |
| resnet_c2f_b_fine03_coarse02 | c2f | Step-R | PASS | 14.2393 |
| resnet_c2f_b_fine03_coarse03 | c2f | Step-R | PASS | 13.9513 |
| resnet_c2f_b_fine03_coarse04 | c2f | Step-R | PASS | 13.9647 |
| resnet_c2f_b_fine03_coarse05 | c2f | Step-R | PASS | 13.8048 |
| resnet_c2f_b_fine04_coarse01 | c2f | Step-R | PASS | 20.2981 |
| resnet_c2f_b_fine04_coarse02 | c2f | Step-R | PASS | 20.2788 |
| resnet_c2f_b_fine04_coarse03 | c2f | Step-R | PASS | 20.6191 |
| resnet_c2f_b_fine04_coarse04 | c2f | Step-R | PASS | 20.2556 |
| resnet_c2f_b_fine04_coarse05 | c2f | Step-R | PASS | 20.1804 |
| resnet_c2f_b_fine05_coarse01 | c2f | Step-R | PASS | 17.6014 |
| resnet_c2f_b_fine05_coarse02 | c2f | Step-R | PASS | 17.5824 |
| resnet_c2f_b_fine05_coarse03 | c2f | Step-R | PASS | 18.3274 |
| resnet_c2f_b_fine05_coarse04 | c2f | Step-R | PASS | 18.4147 |
| resnet_c2f_b_fine05_coarse05 | c2f | Step-R | PASS | 18.1378 |
| resnet_c2f_b_fine06_coarse01 | c2f | Step-R | PASS | 16.0435 |
| resnet_c2f_b_fine06_coarse02 | c2f | Step-R | PASS | 16.2667 |
| resnet_c2f_b_fine06_coarse03 | c2f | Step-R | PASS | 16.1641 |
| resnet_c2f_b_fine06_coarse04 | c2f | Step-R | PASS | 16.0451 |
| resnet_c2f_b_fine06_coarse05 | c2f | Step-R | PASS | 16.0022 |
