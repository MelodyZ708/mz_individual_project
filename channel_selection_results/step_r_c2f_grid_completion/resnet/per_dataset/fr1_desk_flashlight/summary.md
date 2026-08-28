# RESNET Step-R C2F grid completion (fr1_desk_flashlight)

- Matched frames: 573; mapping remains gray + sensor depth.
- This database contains only missing Step-R cells. Existing Step-Q rows are intentionally reused externally and never copied or rerun.
- Full selected grid: 60 C2F pairs + 11 direct parents.
- Reused Step-Q rows: 9; new Step-R rows: 62.
- Persisted new rows: 62/62; statuses: {'PASS': 62}.

| Standard label | Kind | Source | Status | ATE mean (cm) |
|---|---|---|---|---:|
| resnet_direct_fine01 | direct_fine | Step-Q `resnet_conv1_direct_f1_d15_d20_d26_d34` | REUSED |  |
| resnet_direct_fine02 | direct_fine | Step-Q `resnet_conv1_direct_f2_d23_d24_d26_d51_d63` | REUSED |  |
| resnet_direct_fine03 | direct_fine | Step-R | PASS | 8.8357 |
| resnet_direct_fine04 | direct_fine | Step-R | PASS | 11.9759 |
| resnet_direct_fine05 | direct_fine | Step-R | PASS | 10.9510 |
| resnet_direct_fine06 | direct_fine | Step-Q `resnet_conv1_direct_f6_d29_d33_d52` | REUSED |  |
| resnet_direct_coarse01 | direct_coarse | Step-R | PASS | 15.1930 |
| resnet_direct_coarse02 | direct_coarse | Step-R | PASS | 16.8671 |
| resnet_direct_coarse03 | direct_coarse | Step-R | PASS | 28.6973 |
| resnet_direct_coarse04 | direct_coarse | Step-Q `resnet_layer2_direct_c4_d67_d108_d121` | REUSED |  |
| resnet_direct_coarse05 | direct_coarse | Step-Q `resnet_layer2_direct_c5_d108_d121` | REUSED |  |
| resnet_c2f_a_fine01_coarse01 | c2f | Step-R | PASS | 6.7160 |
| resnet_c2f_a_fine01_coarse02 | c2f | Step-R | PASS | 20.7007 |
| resnet_c2f_a_fine01_coarse03 | c2f | Step-R | PASS | 21.9253 |
| resnet_c2f_a_fine01_coarse04 | c2f | Step-R | PASS | 17.8342 |
| resnet_c2f_a_fine01_coarse05 | c2f | Step-R | PASS | 7.1785 |
| resnet_c2f_a_fine02_coarse01 | c2f | Step-R | PASS | 6.7359 |
| resnet_c2f_a_fine02_coarse02 | c2f | Step-R | PASS | 16.4031 |
| resnet_c2f_a_fine02_coarse03 | c2f | Step-R | PASS | 29.6283 |
| resnet_c2f_a_fine02_coarse04 | c2f | Step-Q `resnet_c2f_a_f2_c4_variant_negative` | REUSED |  |
| resnet_c2f_a_fine02_coarse05 | c2f | Step-R | PASS | 6.9996 |
| resnet_c2f_a_fine03_coarse01 | c2f | Step-R | PASS | 18.5982 |
| resnet_c2f_a_fine03_coarse02 | c2f | Step-R | PASS | 16.7314 |
| resnet_c2f_a_fine03_coarse03 | c2f | Step-R | PASS | 8.2068 |
| resnet_c2f_a_fine03_coarse04 | c2f | Step-R | PASS | 17.7114 |
| resnet_c2f_a_fine03_coarse05 | c2f | Step-R | PASS | 6.4370 |
| resnet_c2f_a_fine04_coarse01 | c2f | Step-R | PASS | 11.8992 |
| resnet_c2f_a_fine04_coarse02 | c2f | Step-R | PASS | 13.1660 |
| resnet_c2f_a_fine04_coarse03 | c2f | Step-R | PASS | 24.2790 |
| resnet_c2f_a_fine04_coarse04 | c2f | Step-R | PASS | 21.3796 |
| resnet_c2f_a_fine04_coarse05 | c2f | Step-R | PASS | 9.3163 |
| resnet_c2f_a_fine05_coarse01 | c2f | Step-R | PASS | 12.9009 |
| resnet_c2f_a_fine05_coarse02 | c2f | Step-R | PASS | 9.8646 |
| resnet_c2f_a_fine05_coarse03 | c2f | Step-R | PASS | 33.6898 |
| resnet_c2f_a_fine05_coarse04 | c2f | Step-R | PASS | 19.1091 |
| resnet_c2f_a_fine05_coarse05 | c2f | Step-R | PASS | 7.1071 |
| resnet_c2f_a_fine06_coarse01 | c2f | Step-R | PASS | 14.7164 |
| resnet_c2f_a_fine06_coarse02 | c2f | Step-R | PASS | 12.5569 |
| resnet_c2f_a_fine06_coarse03 | c2f | Step-R | PASS | 22.7723 |
| resnet_c2f_a_fine06_coarse04 | c2f | Step-R | PASS | 18.3620 |
| resnet_c2f_a_fine06_coarse05 | c2f | Step-Q `resnet_c2f_a_f6_c5_positive_synergy` | REUSED |  |
| resnet_c2f_b_fine01_coarse01 | c2f | Step-R | PASS | 7.2238 |
| resnet_c2f_b_fine01_coarse02 | c2f | Step-R | PASS | 11.1160 |
| resnet_c2f_b_fine01_coarse03 | c2f | Step-R | PASS | 21.3900 |
| resnet_c2f_b_fine01_coarse04 | c2f | Step-Q `resnet_c2f_b_f1_c4_negative_high_parent` | REUSED |  |
| resnet_c2f_b_fine01_coarse05 | c2f | Step-R | PASS | 6.7830 |
| resnet_c2f_b_fine02_coarse01 | c2f | Step-R | PASS | 6.7701 |
| resnet_c2f_b_fine02_coarse02 | c2f | Step-R | PASS | 7.1244 |
| resnet_c2f_b_fine02_coarse03 | c2f | Step-R | PASS | 24.8113 |
| resnet_c2f_b_fine02_coarse04 | c2f | Step-Q `resnet_c2f_b_f2_c4_global_best` | REUSED |  |
| resnet_c2f_b_fine02_coarse05 | c2f | Step-R | PASS | 7.1504 |
| resnet_c2f_b_fine03_coarse01 | c2f | Step-R | PASS | 17.7165 |
| resnet_c2f_b_fine03_coarse02 | c2f | Step-R | PASS | 17.6305 |
| resnet_c2f_b_fine03_coarse03 | c2f | Step-R | PASS | 20.2331 |
| resnet_c2f_b_fine03_coarse04 | c2f | Step-R | PASS | 17.4756 |
| resnet_c2f_b_fine03_coarse05 | c2f | Step-R | PASS | 6.3413 |
| resnet_c2f_b_fine04_coarse01 | c2f | Step-R | PASS | 11.0238 |
| resnet_c2f_b_fine04_coarse02 | c2f | Step-R | PASS | 9.7404 |
| resnet_c2f_b_fine04_coarse03 | c2f | Step-R | PASS | 24.9373 |
| resnet_c2f_b_fine04_coarse04 | c2f | Step-R | PASS | 14.2496 |
| resnet_c2f_b_fine04_coarse05 | c2f | Step-R | PASS | 9.5218 |
| resnet_c2f_b_fine05_coarse01 | c2f | Step-R | PASS | 10.7659 |
| resnet_c2f_b_fine05_coarse02 | c2f | Step-R | PASS | 8.3919 |
| resnet_c2f_b_fine05_coarse03 | c2f | Step-R | PASS | 23.1745 |
| resnet_c2f_b_fine05_coarse04 | c2f | Step-R | PASS | 17.3253 |
| resnet_c2f_b_fine05_coarse05 | c2f | Step-R | PASS | 7.4290 |
| resnet_c2f_b_fine06_coarse01 | c2f | Step-R | PASS | 10.0458 |
| resnet_c2f_b_fine06_coarse02 | c2f | Step-R | PASS | 12.8226 |
| resnet_c2f_b_fine06_coarse03 | c2f | Step-R | PASS | 22.0207 |
| resnet_c2f_b_fine06_coarse04 | c2f | Step-R | PASS | 18.1728 |
| resnet_c2f_b_fine06_coarse05 | c2f | Step-R | PASS | 10.0847 |
