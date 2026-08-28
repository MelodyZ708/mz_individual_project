# RESNET Step-R C2F grid completion (fr1_desk_clean)

- Matched frames: 573; mapping remains gray + sensor depth.
- This database contains only missing Step-R cells. Existing Step-Q rows are intentionally reused externally and never copied or rerun.
- Full selected grid: 60 C2F pairs + 11 direct parents.
- Reused Step-Q rows: 9; new Step-R rows: 62.
- Persisted new rows: 62/62; statuses: {'PASS': 62}.

| Standard label | Kind | Source | Status | ATE mean (cm) |
|---|---|---|---|---:|
| resnet_direct_fine01 | direct_fine | Step-Q `resnet_conv1_direct_f1_d15_d20_d26_d34` | REUSED |  |
| resnet_direct_fine02 | direct_fine | Step-Q `resnet_conv1_direct_f2_d23_d24_d26_d51_d63` | REUSED |  |
| resnet_direct_fine03 | direct_fine | Step-R | PASS | 7.7507 |
| resnet_direct_fine04 | direct_fine | Step-R | PASS | 11.5471 |
| resnet_direct_fine05 | direct_fine | Step-R | PASS | 14.7927 |
| resnet_direct_fine06 | direct_fine | Step-Q `resnet_conv1_direct_f6_d29_d33_d52` | REUSED |  |
| resnet_direct_coarse01 | direct_coarse | Step-R | PASS | 19.8825 |
| resnet_direct_coarse02 | direct_coarse | Step-R | PASS | 19.6141 |
| resnet_direct_coarse03 | direct_coarse | Step-R | PASS | 24.6459 |
| resnet_direct_coarse04 | direct_coarse | Step-Q `resnet_layer2_direct_c4_d67_d108_d121` | REUSED |  |
| resnet_direct_coarse05 | direct_coarse | Step-Q `resnet_layer2_direct_c5_d108_d121` | REUSED |  |
| resnet_c2f_a_fine01_coarse01 | c2f | Step-R | PASS | 19.1069 |
| resnet_c2f_a_fine01_coarse02 | c2f | Step-R | PASS | 18.5750 |
| resnet_c2f_a_fine01_coarse03 | c2f | Step-R | PASS | 16.2412 |
| resnet_c2f_a_fine01_coarse04 | c2f | Step-R | PASS | 19.5336 |
| resnet_c2f_a_fine01_coarse05 | c2f | Step-R | PASS | 6.5461 |
| resnet_c2f_a_fine02_coarse01 | c2f | Step-R | PASS | 6.5428 |
| resnet_c2f_a_fine02_coarse02 | c2f | Step-R | PASS | 19.4120 |
| resnet_c2f_a_fine02_coarse03 | c2f | Step-R | PASS | 14.5833 |
| resnet_c2f_a_fine02_coarse04 | c2f | Step-Q `resnet_c2f_a_f2_c4_variant_negative` | REUSED |  |
| resnet_c2f_a_fine02_coarse05 | c2f | Step-R | PASS | 15.4795 |
| resnet_c2f_a_fine03_coarse01 | c2f | Step-R | PASS | 6.6429 |
| resnet_c2f_a_fine03_coarse02 | c2f | Step-R | PASS | 19.1216 |
| resnet_c2f_a_fine03_coarse03 | c2f | Step-R | PASS | 20.6499 |
| resnet_c2f_a_fine03_coarse04 | c2f | Step-R | PASS | 19.1815 |
| resnet_c2f_a_fine03_coarse05 | c2f | Step-R | PASS | 7.0701 |
| resnet_c2f_a_fine04_coarse01 | c2f | Step-R | PASS | 22.7383 |
| resnet_c2f_a_fine04_coarse02 | c2f | Step-R | PASS | 23.2896 |
| resnet_c2f_a_fine04_coarse03 | c2f | Step-R | PASS | 12.3320 |
| resnet_c2f_a_fine04_coarse04 | c2f | Step-R | PASS | 23.2826 |
| resnet_c2f_a_fine04_coarse05 | c2f | Step-R | PASS | 9.9591 |
| resnet_c2f_a_fine05_coarse01 | c2f | Step-R | PASS | 19.6922 |
| resnet_c2f_a_fine05_coarse02 | c2f | Step-R | PASS | 18.4573 |
| resnet_c2f_a_fine05_coarse03 | c2f | Step-R | PASS | 20.0792 |
| resnet_c2f_a_fine05_coarse04 | c2f | Step-R | PASS | 19.6181 |
| resnet_c2f_a_fine05_coarse05 | c2f | Step-R | PASS | 7.9563 |
| resnet_c2f_a_fine06_coarse01 | c2f | Step-R | PASS | 20.0227 |
| resnet_c2f_a_fine06_coarse02 | c2f | Step-R | PASS | 19.3824 |
| resnet_c2f_a_fine06_coarse03 | c2f | Step-R | PASS | 23.9085 |
| resnet_c2f_a_fine06_coarse04 | c2f | Step-R | PASS | 20.2483 |
| resnet_c2f_a_fine06_coarse05 | c2f | Step-Q `resnet_c2f_a_f6_c5_positive_synergy` | REUSED |  |
| resnet_c2f_b_fine01_coarse01 | c2f | Step-R | PASS | 18.9691 |
| resnet_c2f_b_fine01_coarse02 | c2f | Step-R | PASS | 11.3968 |
| resnet_c2f_b_fine01_coarse03 | c2f | Step-R | PASS | 12.5876 |
| resnet_c2f_b_fine01_coarse04 | c2f | Step-Q `resnet_c2f_b_f1_c4_negative_high_parent` | REUSED |  |
| resnet_c2f_b_fine01_coarse05 | c2f | Step-R | PASS | 6.1240 |
| resnet_c2f_b_fine02_coarse01 | c2f | Step-R | PASS | 15.1469 |
| resnet_c2f_b_fine02_coarse02 | c2f | Step-R | PASS | 10.6145 |
| resnet_c2f_b_fine02_coarse03 | c2f | Step-R | PASS | 11.3338 |
| resnet_c2f_b_fine02_coarse04 | c2f | Step-Q `resnet_c2f_b_f2_c4_global_best` | REUSED |  |
| resnet_c2f_b_fine02_coarse05 | c2f | Step-R | PASS | 13.6690 |
| resnet_c2f_b_fine03_coarse01 | c2f | Step-R | PASS | 7.6435 |
| resnet_c2f_b_fine03_coarse02 | c2f | Step-R | PASS | 8.2465 |
| resnet_c2f_b_fine03_coarse03 | c2f | Step-R | PASS | 19.3315 |
| resnet_c2f_b_fine03_coarse04 | c2f | Step-R | PASS | 6.7293 |
| resnet_c2f_b_fine03_coarse05 | c2f | Step-R | PASS | 6.0162 |
| resnet_c2f_b_fine04_coarse01 | c2f | Step-R | PASS | 8.9980 |
| resnet_c2f_b_fine04_coarse02 | c2f | Step-R | PASS | 12.3988 |
| resnet_c2f_b_fine04_coarse03 | c2f | Step-R | PASS | 13.0254 |
| resnet_c2f_b_fine04_coarse04 | c2f | Step-R | PASS | 23.3772 |
| resnet_c2f_b_fine04_coarse05 | c2f | Step-R | PASS | 11.1903 |
| resnet_c2f_b_fine05_coarse01 | c2f | Step-R | PASS | 21.9462 |
| resnet_c2f_b_fine05_coarse02 | c2f | Step-R | PASS | 12.9325 |
| resnet_c2f_b_fine05_coarse03 | c2f | Step-R | PASS | 14.4352 |
| resnet_c2f_b_fine05_coarse04 | c2f | Step-R | PASS | 8.0843 |
| resnet_c2f_b_fine05_coarse05 | c2f | Step-R | PASS | 16.2756 |
| resnet_c2f_b_fine06_coarse01 | c2f | Step-R | PASS | 19.5144 |
| resnet_c2f_b_fine06_coarse02 | c2f | Step-R | PASS | 9.5387 |
| resnet_c2f_b_fine06_coarse03 | c2f | Step-R | PASS | 11.6877 |
| resnet_c2f_b_fine06_coarse04 | c2f | Step-R | PASS | 9.5532 |
| resnet_c2f_b_fine06_coarse05 | c2f | Step-R | PASS | 11.2906 |
