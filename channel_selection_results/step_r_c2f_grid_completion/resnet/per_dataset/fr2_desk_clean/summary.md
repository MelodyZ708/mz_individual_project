# RESNET Step-R C2F grid completion (fr2_desk_clean)

- Matched frames: 2893; mapping remains gray + sensor depth.
- This database contains only missing Step-R cells. Existing Step-Q rows are intentionally reused externally and never copied or rerun.
- Full selected grid: 60 C2F pairs + 11 direct parents.
- Reused Step-Q rows: 9; new Step-R rows: 62.
- Persisted new rows: 62/62; statuses: {'PASS': 62}.

| Standard label | Kind | Source | Status | ATE mean (cm) |
|---|---|---|---|---:|
| resnet_direct_fine01 | direct_fine | Step-Q `resnet_conv1_direct_f1_d15_d20_d26_d34` | REUSED |  |
| resnet_direct_fine02 | direct_fine | Step-Q `resnet_conv1_direct_f2_d23_d24_d26_d51_d63` | REUSED |  |
| resnet_direct_fine03 | direct_fine | Step-R | PASS | 4.6702 |
| resnet_direct_fine04 | direct_fine | Step-R | PASS | 6.4544 |
| resnet_direct_fine05 | direct_fine | Step-R | PASS | 6.2675 |
| resnet_direct_fine06 | direct_fine | Step-Q `resnet_conv1_direct_f6_d29_d33_d52` | REUSED |  |
| resnet_direct_coarse01 | direct_coarse | Step-R | PASS | 16.4851 |
| resnet_direct_coarse02 | direct_coarse | Step-R | PASS | 16.5606 |
| resnet_direct_coarse03 | direct_coarse | Step-R | PASS | 16.2098 |
| resnet_direct_coarse04 | direct_coarse | Step-Q `resnet_layer2_direct_c4_d67_d108_d121` | REUSED |  |
| resnet_direct_coarse05 | direct_coarse | Step-Q `resnet_layer2_direct_c5_d108_d121` | REUSED |  |
| resnet_c2f_a_fine01_coarse01 | c2f | Step-R | PASS | 5.1830 |
| resnet_c2f_a_fine01_coarse02 | c2f | Step-R | PASS | 5.1862 |
| resnet_c2f_a_fine01_coarse03 | c2f | Step-R | PASS | 5.1406 |
| resnet_c2f_a_fine01_coarse04 | c2f | Step-R | PASS | 5.1824 |
| resnet_c2f_a_fine01_coarse05 | c2f | Step-R | PASS | 5.2194 |
| resnet_c2f_a_fine02_coarse01 | c2f | Step-R | PASS | 6.4973 |
| resnet_c2f_a_fine02_coarse02 | c2f | Step-R | PASS | 6.3274 |
| resnet_c2f_a_fine02_coarse03 | c2f | Step-R | PASS | 6.3811 |
| resnet_c2f_a_fine02_coarse04 | c2f | Step-Q `resnet_c2f_a_f2_c4_variant_negative` | REUSED |  |
| resnet_c2f_a_fine02_coarse05 | c2f | Step-R | PASS | 6.3549 |
| resnet_c2f_a_fine03_coarse01 | c2f | Step-R | PASS | 5.5153 |
| resnet_c2f_a_fine03_coarse02 | c2f | Step-R | PASS | 5.4214 |
| resnet_c2f_a_fine03_coarse03 | c2f | Step-R | PASS | 5.4253 |
| resnet_c2f_a_fine03_coarse04 | c2f | Step-R | PASS | 5.4736 |
| resnet_c2f_a_fine03_coarse05 | c2f | Step-R | PASS | 5.3826 |
| resnet_c2f_a_fine04_coarse01 | c2f | Step-R | PASS | 6.6671 |
| resnet_c2f_a_fine04_coarse02 | c2f | Step-R | PASS | 6.8948 |
| resnet_c2f_a_fine04_coarse03 | c2f | Step-R | PASS | 6.7193 |
| resnet_c2f_a_fine04_coarse04 | c2f | Step-R | PASS | 6.8691 |
| resnet_c2f_a_fine04_coarse05 | c2f | Step-R | PASS | 6.8354 |
| resnet_c2f_a_fine05_coarse01 | c2f | Step-R | PASS | 7.6940 |
| resnet_c2f_a_fine05_coarse02 | c2f | Step-R | PASS | 7.9225 |
| resnet_c2f_a_fine05_coarse03 | c2f | Step-R | PASS | 7.5999 |
| resnet_c2f_a_fine05_coarse04 | c2f | Step-R | PASS | 7.6960 |
| resnet_c2f_a_fine05_coarse05 | c2f | Step-R | PASS | 7.6518 |
| resnet_c2f_a_fine06_coarse01 | c2f | Step-R | PASS | 6.3652 |
| resnet_c2f_a_fine06_coarse02 | c2f | Step-R | PASS | 6.4825 |
| resnet_c2f_a_fine06_coarse03 | c2f | Step-R | PASS | 6.4026 |
| resnet_c2f_a_fine06_coarse04 | c2f | Step-R | PASS | 6.6038 |
| resnet_c2f_a_fine06_coarse05 | c2f | Step-Q `resnet_c2f_a_f6_c5_positive_synergy` | REUSED |  |
| resnet_c2f_b_fine01_coarse01 | c2f | Step-R | PASS | 4.5598 |
| resnet_c2f_b_fine01_coarse02 | c2f | Step-R | PASS | 4.5825 |
| resnet_c2f_b_fine01_coarse03 | c2f | Step-R | PASS | 4.4777 |
| resnet_c2f_b_fine01_coarse04 | c2f | Step-Q `resnet_c2f_b_f1_c4_negative_high_parent` | REUSED |  |
| resnet_c2f_b_fine01_coarse05 | c2f | Step-R | PASS | 4.5139 |
| resnet_c2f_b_fine02_coarse01 | c2f | Step-R | PASS | 5.7844 |
| resnet_c2f_b_fine02_coarse02 | c2f | Step-R | PASS | 5.7667 |
| resnet_c2f_b_fine02_coarse03 | c2f | Step-R | PASS | 5.7531 |
| resnet_c2f_b_fine02_coarse04 | c2f | Step-Q `resnet_c2f_b_f2_c4_global_best` | REUSED |  |
| resnet_c2f_b_fine02_coarse05 | c2f | Step-R | PASS | 5.7222 |
| resnet_c2f_b_fine03_coarse01 | c2f | Step-R | PASS | 4.7771 |
| resnet_c2f_b_fine03_coarse02 | c2f | Step-R | PASS | 4.7180 |
| resnet_c2f_b_fine03_coarse03 | c2f | Step-R | PASS | 4.7985 |
| resnet_c2f_b_fine03_coarse04 | c2f | Step-R | PASS | 4.7038 |
| resnet_c2f_b_fine03_coarse05 | c2f | Step-R | PASS | 4.6686 |
| resnet_c2f_b_fine04_coarse01 | c2f | Step-R | PASS | 6.4094 |
| resnet_c2f_b_fine04_coarse02 | c2f | Step-R | PASS | 6.4145 |
| resnet_c2f_b_fine04_coarse03 | c2f | Step-R | PASS | 6.4335 |
| resnet_c2f_b_fine04_coarse04 | c2f | Step-R | PASS | 6.4284 |
| resnet_c2f_b_fine04_coarse05 | c2f | Step-R | PASS | 6.4194 |
| resnet_c2f_b_fine05_coarse01 | c2f | Step-R | PASS | 6.6132 |
| resnet_c2f_b_fine05_coarse02 | c2f | Step-R | PASS | 6.5590 |
| resnet_c2f_b_fine05_coarse03 | c2f | Step-R | PASS | 6.3292 |
| resnet_c2f_b_fine05_coarse04 | c2f | Step-R | PASS | 6.2711 |
| resnet_c2f_b_fine05_coarse05 | c2f | Step-R | PASS | 6.3587 |
| resnet_c2f_b_fine06_coarse01 | c2f | Step-R | PASS | 5.9586 |
| resnet_c2f_b_fine06_coarse02 | c2f | Step-R | PASS | 5.9490 |
| resnet_c2f_b_fine06_coarse03 | c2f | Step-R | PASS | 5.9594 |
| resnet_c2f_b_fine06_coarse04 | c2f | Step-R | PASS | 5.9589 |
| resnet_c2f_b_fine06_coarse05 | c2f | Step-R | PASS | 5.9663 |
