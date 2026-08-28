# RESNET Step-R C2F grid completion (fr2_desk_flashlight)

- Matched frames: 2893; mapping remains gray + sensor depth.
- This database contains only missing Step-R cells. Existing Step-Q rows are intentionally reused externally and never copied or rerun.
- Full selected grid: 60 C2F pairs + 11 direct parents.
- Reused Step-Q rows: 9; new Step-R rows: 62.
- Persisted new rows: 62/62; statuses: {'PASS': 62}.

| Standard label | Kind | Source | Status | ATE mean (cm) |
|---|---|---|---|---:|
| resnet_direct_fine01 | direct_fine | Step-Q `resnet_conv1_direct_f1_d15_d20_d26_d34` | REUSED |  |
| resnet_direct_fine02 | direct_fine | Step-Q `resnet_conv1_direct_f2_d23_d24_d26_d51_d63` | REUSED |  |
| resnet_direct_fine03 | direct_fine | Step-R | PASS | 4.8912 |
| resnet_direct_fine04 | direct_fine | Step-R | PASS | 6.8948 |
| resnet_direct_fine05 | direct_fine | Step-R | PASS | 6.3370 |
| resnet_direct_fine06 | direct_fine | Step-Q `resnet_conv1_direct_f6_d29_d33_d52` | REUSED |  |
| resnet_direct_coarse01 | direct_coarse | Step-R | PASS | 16.7406 |
| resnet_direct_coarse02 | direct_coarse | Step-R | PASS | 17.8252 |
| resnet_direct_coarse03 | direct_coarse | Step-R | PASS | 16.6129 |
| resnet_direct_coarse04 | direct_coarse | Step-Q `resnet_layer2_direct_c4_d67_d108_d121` | REUSED |  |
| resnet_direct_coarse05 | direct_coarse | Step-Q `resnet_layer2_direct_c5_d108_d121` | REUSED |  |
| resnet_c2f_a_fine01_coarse01 | c2f | Step-R | PASS | 5.3049 |
| resnet_c2f_a_fine01_coarse02 | c2f | Step-R | PASS | 5.4214 |
| resnet_c2f_a_fine01_coarse03 | c2f | Step-R | PASS | 5.2646 |
| resnet_c2f_a_fine01_coarse04 | c2f | Step-R | PASS | 5.4624 |
| resnet_c2f_a_fine01_coarse05 | c2f | Step-R | PASS | 5.2103 |
| resnet_c2f_a_fine02_coarse01 | c2f | Step-R | PASS | 6.7697 |
| resnet_c2f_a_fine02_coarse02 | c2f | Step-R | PASS | 6.6943 |
| resnet_c2f_a_fine02_coarse03 | c2f | Step-R | PASS | 6.4219 |
| resnet_c2f_a_fine02_coarse04 | c2f | Step-Q `resnet_c2f_a_f2_c4_variant_negative` | REUSED |  |
| resnet_c2f_a_fine02_coarse05 | c2f | Step-R | PASS | 6.6567 |
| resnet_c2f_a_fine03_coarse01 | c2f | Step-R | PASS | 5.5890 |
| resnet_c2f_a_fine03_coarse02 | c2f | Step-R | PASS | 5.6462 |
| resnet_c2f_a_fine03_coarse03 | c2f | Step-R | PASS | 5.4995 |
| resnet_c2f_a_fine03_coarse04 | c2f | Step-R | PASS | 5.5403 |
| resnet_c2f_a_fine03_coarse05 | c2f | Step-R | PASS | 5.5208 |
| resnet_c2f_a_fine04_coarse01 | c2f | Step-R | PASS | 6.9529 |
| resnet_c2f_a_fine04_coarse02 | c2f | Step-R | PASS | 7.0406 |
| resnet_c2f_a_fine04_coarse03 | c2f | Step-R | PASS | 7.1680 |
| resnet_c2f_a_fine04_coarse04 | c2f | Step-R | PASS | 7.1082 |
| resnet_c2f_a_fine04_coarse05 | c2f | Step-R | PASS | 7.1985 |
| resnet_c2f_a_fine05_coarse01 | c2f | Step-R | PASS | 7.9998 |
| resnet_c2f_a_fine05_coarse02 | c2f | Step-R | PASS | 8.1715 |
| resnet_c2f_a_fine05_coarse03 | c2f | Step-R | PASS | 7.8628 |
| resnet_c2f_a_fine05_coarse04 | c2f | Step-R | PASS | 7.8709 |
| resnet_c2f_a_fine05_coarse05 | c2f | Step-R | PASS | 7.8981 |
| resnet_c2f_a_fine06_coarse01 | c2f | Step-R | PASS | 6.4622 |
| resnet_c2f_a_fine06_coarse02 | c2f | Step-R | PASS | 6.4702 |
| resnet_c2f_a_fine06_coarse03 | c2f | Step-R | PASS | 6.4582 |
| resnet_c2f_a_fine06_coarse04 | c2f | Step-R | PASS | 6.4128 |
| resnet_c2f_a_fine06_coarse05 | c2f | Step-Q `resnet_c2f_a_f6_c5_positive_synergy` | REUSED |  |
| resnet_c2f_b_fine01_coarse01 | c2f | Step-R | PASS | 4.6733 |
| resnet_c2f_b_fine01_coarse02 | c2f | Step-R | PASS | 4.6957 |
| resnet_c2f_b_fine01_coarse03 | c2f | Step-R | PASS | 4.7013 |
| resnet_c2f_b_fine01_coarse04 | c2f | Step-Q `resnet_c2f_b_f1_c4_negative_high_parent` | REUSED |  |
| resnet_c2f_b_fine01_coarse05 | c2f | Step-R | PASS | 4.6414 |
| resnet_c2f_b_fine02_coarse01 | c2f | Step-R | PASS | 5.9131 |
| resnet_c2f_b_fine02_coarse02 | c2f | Step-R | PASS | 5.8608 |
| resnet_c2f_b_fine02_coarse03 | c2f | Step-R | PASS | 5.8543 |
| resnet_c2f_b_fine02_coarse04 | c2f | Step-Q `resnet_c2f_b_f2_c4_global_best` | REUSED |  |
| resnet_c2f_b_fine02_coarse05 | c2f | Step-R | PASS | 5.7978 |
| resnet_c2f_b_fine03_coarse01 | c2f | Step-R | PASS | 4.9401 |
| resnet_c2f_b_fine03_coarse02 | c2f | Step-R | PASS | 4.9244 |
| resnet_c2f_b_fine03_coarse03 | c2f | Step-R | PASS | 4.9099 |
| resnet_c2f_b_fine03_coarse04 | c2f | Step-R | PASS | 4.8998 |
| resnet_c2f_b_fine03_coarse05 | c2f | Step-R | PASS | 4.8672 |
| resnet_c2f_b_fine04_coarse01 | c2f | Step-R | PASS | 6.9170 |
| resnet_c2f_b_fine04_coarse02 | c2f | Step-R | PASS | 6.7979 |
| resnet_c2f_b_fine04_coarse03 | c2f | Step-R | PASS | 6.9193 |
| resnet_c2f_b_fine04_coarse04 | c2f | Step-R | PASS | 6.8016 |
| resnet_c2f_b_fine04_coarse05 | c2f | Step-R | PASS | 6.8269 |
| resnet_c2f_b_fine05_coarse01 | c2f | Step-R | PASS | 6.4236 |
| resnet_c2f_b_fine05_coarse02 | c2f | Step-R | PASS | 6.6035 |
| resnet_c2f_b_fine05_coarse03 | c2f | Step-R | PASS | 6.5449 |
| resnet_c2f_b_fine05_coarse04 | c2f | Step-R | PASS | 6.5257 |
| resnet_c2f_b_fine05_coarse05 | c2f | Step-R | PASS | 6.6539 |
| resnet_c2f_b_fine06_coarse01 | c2f | Step-R | PASS | 6.1431 |
| resnet_c2f_b_fine06_coarse02 | c2f | Step-R | PASS | 6.1096 |
| resnet_c2f_b_fine06_coarse03 | c2f | Step-R | PASS | 6.1377 |
| resnet_c2f_b_fine06_coarse04 | c2f | Step-R | PASS | 6.1351 |
| resnet_c2f_b_fine06_coarse05 | c2f | Step-R | PASS | 6.1015 |
