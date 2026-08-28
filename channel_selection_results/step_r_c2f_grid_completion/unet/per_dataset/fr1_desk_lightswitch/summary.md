# UNET Step-R C2F grid completion (fr1_desk_lightswitch)

- Matched frames: 573; mapping remains gray + sensor depth.
- This database contains only missing Step-R cells. Existing Step-Q rows are intentionally reused externally and never copied or rerun.
- Full selected grid: 40 C2F pairs + 9 direct parents.
- Reused Step-Q rows: 9; new Step-R rows: 40.
- Persisted new rows: 40/40; statuses: {'PASS': 40}.

| Standard label | Kind | Source | Status | ATE mean (cm) |
|---|---|---|---|---:|
| unet_direct_fine01 | direct_fine | Step-Q `unet_enc0_direct_f1_d02_d03_d07_d12_d13_d14` | REUSED |  |
| unet_direct_fine02 | direct_fine | Step-Q `unet_enc0_direct_f2_d03_d07_d12` | REUSED |  |
| unet_direct_fine03 | direct_fine | Step-R | PASS | 6.3905 |
| unet_direct_fine04 | direct_fine | Step-R | PASS | 6.6111 |
| unet_direct_fine05 | direct_fine | Step-Q `unet_enc0_direct_f5_d02_d03_d07_d12_d13` | REUSED |  |
| unet_direct_coarse01 | direct_coarse | Step-R | PASS | 6.7335 |
| unet_direct_coarse02 | direct_coarse | Step-Q `unet_enc1_direct_c2_d00_d05_d06_d17_d18_d30` | REUSED |  |
| unet_direct_coarse03 | direct_coarse | Step-R | PASS | 7.1673 |
| unet_direct_coarse04 | direct_coarse | Step-Q `unet_enc1_direct_c4_d00_d05_d06_d18_d30` | REUSED |  |
| unet_c2f_a_fine01_coarse01 | c2f | Step-R | PASS | 7.0912 |
| unet_c2f_a_fine01_coarse02 | c2f | Step-R | PASS | 7.7116 |
| unet_c2f_a_fine01_coarse03 | c2f | Step-R | PASS | 8.7433 |
| unet_c2f_a_fine01_coarse04 | c2f | Step-Q `unet_c2f_a_f1_c4_negative_high_parent` | REUSED |  |
| unet_c2f_a_fine02_coarse01 | c2f | Step-R | PASS | 7.0387 |
| unet_c2f_a_fine02_coarse02 | c2f | Step-R | PASS | 7.2166 |
| unet_c2f_a_fine02_coarse03 | c2f | Step-R | PASS | 8.6616 |
| unet_c2f_a_fine02_coarse04 | c2f | Step-Q `unet_c2f_a_f2_c4_global_best` | REUSED |  |
| unet_c2f_a_fine03_coarse01 | c2f | Step-R | PASS | 8.0483 |
| unet_c2f_a_fine03_coarse02 | c2f | Step-R | PASS | 6.7074 |
| unet_c2f_a_fine03_coarse03 | c2f | Step-R | PASS | 9.3100 |
| unet_c2f_a_fine03_coarse04 | c2f | Step-R | PASS | 6.3974 |
| unet_c2f_a_fine04_coarse01 | c2f | Step-R | PASS | 8.2706 |
| unet_c2f_a_fine04_coarse02 | c2f | Step-R | PASS | 7.0455 |
| unet_c2f_a_fine04_coarse03 | c2f | Step-R | PASS | 12.6357 |
| unet_c2f_a_fine04_coarse04 | c2f | Step-R | PASS | 7.2947 |
| unet_c2f_a_fine05_coarse01 | c2f | Step-R | PASS | 6.9654 |
| unet_c2f_a_fine05_coarse02 | c2f | Step-R | PASS | 8.7757 |
| unet_c2f_a_fine05_coarse03 | c2f | Step-R | PASS | 8.9114 |
| unet_c2f_a_fine05_coarse04 | c2f | Step-Q `unet_c2f_a_f5_c4_positive_synergy` | REUSED |  |
| unet_c2f_b_fine01_coarse01 | c2f | Step-R | PASS | 9.2086 |
| unet_c2f_b_fine01_coarse02 | c2f | Step-R | PASS | 8.5553 |
| unet_c2f_b_fine01_coarse03 | c2f | Step-R | PASS | 7.9626 |
| unet_c2f_b_fine01_coarse04 | c2f | Step-R | PASS | 8.6600 |
| unet_c2f_b_fine02_coarse01 | c2f | Step-R | PASS | 6.7999 |
| unet_c2f_b_fine02_coarse02 | c2f | Step-Q `unet_c2f_b_f2_c2_variant_contrast` | REUSED |  |
| unet_c2f_b_fine02_coarse03 | c2f | Step-R | PASS | 8.5506 |
| unet_c2f_b_fine02_coarse04 | c2f | Step-R | PASS | 6.6030 |
| unet_c2f_b_fine03_coarse01 | c2f | Step-R | PASS | 8.4477 |
| unet_c2f_b_fine03_coarse02 | c2f | Step-R | PASS | 6.5437 |
| unet_c2f_b_fine03_coarse03 | c2f | Step-R | PASS | 8.6384 |
| unet_c2f_b_fine03_coarse04 | c2f | Step-R | PASS | 7.2297 |
| unet_c2f_b_fine04_coarse01 | c2f | Step-R | PASS | 10.1612 |
| unet_c2f_b_fine04_coarse02 | c2f | Step-R | PASS | 12.1087 |
| unet_c2f_b_fine04_coarse03 | c2f | Step-R | PASS | 9.6102 |
| unet_c2f_b_fine04_coarse04 | c2f | Step-R | PASS | 10.3979 |
| unet_c2f_b_fine05_coarse01 | c2f | Step-R | PASS | 10.2823 |
| unet_c2f_b_fine05_coarse02 | c2f | Step-R | PASS | 8.3290 |
| unet_c2f_b_fine05_coarse03 | c2f | Step-R | PASS | 7.7622 |
| unet_c2f_b_fine05_coarse04 | c2f | Step-R | PASS | 6.4618 |
