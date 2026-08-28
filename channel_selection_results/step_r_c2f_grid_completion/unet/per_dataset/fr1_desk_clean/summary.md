# UNET Step-R C2F grid completion (fr1_desk_clean)

- Matched frames: 573; mapping remains gray + sensor depth.
- This database contains only missing Step-R cells. Existing Step-Q rows are intentionally reused externally and never copied or rerun.
- Full selected grid: 40 C2F pairs + 9 direct parents.
- Reused Step-Q rows: 9; new Step-R rows: 40.
- Persisted new rows: 40/40; statuses: {'PASS': 40}.

| Standard label | Kind | Source | Status | ATE mean (cm) |
|---|---|---|---|---:|
| unet_direct_fine01 | direct_fine | Step-Q `unet_enc0_direct_f1_d02_d03_d07_d12_d13_d14` | REUSED |  |
| unet_direct_fine02 | direct_fine | Step-Q `unet_enc0_direct_f2_d03_d07_d12` | REUSED |  |
| unet_direct_fine03 | direct_fine | Step-R | PASS | 6.8130 |
| unet_direct_fine04 | direct_fine | Step-R | PASS | 12.5513 |
| unet_direct_fine05 | direct_fine | Step-Q `unet_enc0_direct_f5_d02_d03_d07_d12_d13` | REUSED |  |
| unet_direct_coarse01 | direct_coarse | Step-R | PASS | 15.1311 |
| unet_direct_coarse02 | direct_coarse | Step-Q `unet_enc1_direct_c2_d00_d05_d06_d17_d18_d30` | REUSED |  |
| unet_direct_coarse03 | direct_coarse | Step-R | PASS | 10.8645 |
| unet_direct_coarse04 | direct_coarse | Step-Q `unet_enc1_direct_c4_d00_d05_d06_d18_d30` | REUSED |  |
| unet_c2f_a_fine01_coarse01 | c2f | Step-R | PASS | 8.3773 |
| unet_c2f_a_fine01_coarse02 | c2f | Step-R | PASS | 8.1947 |
| unet_c2f_a_fine01_coarse03 | c2f | Step-R | PASS | 10.2516 |
| unet_c2f_a_fine01_coarse04 | c2f | Step-Q `unet_c2f_a_f1_c4_negative_high_parent` | REUSED |  |
| unet_c2f_a_fine02_coarse01 | c2f | Step-R | PASS | 9.0124 |
| unet_c2f_a_fine02_coarse02 | c2f | Step-R | PASS | 8.5257 |
| unet_c2f_a_fine02_coarse03 | c2f | Step-R | PASS | 9.1038 |
| unet_c2f_a_fine02_coarse04 | c2f | Step-Q `unet_c2f_a_f2_c4_global_best` | REUSED |  |
| unet_c2f_a_fine03_coarse01 | c2f | Step-R | PASS | 7.9846 |
| unet_c2f_a_fine03_coarse02 | c2f | Step-R | PASS | 7.7931 |
| unet_c2f_a_fine03_coarse03 | c2f | Step-R | PASS | 9.7937 |
| unet_c2f_a_fine03_coarse04 | c2f | Step-R | PASS | 7.1485 |
| unet_c2f_a_fine04_coarse01 | c2f | Step-R | PASS | 9.5264 |
| unet_c2f_a_fine04_coarse02 | c2f | Step-R | PASS | 8.0357 |
| unet_c2f_a_fine04_coarse03 | c2f | Step-R | PASS | 9.6844 |
| unet_c2f_a_fine04_coarse04 | c2f | Step-R | PASS | 8.3182 |
| unet_c2f_a_fine05_coarse01 | c2f | Step-R | PASS | 8.3196 |
| unet_c2f_a_fine05_coarse02 | c2f | Step-R | PASS | 11.9916 |
| unet_c2f_a_fine05_coarse03 | c2f | Step-R | PASS | 9.8489 |
| unet_c2f_a_fine05_coarse04 | c2f | Step-Q `unet_c2f_a_f5_c4_positive_synergy` | REUSED |  |
| unet_c2f_b_fine01_coarse01 | c2f | Step-R | PASS | 7.6815 |
| unet_c2f_b_fine01_coarse02 | c2f | Step-R | PASS | 6.9185 |
| unet_c2f_b_fine01_coarse03 | c2f | Step-R | PASS | 8.4553 |
| unet_c2f_b_fine01_coarse04 | c2f | Step-R | PASS | 7.0335 |
| unet_c2f_b_fine02_coarse01 | c2f | Step-R | PASS | 7.3370 |
| unet_c2f_b_fine02_coarse02 | c2f | Step-Q `unet_c2f_b_f2_c2_variant_contrast` | REUSED |  |
| unet_c2f_b_fine02_coarse03 | c2f | Step-R | PASS | 8.9748 |
| unet_c2f_b_fine02_coarse04 | c2f | Step-R | PASS | 6.9753 |
| unet_c2f_b_fine03_coarse01 | c2f | Step-R | PASS | 8.9312 |
| unet_c2f_b_fine03_coarse02 | c2f | Step-R | PASS | 9.5206 |
| unet_c2f_b_fine03_coarse03 | c2f | Step-R | PASS | 11.4150 |
| unet_c2f_b_fine03_coarse04 | c2f | Step-R | PASS | 9.0560 |
| unet_c2f_b_fine04_coarse01 | c2f | Step-R | PASS | 9.7803 |
| unet_c2f_b_fine04_coarse02 | c2f | Step-R | PASS | 9.7866 |
| unet_c2f_b_fine04_coarse03 | c2f | Step-R | PASS | 11.3204 |
| unet_c2f_b_fine04_coarse04 | c2f | Step-R | PASS | 8.1607 |
| unet_c2f_b_fine05_coarse01 | c2f | Step-R | PASS | 8.7080 |
| unet_c2f_b_fine05_coarse02 | c2f | Step-R | PASS | 7.7760 |
| unet_c2f_b_fine05_coarse03 | c2f | Step-R | PASS | 8.9250 |
| unet_c2f_b_fine05_coarse04 | c2f | Step-R | PASS | 8.0275 |
