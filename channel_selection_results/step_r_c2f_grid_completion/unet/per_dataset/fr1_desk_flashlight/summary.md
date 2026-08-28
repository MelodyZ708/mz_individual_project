# UNET Step-R C2F grid completion (fr1_desk_flashlight)

- Matched frames: 573; mapping remains gray + sensor depth.
- This database contains only missing Step-R cells. Existing Step-Q rows are intentionally reused externally and never copied or rerun.
- Full selected grid: 40 C2F pairs + 9 direct parents.
- Reused Step-Q rows: 9; new Step-R rows: 40.
- Persisted new rows: 40/40; statuses: {'PASS': 40}.

| Standard label | Kind | Source | Status | ATE mean (cm) |
|---|---|---|---|---:|
| unet_direct_fine01 | direct_fine | Step-Q `unet_enc0_direct_f1_d02_d03_d07_d12_d13_d14` | REUSED |  |
| unet_direct_fine02 | direct_fine | Step-Q `unet_enc0_direct_f2_d03_d07_d12` | REUSED |  |
| unet_direct_fine03 | direct_fine | Step-R | PASS | 5.9763 |
| unet_direct_fine04 | direct_fine | Step-R | PASS | 9.0889 |
| unet_direct_fine05 | direct_fine | Step-Q `unet_enc0_direct_f5_d02_d03_d07_d12_d13` | REUSED |  |
| unet_direct_coarse01 | direct_coarse | Step-R | PASS | 10.0602 |
| unet_direct_coarse02 | direct_coarse | Step-Q `unet_enc1_direct_c2_d00_d05_d06_d17_d18_d30` | REUSED |  |
| unet_direct_coarse03 | direct_coarse | Step-R | PASS | 13.9928 |
| unet_direct_coarse04 | direct_coarse | Step-Q `unet_enc1_direct_c4_d00_d05_d06_d18_d30` | REUSED |  |
| unet_c2f_a_fine01_coarse01 | c2f | Step-R | PASS | 9.2907 |
| unet_c2f_a_fine01_coarse02 | c2f | Step-R | PASS | 8.1456 |
| unet_c2f_a_fine01_coarse03 | c2f | Step-R | PASS | 9.6066 |
| unet_c2f_a_fine01_coarse04 | c2f | Step-Q `unet_c2f_a_f1_c4_negative_high_parent` | REUSED |  |
| unet_c2f_a_fine02_coarse01 | c2f | Step-R | PASS | 8.2108 |
| unet_c2f_a_fine02_coarse02 | c2f | Step-R | PASS | 8.0658 |
| unet_c2f_a_fine02_coarse03 | c2f | Step-R | PASS | 8.5274 |
| unet_c2f_a_fine02_coarse04 | c2f | Step-Q `unet_c2f_a_f2_c4_global_best` | REUSED |  |
| unet_c2f_a_fine03_coarse01 | c2f | Step-R | PASS | 7.7558 |
| unet_c2f_a_fine03_coarse02 | c2f | Step-R | PASS | 7.4010 |
| unet_c2f_a_fine03_coarse03 | c2f | Step-R | PASS | 10.0466 |
| unet_c2f_a_fine03_coarse04 | c2f | Step-R | PASS | 7.3787 |
| unet_c2f_a_fine04_coarse01 | c2f | Step-R | PASS | 8.8122 |
| unet_c2f_a_fine04_coarse02 | c2f | Step-R | PASS | 7.6049 |
| unet_c2f_a_fine04_coarse03 | c2f | Step-R | PASS | 9.3564 |
| unet_c2f_a_fine04_coarse04 | c2f | Step-R | PASS | 7.5316 |
| unet_c2f_a_fine05_coarse01 | c2f | Step-R | PASS | 8.4867 |
| unet_c2f_a_fine05_coarse02 | c2f | Step-R | PASS | 7.8173 |
| unet_c2f_a_fine05_coarse03 | c2f | Step-R | PASS | 9.4729 |
| unet_c2f_a_fine05_coarse04 | c2f | Step-Q `unet_c2f_a_f5_c4_positive_synergy` | REUSED |  |
| unet_c2f_b_fine01_coarse01 | c2f | Step-R | PASS | 7.1835 |
| unet_c2f_b_fine01_coarse02 | c2f | Step-R | PASS | 7.5882 |
| unet_c2f_b_fine01_coarse03 | c2f | Step-R | PASS | 8.6465 |
| unet_c2f_b_fine01_coarse04 | c2f | Step-R | PASS | 7.5091 |
| unet_c2f_b_fine02_coarse01 | c2f | Step-R | PASS | 7.9729 |
| unet_c2f_b_fine02_coarse02 | c2f | Step-Q `unet_c2f_b_f2_c2_variant_contrast` | REUSED |  |
| unet_c2f_b_fine02_coarse03 | c2f | Step-R | PASS | 9.0291 |
| unet_c2f_b_fine02_coarse04 | c2f | Step-R | PASS | 7.5714 |
| unet_c2f_b_fine03_coarse01 | c2f | Step-R | PASS | 8.6300 |
| unet_c2f_b_fine03_coarse02 | c2f | Step-R | PASS | 8.7742 |
| unet_c2f_b_fine03_coarse03 | c2f | Step-R | PASS | 11.5041 |
| unet_c2f_b_fine03_coarse04 | c2f | Step-R | PASS | 9.6391 |
| unet_c2f_b_fine04_coarse01 | c2f | Step-R | PASS | 9.7285 |
| unet_c2f_b_fine04_coarse02 | c2f | Step-R | PASS | 9.9515 |
| unet_c2f_b_fine04_coarse03 | c2f | Step-R | PASS | 10.7942 |
| unet_c2f_b_fine04_coarse04 | c2f | Step-R | PASS | 9.2535 |
| unet_c2f_b_fine05_coarse01 | c2f | Step-R | PASS | 7.9444 |
| unet_c2f_b_fine05_coarse02 | c2f | Step-R | PASS | 8.1676 |
| unet_c2f_b_fine05_coarse03 | c2f | Step-R | PASS | 8.3646 |
| unet_c2f_b_fine05_coarse04 | c2f | Step-R | PASS | 8.2289 |
