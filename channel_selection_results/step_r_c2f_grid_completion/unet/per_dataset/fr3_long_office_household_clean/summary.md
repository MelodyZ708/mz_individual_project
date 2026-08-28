# UNET Step-R C2F grid completion (fr3_long_office_household_clean)

- Matched frames: 2488; mapping remains gray + sensor depth.
- This database contains only missing Step-R cells. Existing Step-Q rows are intentionally reused externally and never copied or rerun.
- Full selected grid: 40 C2F pairs + 9 direct parents.
- Reused Step-Q rows: 9; new Step-R rows: 40.
- Persisted new rows: 40/40; statuses: {'PASS': 40}.

| Standard label | Kind | Source | Status | ATE mean (cm) |
|---|---|---|---|---:|
| unet_direct_fine01 | direct_fine | Step-Q `unet_enc0_direct_f1_d02_d03_d07_d12_d13_d14` | REUSED |  |
| unet_direct_fine02 | direct_fine | Step-Q `unet_enc0_direct_f2_d03_d07_d12` | REUSED |  |
| unet_direct_fine03 | direct_fine | Step-R | PASS | 10.7810 |
| unet_direct_fine04 | direct_fine | Step-R | PASS | 10.3840 |
| unet_direct_fine05 | direct_fine | Step-Q `unet_enc0_direct_f5_d02_d03_d07_d12_d13` | REUSED |  |
| unet_direct_coarse01 | direct_coarse | Step-R | PASS | 15.1487 |
| unet_direct_coarse02 | direct_coarse | Step-Q `unet_enc1_direct_c2_d00_d05_d06_d17_d18_d30` | REUSED |  |
| unet_direct_coarse03 | direct_coarse | Step-R | PASS | 15.1333 |
| unet_direct_coarse04 | direct_coarse | Step-Q `unet_enc1_direct_c4_d00_d05_d06_d18_d30` | REUSED |  |
| unet_c2f_a_fine01_coarse01 | c2f | Step-R | PASS | 10.0738 |
| unet_c2f_a_fine01_coarse02 | c2f | Step-R | PASS | 9.8672 |
| unet_c2f_a_fine01_coarse03 | c2f | Step-R | PASS | 10.6367 |
| unet_c2f_a_fine01_coarse04 | c2f | Step-Q `unet_c2f_a_f1_c4_negative_high_parent` | REUSED |  |
| unet_c2f_a_fine02_coarse01 | c2f | Step-R | PASS | 10.8904 |
| unet_c2f_a_fine02_coarse02 | c2f | Step-R | PASS | 10.2297 |
| unet_c2f_a_fine02_coarse03 | c2f | Step-R | PASS | 10.5115 |
| unet_c2f_a_fine02_coarse04 | c2f | Step-Q `unet_c2f_a_f2_c4_global_best` | REUSED |  |
| unet_c2f_a_fine03_coarse01 | c2f | Step-R | PASS | 10.2750 |
| unet_c2f_a_fine03_coarse02 | c2f | Step-R | PASS | 9.9465 |
| unet_c2f_a_fine03_coarse03 | c2f | Step-R | PASS | 10.6055 |
| unet_c2f_a_fine03_coarse04 | c2f | Step-R | PASS | 10.0799 |
| unet_c2f_a_fine04_coarse01 | c2f | Step-R | PASS | 10.1597 |
| unet_c2f_a_fine04_coarse02 | c2f | Step-R | PASS | 9.7335 |
| unet_c2f_a_fine04_coarse03 | c2f | Step-R | PASS | 10.4298 |
| unet_c2f_a_fine04_coarse04 | c2f | Step-R | PASS | 9.7715 |
| unet_c2f_a_fine05_coarse01 | c2f | Step-R | PASS | 10.0423 |
| unet_c2f_a_fine05_coarse02 | c2f | Step-R | PASS | 9.6387 |
| unet_c2f_a_fine05_coarse03 | c2f | Step-R | PASS | 10.2428 |
| unet_c2f_a_fine05_coarse04 | c2f | Step-Q `unet_c2f_a_f5_c4_positive_synergy` | REUSED |  |
| unet_c2f_b_fine01_coarse01 | c2f | Step-R | PASS | 10.3309 |
| unet_c2f_b_fine01_coarse02 | c2f | Step-R | PASS | 10.3277 |
| unet_c2f_b_fine01_coarse03 | c2f | Step-R | PASS | 10.6309 |
| unet_c2f_b_fine01_coarse04 | c2f | Step-R | PASS | 10.2894 |
| unet_c2f_b_fine02_coarse01 | c2f | Step-R | PASS | 10.2464 |
| unet_c2f_b_fine02_coarse02 | c2f | Step-Q `unet_c2f_b_f2_c2_variant_contrast` | REUSED |  |
| unet_c2f_b_fine02_coarse03 | c2f | Step-R | PASS | 10.2848 |
| unet_c2f_b_fine02_coarse04 | c2f | Step-R | PASS | 10.3806 |
| unet_c2f_b_fine03_coarse01 | c2f | Step-R | PASS | 10.5261 |
| unet_c2f_b_fine03_coarse02 | c2f | Step-R | PASS | 10.6931 |
| unet_c2f_b_fine03_coarse03 | c2f | Step-R | PASS | 10.8245 |
| unet_c2f_b_fine03_coarse04 | c2f | Step-R | PASS | 10.6218 |
| unet_c2f_b_fine04_coarse01 | c2f | Step-R | PASS | 10.1573 |
| unet_c2f_b_fine04_coarse02 | c2f | Step-R | PASS | 9.7383 |
| unet_c2f_b_fine04_coarse03 | c2f | Step-R | PASS | 9.9351 |
| unet_c2f_b_fine04_coarse04 | c2f | Step-R | PASS | 9.7405 |
| unet_c2f_b_fine05_coarse01 | c2f | Step-R | PASS | 9.9684 |
| unet_c2f_b_fine05_coarse02 | c2f | Step-R | PASS | 10.0505 |
| unet_c2f_b_fine05_coarse03 | c2f | Step-R | PASS | 10.0672 |
| unet_c2f_b_fine05_coarse04 | c2f | Step-R | PASS | 10.0122 |
