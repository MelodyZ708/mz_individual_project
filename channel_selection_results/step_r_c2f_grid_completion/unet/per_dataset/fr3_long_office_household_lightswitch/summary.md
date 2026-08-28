# UNET Step-R C2F grid completion (fr3_long_office_household_lightswitch)

- Matched frames: 2488; mapping remains gray + sensor depth.
- This database contains only missing Step-R cells. Existing Step-Q rows are intentionally reused externally and never copied or rerun.
- Full selected grid: 40 C2F pairs + 9 direct parents.
- Reused Step-Q rows: 9; new Step-R rows: 40.
- Persisted new rows: 40/40; statuses: {'PASS': 40}.

| Standard label | Kind | Source | Status | ATE mean (cm) |
|---|---|---|---|---:|
| unet_direct_fine01 | direct_fine | Step-Q `unet_enc0_direct_f1_d02_d03_d07_d12_d13_d14` | REUSED |  |
| unet_direct_fine02 | direct_fine | Step-Q `unet_enc0_direct_f2_d03_d07_d12` | REUSED |  |
| unet_direct_fine03 | direct_fine | Step-R | PASS | 24.1515 |
| unet_direct_fine04 | direct_fine | Step-R | PASS | 23.1239 |
| unet_direct_fine05 | direct_fine | Step-Q `unet_enc0_direct_f5_d02_d03_d07_d12_d13` | REUSED |  |
| unet_direct_coarse01 | direct_coarse | Step-R | PASS | 15.4466 |
| unet_direct_coarse02 | direct_coarse | Step-Q `unet_enc1_direct_c2_d00_d05_d06_d17_d18_d30` | REUSED |  |
| unet_direct_coarse03 | direct_coarse | Step-R | PASS | 17.1518 |
| unet_direct_coarse04 | direct_coarse | Step-Q `unet_enc1_direct_c4_d00_d05_d06_d18_d30` | REUSED |  |
| unet_c2f_a_fine01_coarse01 | c2f | Step-R | PASS | 8.5638 |
| unet_c2f_a_fine01_coarse02 | c2f | Step-R | PASS | 9.0954 |
| unet_c2f_a_fine01_coarse03 | c2f | Step-R | PASS | 14.2646 |
| unet_c2f_a_fine01_coarse04 | c2f | Step-Q `unet_c2f_a_f1_c4_negative_high_parent` | REUSED |  |
| unet_c2f_a_fine02_coarse01 | c2f | Step-R | PASS | 10.5103 |
| unet_c2f_a_fine02_coarse02 | c2f | Step-R | PASS | 10.8157 |
| unet_c2f_a_fine02_coarse03 | c2f | Step-R | PASS | 14.4598 |
| unet_c2f_a_fine02_coarse04 | c2f | Step-Q `unet_c2f_a_f2_c4_global_best` | REUSED |  |
| unet_c2f_a_fine03_coarse01 | c2f | Step-R | PASS | 10.7143 |
| unet_c2f_a_fine03_coarse02 | c2f | Step-R | PASS | 9.8981 |
| unet_c2f_a_fine03_coarse03 | c2f | Step-R | PASS | 13.0690 |
| unet_c2f_a_fine03_coarse04 | c2f | Step-R | PASS | 9.2581 |
| unet_c2f_a_fine04_coarse01 | c2f | Step-R | PASS | 10.4577 |
| unet_c2f_a_fine04_coarse02 | c2f | Step-R | PASS | 10.6681 |
| unet_c2f_a_fine04_coarse03 | c2f | Step-R | PASS | 14.8795 |
| unet_c2f_a_fine04_coarse04 | c2f | Step-R | PASS | 10.3274 |
| unet_c2f_a_fine05_coarse01 | c2f | Step-R | PASS | 10.7446 |
| unet_c2f_a_fine05_coarse02 | c2f | Step-R | PASS | 9.9186 |
| unet_c2f_a_fine05_coarse03 | c2f | Step-R | PASS | 12.0701 |
| unet_c2f_a_fine05_coarse04 | c2f | Step-Q `unet_c2f_a_f5_c4_positive_synergy` | REUSED |  |
| unet_c2f_b_fine01_coarse01 | c2f | Step-R | PASS | 9.7966 |
| unet_c2f_b_fine01_coarse02 | c2f | Step-R | PASS | 8.9158 |
| unet_c2f_b_fine01_coarse03 | c2f | Step-R | PASS | 13.5287 |
| unet_c2f_b_fine01_coarse04 | c2f | Step-R | PASS | 11.0251 |
| unet_c2f_b_fine02_coarse01 | c2f | Step-R | PASS | 12.5458 |
| unet_c2f_b_fine02_coarse02 | c2f | Step-Q `unet_c2f_b_f2_c2_variant_contrast` | REUSED |  |
| unet_c2f_b_fine02_coarse03 | c2f | Step-R | PASS | 11.7118 |
| unet_c2f_b_fine02_coarse04 | c2f | Step-R | PASS | 11.6103 |
| unet_c2f_b_fine03_coarse01 | c2f | Step-R | PASS | 11.5703 |
| unet_c2f_b_fine03_coarse02 | c2f | Step-R | PASS | 12.5083 |
| unet_c2f_b_fine03_coarse03 | c2f | Step-R | PASS | 12.5697 |
| unet_c2f_b_fine03_coarse04 | c2f | Step-R | PASS | 12.0930 |
| unet_c2f_b_fine04_coarse01 | c2f | Step-R | PASS | 14.3684 |
| unet_c2f_b_fine04_coarse02 | c2f | Step-R | PASS | 14.9575 |
| unet_c2f_b_fine04_coarse03 | c2f | Step-R | PASS | 17.5417 |
| unet_c2f_b_fine04_coarse04 | c2f | Step-R | PASS | 14.2479 |
| unet_c2f_b_fine05_coarse01 | c2f | Step-R | PASS | 10.4713 |
| unet_c2f_b_fine05_coarse02 | c2f | Step-R | PASS | 10.0042 |
| unet_c2f_b_fine05_coarse03 | c2f | Step-R | PASS | 15.2758 |
| unet_c2f_b_fine05_coarse04 | c2f | Step-R | PASS | 10.7314 |
