# UNET Step-R C2F grid completion (fr2_desk_lightswitch)

- Matched frames: 2893; mapping remains gray + sensor depth.
- This database contains only missing Step-R cells. Existing Step-Q rows are intentionally reused externally and never copied or rerun.
- Full selected grid: 40 C2F pairs + 9 direct parents.
- Reused Step-Q rows: 9; new Step-R rows: 40.
- Persisted new rows: 40/40; statuses: {'PASS': 40}.

| Standard label | Kind | Source | Status | ATE mean (cm) |
|---|---|---|---|---:|
| unet_direct_fine01 | direct_fine | Step-Q `unet_enc0_direct_f1_d02_d03_d07_d12_d13_d14` | REUSED |  |
| unet_direct_fine02 | direct_fine | Step-Q `unet_enc0_direct_f2_d03_d07_d12` | REUSED |  |
| unet_direct_fine03 | direct_fine | Step-R | PASS | 5.1419 |
| unet_direct_fine04 | direct_fine | Step-R | PASS | 6.7607 |
| unet_direct_fine05 | direct_fine | Step-Q `unet_enc0_direct_f5_d02_d03_d07_d12_d13` | REUSED |  |
| unet_direct_coarse01 | direct_coarse | Step-R | PASS | 6.0189 |
| unet_direct_coarse02 | direct_coarse | Step-Q `unet_enc1_direct_c2_d00_d05_d06_d17_d18_d30` | REUSED |  |
| unet_direct_coarse03 | direct_coarse | Step-R | PASS | 6.3292 |
| unet_direct_coarse04 | direct_coarse | Step-Q `unet_enc1_direct_c4_d00_d05_d06_d18_d30` | REUSED |  |
| unet_c2f_a_fine01_coarse01 | c2f | Step-R | PASS | 4.2986 |
| unet_c2f_a_fine01_coarse02 | c2f | Step-R | PASS | 4.4374 |
| unet_c2f_a_fine01_coarse03 | c2f | Step-R | PASS | 4.3372 |
| unet_c2f_a_fine01_coarse04 | c2f | Step-Q `unet_c2f_a_f1_c4_negative_high_parent` | REUSED |  |
| unet_c2f_a_fine02_coarse01 | c2f | Step-R | PASS | 4.1429 |
| unet_c2f_a_fine02_coarse02 | c2f | Step-R | PASS | 3.8583 |
| unet_c2f_a_fine02_coarse03 | c2f | Step-R | PASS | 4.1530 |
| unet_c2f_a_fine02_coarse04 | c2f | Step-Q `unet_c2f_a_f2_c4_global_best` | REUSED |  |
| unet_c2f_a_fine03_coarse01 | c2f | Step-R | PASS | 4.3947 |
| unet_c2f_a_fine03_coarse02 | c2f | Step-R | PASS | 4.1012 |
| unet_c2f_a_fine03_coarse03 | c2f | Step-R | PASS | 4.2864 |
| unet_c2f_a_fine03_coarse04 | c2f | Step-R | PASS | 4.0706 |
| unet_c2f_a_fine04_coarse01 | c2f | Step-R | PASS | 4.3409 |
| unet_c2f_a_fine04_coarse02 | c2f | Step-R | PASS | 3.9895 |
| unet_c2f_a_fine04_coarse03 | c2f | Step-R | PASS | 4.2563 |
| unet_c2f_a_fine04_coarse04 | c2f | Step-R | PASS | 3.9172 |
| unet_c2f_a_fine05_coarse01 | c2f | Step-R | PASS | 4.1340 |
| unet_c2f_a_fine05_coarse02 | c2f | Step-R | PASS | 3.9565 |
| unet_c2f_a_fine05_coarse03 | c2f | Step-R | PASS | 4.3029 |
| unet_c2f_a_fine05_coarse04 | c2f | Step-Q `unet_c2f_a_f5_c4_positive_synergy` | REUSED |  |
| unet_c2f_b_fine01_coarse01 | c2f | Step-R | PASS | 5.6000 |
| unet_c2f_b_fine01_coarse02 | c2f | Step-R | PASS | 5.7384 |
| unet_c2f_b_fine01_coarse03 | c2f | Step-R | PASS | 4.6512 |
| unet_c2f_b_fine01_coarse04 | c2f | Step-R | PASS | 4.4849 |
| unet_c2f_b_fine02_coarse01 | c2f | Step-R | PASS | 4.4330 |
| unet_c2f_b_fine02_coarse02 | c2f | Step-Q `unet_c2f_b_f2_c2_variant_contrast` | REUSED |  |
| unet_c2f_b_fine02_coarse03 | c2f | Step-R | PASS | 4.4275 |
| unet_c2f_b_fine02_coarse04 | c2f | Step-R | PASS | 4.3204 |
| unet_c2f_b_fine03_coarse01 | c2f | Step-R | PASS | 4.6366 |
| unet_c2f_b_fine03_coarse02 | c2f | Step-R | PASS | 4.5657 |
| unet_c2f_b_fine03_coarse03 | c2f | Step-R | PASS | 4.5912 |
| unet_c2f_b_fine03_coarse04 | c2f | Step-R | PASS | 4.4560 |
| unet_c2f_b_fine04_coarse01 | c2f | Step-R | PASS | 5.4690 |
| unet_c2f_b_fine04_coarse02 | c2f | Step-R | PASS | 5.2067 |
| unet_c2f_b_fine04_coarse03 | c2f | Step-R | PASS | 5.4545 |
| unet_c2f_b_fine04_coarse04 | c2f | Step-R | PASS | 5.1824 |
| unet_c2f_b_fine05_coarse01 | c2f | Step-R | PASS | 5.4797 |
| unet_c2f_b_fine05_coarse02 | c2f | Step-R | PASS | 5.0856 |
| unet_c2f_b_fine05_coarse03 | c2f | Step-R | PASS | 5.3115 |
| unet_c2f_b_fine05_coarse04 | c2f | Step-R | PASS | 4.7356 |
