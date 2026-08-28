# UNET Step-R C2F grid completion (fr2_desk_flashlight)

- Matched frames: 2893; mapping remains gray + sensor depth.
- This database contains only missing Step-R cells. Existing Step-Q rows are intentionally reused externally and never copied or rerun.
- Full selected grid: 40 C2F pairs + 9 direct parents.
- Reused Step-Q rows: 9; new Step-R rows: 40.
- Persisted new rows: 40/40; statuses: {'PASS': 40}.

| Standard label | Kind | Source | Status | ATE mean (cm) |
|---|---|---|---|---:|
| unet_direct_fine01 | direct_fine | Step-Q `unet_enc0_direct_f1_d02_d03_d07_d12_d13_d14` | REUSED |  |
| unet_direct_fine02 | direct_fine | Step-Q `unet_enc0_direct_f2_d03_d07_d12` | REUSED |  |
| unet_direct_fine03 | direct_fine | Step-R | PASS | 3.5523 |
| unet_direct_fine04 | direct_fine | Step-R | PASS | 3.7374 |
| unet_direct_fine05 | direct_fine | Step-Q `unet_enc0_direct_f5_d02_d03_d07_d12_d13` | REUSED |  |
| unet_direct_coarse01 | direct_coarse | Step-R | PASS | 4.9727 |
| unet_direct_coarse02 | direct_coarse | Step-Q `unet_enc1_direct_c2_d00_d05_d06_d17_d18_d30` | REUSED |  |
| unet_direct_coarse03 | direct_coarse | Step-R | PASS | 6.4378 |
| unet_direct_coarse04 | direct_coarse | Step-Q `unet_enc1_direct_c4_d00_d05_d06_d18_d30` | REUSED |  |
| unet_c2f_a_fine01_coarse01 | c2f | Step-R | PASS | 3.4579 |
| unet_c2f_a_fine01_coarse02 | c2f | Step-R | PASS | 3.4300 |
| unet_c2f_a_fine01_coarse03 | c2f | Step-R | PASS | 3.6343 |
| unet_c2f_a_fine01_coarse04 | c2f | Step-Q `unet_c2f_a_f1_c4_negative_high_parent` | REUSED |  |
| unet_c2f_a_fine02_coarse01 | c2f | Step-R | PASS | 3.1518 |
| unet_c2f_a_fine02_coarse02 | c2f | Step-R | PASS | 3.1183 |
| unet_c2f_a_fine02_coarse03 | c2f | Step-R | PASS | 3.2349 |
| unet_c2f_a_fine02_coarse04 | c2f | Step-Q `unet_c2f_a_f2_c4_global_best` | REUSED |  |
| unet_c2f_a_fine03_coarse01 | c2f | Step-R | PASS | 3.5226 |
| unet_c2f_a_fine03_coarse02 | c2f | Step-R | PASS | 3.4020 |
| unet_c2f_a_fine03_coarse03 | c2f | Step-R | PASS | 3.5802 |
| unet_c2f_a_fine03_coarse04 | c2f | Step-R | PASS | 3.4252 |
| unet_c2f_a_fine04_coarse01 | c2f | Step-R | PASS | 3.3044 |
| unet_c2f_a_fine04_coarse02 | c2f | Step-R | PASS | 3.2224 |
| unet_c2f_a_fine04_coarse03 | c2f | Step-R | PASS | 3.3446 |
| unet_c2f_a_fine04_coarse04 | c2f | Step-R | PASS | 3.1689 |
| unet_c2f_a_fine05_coarse01 | c2f | Step-R | PASS | 3.3606 |
| unet_c2f_a_fine05_coarse02 | c2f | Step-R | PASS | 3.3401 |
| unet_c2f_a_fine05_coarse03 | c2f | Step-R | PASS | 3.4751 |
| unet_c2f_a_fine05_coarse04 | c2f | Step-Q `unet_c2f_a_f5_c4_positive_synergy` | REUSED |  |
| unet_c2f_b_fine01_coarse01 | c2f | Step-R | PASS | 3.5563 |
| unet_c2f_b_fine01_coarse02 | c2f | Step-R | PASS | 3.5092 |
| unet_c2f_b_fine01_coarse03 | c2f | Step-R | PASS | 3.5497 |
| unet_c2f_b_fine01_coarse04 | c2f | Step-R | PASS | 3.5133 |
| unet_c2f_b_fine02_coarse01 | c2f | Step-R | PASS | 3.0747 |
| unet_c2f_b_fine02_coarse02 | c2f | Step-Q `unet_c2f_b_f2_c2_variant_contrast` | REUSED |  |
| unet_c2f_b_fine02_coarse03 | c2f | Step-R | PASS | 3.0933 |
| unet_c2f_b_fine02_coarse04 | c2f | Step-R | PASS | 3.0507 |
| unet_c2f_b_fine03_coarse01 | c2f | Step-R | PASS | 3.5204 |
| unet_c2f_b_fine03_coarse02 | c2f | Step-R | PASS | 3.5051 |
| unet_c2f_b_fine03_coarse03 | c2f | Step-R | PASS | 3.5389 |
| unet_c2f_b_fine03_coarse04 | c2f | Step-R | PASS | 3.5173 |
| unet_c2f_b_fine04_coarse01 | c2f | Step-R | PASS | 3.6087 |
| unet_c2f_b_fine04_coarse02 | c2f | Step-R | PASS | 3.5362 |
| unet_c2f_b_fine04_coarse03 | c2f | Step-R | PASS | 3.5855 |
| unet_c2f_b_fine04_coarse04 | c2f | Step-R | PASS | 3.5322 |
| unet_c2f_b_fine05_coarse01 | c2f | Step-R | PASS | 3.4548 |
| unet_c2f_b_fine05_coarse02 | c2f | Step-R | PASS | 3.4451 |
| unet_c2f_b_fine05_coarse03 | c2f | Step-R | PASS | 3.4525 |
| unet_c2f_b_fine05_coarse04 | c2f | Step-R | PASS | 3.4493 |
