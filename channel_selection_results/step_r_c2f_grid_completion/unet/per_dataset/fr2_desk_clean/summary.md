# UNET Step-R C2F grid completion (fr2_desk_clean)

- Matched frames: 2893; mapping remains gray + sensor depth.
- This database contains only missing Step-R cells. Existing Step-Q rows are intentionally reused externally and never copied or rerun.
- Full selected grid: 40 C2F pairs + 9 direct parents.
- Reused Step-Q rows: 9; new Step-R rows: 40.
- Persisted new rows: 40/40; statuses: {'PASS': 40}.

| Standard label | Kind | Source | Status | ATE mean (cm) |
|---|---|---|---|---:|
| unet_direct_fine01 | direct_fine | Step-Q `unet_enc0_direct_f1_d02_d03_d07_d12_d13_d14` | REUSED |  |
| unet_direct_fine02 | direct_fine | Step-Q `unet_enc0_direct_f2_d03_d07_d12` | REUSED |  |
| unet_direct_fine03 | direct_fine | Step-R | PASS | 3.4186 |
| unet_direct_fine04 | direct_fine | Step-R | PASS | 3.8283 |
| unet_direct_fine05 | direct_fine | Step-Q `unet_enc0_direct_f5_d02_d03_d07_d12_d13` | REUSED |  |
| unet_direct_coarse01 | direct_coarse | Step-R | PASS | 4.8043 |
| unet_direct_coarse02 | direct_coarse | Step-Q `unet_enc1_direct_c2_d00_d05_d06_d17_d18_d30` | REUSED |  |
| unet_direct_coarse03 | direct_coarse | Step-R | PASS | 6.0230 |
| unet_direct_coarse04 | direct_coarse | Step-Q `unet_enc1_direct_c4_d00_d05_d06_d18_d30` | REUSED |  |
| unet_c2f_a_fine01_coarse01 | c2f | Step-R | PASS | 3.5007 |
| unet_c2f_a_fine01_coarse02 | c2f | Step-R | PASS | 3.3678 |
| unet_c2f_a_fine01_coarse03 | c2f | Step-R | PASS | 3.4994 |
| unet_c2f_a_fine01_coarse04 | c2f | Step-Q `unet_c2f_a_f1_c4_negative_high_parent` | REUSED |  |
| unet_c2f_a_fine02_coarse01 | c2f | Step-R | PASS | 3.0978 |
| unet_c2f_a_fine02_coarse02 | c2f | Step-R | PASS | 3.0942 |
| unet_c2f_a_fine02_coarse03 | c2f | Step-R | PASS | 3.1978 |
| unet_c2f_a_fine02_coarse04 | c2f | Step-Q `unet_c2f_a_f2_c4_global_best` | REUSED |  |
| unet_c2f_a_fine03_coarse01 | c2f | Step-R | PASS | 3.3935 |
| unet_c2f_a_fine03_coarse02 | c2f | Step-R | PASS | 3.3337 |
| unet_c2f_a_fine03_coarse03 | c2f | Step-R | PASS | 3.4503 |
| unet_c2f_a_fine03_coarse04 | c2f | Step-R | PASS | 3.3412 |
| unet_c2f_a_fine04_coarse01 | c2f | Step-R | PASS | 3.1822 |
| unet_c2f_a_fine04_coarse02 | c2f | Step-R | PASS | 3.1973 |
| unet_c2f_a_fine04_coarse03 | c2f | Step-R | PASS | 3.2781 |
| unet_c2f_a_fine04_coarse04 | c2f | Step-R | PASS | 3.1348 |
| unet_c2f_a_fine05_coarse01 | c2f | Step-R | PASS | 3.2906 |
| unet_c2f_a_fine05_coarse02 | c2f | Step-R | PASS | 3.2846 |
| unet_c2f_a_fine05_coarse03 | c2f | Step-R | PASS | 3.4064 |
| unet_c2f_a_fine05_coarse04 | c2f | Step-Q `unet_c2f_a_f5_c4_positive_synergy` | REUSED |  |
| unet_c2f_b_fine01_coarse01 | c2f | Step-R | PASS | 3.5175 |
| unet_c2f_b_fine01_coarse02 | c2f | Step-R | PASS | 3.4874 |
| unet_c2f_b_fine01_coarse03 | c2f | Step-R | PASS | 3.5155 |
| unet_c2f_b_fine01_coarse04 | c2f | Step-R | PASS | 3.4906 |
| unet_c2f_b_fine02_coarse01 | c2f | Step-R | PASS | 3.0416 |
| unet_c2f_b_fine02_coarse02 | c2f | Step-Q `unet_c2f_b_f2_c2_variant_contrast` | REUSED |  |
| unet_c2f_b_fine02_coarse03 | c2f | Step-R | PASS | 3.0643 |
| unet_c2f_b_fine02_coarse04 | c2f | Step-R | PASS | 3.0162 |
| unet_c2f_b_fine03_coarse01 | c2f | Step-R | PASS | 3.4217 |
| unet_c2f_b_fine03_coarse02 | c2f | Step-R | PASS | 3.3910 |
| unet_c2f_b_fine03_coarse03 | c2f | Step-R | PASS | 3.4336 |
| unet_c2f_b_fine03_coarse04 | c2f | Step-R | PASS | 3.3762 |
| unet_c2f_b_fine04_coarse01 | c2f | Step-R | PASS | 3.6474 |
| unet_c2f_b_fine04_coarse02 | c2f | Step-R | PASS | 3.6025 |
| unet_c2f_b_fine04_coarse03 | c2f | Step-R | PASS | 3.6077 |
| unet_c2f_b_fine04_coarse04 | c2f | Step-R | PASS | 3.5560 |
| unet_c2f_b_fine05_coarse01 | c2f | Step-R | PASS | 3.3920 |
| unet_c2f_b_fine05_coarse02 | c2f | Step-R | PASS | 3.3908 |
| unet_c2f_b_fine05_coarse03 | c2f | Step-R | PASS | 3.4210 |
| unet_c2f_b_fine05_coarse04 | c2f | Step-R | PASS | 3.3666 |
