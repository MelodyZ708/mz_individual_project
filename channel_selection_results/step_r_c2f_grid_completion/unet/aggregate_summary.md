# UNET Step-R reduced complete C2F grid

- Selected full grid: 40 C2F pairs + 9 direct parents; no gray baseline.
- Reused from Step-Q without rerun: 9 configurations per dataset.
- New Step-R cells per dataset: 40.
- C2F effects are computed only against direct parents on the same sequence; negative delta means lower/better C2F ATE.
- Current merged status counts: {'PASS': 441}.

| C2F | Variant | F rank | C rank | PASS | Comparable | Beats better parent | Median Δ (%) | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| unet_c2f_a_fine04_coarse02 | A | 4 | 2 | 9/9 | 9 | 8/9 | -13.78 | Step-R:new |
| unet_c2f_a_fine05_coarse04 | A | 5 | 4 | 9/9 | 9 | 8/9 | -4.22 | Step-Q:unet_c2f_a_f5_c4_positive_synergy |
| unet_c2f_a_fine01_coarse02 | A | 1 | 2 | 9/9 | 9 | 8/9 | -4.02 | Step-R:new |
| unet_c2f_b_fine05_coarse04 | B | 5 | 4 | 9/9 | 9 | 8/9 | -3.71 | Step-R:new |
| unet_c2f_a_fine02_coarse04 | A | 2 | 4 | 9/9 | 9 | 8/9 | -2.33 | Step-Q:unet_c2f_a_f2_c4_global_best |
| unet_c2f_a_fine04_coarse01 | A | 4 | 1 | 9/9 | 9 | 7/9 | -11.59 | Step-R:new |
| unet_c2f_b_fine04_coarse01 | B | 4 | 1 | 9/9 | 9 | 7/9 | -4.73 | Step-R:new |
| unet_c2f_a_fine01_coarse04 | A | 1 | 4 | 9/9 | 9 | 7/9 | -3.73 | Step-Q:unet_c2f_a_f1_c4_negative_high_parent |
| unet_c2f_a_fine05_coarse02 | A | 5 | 2 | 9/9 | 9 | 7/9 | -3.72 | Step-R:new |
| unet_c2f_b_fine01_coarse04 | B | 1 | 4 | 9/9 | 9 | 7/9 | -1.31 | Step-R:new |
| unet_c2f_b_fine02_coarse04 | B | 2 | 4 | 9/9 | 9 | 7/9 | -0.75 | Step-R:new |
| unet_c2f_a_fine04_coarse04 | A | 4 | 4 | 9/9 | 9 | 6/9 | -8.54 | Step-R:new |
| unet_c2f_b_fine04_coarse02 | B | 4 | 2 | 9/9 | 9 | 6/9 | -5.38 | Step-R:new |
| unet_c2f_a_fine03_coarse02 | A | 3 | 2 | 9/9 | 9 | 6/9 | -4.23 | Step-R:new |
| unet_c2f_b_fine01_coarse01 | B | 1 | 1 | 9/9 | 9 | 6/9 | -3.08 | Step-R:new |
| unet_c2f_a_fine01_coarse01 | A | 1 | 1 | 9/9 | 9 | 6/9 | -2.86 | Step-R:new |
| unet_c2f_a_fine03_coarse04 | A | 3 | 4 | 9/9 | 9 | 6/9 | -2.26 | Step-R:new |
| unet_c2f_b_fine02_coarse01 | B | 2 | 1 | 9/9 | 9 | 6/9 | -2.14 | Step-R:new |
| unet_c2f_a_fine02_coarse02 | A | 2 | 2 | 9/9 | 9 | 6/9 | -1.67 | Step-R:new |
| unet_c2f_b_fine01_coarse02 | B | 1 | 2 | 9/9 | 9 | 6/9 | -1.42 | Step-R:new |
| unet_c2f_a_fine03_coarse01 | A | 3 | 1 | 9/9 | 9 | 6/9 | -0.84 | Step-R:new |
| unet_c2f_b_fine05_coarse01 | B | 5 | 1 | 9/9 | 9 | 6/9 | -0.55 | Step-R:new |
| unet_c2f_a_fine04_coarse03 | A | 4 | 3 | 9/9 | 9 | 5/9 | -10.51 | Step-R:new |
| unet_c2f_b_fine04_coarse04 | B | 4 | 4 | 9/9 | 9 | 5/9 | -5.49 | Step-R:new |
| unet_c2f_a_fine05_coarse01 | A | 5 | 1 | 9/9 | 9 | 5/9 | -2.88 | Step-R:new |
| unet_c2f_b_fine04_coarse03 | B | 4 | 3 | 9/9 | 9 | 5/9 | -1.11 | Step-R:new |
| unet_c2f_b_fine03_coarse04 | B | 3 | 4 | 9/9 | 9 | 5/9 | -0.99 | Step-R:new |
| unet_c2f_b_fine03_coarse01 | B | 3 | 1 | 9/9 | 9 | 5/9 | -0.90 | Step-R:new |
| unet_c2f_b_fine03_coarse02 | B | 3 | 2 | 9/9 | 9 | 5/9 | -0.81 | Step-R:new |
| unet_c2f_b_fine05_coarse02 | B | 5 | 2 | 9/9 | 9 | 5/9 | -0.70 | Step-R:new |
| unet_c2f_b_fine01_coarse03 | B | 1 | 3 | 9/9 | 9 | 5/9 | -0.28 | Step-R:new |
| unet_c2f_b_fine02_coarse03 | B | 2 | 3 | 9/9 | 9 | 5/9 | -0.18 | Step-R:new |
| unet_c2f_b_fine02_coarse02 | B | 2 | 2 | 9/9 | 9 | 4/9 | +0.01 | Step-Q:unet_c2f_b_f2_c2_variant_contrast |
| unet_c2f_b_fine05_coarse03 | B | 5 | 3 | 9/9 | 9 | 4/9 | +0.43 | Step-R:new |
| unet_c2f_b_fine03_coarse03 | B | 3 | 3 | 9/9 | 9 | 3/9 | +0.40 | Step-R:new |
| unet_c2f_a_fine05_coarse03 | A | 5 | 3 | 9/9 | 9 | 3/9 | +0.53 | Step-R:new |
| unet_c2f_a_fine02_coarse01 | A | 2 | 1 | 9/9 | 9 | 3/9 | +0.78 | Step-R:new |
| unet_c2f_a_fine03_coarse03 | A | 3 | 3 | 9/9 | 9 | 3/9 | +0.93 | Step-R:new |
| unet_c2f_a_fine01_coarse03 | A | 1 | 3 | 9/9 | 9 | 2/9 | +3.46 | Step-R:new |
| unet_c2f_a_fine02_coarse03 | A | 2 | 3 | 9/9 | 9 | 2/9 | +4.67 | Step-R:new |
