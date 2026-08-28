# RESNET Step-R reduced complete C2F grid

- Selected full grid: 60 C2F pairs + 11 direct parents; no gray baseline.
- Reused from Step-Q without rerun: 9 configurations per dataset.
- New Step-R cells per dataset: 62.
- C2F effects are computed only against direct parents on the same sequence; negative delta means lower/better C2F ATE.
- Current merged status counts: {'FAIL_TRACKING_NAN': 45, 'PASS': 594}.

| C2F | Variant | F rank | C rank | PASS | Comparable | Beats better parent | Median Δ (%) | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| resnet_c2f_b_fine03_coarse05 | B | 3 | 5 | 9/9 | 9 | 7/9 | -4.73 | Step-R:new |
| resnet_c2f_b_fine04_coarse05 | B | 4 | 5 | 9/9 | 9 | 7/9 | -0.98 | Step-R:new |
| resnet_c2f_a_fine05_coarse05 | A | 5 | 5 | 9/9 | 9 | 5/9 | -2.67 | Step-R:new |
| resnet_c2f_b_fine04_coarse01 | B | 4 | 1 | 9/9 | 9 | 5/9 | -0.70 | Step-R:new |
| resnet_c2f_b_fine04_coarse02 | B | 4 | 2 | 9/9 | 9 | 5/9 | -0.62 | Step-R:new |
| resnet_c2f_b_fine04_coarse04 | B | 4 | 4 | 9/9 | 9 | 5/9 | -0.40 | Step-R:new |
| resnet_c2f_b_fine02_coarse02 | B | 2 | 2 | 9/9 | 9 | 5/9 | -0.08 | Step-R:new |
| resnet_c2f_b_fine06_coarse04 | B | 6 | 4 | 8/9 | 8 | 5/8 | -0.04 | Step-R:new |
| resnet_c2f_b_fine06_coarse03 | B | 6 | 3 | 8/9 | 8 | 5/8 | -0.01 | Step-R:new |
| resnet_c2f_b_fine05_coarse02 | B | 5 | 2 | 8/9 | 8 | 4/8 | -1.96 | Step-R:new |
| resnet_c2f_a_fine06_coarse05 | A | 6 | 5 | 8/9 | 8 | 4/8 | -0.27 | Step-Q:resnet_c2f_a_f6_c5_positive_synergy |
| resnet_c2f_b_fine06_coarse05 | B | 6 | 5 | 8/9 | 8 | 4/8 | -0.25 | Step-R:new |
| resnet_c2f_b_fine01_coarse05 | B | 1 | 5 | 7/9 | 7 | 4/7 | -0.20 | Step-R:new |
| resnet_c2f_b_fine05_coarse04 | B | 5 | 4 | 9/9 | 9 | 4/9 | +0.06 | Step-R:new |
| resnet_c2f_b_fine06_coarse02 | B | 6 | 2 | 8/9 | 8 | 4/8 | +0.22 | Step-R:new |
| resnet_c2f_a_fine04_coarse05 | A | 4 | 5 | 9/9 | 9 | 4/9 | +0.87 | Step-R:new |
| resnet_c2f_b_fine05_coarse05 | B | 5 | 5 | 9/9 | 9 | 4/9 | +1.46 | Step-R:new |
| resnet_c2f_b_fine02_coarse05 | B | 2 | 5 | 9/9 | 9 | 4/9 | +1.86 | Step-R:new |
| resnet_c2f_b_fine06_coarse01 | B | 6 | 1 | 8/9 | 8 | 3/8 | +0.42 | Step-R:new |
| resnet_c2f_b_fine03_coarse04 | B | 3 | 4 | 8/9 | 8 | 3/8 | +0.45 | Step-R:new |
| resnet_c2f_b_fine04_coarse03 | B | 4 | 3 | 9/9 | 9 | 3/9 | +1.90 | Step-R:new |
| resnet_c2f_b_fine05_coarse03 | B | 5 | 3 | 9/9 | 9 | 3/9 | +3.28 | Step-R:new |
| resnet_c2f_a_fine04_coarse01 | A | 4 | 1 | 9/9 | 9 | 3/9 | +3.30 | Step-R:new |
| resnet_c2f_b_fine05_coarse01 | B | 5 | 1 | 8/9 | 8 | 3/8 | +3.44 | Step-R:new |
| resnet_c2f_b_fine02_coarse04 | B | 2 | 4 | 9/9 | 9 | 3/9 | +4.30 | Step-Q:resnet_c2f_b_f2_c4_global_best |
| resnet_c2f_a_fine03_coarse05 | A | 3 | 5 | 9/9 | 9 | 3/9 | +7.31 | Step-R:new |
| resnet_c2f_b_fine03_coarse01 | B | 3 | 1 | 8/9 | 8 | 2/8 | +2.16 | Step-R:new |
| resnet_c2f_b_fine03_coarse02 | B | 3 | 2 | 9/9 | 9 | 2/9 | +2.37 | Step-R:new |
| resnet_c2f_b_fine02_coarse01 | B | 2 | 1 | 9/9 | 9 | 2/9 | +2.97 | Step-R:new |
| resnet_c2f_b_fine02_coarse03 | B | 2 | 3 | 9/9 | 9 | 2/9 | +4.42 | Step-R:new |
| resnet_c2f_a_fine04_coarse03 | A | 4 | 3 | 9/9 | 9 | 2/9 | +6.80 | Step-R:new |
| resnet_c2f_a_fine04_coarse02 | A | 4 | 2 | 8/9 | 8 | 2/8 | +7.66 | Step-R:new |
| resnet_c2f_a_fine04_coarse04 | A | 4 | 4 | 8/9 | 8 | 2/8 | +9.45 | Step-R:new |
| resnet_c2f_a_fine06_coarse04 | A | 6 | 4 | 8/9 | 8 | 2/8 | +13.34 | Step-R:new |
| resnet_c2f_a_fine06_coarse03 | A | 6 | 3 | 8/9 | 8 | 2/8 | +14.53 | Step-R:new |
| resnet_c2f_a_fine02_coarse01 | A | 2 | 1 | 9/9 | 9 | 2/9 | +15.66 | Step-R:new |
| resnet_c2f_a_fine05_coarse03 | A | 5 | 3 | 9/9 | 9 | 2/9 | +24.08 | Step-R:new |
| resnet_c2f_a_fine05_coarse04 | A | 5 | 4 | 9/9 | 9 | 2/9 | +24.21 | Step-R:new |
| resnet_c2f_a_fine05_coarse02 | A | 5 | 2 | 8/9 | 8 | 2/8 | +25.59 | Step-R:new |
| resnet_c2f_b_fine03_coarse03 | B | 3 | 3 | 8/9 | 8 | 1/8 | +3.44 | Step-R:new |
| resnet_c2f_b_fine01_coarse03 | B | 1 | 3 | 7/9 | 7 | 1/7 | +5.50 | Step-R:new |
| resnet_c2f_a_fine06_coarse02 | A | 6 | 2 | 8/9 | 8 | 1/8 | +11.50 | Step-R:new |
| resnet_c2f_a_fine03_coarse03 | A | 3 | 3 | 8/9 | 8 | 1/8 | +14.30 | Step-R:new |
| resnet_c2f_a_fine02_coarse05 | A | 2 | 5 | 9/9 | 9 | 1/9 | +15.11 | Step-R:new |
| resnet_c2f_a_fine01_coarse05 | A | 1 | 5 | 7/9 | 7 | 1/7 | +15.19 | Step-R:new |
| resnet_c2f_a_fine06_coarse01 | A | 6 | 1 | 7/9 | 7 | 1/7 | +16.10 | Step-R:new |
| resnet_c2f_a_fine03_coarse04 | A | 3 | 4 | 9/9 | 9 | 1/9 | +17.20 | Step-R:new |
| resnet_c2f_a_fine02_coarse04 | A | 2 | 4 | 9/9 | 9 | 1/9 | +20.39 | Step-Q:resnet_c2f_a_f2_c4_variant_negative |
| resnet_c2f_a_fine03_coarse01 | A | 3 | 1 | 8/9 | 8 | 1/8 | +22.38 | Step-R:new |
| resnet_c2f_a_fine05_coarse01 | A | 5 | 1 | 8/9 | 8 | 1/8 | +24.50 | Step-R:new |
| resnet_c2f_b_fine01_coarse04 | B | 1 | 4 | 7/9 | 7 | 0/7 | +6.32 | Step-Q:resnet_c2f_b_f1_c4_negative_high_parent |
| resnet_c2f_b_fine01_coarse02 | B | 1 | 2 | 7/9 | 7 | 0/7 | +6.77 | Step-R:new |
| resnet_c2f_b_fine01_coarse01 | B | 1 | 1 | 7/9 | 7 | 0/7 | +9.59 | Step-R:new |
| resnet_c2f_a_fine01_coarse01 | A | 1 | 1 | 7/9 | 7 | 0/7 | +37.03 | Step-R:new |
| resnet_c2f_a_fine03_coarse02 | A | 3 | 2 | 8/9 | 8 | 0/8 | +42.80 | Step-R:new |
| resnet_c2f_a_fine02_coarse03 | A | 2 | 3 | 9/9 | 9 | 0/9 | +56.45 | Step-R:new |
| resnet_c2f_a_fine01_coarse02 | A | 1 | 2 | 7/9 | 7 | 0/7 | +59.88 | Step-R:new |
| resnet_c2f_a_fine01_coarse04 | A | 1 | 4 | 7/9 | 7 | 0/7 | +61.02 | Step-R:new |
| resnet_c2f_a_fine01_coarse03 | A | 1 | 3 | 7/9 | 7 | 0/7 | +65.71 | Step-R:new |
| resnet_c2f_a_fine02_coarse02 | A | 2 | 2 | 9/9 | 9 | 0/9 | +76.78 | Step-R:new |
