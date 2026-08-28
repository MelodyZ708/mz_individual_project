# UNET multi-sequence C2F parent comparison

- Scope: fr1/fr2/fr3 × clean/lightswitch/flashlight = nine full sequences.
- Configurations: 10 per sequence (5 direct parents, 4 C2F cells, one gray baseline).
- Mapping remains gray with sensor depth; only tracking features change.
- Primary metric: historical keyframe `evo_ape --align --correct_scale` translation ATE mean.
- Interpretation rule: assess C2F using within-dataset deltas against its direct fine parent and the better of both direct parents. Do not average raw ATE across fr1/fr2/fr3.
- Current cell statuses: {'FAIL_TRACKING_NAN': 2, 'PASS': 88}.

## Per-dataset winner among the focused comparison set

| Dataset | Winner | Historical ATE mean (cm) |
|---|---|---:|
| fr1_desk_clean | unet_c2f_a_f1_c4_negative_high_parent | 6.4992 |
| fr1_desk_lightswitch | unet_c2f_a_f2_c4_global_best | 5.8765 |
| fr1_desk_flashlight | unet_c2f_a_f2_c4_global_best | 7.9202 |
| fr2_desk_clean | unet_c2f_a_f2_c4_global_best | 3.0273 |
| fr2_desk_lightswitch | unet_c2f_a_f2_c4_global_best | 3.8588 |
| fr2_desk_flashlight | unet_enc0_direct_f2_d03_d07_d12 | 3.0736 |
| fr3_long_office_household_clean | unet_c2f_a_f5_c4_positive_synergy | 9.6901 |
| fr3_long_office_household_lightswitch | unet_c2f_a_f1_c4_negative_high_parent | 9.1046 |
| fr3_long_office_household_flashlight | unet_enc0_direct_f5_d02_d03_d07_d12_d13 | 9.7461 |

## C2F effect summary

| C2F configuration | Variant | Comparable pairs | Beats fine | Beats better direct parent | Median Δ vs fine (%) | Median Δ vs better parent (%) |
|---|---|---:|---:|---:|---:|---:|
| unet_c2f_a_f5_c4_positive_synergy | A | 9 | 8 | 8 | -4.22 | -4.22 |
| unet_c2f_a_f2_c4_global_best | A | 9 | 8 | 8 | -2.33 | -2.33 |
| unet_c2f_a_f1_c4_negative_high_parent | A | 9 | 7 | 7 | -3.73 | -3.73 |
| unet_c2f_b_f2_c2_variant_contrast | B | 9 | 5 | 4 | -2.46 | 0.01 |

Negative Δ means C2F is lower/better than the referenced direct parent on that dataset.

## Files

- `dataset_scorecard.csv`: one row per configuration × dataset, with complete diagnostics.
- `c2f_pairwise_comparison.csv`: the primary evidence table: C2F, both parents, absolute and percentage deltas.
- `c2f_effect_summary.csv`: win/loss counts and median within-sequence deltas for each focused C2F configuration.
- `ate_mean_matrix.csv`: presentation-oriented wide table; C2F rows also include their per-dataset parent deltas.
