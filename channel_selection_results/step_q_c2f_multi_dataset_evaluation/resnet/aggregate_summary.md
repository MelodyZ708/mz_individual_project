# RESNET multi-sequence C2F parent comparison

- Scope: fr1/fr2/fr3 × clean/lightswitch/flashlight = nine full sequences.
- Configurations: 10 per sequence (5 direct parents, 4 C2F cells, one gray baseline).
- Mapping remains gray with sensor depth; only tracking features change.
- Primary metric: historical keyframe `evo_ape --align --correct_scale` translation ATE mean.
- Interpretation rule: assess C2F using within-dataset deltas against its direct fine parent and the better of both direct parents. Do not average raw ATE across fr1/fr2/fr3.
- Current cell statuses: {'FAIL_TRACKING_NAN': 7, 'PASS': 83}.

## Per-dataset winner among the focused comparison set

| Dataset | Winner | Historical ATE mean (cm) |
|---|---|---:|
| fr1_desk_clean | gray_baseline | 7.3847 |
| fr1_desk_lightswitch | resnet_c2f_b_f2_c4_global_best | 9.3293 |
| fr1_desk_flashlight | resnet_conv1_direct_f1_d15_d20_d26_d34 | 6.3644 |
| fr2_desk_clean | resnet_conv1_direct_f1_d15_d20_d26_d34 | 4.5310 |
| fr2_desk_lightswitch | resnet_c2f_b_f2_c4_global_best | 6.5996 |
| fr2_desk_flashlight | gray_baseline | 4.6496 |
| fr3_long_office_household_clean | gray_baseline | 11.2618 |
| fr3_long_office_household_lightswitch | resnet_conv1_direct_f2_d23_d24_d26_d51_d63 | 10.2158 |
| fr3_long_office_household_flashlight | gray_baseline | 10.5942 |

## C2F effect summary

| C2F configuration | Variant | Comparable pairs | Beats fine | Beats better direct parent | Median Δ vs fine (%) | Median Δ vs better parent (%) |
|---|---|---:|---:|---:|---:|---:|
| resnet_c2f_a_f6_c5_positive_synergy | A | 8 | 4 | 4 | -0.27 | -0.27 |
| resnet_c2f_b_f2_c4_global_best | B | 9 | 3 | 3 | 4.30 | 4.30 |
| resnet_c2f_a_f2_c4_variant_negative | A | 9 | 1 | 1 | 20.39 | 20.39 |
| resnet_c2f_b_f1_c4_negative_high_parent | B | 7 | 0 | 0 | 6.32 | 6.32 |

Negative Δ means C2F is lower/better than the referenced direct parent on that dataset.

## Files

- `dataset_scorecard.csv`: one row per configuration × dataset, with complete diagnostics.
- `c2f_pairwise_comparison.csv`: the primary evidence table: C2F, both parents, absolute and percentage deltas.
- `c2f_effect_summary.csv`: win/loss counts and median within-sequence deltas for each focused C2F configuration.
- `ate_mean_matrix.csv`: presentation-oriented wide table; C2F rows also include their per-dataset parent deltas.
