# U-Net Enc0/Enc1 multi-dataset evaluation

- Scope: 3 TUM dataset families × clean/lightswitch/flashlight = 9 full sequences.
- Configurations: 13 selected U-Net candidates; one run per non-excluded cell; nominal total = 117.
- Safety scope change: Enc0-all16 and Enc1-all32 are explicitly omitted only on fr3 lightswitch after the Enc0-all16 run coincided with NVIDIA Xid 79 / PCIe receiver error.
- Tracking: U-Net Enc0 or Enc1 selected channels. Mapping remains gray with sensor depth.
- Primary metric: historical keyframe `evo_ape --align --correct_scale` ATE mean.
- All raw trajectory diagnostics are retained in each `per_dataset/*/evaluations.sqlite3` database.
- Current cell statuses: {'FAIL_TRACKING_NAN': 5, 'PASS': 110, 'SKIPPED_BY_SAFETY': 2}.
- Do not average raw ATE values across sequences; use within-dataset ranks and pass counts for cross-sequence comparison.

## Current dataset winners

| Dataset | Winner | Historical ATE mean (cm) |
|---|---|---:|
| fr1_desk_clean | enc0_k02_d02_d14 (enc0:2,14) | 5.2678 |
| fr1_desk_lightswitch | enc0_global_rank01_k06_d02_d03_d07_d12_d13_d14 (enc0:2,3,7,12,13,14) | 5.9208 |
| fr1_desk_flashlight | enc0_global_rank03_k04_d02_d03_d12_d14 (enc0:2,3,12,14) | 5.9763 |
| fr2_desk_clean | enc0_global_rank02_k03_d03_d07_d12 (enc0:3,7,12) | 3.0303 |
| fr2_desk_lightswitch | enc0_global_rank03_k04_d02_d03_d12_d14 (enc0:2,3,12,14) | 5.1419 |
| fr2_desk_flashlight | enc0_global_rank02_k03_d03_d07_d12 (enc0:3,7,12) | 3.0736 |
| fr3_long_office_household_clean | enc0_global_rank01_k06_d02_d03_d07_d12_d13_d14 (enc0:2,3,7,12,13,14) | 10.2805 |
| fr3_long_office_household_lightswitch | enc1_global_rank02_k06_d00_d05_d06_d17_d18_d30 (enc1:0,5,6,17,18,30) | 11.3039 |
| fr3_long_office_household_flashlight | enc0_global_rank01_k06_d02_d03_d07_d12_d13_d14 (enc0:2,3,7,12,13,14) | 10.0373 |

## Candidate robustness summary

| Candidate | Enc | Channels | PASS / 9 | Mean rank on PASS | Datasets won |
|---|---:|---|---:|---:|---:|
| enc0_global_rank03_k04_d02_d03_d12_d14 | 0 | [2,3,12,14] | 9 | 2.556 | 2 |
| enc0_global_rank02_k03_d03_d07_d12 | 0 | [3,7,12] | 9 | 3.333 | 2 |
| enc0_global_rank01_k06_d02_d03_d07_d12_d13_d14 | 0 | [2,3,7,12,13,14] | 9 | 3.333 | 3 |
| enc0_bqs_top5_d00_d03_d10_d14_d15 | 0 | [0,3,10,14,15] | 9 | 6.556 | 0 |
| enc1_global_rank02_k06_d00_d05_d06_d17_d18_d30 | 1 | [0,5,6,17,18,30] | 9 | 6.889 | 1 |
| enc1_k02_d00_d05 | 1 | [0,5] | 9 | 8.000 | 0 |
| enc1_global_rank01_k06_d05_d06_d17_d18_d28_d30 | 1 | [5,6,17,18,28,30] | 9 | 8.111 | 0 |
| enc1_global_rank03_k04_d00_d05_d18_d30 | 1 | [0,5,18,30] | 9 | 8.778 | 0 |
| enc1_bqs_top5_d04_d09_d10_d15_d30 | 1 | [4,9,10,15,30] | 9 | 11.111 | 0 |
| enc0_all16 | 0 | [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] | 8 | 6.000 | 0 |
| enc0_k02_d02_d14 | 0 | [2,14] | 8 | 6.250 | 1 |
| enc1_all32 | 1 | [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31] | 8 | 8.375 | 0 |
| enc0_k01_single_d03 | 0 | [3] | 5 | 7.800 | 0 |
