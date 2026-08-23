# UNet direct full-sequence greedy search

Primary metric: keyframe evo_ape ATE mean after alignment and scale correction.
All new candidates were ranked only by full-sequence tracking, not MVS/BQS.

| Tags | Channels | PASS | ATE mean cm | ATE std cm |
|---|---|---:|---:|---:|
| Lstar_single | [d5,d6,d17,d18,d28,d30] | 3/3 | 6.7335 | 0.0000 |
| G6 | [d0,d5,d6,d17,d18,d30] | 3/3 | 6.9553 | 0.0000 |
| G4 | [d0,d5,d18,d30] | 3/3 | 7.1673 | 0.0000 |
| G5 | [d0,d5,d6,d18,d30] | 3/3 | 7.6408 | 0.0000 |
| G2 | [d0,d5] | 3/3 | 8.0837 | 0.0000 |
| Rstar_single | [d0,d4,d10,d22,d26] | 3/3 | 8.3566 | 0.0000 |
| G3 | [d0,d5,d18] | 3/3 | 8.4908 | 0.0000 |
| R4 | [d0,d4,d13,d26] | 3/3 | 10.4465 | 0.0000 |
| B4_historical_bqs_prefix | [d4,d9,d10,d15] | 3/3 | 17.8502 | 0.0000 |
| B5_historical_bqs_top5 | [d4,d9,d10,d15,d30] | 3/3 | 18.4940 | 0.0000 |
| unet_all32 | all32 | 3/3 | 18.9322 | 0.0000 |
| gray_current | gray | 0/3 |  |  |
