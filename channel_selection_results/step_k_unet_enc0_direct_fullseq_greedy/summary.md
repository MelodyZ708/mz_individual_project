# UNet direct full-sequence greedy search

Primary metric: keyframe evo_ape ATE mean after alignment and scale correction.
All new candidates were ranked only by full-sequence tracking, not MVS/BQS.

| Tags | Channels | PASS | ATE mean cm | ATE std cm |
|---|---|---:|---:|---:|
| G6+Lstar_single | [d2,d3,d7,d12,d13,d14] | 3/3 | 5.9208 | 0.0000 |
| G3 | [d3,d7,d12] | 3/3 | 6.0004 | 0.0000 |
| G4 | [d2,d3,d12,d14] | 3/3 | 6.3905 | 0.0000 |
| Rstar_single | [d2,d4,d6] | 3/3 | 6.6111 | 0.0000 |
| G5 | [d2,d3,d7,d12,d13] | 3/3 | 7.0959 | 0.0000 |
| R4 | [d0,d2,d3,d11] | 3/3 | 7.4177 | 0.0000 |
| G2 | [d2,d14] | 3/3 | 10.3332 | 0.0000 |
| unet_enc0_all16 | all16 | 3/3 | 12.9502 | 0.0000 |
| B5_historical_bqs_top5 | [d0,d3,d10,d14,d15] | 3/3 | 13.7216 | 0.0000 |
| B3_historical_bqs_top3 | [d0,d10,d15] | 3/3 | 23.0342 | 0.0000 |
| G1 | [d3] | 3/3 | 24.5318 | 0.0000 |
| gray_current | gray | 0/3 |  |  |
