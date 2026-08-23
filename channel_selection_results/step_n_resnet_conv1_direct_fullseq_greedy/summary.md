# ResNet Conv1 direct full-sequence greedy search

Primary metric: keyframe evo_ape ATE mean after alignment and scale correction.
All new candidates were ranked only by full-sequence tracking; correlation clustering is retained only as a historical anchor.

| Tags | Channels | PASS | ATE mean cm | ATE std cm |
|---|---|---:|---:|---:|
| G4+Lstar_single | [d15,d20,d26,d34] | 3/3 | 10.3758 | 0.0000 |
| G5 | [d23,d24,d26,d51,d63] | 3/3 | 10.4535 | 0.0000 |
| G6 | [d15,d20,d26,d34,d45,d53] | 3/3 | 11.6218 | 0.0000 |
| G2 | [d33,d52] | 3/3 | 13.9810 | 0.0000 |
| H4_historical_correlation_search | [d5,d6,d24,d29] | 3/3 | 14.0623 | 0.0000 |
| G3 | [d29,d33,d52] | 3/3 | 14.6721 | 0.0000 |
| H4_historical_cnn_baseline | [d5,d29,d40,d52] | 3/3 | 15.1682 | 0.0000 |
| R4+Rstar_single | [d0,d15,d20,d59] | 3/3 | 16.6402 | 0.0000 |
| resnet_conv1_all64 | all64 | 3/3 | 28.3611 | 0.0000 |
| gray_current | gray | 0/3 |  |  |
