# ResNet Layer2 direct full-sequence greedy search

Primary metric: keyframe evo_ape ATE mean after alignment and scale correction.
All new candidates were ranked only by full-sequence tracking; Layer2 correlation outputs are not used as a seed, representative constraint or ranking signal.

| Tags | Channels | PASS | ATE mean cm | ATE std cm |
|---|---|---:|---:|---:|
| G5+Lstar_single | [d41,d60,d67,d108,d121] | 3/3 | 12.6100 | 0.0000 |
| G4 | [d60,d67,d108,d121] | 3/3 | 12.6714 | 0.0000 |
| G6 | [d41,d60,d67,d95,d108,d121] | 3/3 | 12.7413 | 0.0000 |
| G3 | [d67,d108,d121] | 3/3 | 15.9524 | 0.0000 |
| G2 | [d108,d121] | 3/3 | 16.6202 | 0.0000 |
| R4+Rstar_single | [d37,d74,d119,d121] | 3/3 | 20.3798 | 0.0000 |
| G1 | [d121] | 3/3 | 27.5292 | 0.0000 |
| resnet_layer2_all128 | all128 | 3/3 | 40.8481 | 0.0000 |
| gray_current | gray | 0/3 |  |  |
