# Selected U-Net tracking feature-map visualisations

- Frames 246/250/254 are the before/peak/after samples of the selected MVS turn-on challenge.
- `enc0/` contains [3], [2,14], [3,7,12], and [2,3,7,12,13,14].
- `enc1/` contains [0,5] and [5,6,17,18,28,30].
- `clean_lightswitch_post_activation/` is the primary ResNet-style view: one PNG per group per frame, with Clean / Lightswitch / |Light-clean| columns.
- U-Net ResidualConv uses LeakyReLU. The maps are actual post-LeakyReLU tracking features, not standard-ReLU-clipped surrogates.
- Clean/lightswitch share a scale within each channel; different channels retain independent scales.
- `.npz` files preserve native maps for later quantitative analysis.
