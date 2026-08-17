# ResNet Conv1 direct full-sequence greedy search

This Step N experiment makes the earliest ResNet layer (project shorthand:
`conv0`; COMO layer name: `conv1`) directly comparable with the UNet Enc1
search. It evaluates the 64 post-ReLU Conv1 channels on the complete
`fr1/desk_lightswitch` sequence.

The candidate-selection protocol is intentionally the same as UNet Enc1:

- Full-sequence keyframe `evo_ape` translation ATE mean, with alignment and
  scale correction, is the primary ranking metric.
- Evaluate all 64 singleton channels. Use the strongest PASS singleton runs as
  2--4 greedy starts, then extend every path up to six channels.
- If singleton runs are absent or too sparse, exhaustively evaluate all
  `C(64,2)=2,016` channel pairs and use PASS pairs to provide/augment seeds.
- Add an equal-budget random control, a one-channel swap audit around the
  provisional greedy best, and three total evaluations for final candidates.
- Keep both adaptive `Gstar` and the fixed four-channel `G4` endpoint.

The earlier correlation-clustering output is not a constraint, seed source, or
ranking criterion. Its historical four-channel result and `[5,29,40,52]` are
only anchor controls in the final report.

The SQLite database in the results directory is authoritative and resumable.
The launcher snapshots it before each execution and restores the shared COMO
configuration after every run.

Run from the project root:

```bash
./channel_selection_pipeline/scripts/step_n_resnet_conv1_feature_greedy_search/run_resnet_conv1_direct_fullseq_greedy.sh \
  --stage all \
  --execute
```

Default result directory:
`channel_selection_results/step_n_resnet_conv1_direct_fullseq_greedy/`.
