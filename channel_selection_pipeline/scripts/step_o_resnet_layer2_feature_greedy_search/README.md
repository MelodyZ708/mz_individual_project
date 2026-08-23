# ResNet Layer2 direct full-sequence greedy search

This Step O experiment applies the same direct full-sequence greedy protocol
used for UNet Enc1 and ResNet Conv1 to all **128 post-ReLU ResNet-18 Layer2
channels**. It uses the full 573-frame `fr1/desk_lightswitch` sequence;
mapping remains gray with matched sensor depth.

The primary score is the established historical keyframe `evo_ape` translation
ATE mean after alignment and scale correction. The protocol evaluates K=1--6:

- all Layer2 singletons;
- 2--4 multi-start forward-greedy paths through K=6;
- exhaustive pair rescue when singleton seeds are absent or sparse;
- equal-budget random control and one-channel swap audit;
- three total observations for final candidates, retaining both adaptive Gstar
  and fixed K=4 G4.

There is no prior full-sequence Layer2 subset to reuse as an anchor. Anchors
are gray and all-128 Layer2. If both anchors and all singletons fail, the
runner deliberately retains the same pair-rescue method: it provisionally uses
four pair-start paths after exhaustive pair evaluation. This changes no score
or channel ranking and prevents a missing anchor from blocking recovery.

The SQLite database is authoritative and resumable. The launcher snapshots it
before execution and the shared COMO configuration is restored after every
run.

Run from the project root:

```bash
./channel_selection_pipeline/scripts/step_o_resnet_layer2_feature_greedy_search/run_resnet_layer2_direct_fullseq_greedy.sh \
  --stage all \
  --execute
```

Results: `channel_selection_results/step_o_resnet_layer2_direct_fullseq_greedy/`.

The default 36-hour batch boundary is deliberate. Re-run the identical command
after a clean boundary or interruption; saved candidate/replicate rows are
reused and only missing work is evaluated.
