# RASA-CGSR KGE

This folder is copied from `msrsc_baseline` and keeps the original baseline
files untouched. The new code adds two optional modules to `MSDCSE`:

- RASA: relation-aware structural anchor enhancement.
- CGSR: context-guided scale routing for multi-scale relation-specific
  convolution branches.

Both modules are disabled by default. Running `learn.py` without
`--use_anchor` and `--use_scale_router` follows the original baseline path.

## Main Switches

```bash
--use_anchor          # enable relation-aware structural anchor
--use_scale_router    # enable context-guided branch routing
--max_neighbors 16    # relation-balanced neighbors stored for each entity
--ce_weight_source train
```

`--ce_weight_source test` is the default to preserve baseline behavior, but
the provided scripts use `train` to avoid using test-set statistics.

## Recommended Scripts

- `wn18rr_rasa_cgsr.ps1`: sparse graph, stronger PMI/hub correction.
- `fb237_rasa_cgsr.ps1`: relation-rich graph, larger neighbor budget.
- `yago3_10_rasa_cgsr.ps1`: large graph with strong hubs, conservative batch
  size and stronger hub penalty.
- `umls_rasa_cgsr.ps1`: dense small graph, stronger dropout.
- `kinship_rasa_cgsr.ps1`: dense small graph dominated by N-N relations,
  stronger dropout and smoother routing.

## Implementation Notes

`datasets.py` builds graph context only from training triples and their
reciprocal triples. `models.py` stores the resulting tensors as non-persistent
buffers, so checkpoints only contain learnable parameters. When testing a model
trained with the new modules, pass the same module flags so the architecture is
rebuilt before loading the checkpoint.
