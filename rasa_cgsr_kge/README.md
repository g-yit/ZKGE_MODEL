# CGSR-KGE

This folder is copied from `msrsc_baseline` and keeps the original baseline
files untouched. The current version keeps only the validated module:

- CGSR: context-guided scale routing for multi-scale relation-specific
  convolution branches.

Running `learn.py` without `--use_scale_router` follows the original baseline
path. Enabling `--use_scale_router` adds a residual branch router driven by
relation embeddings and relation-pattern statistics.

## Main Switches

```bash
--use_scale_router
--router_temperature 1.0
--router_min_branch_weight 0.02
--module_warmup_epochs 0
--module_ramp_epochs 1
--ce_weight_source train
```

`--ce_weight_source test` is the default to preserve baseline behavior, but
the provided scripts use `train` to avoid using test-set statistics.

## Recommended Scripts

- `wn18rr_router_only.ps1`
- `fb237_router_only.ps1`
- `yago3_10_router_only.ps1`
- `umls_router_only.ps1`
- `kinship_router_only.ps1`

## Implementation Notes

`datasets.py` builds relation statistics only from training triples and their
reciprocal triples. The router receives:

- relation frequency;
- tails-per-head and heads-per-tail;
- head/tail entropy;
- 1-1, 1-N, N-1, N-N relation type indicators;
- inverse-relation indicator.

By default, CGSR uses direct softmax branch weights with
`--router_temperature 1.0` and no warmup (`--module_warmup_epochs 0`,
`--module_ramp_epochs 1`). To run the older baseline-preserving residual
variant, add `--use_router_residual`.
