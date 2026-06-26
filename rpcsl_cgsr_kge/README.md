# RPCSL-CGSR KGE

This folder is copied from `rasa_cgsr_kge`; the previous folders are untouched.
The default behavior is still the CGSR baseline unless `--use_rpcsl` is passed.

## Modules

- CGSR: relation-pattern guided scale routing over convolution branches.
- RPCSL: relation-pattern calibrated set-wise learning for incomplete KGs.

The two modules form one story: relation patterns guide both representation
interaction scales and supervision softness.

## RPCSL Switches

```bash
--use_rpcsl
--rpcsl_filter_positives
--rpcsl_max_pos 32
--rpcsl_eps_min 0.0
--rpcsl_eps_max 0.20
--rpcsl_strength 1.0
--rpcsl_warmup_epochs 0
--rpcsl_ramp_epochs 1
```

RPCSL stores train positives for each query `(h, r)` and optimizes a residual
mixture of filtered CE and set-wise positive likelihood:

```text
L = (1 - epsilon_r) * L_filtered_CE + epsilon_r * L_set
```

`epsilon_r` is computed from train-only relation statistics: tails-per-head,
tail entropy, and 1-N/N-N indicators. This keeps 1-1 relations sharp while
softening supervision for multi-answer relations.

## Scripts

CGSR-only:

- `wn18rr_router_only.ps1`
- `fb237_router_only.ps1`
- `yago3_10_router_only.ps1`
- `umls_router_only.ps1`
- `kinship_router_only.ps1`

CGSR + RPCSL:

- `wn18rr_rpcsl_cgsr.ps1`
- `fb237_rpcsl_cgsr.ps1`
- `yago3_10_rpcsl_cgsr.ps1`
- `umls_rpcsl_cgsr.ps1`
- `kinship_rpcsl_cgsr.ps1`

## Safety Notes

If RPCSL hurts a dataset, remove `--use_rpcsl` and the model returns to
CGSR-only. For conservative RPCSL, reduce `--rpcsl_eps_max` and
`--rpcsl_strength`, or increase `--rpcsl_warmup_epochs`.
