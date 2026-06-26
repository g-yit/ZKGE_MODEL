# CGSR KGE

This folder keeps the context-guided scale routing code copied from
`rasa_cgsr_kge`; the previous folders are untouched.

## Modules

- CGSR: relation-pattern guided scale routing over convolution branches.

## Scale Router Switches

```bash
--use_scale_router
--module_warmup_epochs 0
--module_ramp_epochs 1
--router_hidden 0
--router_dropout 0.1
--router_temperature 1.0
--router_min_branch_weight 0.0
--router_residual_init 0.10
--use_router_residual
```

The default path is the original MSDCSE branch aggregation. Passing
`--use_scale_router` enables relation-pattern guided branch weighting.

## Scripts

- `wn18rr_router_only.ps1`
- `fb237_router_only.ps1`
- `yago3_10_router_only.ps1`
- `umls_router_only.ps1`
- `kinship_router_only.ps1`

## Safety Notes

If the router is not enabled, the model follows the original baseline behavior.
