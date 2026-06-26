# RCEM-CGSR KGE

This folder is copied from `rasa_cgsr_kge` and keeps the previous folders
untouched. It adds two relation-conditioned modules:

- CGSR: context-guided scale routing for multi-scale relation-specific
  convolution branches.
- RCEM: relation-conditioned evidence memory that injects train-only path
  composition evidence and implicit entity-role type evidence into final logits.

Running `learn.py` without `--use_scale_router` and `--use_rcem` follows the
original baseline path. Enabling `--use_rcem` adds a small gated residual term
to candidate logits:

```text
s_final(h, r, t) = s_embed(h, r, t)
                 + g_path(r, h) * E_path(h, r, t)
                 + g_type(r, h) * E_type(r, t)
```

All evidence is built from training triples and their reciprocal triples only.

## Main Switches

```bash
--use_scale_router
--router_temperature 1.0
--router_min_branch_weight 0.02
--module_warmup_epochs 0
--module_ramp_epochs 1
--use_rcem
--rcem_max_rules 8
--rcem_max_candidates 32
--rcem_min_rule_support 3
--rcem_max_rule_degree 64
--rcem_path_strength 0.10
--rcem_type_strength 0.04
--rcem_path_gate_init 0.05
--rcem_type_gate_init 0.05
--rcem_warmup_epochs 0
--rcem_ramp_epochs 5
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
- `wn18rr_rcem_cgsr.ps1`
- `fb237_rcem_cgsr.ps1`
- `yago3_10_rcem_cgsr.ps1`
- `umls_rcem_cgsr.ps1`
- `kinship_rcem_cgsr.ps1`

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

RCEM has two evidence sources:

- path evidence mines high-support two-hop relation compositions
  `r1(h,z) + r2(z,t) -> r(h,t)`, stores only top candidates for each query,
  and injects them with `scatter_add`;
- type evidence builds unsupervised entity role signatures from incoming and
  outgoing relation distributions, then scores relation-tail compatibility.

The evidence gates are initialized with small probabilities and can be ramped
with `--rcem_warmup_epochs` and `--rcem_ramp_epochs`. This makes the new module
residual and conservative at the beginning of training.
