# RCEM-CGSR KGE

This folder is copied from `rasa_cgsr_kge` and keeps the previous folders
untouched. It adds two context-conditioned modules:

- CGSR: hierarchical query-content scale routing for multi-scale
  relation-specific convolution branches.
- RCEM: query- and evidence-conditioned memory that injects train-only path
  composition evidence and implicit entity-role type evidence into final logits.

Running `learn.py` without `--use_scale_router` and `--use_rcem` follows the
original baseline path. Enabling `--use_rcem` adds a small gated residual term
to candidate logits:

```text
s_final(h, r, t) = s_embed(h, r, t)
                 + g_path(q, evidence) * E_path(h, r, t)
                 + g_type(q, evidence) * E_type(r, t)
```

All evidence is built from training triples and their reciprocal triples only.

## Main Switches and Core Hyperparameters

Only seven numerical module hyperparameters are exposed for experiments. The
remaining capacity, stability, and preprocessing settings use fixed defaults
recorded in every saved `config.json`.

```bash
--context_hidden 0
--use_scale_router
--router_temperature 1.0
--router_content_scale 0.25
--use_rcem
--rcem_max_rules 8
--rcem_standard_confidence_weight 0.3
--rcem_path_strength 0.10
--rcem_type_strength 0.04
--ce_weight_source train
```

The seven experimental dimensions are:

- `context_hidden`: shared CGSR-router and RCEM-gate hidden size; `0` selects
  `min(rank, 128)` automatically;
- `router_temperature`: sharpness of the multi-branch routing distribution;
- `router_content_scale`: initial trainable strength of branch-content
  feedback; exactly `0` is a strict content-feedback ablation;
- `rcem_max_rules`: maximum retained two-hop rules for each target relation;
- `rcem_standard_confidence_weight`: interpolation between standard and PCA
  rule confidence;
- `rcem_path_strength`: maximum gated path-evidence logit contribution;
- `rcem_type_strength`: maximum gated type-evidence logit contribution.

Recommended compact search ranges are:

```text
context_hidden:                      [64, 128, 256]
router_temperature:                  [0.75, 1.0, 1.5]
router_content_scale:                [0.10, 0.25, 0.50]
rcem_max_rules:                      [4, 8, 16]
rcem_standard_confidence_weight:     [0.0, 0.3, 0.5]
rcem_path_strength:                  [0.05, 0.10, 0.20]
rcem_type_strength:                  [0.02, 0.04, 0.08]
```

Use a value of `0` for `router_content_scale`, `rcem_path_strength`, or
`rcem_type_strength` when a strict contribution ablation is required.

Fixed engineering defaults are:

```text
CGSR: dropout=0.10, min_branch_weight=0.02, residual_init=0.10
RCEM: max_candidates=32, min_rule_support=3, max_rule_degree=64,
      rule_smoothing=1.0, gate_dropout=0.05,
      path_gate_init=0.05, type_gate_init=0.05
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

`datasets.py` builds every routing feature from training triples only. The
router receives:

- head entity and relation embeddings;
- element-wise head-relation interactions;
- relation frequency;
- tails-per-head and heads-per-tail;
- head/tail entropy;
- 1-1, 1-N, N-1, N-N relation type indicators;
- inverse-relation indicator;
- directed entity degree, relation diversity, role entropy, and role balance;
- pooled content responses from every convolution branch.

CGSR first predicts relation-level scale priors, adapts them with query and
entity context, and then uses pooled branch responses as content feedback. By
default it returns query-dependent residual gains whose mean is exactly one,
preserving the baseline feature scale. Use `--no_router_residual` to apply the
full mean-preserving gains directly.

RCEM has two evidence sources:

- path evidence mines two-hop relation compositions
  `r1(h,z) + r2(z,t) -> r(h,t)`. Rule support is deduplicated by `(h,t)` and
  scored with standard confidence, PCA confidence, and head coverage. Path
  multiplicity is log-compressed, relation-calibrated, and stored only for top
  candidates before injection with `scatter_add`;
- type evidence builds unsupervised entity role signatures from incoming and
  outgoing relation distributions, then scores relation-tail compatibility.

The evidence gates consume the encoded query, head-relation interaction,
relation statistics, path-candidate concentration, and type-prototype
reliability. They are initialized with small probabilities, keeping the
residual contribution conservative at the beginning of training.
