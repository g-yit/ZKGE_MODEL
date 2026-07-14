# RCEM-CGSR KGE Improved

This directory is an independent improved copy of `rcem_cgsr_kge`.  The
original directory is intentionally untouched.

The implementation keeps the original MSDCSE training/evaluation interface and
adds a relation-heterogeneity-aware dual calibration mechanism:

- a shared relation-context encoder built only from training triples;
- query-conditioned, energy-preserving multi-scale routing;
- competitive base/path/role evidence gates;
- query-level path-quality and role-quality features;
- a small learnable calibrator for offline path candidates;
- reciprocal-cycle filtering during path-rule mining;
- optional gradient clipping for stable training.

The final score is conceptually:

```text
s(h,r,t) = s_base(h,r,t)
          + beta_path(h,r) * path_strength * E_path(h,r,t)
          + beta_role(h,r) * role_strength * E_role(r,t)
```

`beta_base`, `beta_path`, and `beta_role` are produced by a softmax gate, so
the model explicitly decides whether the base embedding score, path evidence,
or role evidence should be trusted for the current query.

## Data preparation

Run from this directory:

```bash
python process_datasets.py --src_root ../../src_data --out_root ../../data
```

The script now accepts the command-line arguments used by the grid scripts. It
creates `../../data/{WN18RR,FB237,YAGO3-10,UMLS,KINSHIP}`.

## Recommended run

The existing PowerShell scripts can be used without changing their training
interface. The improved defaults are:

- training-only CE class weights;
- query-conditioned router and evidence gate;
- energy-preserving routing;
- inverse-cycle filtering during path mining;
- path-candidate calibration;
- gradient clipping at 1.0.

Useful ablations include:

```text
--no_router_query_context
--no_router_energy_preserving
--rcem_no_query_gate
--rcem_no_candidate_calibrator
--rcem_no_path
--rcem_no_type
```

For a strict comparison with the earlier direct-softmax router, use
`--no_router_energy_preserving`. For a clean paper protocol, keep the default
`--ce_weight_source train`; using `test` introduces test-split statistics into
training.

## Main files

- `models.py`: shared relation context, adaptive router, query-level RCEM gate,
  and path calibrator.
- `datasets.py`: relation statistics, reciprocal-cycle-safe evidence mining,
  and role evidence construction.
- `learn.py`: training entry point and all new switches.
- `optimizers.py`: original optimizer loop plus gradient clipping.
- `analyze_rcar_wn18rr.py`: routing/evidence analysis updated for the
  base/path/role gate.
