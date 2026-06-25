# baseline_improve

This directory implements an improved version of `msrsc_baseline` with three
KG-specific switches for ablation:

1. Relation-set aware multi-positive objective (`--loss_mode soft_ce`)
2. Relation-cardinality adaptive multi-scale router (`--use_router` / `--no_router`)
3. Relation-aware anchor prototype enhancement (`--use_anchor` / `--no_anchor`)

## Prepare data

Run from this directory:

```bash
python process_datasets.py --datasets KINSHIP UMLS WN18RR FB237 YAGO3-10 --force
```

## Full model

```bash
python learn.py --dataset KINSHIP \
  --model MSRSCImprove \
  --rank 400 --k_w 20 --k_h 20 --output_channel 4 \
  --filter_size_list "[(1,3),(3,3),(1,5)]" \
  --loss_mode soft_ce --max_positives 64 \
  --use_router --router_hidden 64 --router_temperature 1.0 \
  --use_anchor --anchor_topk 8 --anchor_alpha 0.25 \
  --input_drop 0.3 --hidden_drop 0.1 --feature_map_drop 0.4 \
  --optimizer Adam --learning_rate 0.001 --weight_decay 0.005 \
  --valid 10 --max_epochs 200 --batch_size 800 --seed 42 \
  -train -test -save -id full
```

## Ablations

Original single-positive loss:

```bash
python learn.py --dataset KINSHIP --loss_mode ce --no_router --no_anchor -train -test -save -id base_ce
```

Remove router:

```bash
python learn.py --dataset KINSHIP --loss_mode soft_ce --no_router --use_anchor -train -test -save -id no_router
```

Remove anchor enhancement:

```bash
python learn.py --dataset KINSHIP --loss_mode soft_ce --use_router --no_anchor -train -test -save -id no_anchor
```

Only multi-positive objective:

```bash
python learn.py --dataset KINSHIP --loss_mode soft_ce --no_router --no_anchor -train -test -save -id only_multi_pos
```

Notes:

- `--loss_mode soft_ce` is the recommended efficient multi-positive objective.
- `--loss_mode bce` is available but usually consumes more GPU memory.
- `--amp` is enabled by default; use `--no_amp` for exact FP32 runs.
- `get_weight()` is train-based in this version to avoid test leakage.
