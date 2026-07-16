$ErrorActionPreference = "Stop"

python learn.py `
    --dataset WN18RR `
    --model MSDCSE `
    --regularizer NA `
    --optimizer Adam `
    --rank 400 `
    --k_w 20 `
    --k_h 20 `
    --output_channel 4 `
    --filter_size_list "[(1,3),(3,3),(1,5)]" `
    --input_drop 0.30 `
    --hidden_drop 0.20 `
    --feature_map_drop 0.35 `
    --active_fn selu `
    --init_fn kaiming_normal `
    --ce_weight_source train `
    --learning_rate 0.001 `
    --weight_decay 0.0005 `
    --factor 0.5 `
    --patience 5 `
    --min_lr 0.00001 `
    --valid 5 `
    --max_epochs 200 `
    --batch_size 1500 `
    --seed 42 `
    --verbose 1 `
    --context_hidden 128 `
    --use_scale_router `
    --router_temperature 1.0 `
    --router_content_scale 0.25 `
    --use_rcem `
    --rcem_max_rules 8 `
    --rcem_standard_confidence_weight 0.3 `
    --rcem_path_strength 0.10 `
    --rcem_type_strength 0.04 `
    -train `
    -save `
    -id wn18rr_current_rcem_cgsr

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
