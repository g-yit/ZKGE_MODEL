$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

# The improved project is nested below rcem_cgsr_kge.  Data remains shared with
# the original project and is therefore two levels above this script.
if (-not (Test-Path "../../data/WN18RR/train.pickle")) {
    python process_datasets.py `
        --src_root "../../src_data" `
        --out_root "../../data" `
        --datasets WN18RR
}

# Start from the strongest historical WN18RR configuration:
# rank=400, three convolution scales, Adam, lr=1e-3, batch_size=1500.
# The new code keeps this capacity and adds conservative, query-aware
# calibration.  Training-only CE weights are used to avoid test-stat leakage.
python learn.py --dataset WN18RR `
        --model MSDCSE `
        --regularizer NA `
        --optimizer Adam `
        --rank 400 `
        --k_w 20 `
        --k_h 20 `
        --output_channel 8 `
        --filter_size_list "[(1,3),(3,3),(1,5)]" `
        --input_drop 0.20 `
        --hidden_drop 0.45 `
        --feature_map_drop 0.20 `
        --active_fn "selu" `
        --init_fn "kaiming_normal" `
        --ce_weight_source train `
        --seed 42 `
        --valid 10 `
        --max_epochs 400 `
        --batch_size 1500 `
        --learning_rate 0.001 `
        --weight_decay 0.0005 `
        --factor 0.8 `
        --patience 5 `
        --min_lr 0.00001 `
        --grad_clip 5.0 `
        --use_scale_router `
        --router_hidden 0 `
        --router_dropout 0.10 `
        --router_temperature 0.95 `
        --router_min_branch_weight 0.02 `
        --module_warmup_epochs 0 `
        --module_ramp_epochs 1 `
        --relation_context_dim 0 `
        --relation_context_dropout 0.10 `
        --use_rcem `
        --rcem_max_rules 4 `
        --rcem_max_candidates 16 `
        --rcem_min_rule_support 8 `
        --rcem_max_rule_degree 32 `
        --rcem_path_strength 0.04 `
        --rcem_type_strength 0.012 `
        --rcem_path_gate_init 0.02 `
        --rcem_type_gate_init 0.008 `
        --rcem_warmup_epochs 20 `
        --rcem_ramp_epochs 40 `
        --rcem_gate_hidden 0 `
        --rcem_gate_dropout 0.05 `
        --rcem_calibrator_hidden 64 `
        --rcem_calibrator_strength 0.10 `
        -train `
        -save `
        -id wn18rr_improved_best_refine
