$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Test-Path "../data/WN18RR/train.pickle")) {
    python process_datasets.py `
        --src_root "../src_data" `
        --out_root "../data" `
        --datasets WN18RR
}

python learn.py --dataset WN18RR `
        --model MSDCSE `
        --regularizer NA `
        --optimizer Adam `
        --max_epochs 400 `
        --valid 10 `
        --rank 400 `
        --batch_size 1500 `
        --reg 0 `
        --init 0.001 `
        --learning_rate 0.001 `
        --decay1 0.9 `
        --decay2 0.999 `
        --name WN18RR `
        -weight `
        --ce_weight_source test `
        --save_path "./logs/" `
        --negative_sample_size 200 `
        --out_size 4000 `
        --min_lr 0.00001 `
        --input_drop 0.2 `
        --hidden_drop 0.45 `
        --feature_map_drop 0.2 `
        --weight_decay 0.0005 `
        --factor 0.8 `
        --verbose 1 `
        --patience 5 `
        --momentum 0.9 `
        --output_channel 8 `
        --k_w 20 `
        --k_h 20 `
        --seed 42 `
        --active_fn "selu" `
        --init_fn "kaiming_normal" `
        --filter_size_list "[(1,3),(3,3),(1,5)]" `
        --use_scale_router `
        --module_warmup_epochs 0 `
        --module_ramp_epochs 1 `
        --router_hidden 0 `
        --router_dropout 0.1 `
        --router_temperature 0.95 `
        --router_min_branch_weight 0.02 `
        --router_residual_init 0.1 `
        --use_rcem `
        --rcem_max_rules 4 `
        --rcem_max_candidates 16 `
        --rcem_min_rule_support 8 `
        --rcem_max_rule_degree 32 `
        --rcem_warmup_epochs 20 `
        --rcem_ramp_epochs 40 `
        --rcem_gate_hidden 0 `
        --rcem_gate_dropout 0.05 `
        --rcem_path_strength 0.04 `
        --rcem_type_strength 0.012 `
        --rcem_path_gate_init 0.02 `
        --rcem_type_gate_init 0.008 `
        -train `
        -save `
        -id wn18rr_refine16_07_r_t095_pt_type012
