$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Test-Path "../data/WN18RR/train.pickle")) {
    python process_datasets.py `
        --src_root "../src_data" `
        --out_root "../data" `
        --datasets WN18RR
}

# WN18RR is sparse: about 40K entities, 11 original relations, and most (h,r)
# queries have a single observed tail. The improved modules are therefore kept
# conservative: small multi-positive cap, lightweight relation router, and weak
# relation-anchor residual.
python learn.py --dataset WN18RR `
        --data_path "../data" `
        --min_lr 0.00001 `
        --model MSRSCImprove `
        --active_fn "selu" `
        --init_fn "kaiming_normal" `
        --rank 400 `
        --k_w 20 `
        --k_h 20 `
        --output_channel 8 `
        --filter_size_list "[(1,3),(3,3),(1,5)]" `
        --input_drop 0.2 `
        --hidden_drop 0.45 `
        --feature_map_drop 0.2 `
        --loss_mode soft_ce `
        --max_positives 32 `
        --label_smoothing 0.02 `
        --use_router `
        --router_hidden 32 `
        --router_temperature 0.75 `
        --use_anchor `
        --anchor_topk 4 `
        --anchor_alpha 0.12 `
        --anchor_dropout 0.2 `
        --optimizer Adam `
        --learning_rate 0.0003 `
        --weight_decay 5e-4 `
        --factor 0.5 `
        --verbose 1 `
        --patience 5 `
        --valid 20 `
        --max_epochs 200 `
        --batch_size 500 `
        --eval_batch_size 256 `
        --grad_clip 1.0 `
        --seed 42 `
        -train `
        -save `
        -id wn18rr_struct_adapt
