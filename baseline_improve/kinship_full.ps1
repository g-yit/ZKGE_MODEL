$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Test-Path "../data/KINSHIP/train.pickle")) {
    python process_datasets.py `
        --src_root "../src_data" `
        --out_root "../data" `
        --datasets KINSHIP
}

python learn.py --dataset KINSHIP `
        --data_path "../data" `
        --min_lr 0.00001 `
        --model MSRSCImprove `
        --active_fn "selu" --init_fn "kaiming_normal" `
        --rank 400 --k_w 20 --k_h 20 --output_channel 2 `
        --filter_size_list "[(1,3),(3,3),(1,5)]" `
        --input_drop 0.3 --hidden_drop 0.1 --feature_map_drop 0.4 `
        --loss_mode soft_ce --max_positives 64 `
        --use_router --router_hidden 64 --router_temperature 1.0 `
        --optimizer Adam --weight_decay 5e-3 `
        --factor 0.5 --verbose 1 `
        --patience 5 `
        --valid 10 `
        --max_epochs 200 `
        --batch_size 800 `
        --seed 42 `
        --learning_rate 0.001 `
        -train -test -id kinship_full -save
