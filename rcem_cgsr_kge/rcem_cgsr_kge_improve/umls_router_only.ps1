Set-Location $PSScriptRoot

python learn.py --dataset UMLS `
        --model MSDCSE `
        --regularizer NA `
        --optimizer Adam `
        --rank 400 --k_w 20 --k_h 20 --output_channel 4 `
        --filter_size_list "[(1,3),(3,3),(1,5)]" `
        --input_drop 0.25 --hidden_drop 0.25 --feature_map_drop 0.45 `
        --active_fn "selu" --init_fn "kaiming_normal" `
        --use_scale_router `
        --router_dropout 0.20 --router_temperature 1.00 --router_min_branch_weight 0.03 `
        --ce_weight_source train `
        --learning_rate 0.001 --weight_decay 0.001 `
        --factor 0.5 --patience 8 --min_lr 0.00001 `
        --valid 10 --max_epochs 250 --batch_size 800 `
        --seed 42 --verbose 1 `
        -train -save -id umls_router_only
