Set-Location $PSScriptRoot

python learn.py --dataset FB237 `
        --model MSDCSE `
        --regularizer NA `
        --optimizer Adam `
        --rank 400 --k_w 20 --k_h 20 --output_channel 4 `
        --filter_size_list "[(1,3),(3,3),(1,5)]" `
        --input_drop 0.35 --hidden_drop 0.20 --feature_map_drop 0.40 `
        --active_fn "selu" --init_fn "kaiming_normal" `
        --use_scale_router `
        --router_dropout 0.15 --router_temperature 1.00 --router_min_branch_weight 0.02 `
        --ce_weight_source train `
        --learning_rate 0.001 --weight_decay 0.0005 `
        --factor 0.5 --patience 5 --min_lr 0.00001 `
        --valid 5 --max_epochs 200 --batch_size 512 `
        --seed 42 --verbose 1 `
        -train -save -id fb237_router_only
