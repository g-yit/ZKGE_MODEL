python learn.py --dataset KINSHIP `
        --min_lr 0.00001 `
        --model MSDCSE `
        --active_fn "selu" --init_fn "kaiming_normal" `
        --rank 400 --k_w 20 --k_h 20 --output_channel 4 `
        --filter_size_list "[(1,3),(3,3),(1,5)]" `
        --input_drop 0.3 --hidden_drop 0.1 --feature_map_drop 0.4 `
        --optimizer Adam  --weight_decay 5e-3 `
        --factor 0.5 --verbose 1 `
        --patience 5 `
        --valid 10 `
        --max_epochs 200 `
        --batch_size 800 `
        --seed 42 `
        --learning_rate 0.001 `
        -train -id aaa_uml -save

