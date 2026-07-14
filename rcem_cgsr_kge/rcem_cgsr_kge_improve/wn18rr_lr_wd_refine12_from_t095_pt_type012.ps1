$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Test-Path "../../data/WN18RR/train.pickle")) {
    python process_datasets.py `
        --src_root "../../src_data" `
        --out_root "../../data" `
        --datasets WN18RR
}

$baseArgs = @(
    "--dataset", "WN18RR",
    "--model", "MSDCSE",
    "--regularizer", "NA",
    "--optimizer", "Adam",
    "--max_epochs", "400",
    "--valid", "10",
    "--rank", "400",
    "--batch_size", "1500",
    "--reg", "0",
    "--init", "0.001",
    "--decay1", "0.9",
    "--decay2", "0.999",
    "--name", "WN18RR",
        "--ce_weight_source", "train",
    "--save_path", "./logs/",
    "--negative_sample_size", "200",
    "--out_size", "4000",
    "--min_lr", "0.00001",
    "--input_drop", "0.2",
    "--hidden_drop", "0.45",
    "--feature_map_drop", "0.2",
    "--factor", "0.8",
    "--verbose", "1",
    "--patience", "5",
    "--momentum", "0.9",
    "--output_channel", "8",
    "--k_w", "20",
    "--k_h", "20",
    "--seed", "42",
    "--active_fn", "selu",
    "--init_fn", "kaiming_normal",
    "--filter_size_list", "[(1,3),(3,3),(1,5)]",
    "--use_scale_router",
    "--module_warmup_epochs", "0",
    "--module_ramp_epochs", "1",
    "--router_hidden", "0",
    "--router_dropout", "0.1",
    "--router_temperature", "0.95",
    "--router_min_branch_weight", "0.02",
    "--router_residual_init", "0.1",
    "--use_rcem",
    "--rcem_max_rules", "4",
    "--rcem_max_candidates", "16",
    "--rcem_min_rule_support", "8",
    "--rcem_max_rule_degree", "32",
    "--rcem_warmup_epochs", "20",
    "--rcem_ramp_epochs", "40",
    "--rcem_gate_hidden", "0",
    "--rcem_gate_dropout", "0.05",
    "--rcem_path_strength", "0.04",
    "--rcem_type_strength", "0.012",
    "--rcem_path_gate_init", "0.02",
    "--rcem_type_gate_init", "0.008",
    "-train",
    "-save"
)

# Local micro-search around the current best:
# learning_rate=0.001, weight_decay=0.0005.
# The exact best pair is intentionally not repeated.
$learningRates = @(
    @{ Name = "lr090"; Value = "0.00090" },
    @{ Name = "lr095"; Value = "0.00095" },
    @{ Name = "lr105"; Value = "0.00105" },
    @{ Name = "lr110"; Value = "0.00110" }
)

$weightDecays = @(
    @{ Name = "wd040"; Value = "0.00040" },
    @{ Name = "wd050"; Value = "0.00050" },
    @{ Name = "wd060"; Value = "0.00060" }
)

$runIndex = 0
foreach ($lr in $learningRates) {
    foreach ($wd in $weightDecays) {
        $runIndex += 1
        $runId = "wn18rr_lrwdr12_{0:D2}_{1}_{2}" -f $runIndex, $lr.Name, $wd.Name
        $argsList = @()
        $argsList += $baseArgs
        $argsList += @(
            "--learning_rate", $lr.Value,
            "--weight_decay", $wd.Value,
            "-id", $runId
        )

        Write-Host ""
        Write-Host "========== Running $runIndex / 12: $runId =========="
        Write-Host "learning_rate=$($lr.Value), weight_decay=$($wd.Value)"
        & python learn.py @argsList
    }
}
