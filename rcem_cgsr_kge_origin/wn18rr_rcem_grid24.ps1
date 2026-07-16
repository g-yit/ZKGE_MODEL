$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Test-Path "../data/WN18RR/train.pickle")) {
    python process_datasets.py `
        --src_root "../src_data" `
        --out_root "../data" `
        --datasets WN18RR
}

$baseArgs = @(
    "--dataset", "WN18RR",
    "--min_lr", "0.00001",
    "--model", "MSDCSE",
    "--active_fn", "selu",
    "--init_fn", "kaiming_normal",
    "--rank", "400",
    "--k_w", "20",
    "--k_h", "20",
    "--output_channel", "8",
    "--filter_size_list", "[(1,3),(3,3),(1,5)]",
    "--input_drop", "0.2",
    "--hidden_drop", "0.45",
    "--feature_map_drop", "0.2",
    "--ce_weight_source", "test",
    "--seed", "42",
    "--valid", "10",
    "--max_epochs", "400",
    "--batch_size", "1500",
    "--learning_rate", "0.001",
    "--weight_decay", "0.0005",
    "--optimizer", "Adam",
    "-train",
    "-save"
)

$routerProfiles = @(
    @{
        Name = "r_base"
        Args = @(
            "--use_scale_router",
            "--router_dropout", "0.10",
            "--router_temperature", "1.00",
            "--router_min_branch_weight", "0.02",
            "--module_warmup_epochs", "0",
            "--module_ramp_epochs", "1"
        )
    },
    @{
        Name = "r_smooth"
        Args = @(
            "--use_scale_router",
            "--router_dropout", "0.05",
            "--router_temperature", "1.50",
            "--router_min_branch_weight", "0.05",
            "--module_warmup_epochs", "3",
            "--module_ramp_epochs", "15"
        )
    },
    @{
        Name = "r_res005"
        Args = @(
            "--use_scale_router",
            "--use_router_residual",
            "--router_residual_init", "0.05",
            "--router_dropout", "0.05",
            "--router_temperature", "1.00",
            "--router_min_branch_weight", "0.02",
            "--module_warmup_epochs", "0",
            "--module_ramp_epochs", "1"
        )
    }
)

$rcemProfiles = @(
    @{
        Name = "p_vweak"
        NoPath = $false
        NoType = $true
        MaxRules = "3"
        MaxCandidates = "8"
        MinSupport = "10"
        MaxDegree = "24"
        PathStrength = "0.025"
        TypeStrength = "0.00"
        PathGate = "0.01"
        TypeGate = "0.01"
        Warmup = "30"
        Ramp = "50"
        GateDropout = "0.05"
    },
    @{
        Name = "p_weak"
        NoPath = $false
        NoType = $true
        MaxRules = "4"
        MaxCandidates = "16"
        MinSupport = "8"
        MaxDegree = "32"
        PathStrength = "0.04"
        TypeStrength = "0.00"
        PathGate = "0.02"
        TypeGate = "0.01"
        Warmup = "20"
        Ramp = "40"
        GateDropout = "0.05"
    },
    @{
        Name = "p_mid"
        NoPath = $false
        NoType = $true
        MaxRules = "6"
        MaxCandidates = "16"
        MinSupport = "6"
        MaxDegree = "32"
        PathStrength = "0.06"
        TypeStrength = "0.00"
        PathGate = "0.03"
        TypeGate = "0.01"
        Warmup = "15"
        Ramp = "30"
        GateDropout = "0.05"
    },
    @{
        Name = "p_mid_deg48"
        NoPath = $false
        NoType = $true
        MaxRules = "6"
        MaxCandidates = "24"
        MinSupport = "6"
        MaxDegree = "48"
        PathStrength = "0.06"
        TypeStrength = "0.00"
        PathGate = "0.03"
        TypeGate = "0.01"
        Warmup = "15"
        Ramp = "30"
        GateDropout = "0.05"
    },
    @{
        Name = "pt_tiny"
        NoPath = $false
        NoType = $false
        MaxRules = "4"
        MaxCandidates = "16"
        MinSupport = "8"
        MaxDegree = "32"
        PathStrength = "0.04"
        TypeStrength = "0.015"
        PathGate = "0.02"
        TypeGate = "0.01"
        Warmup = "20"
        Ramp = "40"
        GateDropout = "0.05"
    },
    @{
        Name = "pt_weak"
        NoPath = $false
        NoType = $false
        MaxRules = "6"
        MaxCandidates = "24"
        MinSupport = "6"
        MaxDegree = "32"
        PathStrength = "0.06"
        TypeStrength = "0.02"
        PathGate = "0.03"
        TypeGate = "0.015"
        Warmup = "20"
        Ramp = "35"
        GateDropout = "0.05"
    },
    @{
        Name = "p_more"
        NoPath = $false
        NoType = $true
        MaxRules = "8"
        MaxCandidates = "24"
        MinSupport = "5"
        MaxDegree = "48"
        PathStrength = "0.08"
        TypeStrength = "0.00"
        PathGate = "0.03"
        TypeGate = "0.01"
        Warmup = "10"
        Ramp = "30"
        GateDropout = "0.05"
    },
    @{
        Name = "t_only"
        NoPath = $true
        NoType = $false
        MaxRules = "4"
        MaxCandidates = "16"
        MinSupport = "8"
        MaxDegree = "32"
        PathStrength = "0.00"
        TypeStrength = "0.02"
        PathGate = "0.01"
        TypeGate = "0.01"
        Warmup = "20"
        Ramp = "40"
        GateDropout = "0.05"
    }
)

$runIndex = 0
foreach ($router in $routerProfiles) {
    foreach ($rcem in $rcemProfiles) {
        $runIndex += 1
        $runId = "wn18rr_grid24_{0:D2}_{1}_{2}" -f $runIndex, $router.Name, $rcem.Name
        $argsList = @()
        $argsList += $baseArgs
        $argsList += $router.Args
        $argsList += @(
            "--use_rcem",
            "--rcem_max_rules", $rcem.MaxRules,
            "--rcem_max_candidates", $rcem.MaxCandidates,
            "--rcem_min_rule_support", $rcem.MinSupport,
            "--rcem_max_rule_degree", $rcem.MaxDegree,
            "--rcem_path_strength", $rcem.PathStrength,
            "--rcem_type_strength", $rcem.TypeStrength,
            "--rcem_path_gate_init", $rcem.PathGate,
            "--rcem_type_gate_init", $rcem.TypeGate,
            "--rcem_warmup_epochs", $rcem.Warmup,
            "--rcem_ramp_epochs", $rcem.Ramp,
            "--rcem_gate_dropout", $rcem.GateDropout,
            "-id", $runId
        )
        if ($rcem.NoPath) {
            $argsList += "--rcem_no_path"
        }
        if ($rcem.NoType) {
            $argsList += "--rcem_no_type"
        }

        Write-Host ""
        Write-Host "========== Running $runIndex / 24: $runId =========="
        & python learn.py @argsList
    }
}
