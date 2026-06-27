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

# Previous best neighborhood:
# r_base: dropout=0.10, temperature=1.00, min_branch_weight=0.02, no router warmup.
# These four profiles keep that behavior and only make tiny local changes.
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
        Name = "r_t095"
        Args = @(
            "--use_scale_router",
            "--router_dropout", "0.10",
            "--router_temperature", "0.95",
            "--router_min_branch_weight", "0.02",
            "--module_warmup_epochs", "0",
            "--module_ramp_epochs", "1"
        )
    },
    @{
        Name = "r_t110"
        Args = @(
            "--use_scale_router",
            "--router_dropout", "0.10",
            "--router_temperature", "1.10",
            "--router_min_branch_weight", "0.02",
            "--module_warmup_epochs", "0",
            "--module_ramp_epochs", "1"
        )
    },
    @{
        Name = "r_mw025"
        Args = @(
            "--use_scale_router",
            "--router_dropout", "0.10",
            "--router_temperature", "1.00",
            "--router_min_branch_weight", "0.025",
            "--module_warmup_epochs", "0",
            "--module_ramp_epochs", "1"
        )
    }
)

# Previous best RCEM point:
# pt_tiny: rules=4, candidates=16, support=8, degree=32,
#          path=0.04, type=0.015, gates=(0.02, 0.01), warmup/ramp=(20, 40).
# All profiles below are new, non-duplicate local variants around that point.
$rcemProfiles = @(
    @{
        Name = "pt_sup9"
        NoPath = $false
        NoType = $false
        MaxRules = "4"
        MaxCandidates = "16"
        MinSupport = "9"
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
        Name = "pt_path035"
        NoPath = $false
        NoType = $false
        MaxRules = "4"
        MaxCandidates = "16"
        MinSupport = "8"
        MaxDegree = "32"
        PathStrength = "0.035"
        TypeStrength = "0.015"
        PathGate = "0.018"
        TypeGate = "0.01"
        Warmup = "20"
        Ramp = "40"
        GateDropout = "0.05"
    },
    @{
        Name = "pt_type012"
        NoPath = $false
        NoType = $false
        MaxRules = "4"
        MaxCandidates = "16"
        MinSupport = "8"
        MaxDegree = "32"
        PathStrength = "0.04"
        TypeStrength = "0.012"
        PathGate = "0.02"
        TypeGate = "0.008"
        Warmup = "20"
        Ramp = "40"
        GateDropout = "0.05"
    },
    @{
        Name = "pt_cand12_w25"
        NoPath = $false
        NoType = $false
        MaxRules = "4"
        MaxCandidates = "12"
        MinSupport = "8"
        MaxDegree = "32"
        PathStrength = "0.04"
        TypeStrength = "0.015"
        PathGate = "0.02"
        TypeGate = "0.01"
        Warmup = "25"
        Ramp = "45"
        GateDropout = "0.05"
    }
)

$runIndex = 0
foreach ($router in $routerProfiles) {
    foreach ($rcem in $rcemProfiles) {
        $runIndex += 1
        $runId = "wn18rr_refine16_{0:D2}_{1}_{2}" -f $runIndex, $router.Name, $rcem.Name
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
        Write-Host "========== Running $runIndex / 16: $runId =========="
        & python learn.py @argsList
    }
}
