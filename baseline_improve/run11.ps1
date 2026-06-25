# ============================================================
# SCOPE joint-search script: 16 configurations, 200 epochs each
# Dataset: WN18RR
# Model: MSDCSE
# Modules: CSCC + SGHNCL + GSPCL
# ============================================================

$ErrorActionPreference = "Stop"

$Python = "python"
$Script = "learn.py"

$LogDir = "logs_scope_joint_16x200"
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

# -----------------------------
# Fixed backbone settings
# -----------------------------
$BaseArgs = @(
    $Script,
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
    "--optimizer", "Adam",
    "--learning_rate", "0.0003",
    "--weight_decay", "5e-4",
    "--factor", "0.5",
    "--verbose", "1",
    "--patience", "5",
    "--valid", "20",
    "--max_epochs", "200",
    "--batch_size", "500",
    "--seed", "42",
    "-train",
    "-save"
)

# ============================================================
# 16 search configurations
# ============================================================
$Configs = @(

    # -----------------------------
    # Group A: stable low-weight joint setting
    # -----------------------------
    @{
        id="scope16_A01_cs02_hn005_gs005_tauhn02_hk8_gsk8_taug05"
        lambda_cscc="0.2"; cscc_proj_dim="64"; cscc_tau="0.8"
        lambda_sghncl="0.05"; sghncl_proj_dim="128"; sghncl_tau="0.2"; hard_neg_k="8"
        lambda_gspcl="0.05"; gspcl_top_k="8"; gspcl_tau="0.5"
    },

    @{
        id="scope16_A02_cs03_hn005_gs005_tauhn02_hk8_gsk8_taug05"
        lambda_cscc="0.3"; cscc_proj_dim="64"; cscc_tau="0.8"
        lambda_sghncl="0.05"; sghncl_proj_dim="128"; sghncl_tau="0.2"; hard_neg_k="8"
        lambda_gspcl="0.05"; gspcl_top_k="8"; gspcl_tau="0.5"
    },

    @{
        id="scope16_A03_cs02_hn01_gs005_tauhn02_hk8_gsk8_taug05"
        lambda_cscc="0.2"; cscc_proj_dim="64"; cscc_tau="0.8"
        lambda_sghncl="0.1"; sghncl_proj_dim="128"; sghncl_tau="0.2"; hard_neg_k="8"
        lambda_gspcl="0.05"; gspcl_top_k="8"; gspcl_tau="0.5"
    },

    @{
        id="scope16_A04_cs02_hn005_gs01_tauhn02_hk8_gsk8_taug07"
        lambda_cscc="0.2"; cscc_proj_dim="64"; cscc_tau="0.8"
        lambda_sghncl="0.05"; sghncl_proj_dim="128"; sghncl_tau="0.2"; hard_neg_k="8"
        lambda_gspcl="0.1"; gspcl_top_k="8"; gspcl_tau="0.7"
    },

    # -----------------------------
    # Group B: CSCC-dominant setting
    # -----------------------------
    @{
        id="scope16_B01_cs04_hn005_gs005_tauhn02_hk8_gsk8_taug05"
        lambda_cscc="0.4"; cscc_proj_dim="64"; cscc_tau="0.8"
        lambda_sghncl="0.05"; sghncl_proj_dim="128"; sghncl_tau="0.2"; hard_neg_k="8"
        lambda_gspcl="0.05"; gspcl_top_k="8"; gspcl_tau="0.5"
    },

    @{
        id="scope16_B02_cs04_hn005_gs01_tauhn02_hk8_gsk8_taug07"
        lambda_cscc="0.4"; cscc_proj_dim="64"; cscc_tau="0.8"
        lambda_sghncl="0.05"; sghncl_proj_dim="128"; sghncl_tau="0.2"; hard_neg_k="8"
        lambda_gspcl="0.1"; gspcl_top_k="8"; gspcl_tau="0.7"
    },

    @{
        id="scope16_B03_cs05_hn003_gs003_tauhn02_hk8_gsk8_taug07"
        lambda_cscc="0.5"; cscc_proj_dim="64"; cscc_tau="0.8"
        lambda_sghncl="0.03"; sghncl_proj_dim="128"; sghncl_tau="0.2"; hard_neg_k="8"
        lambda_gspcl="0.03"; gspcl_top_k="8"; gspcl_tau="0.7"
    },

    @{
        id="scope16_B04_cs03_hn01_gs005_tauhn03_hk8_gsk8_taug05"
        lambda_cscc="0.3"; cscc_proj_dim="64"; cscc_tau="0.8"
        lambda_sghncl="0.1"; sghncl_proj_dim="128"; sghncl_tau="0.3"; hard_neg_k="8"
        lambda_gspcl="0.05"; gspcl_top_k="8"; gspcl_tau="0.5"
    },

    # -----------------------------
    # Group C: SGHNCL-focused setting
    # -----------------------------
    @{
        id="scope16_C01_cs01_hn01_gs005_tauhn02_hk8_gsk8_taug05"
        lambda_cscc="0.1"; cscc_proj_dim="64"; cscc_tau="0.8"
        lambda_sghncl="0.1"; sghncl_proj_dim="128"; sghncl_tau="0.2"; hard_neg_k="8"
        lambda_gspcl="0.05"; gspcl_top_k="8"; gspcl_tau="0.5"
    },

    @{
        id="scope16_C02_cs01_hn01_gs005_tauhn03_hk8_gsk8_taug05"
        lambda_cscc="0.1"; cscc_proj_dim="64"; cscc_tau="0.8"
        lambda_sghncl="0.1"; sghncl_proj_dim="128"; sghncl_tau="0.3"; hard_neg_k="8"
        lambda_gspcl="0.05"; gspcl_top_k="8"; gspcl_tau="0.5"
    },

    @{
        id="scope16_C03_cs02_hn01_gs005_tauhn02_hk16_gsk8_taug05"
        lambda_cscc="0.2"; cscc_proj_dim="64"; cscc_tau="0.8"
        lambda_sghncl="0.1"; sghncl_proj_dim="128"; sghncl_tau="0.2"; hard_neg_k="16"
        lambda_gspcl="0.05"; gspcl_top_k="8"; gspcl_tau="0.5"
    },

    @{
        id="scope16_C04_cs02_hn007_gs005_tauhn015_hk8_gsk8_taug05"
        lambda_cscc="0.2"; cscc_proj_dim="64"; cscc_tau="0.8"
        lambda_sghncl="0.07"; sghncl_proj_dim="128"; sghncl_tau="0.15"; hard_neg_k="8"
        lambda_gspcl="0.05"; gspcl_top_k="8"; gspcl_tau="0.5"
    },

    # -----------------------------
    # Group D: GSPCL-focused setting
    # -----------------------------
    @{
        id="scope16_D01_cs02_hn005_gs01_tauhn02_hk8_gsk16_taug07"
        lambda_cscc="0.2"; cscc_proj_dim="64"; cscc_tau="0.8"
        lambda_sghncl="0.05"; sghncl_proj_dim="128"; sghncl_tau="0.2"; hard_neg_k="8"
        lambda_gspcl="0.1"; gspcl_top_k="16"; gspcl_tau="0.7"
    },

    @{
        id="scope16_D02_cs03_hn005_gs01_tauhn02_hk8_gsk16_taug07"
        lambda_cscc="0.3"; cscc_proj_dim="64"; cscc_tau="0.8"
        lambda_sghncl="0.05"; sghncl_proj_dim="128"; sghncl_tau="0.2"; hard_neg_k="8"
        lambda_gspcl="0.1"; gspcl_top_k="16"; gspcl_tau="0.7"
    },

    @{
        id="scope16_D03_cs02_hn003_gs015_tauhn02_hk8_gsk16_taug08"
        lambda_cscc="0.2"; cscc_proj_dim="64"; cscc_tau="0.8"
        lambda_sghncl="0.03"; sghncl_proj_dim="128"; sghncl_tau="0.2"; hard_neg_k="8"
        lambda_gspcl="0.15"; gspcl_top_k="16"; gspcl_tau="0.8"
    },

    @{
        id="scope16_D04_cs01_hn005_gs015_tauhn02_hk8_gsk16_taug08"
        lambda_cscc="0.1"; cscc_proj_dim="64"; cscc_tau="0.8"
        lambda_sghncl="0.05"; sghncl_proj_dim="128"; sghncl_tau="0.2"; hard_neg_k="8"
        lambda_gspcl="0.15"; gspcl_top_k="16"; gspcl_tau="0.8"
    }
)

# ============================================================
# Run experiments
# ============================================================
foreach ($cfg in $Configs) {

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Running experiment: $($cfg.id)"
    Write-Host "============================================================"

    $ExpArgs = @(
        "--use_cscc",
        "--lambda_cscc", $cfg.lambda_cscc,
        "--cscc_proj_dim", $cfg.cscc_proj_dim,
        "--cscc_tau", $cfg.cscc_tau,

        "--use_sghncl",
        "--lambda_sghncl", $cfg.lambda_sghncl,
        "--sghncl_proj_dim", $cfg.sghncl_proj_dim,
        "--sghncl_tau", $cfg.sghncl_tau,
        "--hard_neg_k", $cfg.hard_neg_k,

        "--use_gspcl",
        "--lambda_gspcl", $cfg.lambda_gspcl,
        "--gspcl_top_k", $cfg.gspcl_top_k,
        "--gspcl_tau", $cfg.gspcl_tau,

        "-id", $cfg.id
    )

    $RunArgs = $BaseArgs + $ExpArgs
    $LogFile = Join-Path $LogDir "$($cfg.id).log"

    Write-Host "Command:"
    Write-Host "$Python $($RunArgs -join ' ')"
    Write-Host "Log file: $LogFile"

    & $Python @RunArgs 2>&1 | Tee-Object -FilePath $LogFile

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Experiment failed: $($cfg.id)" -ForegroundColor Red
        Write-Host "Continuing to next configuration..."
    }
    else {
        Write-Host "Finished experiment: $($cfg.id)" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "All 16 experiments have finished."