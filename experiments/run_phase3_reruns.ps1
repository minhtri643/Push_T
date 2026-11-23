# Re-measurements so that every stage has numbers on the same paired protocol (95 starts, 300 steps,
# success = coverage > 0.95). Runs one GPU job at a time. Logs: results/benchmark/logs/<label>.log
#     powershell -ExecutionPolicy Bypass -File experiments/run_phase3_reruns.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

function Invoke-Run($label, [string[]]$cmd) {
    Write-Host "=== $label ($(Get-Date -Format HH:mm:ss))"
    & $py @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning|warn|frozen successfully' } | Tee-Object -FilePath "results\benchmark\logs\$label.log" | Out-Host
}

# E0: original baseline models on the standard protocol
Invoke-Run "E0_baseline__dit_id" @("scripts/5_evaluation/benchmark.py", "--policy", "dit_id", "--ckpt-dir", "checkpoints/E0_baseline", "--label", "E0_baseline__dit_id")

# E2: hand-written controllers (7 configurations, one log)
Invoke-Run "E2_controllers" @("experiments/E2_handcrafted_controllers/code/benchmark_controllers.py")

# Diffusion-noise repeats (offsets 1 and 2) of the main configurations
foreach ($o in 1, 2) {
    Invoke-Run "E4_final__dit_wm__noise$o" @("scripts/5_evaluation/benchmark.py", "--policy", "dit_wm", "--seed-offset", "$o", "--label", "E4_final__dit_wm__noise$o")
    Invoke-Run "E4_final__dit_id__noise$o" @("scripts/5_evaluation/benchmark.py", "--policy", "dit_id", "--seed-offset", "$o", "--label", "E4_final__dit_id__noise$o")
    foreach ($v in @(@("E5b_jepa", "jepa_wm"), @("E5c_dinov2_frozen", "dinov2_hf_wm"), @("E5d_vjepa2_frozen", "vjepa2_wm"))) {
        $label = "$($v[0])__dit_id_vision__noise$o"
        Invoke-Run $label @("scripts/5_evaluation/benchmark.py", "--policy", "dit_id_vision", "--vision-backend", $v[1], "--seed-offset", "$o", "--label", $label)
    }
}
Write-Host "ALL DONE $(Get-Date -Format HH:mm:ss)"
