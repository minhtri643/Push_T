# Run from the repo root with the (pusht) environment active. About 80 minutes in total.
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path outputs\logs | Out-Null

Write-Host "=== A4: DDIM 4 and 8 steps ==="
python -m scripts.sweep_eval --name a4 --configs 8:4 8:8 2>&1 | Tee-Object -FilePath outputs\logs\a4.log

Write-Host "=== A1: random-init encoder (train DiT, then evaluate) ==="
python -m scripts.train_policy --run_name dit_random_init --latents data/cache/latents_random_init.npy --overwrite 2>&1 | Tee-Object -FilePath outputs\logs\a1_train.log
python -m scripts.sweep_eval --policy_run dit_random_init --wm_random_init --name a1 --configs 8:16 2>&1 | Tee-Object -FilePath outputs\logs\a1_eval.log

Write-Host "=== A2: no proprioception (train DiT, then evaluate) ==="
python -m scripts.train_policy --run_name dit_no_proprio --override policy.use_proprio=false --overwrite 2>&1 | Tee-Object -FilePath outputs\logs\a2_train.log
python -m scripts.sweep_eval --policy_run dit_no_proprio --name a2 --configs 8:16 2>&1 | Tee-Object -FilePath outputs\logs\a2_eval.log

Write-Host "=== ALL DONE ==="
