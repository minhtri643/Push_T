# Runs one step of the original pipeline inside E7 only.
# Usage (from anywhere):  powershell -File experiments/E7_self_supervised_jepa_dit/run_e7.ps1 scripts.train_world_model --run_name wm_main
# cwd = E7/run (configs, data cache, outputs stay inside E7); code = E7/code (unchanged, commit ebe99bd);
# extra packages = E7/_deps (pandas, scikit-learn, av); everything else comes from the main .venv.
# TF32 is switched off: PyTorch allows TF32 in cuDNN convolutions by default on Ampere GPUs (RTX 3050 here), while
# the GTX 1650 of the original run has no TF32, i.e. its runs are strict fp32. With TF32 on, the code's own export check fails
# (batch-size dependent latents, max diff 7.7e-3 > 1e-3); with TF32 off the difference is 2.1e-5.
$E7 = $PSScriptRoot
$py = Join-Path $E7 "..\..\.venv\Scripts\python.exe"
$env:PYTHONPATH = "$E7\_deps;$E7\code"
$env:PYTHONUTF8 = "1"
$launch = "import runpy, sys, torch; torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False; m = sys.argv[1]; sys.argv = [m] + sys.argv[2:]; runpy.run_module(m, run_name='__main__', alter_sys=True)"
Push-Location (Join-Path $E7 "run")
try { & $py -c $launch @args; exit $LASTEXITCODE } finally { Pop-Location }
