param(
    [string]$Venv = "C:\fern\FERN_V2\venv\Scripts\python.exe",
    [string]$Root = "C:\fern\FERN_V2"
)

Set-Location -LiteralPath $Root
$log = "C:\fern\FERN_V2\run_nightly.log"
$py = $Venv

function Log { param($msg) "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Out-File -Append -LiteralPath $log; Write-Host $msg }

Log "=== Pipeline start ==="

# Step 1: 5-fold CV front-only (video-level grouping)
Log "Step 1: 5-fold CV front-only (group_by=video, 50 epochs)"
$start = Get-Date
& $py src/kfold_cv.py --skeleton_dir data/skeletons/front --label_dir data/labels/front --n_cameras 1 --group_by video --epochs 50 --k_folds 5 --device cuda 2>&1 | Out-File -Append -LiteralPath $log
$elapsed = (Get-Date) - $start
Log "Step 1 done in $($elapsed.TotalMinutes.ToString('0.0')) min"

# Step 2: 5-fold CV front+45 (subject-level grouping)
Log "Step 2: 5-fold CV front+45 (group_by=subject, n_cameras=2, 50 epochs)"
$start = Get-Date
& $py src/kfold_cv.py --skeleton_dir data/skeletons/front_plus_45 --label_dir data/labels/front_plus_45 --n_cameras 2 --group_by subject --epochs 50 --k_folds 5 --device cuda 2>&1 | Out-File -Append -LiteralPath $log
$elapsed = (Get-Date) - $start
Log "Step 2 done in $($elapsed.TotalMinutes.ToString('0.0')) min"

# Step 3: Train front-only production model (train_all)
Log "Step 3: Train front-only production model (200 epochs, train_all)"
$start = Get-Date
& $py src/train_v2.py --skeleton_dir data/skeletons/front --label_dir data/labels/front --output_dir models_final --log_dir logs_final --epochs 200 --warmup_epochs 20 --batch_size 32 --window_size 60 --stride 15 --lr 3e-4 --weight_decay 1e-2 --dropout 0.6 --cnn_out 64 --lstm_hidden 0 --device cuda --num_workers 4 --train_all 2>&1 | Out-File -Append -LiteralPath $log
$elapsed = (Get-Date) - $start
Log "Step 3 done in $($elapsed.TotalMinutes.ToString('0.0')) min"

# Step 4: Train front+45 Phase 1 model (train_all, n_cameras=2)
Log "Step 4: Train front+45 Phase 1 (200 epochs, train_all, n_cameras=2)"
$start = Get-Date
& $py src/train_v2.py --skeleton_dir data/skeletons/front_plus_45 --label_dir data/labels/front_plus_45 --output_dir models_final_v2 --log_dir logs_final_v2 --epochs 200 --warmup_epochs 20 --batch_size 32 --window_size 60 --stride 15 --lr 3e-4 --weight_decay 1e-2 --dropout 0.6 --cnn_out 64 --lstm_hidden 0 --device cuda --num_workers 4 --n_cameras 2 --train_all 2>&1 | Out-File -Append -LiteralPath $log
$elapsed = (Get-Date) - $start
Log "Step 4 done in $($elapsed.TotalMinutes.ToString('0.0')) min"

# Step 5: Export ONNX + test for both models
Log "Step 5: Export + test ONNX models"
& $py src/export_onnx.py --checkpoint_path models_final/fern_v2_latest.pth --output_path models_final/fern_v2.onnx 2>&1 | Out-File -Append -LiteralPath $log
& $py src/export_onnx.py --checkpoint_path models_final_v2/fern_v2_latest.pth --output_path models_final_v2/fern_v2.onnx 2>&1 | Out-File -Append -LiteralPath $log

Log "--- ONNX test: front-only ---"
& $py src/test_onnx.py --onnx_path models_final/fern_v2.onnx --skeleton_dir data/skeletons/front --label_dir data/labels/front --window_size 60 --stride 15 2>&1 | Out-File -Append -LiteralPath $log

Log "--- ONNX test: front+45 Phase 1 ---"
& $py src/test_onnx.py --onnx_path models_final_v2/fern_v2.onnx --skeleton_dir data/skeletons/front_plus_45 --label_dir data/labels/front_plus_45 --n_cameras 2 --window_size 60 --stride 15 2>&1 | Out-File -Append -LiteralPath $log

# Step 6: Commit and push results
Log "Step 6: Commit and push results"
git add -A 2>&1 | Out-File -Append -LiteralPath $log
git -c "core.autocrlf=true" commit -m "Nightly re-baseline after fix.txt fixes" 2>&1 | Out-File -Append -LiteralPath $log
git pull --rebase 2>&1 | Out-File -Append -LiteralPath $log
git push 2>&1 | Out-File -Append -LiteralPath $log

Log "=== Pipeline complete ==="
