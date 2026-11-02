$Host.UI.RawUI.WindowTitle = "osu! temporal YOLO training"
Set-Location -LiteralPath "C:\Gamma\Osu_module"
Write-Host "Temporal YOLO: resuming from the last completed epoch" -ForegroundColor Cyan
Write-Host "Low CPU priority, workers=0, GPU yield enabled. Closing this window stops training." -ForegroundColor DarkGray
& python -u main.py train `
  --data "C:\Gamma\Osu_module\data\yolo_temporal_real" `
  --name osu_temporal `
  --epochs 30 `
  --batch 8 `
  --workers 0 `
  --nice `
  --resume
$code = $LASTEXITCODE
Write-Host "Training exited with code $code" -ForegroundColor $(if ($code -eq 0) { "Green" } else { "Red" })
Read-Host "Press Enter to close"
