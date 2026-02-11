# start_fresh.ps1
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "STARTING FRESH IT TASK MANAGER" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Clean up any existing files
Write-Host "Cleaning up old files..." -ForegroundColor Yellow
$files_to_delete = @('task_manager.db', 'task_manager.db-wal', 'task_manager.db-shm')
foreach ($file in $files_to_delete) {
    if (Test-Path $file) {
        Remove-Item -Force $file -ErrorAction SilentlyContinue
        Write-Host "  Deleted: $file" -ForegroundColor Green
    }
}

# Clean cache
if (Test-Path "__pycache__") {
    Remove-Item -Force -Recurse "__pycache__" -ErrorAction SilentlyContinue
    Write-Host "  Cleaned pycache" -ForegroundColor Green
}

Get-ChildItem -Path . -Include "__pycache__" -Recurse -Directory | ForEach-Object {
    Remove-Item -Force -Recurse $_.FullName -ErrorAction SilentlyContinue
}

# Start the application
Write-Host "`nStarting Flask application..." -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Cyan
python app.py