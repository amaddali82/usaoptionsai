# Stop USA Options AI System

Write-Host "Stopping USA Options AI System..." -ForegroundColor Yellow

# Stop all PowerShell jobs
Write-Host "`nStopping background jobs..." -ForegroundColor Gray
Get-Job | Stop-Job
Get-Job | Remove-Job
Write-Host "  Background jobs stopped" -ForegroundColor Green

# Stop Docker services
Write-Host "`nStopping Docker services..." -ForegroundColor Gray
docker-compose down

Write-Host "`n${GREEN}System stopped successfully${NC}" -ForegroundColor Green
