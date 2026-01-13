# USA Options AI System - Startup Script
# This script starts all components of the system in the correct order

param(
    [switch]$SkipDocker,
    [switch]$SkipIngestion,
    [switch]$SkipProcessing,
    [switch]$SkipModels,
    [switch]$SkipDashboard
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "USA Options AI System Startup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if Docker is running
if (-not $SkipDocker) {
    Write-Host "[1/6] Checking Docker..." -ForegroundColor Yellow
    
    $dockerRunning = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Docker is not running. Please start Docker Desktop." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "  Docker is running" -ForegroundColor Green
    
    # Start Docker Compose services
    Write-Host "`n[2/6] Starting Docker services..." -ForegroundColor Yellow
    Write-Host "  This may take a few minutes on first run..." -ForegroundColor Gray
    
    docker-compose up -d
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to start Docker services" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "  Docker services started successfully" -ForegroundColor Green
    
    # Wait for services to be ready
    Write-Host "`n  Waiting for services to be ready..." -ForegroundColor Gray
    Start-Sleep -Seconds 30
    
    # Check service health
    Write-Host "`n  Checking service health:" -ForegroundColor Gray
    
    $services = @(
        @{Name="Kafka"; Port=9092},
        @{Name="InfluxDB"; Port=8086},
        @{Name="TimescaleDB"; Port=5432},
        @{Name="Redis"; Port=6379},
        @{Name="Grafana"; Port=3000}
    )
    
    foreach ($service in $services) {
        $result = Test-NetConnection -ComputerName localhost -Port $service.Port -WarningAction SilentlyContinue
        if ($result.TcpTestSucceeded) {
            Write-Host "    ✓ $($service.Name) (port $($service.Port))" -ForegroundColor Green
        } else {
            Write-Host "    ✗ $($service.Name) (port $($service.Port)) - NOT READY" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "[1-2/6] Skipping Docker services" -ForegroundColor Gray
}

# Initialize Kafka topics
Write-Host "`n[3/6] Initializing Kafka topics..." -ForegroundColor Yellow

if (-not $SkipDocker) {
    python scripts/init_kafka_topics.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Kafka topics initialized" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: Failed to initialize Kafka topics" -ForegroundColor Yellow
    }
}

# Start data ingestion
if (-not $SkipIngestion) {
    Write-Host "`n[4/6] Starting data ingestion service..." -ForegroundColor Yellow
    
    $ingestionJob = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        python data_ingestion/main.py
    }
    
    Write-Host "  Data ingestion service started (Job ID: $($ingestionJob.Id))" -ForegroundColor Green
} else {
    Write-Host "`n[4/6] Skipping data ingestion" -ForegroundColor Gray
}

# Start stream processing
if (-not $SkipProcessing) {
    Write-Host "`n[5/6] Starting stream processing jobs..." -ForegroundColor Yellow
    
    # Submit Spark jobs
    Write-Host "  Submitting feature extraction job..." -ForegroundColor Gray
    $featureJob = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        spark-submit `
            --master spark://localhost:7077 `
            --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 `
            stream_processing/spark_jobs/feature_extraction.py
    }
    
    Start-Sleep -Seconds 5
    
    Write-Host "  Submitting Greeks calculator job..." -ForegroundColor Gray
    $greeksJob = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        spark-submit `
            --master spark://localhost:7077 `
            --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 `
            stream_processing/spark_jobs/greeks_calculator.py
    }
    
    Write-Host "  Stream processing jobs started" -ForegroundColor Green
    Write-Host "    Feature extraction (Job ID: $($featureJob.Id))" -ForegroundColor Gray
    Write-Host "    Greeks calculator (Job ID: $($greeksJob.Id))" -ForegroundColor Gray
} else {
    Write-Host "`n[5/6] Skipping stream processing" -ForegroundColor Gray
}

# Start prediction service
if (-not $SkipModels) {
    Write-Host "`n[6/6] Starting ML prediction service..." -ForegroundColor Yellow
    
    $predictionJob = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        python models/realtime_prediction.py
    }
    
    Write-Host "  Prediction service started (Job ID: $($predictionJob.Id))" -ForegroundColor Green
} else {
    Write-Host "`n[6/6] Skipping ML models" -ForegroundColor Gray
}

# Start dashboard
if (-not $SkipDashboard) {
    Write-Host "`n[OPTIONAL] Starting Dash visualization dashboard..." -ForegroundColor Yellow
    
    $dashJob = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        python visualization/dash_app.py
    }
    
    Write-Host "  Dash dashboard started (Job ID: $($dashJob.Id))" -ForegroundColor Green
    Write-Host "  Access at: http://localhost:8050" -ForegroundColor Cyan
} else {
    Write-Host "`n[OPTIONAL] Skipping Dash dashboard" -ForegroundColor Gray
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "System Startup Complete!" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Service URLs:" -ForegroundColor Yellow
Write-Host "  Grafana:        http://localhost:3000 (admin/admin)" -ForegroundColor White
Write-Host "  Dash Dashboard: http://localhost:8050" -ForegroundColor White
Write-Host "  Jupyter:        http://localhost:8888" -ForegroundColor White
Write-Host "  Spark UI:       http://localhost:8080" -ForegroundColor White
Write-Host "  InfluxDB:       http://localhost:8086" -ForegroundColor White

Write-Host "`nRunning Jobs:" -ForegroundColor Yellow
Get-Job | Format-Table Id, Name, State -AutoSize

Write-Host "`nUseful Commands:" -ForegroundColor Yellow
Write-Host "  Check job output: Receive-Job -Id <JobId> -Keep" -ForegroundColor Gray
Write-Host "  Stop all jobs:    Get-Job | Stop-Job; Get-Job | Remove-Job" -ForegroundColor Gray
Write-Host "  Stop Docker:      docker-compose down" -ForegroundColor Gray
Write-Host "  View logs:        docker-compose logs -f <service-name>" -ForegroundColor Gray

Write-Host "`nPress Ctrl+C to stop all services`n" -ForegroundColor Yellow

# Keep script running
try {
    while ($true) {
        Start-Sleep -Seconds 10
        
        # Check if any jobs failed
        $failedJobs = Get-Job | Where-Object { $_.State -eq "Failed" }
        if ($failedJobs) {
            Write-Host "`nWARNING: Some jobs have failed:" -ForegroundColor Yellow
            $failedJobs | Format-Table Id, Name, State -AutoSize
        }
    }
} finally {
    Write-Host "`nShutting down..." -ForegroundColor Yellow
    Get-Job | Stop-Job
    Get-Job | Remove-Job
    Write-Host "Cleanup complete" -ForegroundColor Green
}
