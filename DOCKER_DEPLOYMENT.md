# USA Options AI - Docker Deployment Guide

## 🚀 Quick Start

### Start the Dashboard
```bash
docker-compose up -d dashboard
```

### View Dashboard
Open browser to: **http://localhost:8050**

### View Logs
```bash
docker-compose logs -f dashboard
```

### Stop Dashboard
```bash
docker-compose stop dashboard
```

### Restart Dashboard
```bash
docker-compose restart dashboard
```

### Rebuild and Restart
```bash
docker-compose build dashboard
docker-compose up -d dashboard
```

## 📦 What's Deployed

- **Container Name**: `usaoptionsai-dashboard`
- **Port**: 8050
- **Image**: Built from local Dockerfile
- **Volumes Mounted**:
  - `./data` → `/app/data` (Stock data)
  - `./saved_models` → `/app/saved_models` (ML models)
  - `./visualization` → `/app/visualization` (Dashboard code)
  - `./config` → `/app/config` (Configuration)

## 🎯 Dashboard Features

### Home Page (Default)
- **Options Trading Strategies Table** for ALL 83 stocks
- 16 columns with complete options analysis
- 3 expiry dates per stock (7/14/30 days)
- Sortable and filterable table
- Color-coded risk levels and option types

### Stock Details (Select from Dropdown)
- Price history charts with ML predictions
- Technical indicators (RSI, Moving Averages)
- Prediction accuracy metrics (MAE, MAPE, RMSE)
- 3 detailed options strategies per stock
- Real-time auto-refresh (60 seconds)

## 📊 Table Columns

1. **Symbol** - Stock ticker
2. **Name** - Company name
3. **Option Type** - CALL (bullish) or PUT (bearish)
4. **Expiry Date** - Option expiration
5. **Stock Price** - Current market price
6. **Strike Price** - 2% OTM strike
7-9. **Target 1/2/3** - Conservative/Moderate/Aggressive targets
10-12. **Confidence 1/2/3** - ML confidence levels
13. **Capital Required** - Estimated premium per contract
14. **Risk Level** - LOW/MEDIUM/HIGH
15. **Risk/Reward** - Profit potential ratio
16. **Predicted Move** - Expected price change %

## 🔧 Maintenance Commands

### Check Container Status
```bash
docker ps --filter name=usaoptionsai-dashboard
```

### View Container Details
```bash
docker inspect usaoptionsai-dashboard
```

### Access Container Shell
```bash
docker exec -it usaoptionsai-dashboard bash
```

### Remove Container and Image
```bash
docker-compose down dashboard
docker rmi usaoptionsai-dashboard
```

### Full Clean Rebuild
```bash
docker-compose down dashboard
docker system prune -f
docker-compose build --no-cache dashboard
docker-compose up -d dashboard
```

## 📈 System Requirements

- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
- **8GB RAM** minimum (16GB recommended)
- **10GB Disk Space** for models and data
- **Python 3.11** base image

## 🔐 Data Persistence

All data persists outside the container:
- Models: `./saved_models/` (83 trained models)
- Data: `./data/` (stock historical data)
- Logs: Container logs via `docker-compose logs`

## 🐛 Troubleshooting

### Dashboard Not Loading
```bash
# Check if container is running
docker ps | grep dashboard

# Check logs for errors
docker-compose logs dashboard

# Restart container
docker-compose restart dashboard
```

### Port Already in Use
```bash
# Stop local Python process
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# Or change port in docker-compose.yml
ports:
  - "8051:8050"  # Change 8051 to any available port
```

### Container Crashes
```bash
# View full logs
docker-compose logs --tail=100 dashboard

# Check container exit code
docker inspect usaoptionsai-dashboard --format='{{.State.ExitCode}}'
```

## 🚀 Production Deployment

For production, consider:

1. **Use Gunicorn** instead of Flask dev server
2. **Add Nginx** reverse proxy
3. **Enable HTTPS** with SSL certificates
4. **Set up monitoring** (Prometheus/Grafana)
5. **Configure backups** for models and data
6. **Use docker-compose.prod.yml** with production settings

## 📝 Git Commits

Latest commits:
```
060a5e8 - Add minimal requirements for dashboard Docker deployment
7b838a3 - Add Docker support for ML Options Trading Dashboard
6e41b60 - Options Trading Strategies Dashboard - Complete implementation
```

## ✅ Deployment Checklist

- [x] Code committed to git repository
- [x] Dockerfile created with minimal dependencies
- [x] docker-compose.yml updated with dashboard service
- [x] Docker image built successfully
- [x] Container running on port 8050
- [x] Dashboard accessible at http://localhost:8050
- [x] 83 ML models loaded
- [x] Options strategies table displaying all stocks
- [x] Auto-refresh working (60 seconds)

---

**Status**: ✅ DEPLOYED AND RUNNING

**Access**: http://localhost:8050
