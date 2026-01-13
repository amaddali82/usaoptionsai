# Getting Started with USA Options AI

This guide will help you get the system up and running quickly.

## Prerequisites

1. **Docker & Docker Compose** - [Install Docker](https://docs.docker.com/get-docker/)
2. **Python 3.9+** - [Install Python](https://www.python.org/downloads/)
3. **Git** - [Install Git](https://git-scm.com/downloads)

## Quick Start (5 minutes)

### Step 1: Clone and Setup

```bash
cd c:\usaoptionsai
```

### Step 2: Configure API Keys

Copy the example configuration and add your API keys:

```bash
copy config\api_config.example.yaml config\api_config.yaml
notepad config\api_config.yaml
```

**Free API Keys** (Get these first):
- **Yahoo Finance**: No key needed (via yfinance)
- **Alpha Vantage**: Get free key at https://www.alphavantage.co/support/#api-key
- **Polygon.io**: Get free tier at https://polygon.io/

### Step 3: Install Python Dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 4: Start Infrastructure Services

```bash
docker-compose up -d
```

This starts:
- Kafka (message streaming)
- InfluxDB (time-series database)
- TimescaleDB (relational database)
- Redis (caching)
- Grafana (visualization)
- Spark (stream processing)
- Jupyter (notebooks)

**Verify services are running:**
```bash
docker-compose ps
```

### Step 5: Initialize Kafka Topics

```bash
python scripts\init_kafka_topics.py
```

### Step 6: Test Data Ingestion

```bash
python -m data_ingestion.main --symbols AAPL MSFT TSLA --test
```

### Step 7: Open Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Jupyter**: http://localhost:8888
- **Spark UI**: http://localhost:8080
- **Prometheus**: http://localhost:9090

## What's Next?

### Train Your First Model

```bash
# Train a simple LSTM model for short-term predictions
python -m models.train --model lstm --symbols AAPL --timeframe short
```

### Start Real-Time Ingestion

```bash
# Stream live market data
python -m data_ingestion.main --symbols AAPL MSFT GOOGL TSLA --continuous
```

### Run Backtests

```bash
# Backtest a strategy
python -m backtesting.run --strategy momentum --symbols SPY --start 2023-01-01 --end 2024-01-01
```

## Architecture Overview

```
┌─────────────┐
│   APIs      │ → Polygon.io, Alpha Vantage, Yahoo Finance
└──────┬──────┘
       ↓
┌─────────────┐
│   Kafka     │ → Real-time event streaming
└──────┬──────┘
       ↓
┌─────────────┐
│   Spark     │ → Stream processing & feature engineering
└──────┬──────┘
       ↓
┌─────────────┐
│ Databases   │ → InfluxDB (time-series), TimescaleDB (relational)
└──────┬──────┘
       ↓
┌─────────────┐
│ ML Models   │ → LSTM, CNN, Transformer, Black-Scholes
└──────┬──────┘
       ↓
┌─────────────┐
│ Signals     │ → Buy/sell/hold recommendations
└──────┬──────┘
       ↓
┌─────────────┐
│  Grafana    │ → Real-time dashboards
└─────────────┘
```

## Common Use Cases

### 1. Monitor High-Volatility Stocks

```python
from data_ingestion.api_clients import YahooFinanceClient
from models.quantitative import BlackScholesModel

# Get options chain
client = YahooFinanceClient()
options = client.get_options_chain('TSLA')

# Calculate IV for each contract
bs = BlackScholesModel()
for index, row in options['calls'].iterrows():
    iv = bs.implied_volatility(
        market_price=row['lastPrice'],
        S=row['underlyingPrice'],
        K=row['strike'],
        T=row['daysToExpiration']/365,
        r=0.05
    )
    print(f"Strike {row['strike']}: IV = {iv:.2%}")
```

### 2. Predict Short-Term Price Movement

```python
from models.short_term import LSTMPredictor

# Load trained model
predictor = LSTMPredictor.load('models/saved_models/lstm_short_term.h5')

# Make prediction
prediction = predictor.predict('AAPL', horizon='1d')
print(f"Predicted move: {prediction['direction']} ({prediction['confidence']:.1%} confidence)")
```

### 3. Generate Trading Signals

```python
from recommendation_engine import SignalGenerator

generator = SignalGenerator()
signals = generator.generate_signals(['AAPL', 'MSFT', 'GOOGL'])

for signal in signals:
    print(f"{signal['symbol']}: {signal['action']} @ {signal['target_price']}")
```

## Troubleshooting

### Services won't start

```bash
# Check logs
docker-compose logs kafka
docker-compose logs influxdb

# Restart specific service
docker-compose restart kafka
```

### API rate limits

- Free tier Alpha Vantage: 5 requests/minute
- Increase `sleep` between requests in config
- Use caching to reduce API calls

### Out of memory

- Reduce `num_simulations` in Monte Carlo config
- Reduce `batch_size` in ML model config
- Allocate more memory to Docker

## Learning Resources

- **Options Trading Basics**: [Investopedia Options](https://www.investopedia.com/options-basics-tutorial-4583012)
- **Black-Scholes Model**: [Options Pricing](https://www.investopedia.com/terms/b/blackscholes.asp)
- **Machine Learning for Finance**: [ML Finance Books](https://www.amazon.com/s?k=machine+learning+finance)
- **Apache Kafka**: [Kafka Documentation](https://kafka.apache.org/documentation/)
- **Apache Spark**: [Spark Documentation](https://spark.apache.org/docs/latest/)

## Support

- **Documentation**: See `docs/` folder
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

## Next Steps

1. **Explore Jupyter Notebooks**: Open http://localhost:8888 and explore example notebooks
2. **Customize Models**: Edit `config/model_config.yaml` to tune hyperparameters
3. **Add New Strategies**: Create custom strategies in `recommendation_engine/strategies/`
4. **Scale Up**: Deploy to Kubernetes for production (see `kubernetes/`)

---

**Happy Trading! 📈**

⚠️ **Disclaimer**: This is for educational purposes only. Options trading involves substantial risk. Always do your own research and consult with financial advisors.
