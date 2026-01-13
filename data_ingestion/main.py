"""
Main Data Ingestion Orchestrator
Coordinates data collection from multiple sources and streams to Kafka
"""
import logging
import yaml
import time
from typing import List, Dict, Any
from datetime import datetime, timedelta
import schedule

from data_ingestion.api_clients import PolygonClient, AlphaVantageClient, YahooFinanceClient
from data_ingestion.kafka_producers import OptionsDataProducer, StockPriceProducer, NewsProducer
from data_ingestion.data_validators import DataValidator


class DataIngestionOrchestrator:
    """Orchestrates data ingestion from multiple sources"""
    
    def __init__(self, config_path: str = "config/"):
        """
        Initialize orchestrator
        
        Args:
            config_path: Path to configuration directory
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        
        # Load configurations
        self.api_config = self._load_config("api_config.yaml")
        self.kafka_config = self._load_config("kafka_config.yaml")
        
        # Initialize API clients
        self.polygon_client = None
        self.alpha_vantage_client = None
        self.yahoo_client = YahooFinanceClient()
        
        self._init_api_clients()
        
        # Initialize Kafka producers
        self._init_kafka_producers()
        
        # Initialize validator
        self.validator = DataValidator()
        
        self.logger.info("Data ingestion orchestrator initialized")
    
    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        config_file = f"{self.config_path}/{filename}"
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded configuration from {config_file}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load {config_file}: {e}")
            raise
    
    def _init_api_clients(self):
        """Initialize API clients based on configuration"""
        # Polygon.io
        if 'polygon' in self.api_config and self.api_config['polygon'].get('api_key'):
            self.polygon_client = PolygonClient(
                api_key=self.api_config['polygon']['api_key'],
                rate_limit=self.api_config['polygon'].get('rate_limit', 5)
            )
            self.logger.info("Initialized Polygon.io client")
        
        # Alpha Vantage
        if 'alpha_vantage' in self.api_config and self.api_config['alpha_vantage'].get('api_key'):
            self.alpha_vantage_client = AlphaVantageClient(
                api_key=self.api_config['alpha_vantage']['api_key'],
                rate_limit=self.api_config['alpha_vantage'].get('rate_limit', 5)
            )
            self.logger.info("Initialized Alpha Vantage client")
    
    def _init_kafka_producers(self):
        """Initialize Kafka producers"""
        brokers = self.kafka_config['brokers']
        
        self.options_producer = OptionsDataProducer(
            bootstrap_servers=brokers,
            topic_prefix=self.kafka_config['topics']['raw_options_chains'].rsplit('.', 1)[0]
        )
        
        self.price_producer = StockPriceProducer(
            bootstrap_servers=brokers,
            topic_prefix=self.kafka_config['topics']['raw_stock_prices'].rsplit('.', 1)[0]
        )
        
        self.news_producer = NewsProducer(
            bootstrap_servers=brokers,
            topic_prefix=self.kafka_config['topics']['raw_news_sentiment'].rsplit('.', 1)[0]
        )
        
        self.logger.info("Initialized Kafka producers")
    
    def fetch_and_stream_options(self, symbols: List[str]):
        """
        Fetch options data for symbols and stream to Kafka
        
        Args:
            symbols: List of stock tickers
        """
        self.logger.info(f"Fetching options data for {len(symbols)} symbols")
        
        for symbol in symbols:
            try:
                # Try Polygon first
                if self.polygon_client:
                    data = self.polygon_client.get_options_chain(symbol)
                elif self.alpha_vantage_client:
                    data = self.alpha_vantage_client.get_option_chain(symbol)
                else:
                    # Fall back to Yahoo Finance
                    data = self.yahoo_client.get_options_chain(symbol)
                
                # Validate data
                is_valid, errors = self.validator.validate_options_chain({
                    'symbol': symbol,
                    'data': data
                })
                
                if is_valid:
                    # Stream to Kafka
                    self.options_producer.send_options_chain(symbol, data)
                    self.logger.info(f"Streamed options chain for {symbol}")
                else:
                    self.logger.warning(f"Validation failed for {symbol}: {errors}")
                
            except Exception as e:
                self.logger.error(f"Failed to fetch options for {symbol}: {e}")
    
    def fetch_and_stream_prices(self, symbols: List[str], interval: str = "1min"):
        """
        Fetch stock prices and stream to Kafka
        
        Args:
            symbols: List of stock tickers
            interval: Time interval (1min, 5min, etc.)
        """
        self.logger.info(f"Fetching price data for {len(symbols)} symbols")
        
        for symbol in symbols:
            try:
                # Use Yahoo Finance for reliability
                data = self.yahoo_client.get_historical_prices(
                    symbol,
                    period="1d",
                    interval=interval
                )
                
                if not data.empty:
                    # Get latest bar
                    latest = data.iloc[-1].to_dict()
                    
                    # Validate
                    is_valid, errors = self.validator.validate_stock_price(latest)
                    
                    if is_valid:
                        self.price_producer.send_ohlcv(symbol, latest)
                        self.logger.info(f"Streamed price data for {symbol}")
                    else:
                        self.logger.warning(f"Validation failed for {symbol}: {errors}")
                
            except Exception as e:
                self.logger.error(f"Failed to fetch prices for {symbol}: {e}")
    
    def fetch_and_stream_news(self, symbols: List[str]):
        """
        Fetch news and sentiment data
        
        Args:
            symbols: List of stock tickers
        """
        self.logger.info(f"Fetching news for {len(symbols)} symbols")
        
        if self.alpha_vantage_client:
            try:
                # Get news for all symbols
                tickers = ','.join(symbols)
                news_data = self.alpha_vantage_client.get_news_sentiment(
                    tickers=tickers,
                    limit=50
                )
                
                if 'feed' in news_data:
                    for article in news_data['feed']:
                        self.news_producer.send_news_article(article)
                    
                    self.logger.info(f"Streamed {len(news_data['feed'])} news articles")
            
            except Exception as e:
                self.logger.error(f"Failed to fetch news: {e}")
    
    def run_continuous(self, symbols: List[str], 
                      interval_seconds: int = 60):
        """
        Run continuous data ingestion
        
        Args:
            symbols: List of stock tickers to monitor
            interval_seconds: Update interval in seconds
        """
        self.logger.info(f"Starting continuous ingestion for {len(symbols)} symbols")
        
        def job():
            self.fetch_and_stream_prices(symbols, interval="1min")
            self.fetch_and_stream_options(symbols)
        
        # Schedule jobs
        schedule.every(interval_seconds).seconds.do(job)
        
        # News updates less frequently
        schedule.every(15).minutes.do(
            lambda: self.fetch_and_stream_news(symbols)
        )
        
        # Run indefinitely
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Stopping continuous ingestion")
            self.shutdown()
    
    def shutdown(self):
        """Gracefully shutdown all producers"""
        self.logger.info("Shutting down data ingestion orchestrator")
        
        if self.options_producer:
            self.options_producer.close()
        
        if self.price_producer:
            self.price_producer.close()
        
        if self.news_producer:
            self.news_producer.close()
        
        self.logger.info("Shutdown complete")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    orchestrator = DataIngestionOrchestrator(config_path="./config")
    
    # Define symbols to monitor
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'NVDA', 'META', 'SPY', 'QQQ', 'IWM'
    ]
    
    # Run continuous ingestion
    orchestrator.run_continuous(symbols, interval_seconds=60)
