"""
Kafka Producer for Options Data
Streams options chains, quotes, and Greeks to Kafka topics
"""
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError


class OptionsDataProducer:
    """Producer for options market data"""
    
    def __init__(self, bootstrap_servers: List[str], 
                 topic_prefix: str = "raw.options"):
        """
        Initialize options data producer
        
        Args:
            bootstrap_servers: List of Kafka broker addresses
            topic_prefix: Prefix for Kafka topics
        """
        self.logger = logging.getLogger(__name__)
        self.topic_prefix = topic_prefix
        
        # Create Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',
            retries=3,
            compression_type='gzip',
            max_in_flight_requests_per_connection=5,
            enable_idempotence=True
        )
        
        self.logger.info("Options data producer initialized")
    
    def send_options_chain(self, symbol: str, data: Dict[str, Any], 
                          topic: str = "chains"):
        """
        Send options chain data to Kafka
        
        Args:
            symbol: Stock ticker
            data: Options chain data
            topic: Kafka topic suffix
        """
        full_topic = f"{self.topic_prefix}.{topic}"
        
        # Add metadata
        message = {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'data': data
        }
        
        try:
            future = self.producer.send(
                full_topic,
                key=symbol,
                value=message
            )
            
            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            
            self.logger.debug(
                f"Sent options chain for {symbol} to {record_metadata.topic} "
                f"partition {record_metadata.partition} offset {record_metadata.offset}"
            )
            
        except KafkaError as e:
            self.logger.error(f"Failed to send options chain for {symbol}: {e}")
            raise
    
    def send_option_quote(self, options_ticker: str, quote_data: Dict[str, Any],
                         topic: str = "quotes"):
        """
        Send individual option quote
        
        Args:
            options_ticker: Options contract ticker
            quote_data: Quote data (bid, ask, last, volume, etc.)
            topic: Kafka topic suffix
        """
        full_topic = f"{self.topic_prefix}.{topic}"
        
        message = {
            'options_ticker': options_ticker,
            'timestamp': datetime.utcnow().isoformat(),
            'data': quote_data
        }
        
        try:
            self.producer.send(
                full_topic,
                key=options_ticker,
                value=message
            )
            
        except KafkaError as e:
            self.logger.error(f"Failed to send quote for {options_ticker}: {e}")
            raise
    
    def send_greeks(self, options_ticker: str, greeks: Dict[str, float],
                   topic: str = "greeks"):
        """
        Send option Greeks
        
        Args:
            options_ticker: Options contract ticker
            greeks: Greeks data (delta, gamma, theta, vega, rho)
            topic: Kafka topic suffix
        """
        full_topic = f"{self.topic_prefix}.{topic}"
        
        message = {
            'options_ticker': options_ticker,
            'timestamp': datetime.utcnow().isoformat(),
            'greeks': greeks
        }
        
        try:
            self.producer.send(
                full_topic,
                key=options_ticker,
                value=message
            )
            
        except KafkaError as e:
            self.logger.error(f"Failed to send Greeks for {options_ticker}: {e}")
            raise
    
    def send_batch(self, messages: List[Dict[str, Any]], topic: str = "batch"):
        """
        Send batch of options data
        
        Args:
            messages: List of option data messages
            topic: Kafka topic suffix
        """
        full_topic = f"{self.topic_prefix}.{topic}"
        
        for msg in messages:
            try:
                self.producer.send(
                    full_topic,
                    key=msg.get('symbol') or msg.get('options_ticker'),
                    value=msg
                )
            except KafkaError as e:
                self.logger.error(f"Failed to send batch message: {e}")
        
        self.producer.flush()
        self.logger.info(f"Sent batch of {len(messages)} messages")
    
    def close(self):
        """Close producer and flush pending messages"""
        self.logger.info("Closing options data producer")
        self.producer.flush()
        self.producer.close()


class StockPriceProducer:
    """Producer for stock price data"""
    
    def __init__(self, bootstrap_servers: List[str], 
                 topic_prefix: str = "raw.stock"):
        """
        Initialize stock price producer
        
        Args:
            bootstrap_servers: List of Kafka broker addresses
            topic_prefix: Prefix for Kafka topics
        """
        self.logger = logging.getLogger(__name__)
        self.topic_prefix = topic_prefix
        
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',
            retries=3,
            compression_type='gzip'
        )
        
        self.logger.info("Stock price producer initialized")
    
    def send_price_data(self, symbol: str, price_data: Dict[str, Any],
                       topic: str = "prices"):
        """
        Send stock price data
        
        Args:
            symbol: Stock ticker
            price_data: OHLCV or tick data
            topic: Kafka topic suffix
        """
        full_topic = f"{self.topic_prefix}.{topic}"
        
        message = {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'data': price_data
        }
        
        try:
            self.producer.send(
                full_topic,
                key=symbol,
                value=message
            )
            
        except KafkaError as e:
            self.logger.error(f"Failed to send price data for {symbol}: {e}")
            raise
    
    def send_tick(self, symbol: str, tick: Dict[str, Any]):
        """Send real-time tick data"""
        self.send_price_data(symbol, tick, topic="ticks")
    
    def send_ohlcv(self, symbol: str, bar: Dict[str, Any]):
        """Send OHLCV bar data"""
        self.send_price_data(symbol, bar, topic="bars")
    
    def close(self):
        """Close producer"""
        self.logger.info("Closing stock price producer")
        self.producer.flush()
        self.producer.close()


class NewsProducer:
    """Producer for news and sentiment data"""
    
    def __init__(self, bootstrap_servers: List[str],
                 topic_prefix: str = "raw.news"):
        """
        Initialize news producer
        
        Args:
            bootstrap_servers: List of Kafka broker addresses
            topic_prefix: Prefix for Kafka topics
        """
        self.logger = logging.getLogger(__name__)
        self.topic_prefix = topic_prefix
        
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',
            retries=3,
            compression_type='gzip'
        )
        
        self.logger.info("News producer initialized")
    
    def send_news_article(self, article: Dict[str, Any],
                         topic: str = "articles"):
        """
        Send news article with sentiment
        
        Args:
            article: Article data with headline, content, sentiment
            topic: Kafka topic suffix
        """
        full_topic = f"{self.topic_prefix}.{topic}"
        
        message = {
            'timestamp': datetime.utcnow().isoformat(),
            'article': article
        }
        
        # Use article ID or headline as key
        key = article.get('id') or article.get('headline', '')[:50]
        
        try:
            self.producer.send(
                full_topic,
                key=key,
                value=message
            )
            
        except KafkaError as e:
            self.logger.error(f"Failed to send news article: {e}")
            raise
    
    def send_sentiment_score(self, symbol: str, sentiment: Dict[str, Any],
                            topic: str = "sentiment"):
        """
        Send aggregated sentiment score for a symbol
        
        Args:
            symbol: Stock ticker
            sentiment: Sentiment scores and metadata
            topic: Kafka topic suffix
        """
        full_topic = f"{self.topic_prefix}.{topic}"
        
        message = {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'sentiment': sentiment
        }
        
        try:
            self.producer.send(
                full_topic,
                key=symbol,
                value=message
            )
            
        except KafkaError as e:
            self.logger.error(f"Failed to send sentiment for {symbol}: {e}")
            raise
    
    def close(self):
        """Close producer"""
        self.logger.info("Closing news producer")
        self.producer.flush()
        self.producer.close()
