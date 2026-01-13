"""
Data Ingestion Package
Handles real-time and historical data acquisition from multiple APIs
"""

from .api_clients import PolygonClient, AlphaVantageClient, YahooFinanceClient
from .kafka_producers import OptionsDataProducer, StockPriceProducer, NewsProducer
from .data_validators import DataValidator

__all__ = [
    'PolygonClient',
    'AlphaVantageClient',
    'YahooFinanceClient',
    'OptionsDataProducer',
    'StockPriceProducer',
    'NewsProducer',
    'DataValidator'
]
