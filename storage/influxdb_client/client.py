"""
InfluxDB Client for Time Series Data Storage
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS


class InfluxDBManager:
    """Manager for InfluxDB operations"""
    
    def __init__(self, url: str, token: str, org: str, bucket: str):
        """
        Initialize InfluxDB client
        
        Args:
            url: InfluxDB URL
            token: Authentication token
            org: Organization name
            bucket: Default bucket name
        """
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.logger = logging.getLogger(__name__)
        
        # Create client
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        
        self.logger.info(f"Connected to InfluxDB at {url}")
    
    def write_price_data(self, symbol: str, timestamp: datetime,
                        open_price: float, high: float, low: float,
                        close: float, volume: int, bucket: Optional[str] = None):
        """
        Write stock price data
        
        Args:
            symbol: Stock ticker
            timestamp: Data timestamp
            open_price, high, low, close: OHLC prices
            volume: Trading volume
            bucket: Target bucket (uses default if None)
        """
        bucket = bucket or self.bucket
        
        point = Point("stock_price") \
            .tag("symbol", symbol) \
            .field("open", open_price) \
            .field("high", high) \
            .field("low", low) \
            .field("close", close) \
            .field("volume", volume) \
            .time(timestamp, WritePrecision.NS)
        
        try:
            self.write_api.write(bucket=bucket, org=self.org, record=point)
            self.logger.debug(f"Wrote price data for {symbol}")
        except Exception as e:
            self.logger.error(f"Failed to write price data: {e}")
            raise
    
    def write_option_data(self, symbol: str, options_ticker: str,
                         timestamp: datetime, strike: float, expiration: str,
                         option_type: str, bid: float, ask: float,
                         last: float, volume: int, open_interest: int,
                         implied_volatility: Optional[float] = None,
                         bucket: Optional[str] = None):
        """Write options data"""
        bucket = bucket or self.bucket
        
        point = Point("option_quote") \
            .tag("symbol", symbol) \
            .tag("options_ticker", options_ticker) \
            .tag("option_type", option_type) \
            .tag("expiration", expiration) \
            .field("strike", strike) \
            .field("bid", bid) \
            .field("ask", ask) \
            .field("last", last) \
            .field("volume", volume) \
            .field("open_interest", open_interest)
        
        if implied_volatility is not None:
            point = point.field("implied_volatility", implied_volatility)
        
        point = point.time(timestamp, WritePrecision.NS)
        
        try:
            self.write_api.write(bucket=bucket, org=self.org, record=point)
        except Exception as e:
            self.logger.error(f"Failed to write option data: {e}")
            raise
    
    def write_greeks(self, options_ticker: str, timestamp: datetime,
                    delta: float, gamma: float, theta: float,
                    vega: float, rho: Optional[float] = None,
                    bucket: Optional[str] = None):
        """Write option Greeks"""
        bucket = bucket or self.bucket
        
        point = Point("option_greeks") \
            .tag("options_ticker", options_ticker) \
            .field("delta", delta) \
            .field("gamma", gamma) \
            .field("theta", theta) \
            .field("vega", vega)
        
        if rho is not None:
            point = point.field("rho", rho)
        
        point = point.time(timestamp, WritePrecision.NS)
        
        try:
            self.write_api.write(bucket=bucket, org=self.org, record=point)
        except Exception as e:
            self.logger.error(f"Failed to write Greeks: {e}")
            raise
    
    def write_technical_indicators(self, symbol: str, timestamp: datetime,
                                   indicators: Dict[str, float],
                                   bucket: Optional[str] = None):
        """
        Write technical indicators
        
        Args:
            symbol: Stock ticker
            timestamp: Data timestamp
            indicators: Dictionary of indicator name -> value
        """
        bucket = bucket or self.bucket
        
        point = Point("technical_indicators") \
            .tag("symbol", symbol)
        
        for name, value in indicators.items():
            if value is not None:
                point = point.field(name, float(value))
        
        point = point.time(timestamp, WritePrecision.NS)
        
        try:
            self.write_api.write(bucket=bucket, org=self.org, record=point)
        except Exception as e:
            self.logger.error(f"Failed to write indicators: {e}")
            raise
    
    def write_prediction(self, symbol: str, timestamp: datetime,
                        model_name: str, horizon: str,
                        predicted_price: float, confidence: float,
                        lower_bound: Optional[float] = None,
                        upper_bound: Optional[float] = None,
                        bucket: Optional[str] = None):
        """Write model prediction"""
        bucket = bucket or self.bucket
        
        point = Point("prediction") \
            .tag("symbol", symbol) \
            .tag("model", model_name) \
            .tag("horizon", horizon) \
            .field("predicted_price", predicted_price) \
            .field("confidence", confidence)
        
        if lower_bound is not None:
            point = point.field("lower_bound", lower_bound)
        if upper_bound is not None:
            point = point.field("upper_bound", upper_bound)
        
        point = point.time(timestamp, WritePrecision.NS)
        
        try:
            self.write_api.write(bucket=bucket, org=self.org, record=point)
        except Exception as e:
            self.logger.error(f"Failed to write prediction: {e}")
            raise
    
    def query_price_history(self, symbol: str, start_time: str,
                           stop_time: str, bucket: Optional[str] = None) -> List[Dict]:
        """
        Query historical price data
        
        Args:
            symbol: Stock ticker
            start_time: Start time (RFC3339 format)
            stop_time: Stop time (RFC3339 format)
            
        Returns:
            List of price records
        """
        bucket = bucket or self.bucket
        
        query = f'''
        from(bucket: "{bucket}")
            |> range(start: {start_time}, stop: {stop_time})
            |> filter(fn: (r) => r["_measurement"] == "stock_price")
            |> filter(fn: (r) => r["symbol"] == "{symbol}")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        try:
            result = self.query_api.query(query, org=self.org)
            records = []
            
            for table in result:
                for record in table.records:
                    records.append({
                        'time': record.get_time(),
                        'symbol': record.values.get('symbol'),
                        'open': record.values.get('open'),
                        'high': record.values.get('high'),
                        'low': record.values.get('low'),
                        'close': record.values.get('close'),
                        'volume': record.values.get('volume')
                    })
            
            return records
        
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            raise
    
    def query_latest_price(self, symbol: str, bucket: Optional[str] = None) -> Optional[Dict]:
        """Get latest price for a symbol"""
        bucket = bucket or self.bucket
        
        query = f'''
        from(bucket: "{bucket}")
            |> range(start: -1h)
            |> filter(fn: (r) => r["_measurement"] == "stock_price")
            |> filter(fn: (r) => r["symbol"] == "{symbol}")
            |> last()
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        try:
            result = self.query_api.query(query, org=self.org)
            
            for table in result:
                for record in table.records:
                    return {
                        'time': record.get_time(),
                        'symbol': record.values.get('symbol'),
                        'close': record.values.get('close'),
                        'volume': record.values.get('volume')
                    }
            
            return None
        
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return None
    
    def close(self):
        """Close client connection"""
        self.client.close()
        self.logger.info("InfluxDB connection closed")
