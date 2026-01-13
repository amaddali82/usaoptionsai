"""
TimescaleDB Client for Relational Time Series Data
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from contextlib import contextmanager


class TimescaleDBManager:
    """Manager for TimescaleDB operations"""
    
    def __init__(self, host: str, port: int, database: str,
                 user: str, password: str):
        """
        Initialize TimescaleDB client
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Username
            password: Password
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.logger = logging.getLogger(__name__)
        
        # Test connection
        self._test_connection()
        
        # Initialize schema
        self._init_schema()
        
        self.logger.info(f"Connected to TimescaleDB at {host}:{port}/{database}")
    
    def _test_connection(self):
        """Test database connection"""
        try:
            conn = self._get_connection()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise
    
    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )
    
    @contextmanager
    def _get_cursor(self, dict_cursor: bool = False):
        """Context manager for database cursor"""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor if dict_cursor else None)
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def _init_schema(self):
        """Initialize database schema and hypertables"""
        
        with self._get_cursor() as cursor:
            # Create extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            
            # Stock prices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_prices (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(10) NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume BIGINT,
                    PRIMARY KEY (time, symbol)
                );
            """)
            
            # Convert to hypertable
            cursor.execute("""
                SELECT create_hypertable('stock_prices', 'time', 
                    if_not_exists => TRUE);
            """)
            
            # Options quotes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS option_quotes (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(10) NOT NULL,
                    options_ticker VARCHAR(50) NOT NULL,
                    strike DOUBLE PRECISION NOT NULL,
                    expiration DATE NOT NULL,
                    option_type VARCHAR(4) NOT NULL,
                    bid DOUBLE PRECISION,
                    ask DOUBLE PRECISION,
                    last DOUBLE PRECISION,
                    volume BIGINT,
                    open_interest BIGINT,
                    implied_volatility DOUBLE PRECISION,
                    PRIMARY KEY (time, options_ticker)
                );
            """)
            
            cursor.execute("""
                SELECT create_hypertable('option_quotes', 'time',
                    if_not_exists => TRUE);
            """)
            
            # Option Greeks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS option_greeks (
                    time TIMESTAMPTZ NOT NULL,
                    options_ticker VARCHAR(50) NOT NULL,
                    delta DOUBLE PRECISION,
                    gamma DOUBLE PRECISION,
                    theta DOUBLE PRECISION,
                    vega DOUBLE PRECISION,
                    rho DOUBLE PRECISION,
                    PRIMARY KEY (time, options_ticker)
                );
            """)
            
            cursor.execute("""
                SELECT create_hypertable('option_greeks', 'time',
                    if_not_exists => TRUE);
            """)
            
            # Technical indicators table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(10) NOT NULL,
                    sma_5 DOUBLE PRECISION,
                    sma_10 DOUBLE PRECISION,
                    sma_20 DOUBLE PRECISION,
                    sma_50 DOUBLE PRECISION,
                    rsi DOUBLE PRECISION,
                    macd DOUBLE PRECISION,
                    macd_signal DOUBLE PRECISION,
                    bb_upper DOUBLE PRECISION,
                    bb_lower DOUBLE PRECISION,
                    volatility DOUBLE PRECISION,
                    PRIMARY KEY (time, symbol)
                );
            """)
            
            cursor.execute("""
                SELECT create_hypertable('technical_indicators', 'time',
                    if_not_exists => TRUE);
            """)
            
            # Model predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(10) NOT NULL,
                    model_name VARCHAR(50) NOT NULL,
                    horizon VARCHAR(20) NOT NULL,
                    target_time TIMESTAMPTZ NOT NULL,
                    predicted_price DOUBLE PRECISION,
                    confidence DOUBLE PRECISION,
                    lower_bound DOUBLE PRECISION,
                    upper_bound DOUBLE PRECISION,
                    PRIMARY KEY (prediction_time, symbol, model_name, horizon)
                );
            """)
            
            cursor.execute("""
                SELECT create_hypertable('predictions', 'prediction_time',
                    if_not_exists => TRUE);
            """)
            
            # Trading signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(10) NOT NULL,
                    signal_type VARCHAR(10) NOT NULL,
                    strategy VARCHAR(50) NOT NULL,
                    target_price DOUBLE PRECISION,
                    stop_loss DOUBLE PRECISION,
                    confidence DOUBLE PRECISION,
                    metadata JSONB,
                    PRIMARY KEY (time, symbol, strategy)
                );
            """)
            
            cursor.execute("""
                SELECT create_hypertable('trading_signals', 'time',
                    if_not_exists => TRUE);
            """)
            
            # Create indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol ON stock_prices(symbol, time DESC);",
                "CREATE INDEX IF NOT EXISTS idx_option_quotes_symbol ON option_quotes(symbol, time DESC);",
                "CREATE INDEX IF NOT EXISTS idx_option_quotes_expiration ON option_quotes(expiration, time DESC);",
                "CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol, prediction_time DESC);",
                "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON trading_signals(symbol, time DESC);"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
        
        self.logger.info("Database schema initialized")
    
    def insert_price_data(self, data: List[Tuple]):
        """
        Bulk insert stock price data
        
        Args:
            data: List of tuples (time, symbol, open, high, low, close, volume)
        """
        with self._get_cursor() as cursor:
            execute_values(
                cursor,
                """
                INSERT INTO stock_prices (time, symbol, open, high, low, close, volume)
                VALUES %s
                ON CONFLICT (time, symbol) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume;
                """,
                data
            )
        self.logger.debug(f"Inserted {len(data)} price records")
    
    def insert_option_data(self, data: List[Tuple]):
        """
        Bulk insert option quote data
        
        Args:
            data: List of tuples (time, symbol, options_ticker, strike, expiration,
                                  option_type, bid, ask, last, volume, open_interest, iv)
        """
        with self._get_cursor() as cursor:
            execute_values(
                cursor,
                """
                INSERT INTO option_quotes (time, symbol, options_ticker, strike, expiration,
                                          option_type, bid, ask, last, volume, open_interest,
                                          implied_volatility)
                VALUES %s
                ON CONFLICT (time, options_ticker) DO UPDATE SET
                    bid = EXCLUDED.bid,
                    ask = EXCLUDED.ask,
                    last = EXCLUDED.last,
                    volume = EXCLUDED.volume,
                    open_interest = EXCLUDED.open_interest,
                    implied_volatility = EXCLUDED.implied_volatility;
                """,
                data
            )
        self.logger.debug(f"Inserted {len(data)} option records")
    
    def query_price_history(self, symbol: str, start_time: datetime,
                           end_time: datetime) -> List[Dict]:
        """Query historical stock prices"""
        with self._get_cursor(dict_cursor=True) as cursor:
            cursor.execute("""
                SELECT time, symbol, open, high, low, close, volume
                FROM stock_prices
                WHERE symbol = %s AND time >= %s AND time <= %s
                ORDER BY time ASC;
            """, (symbol, start_time, end_time))
            
            return cursor.fetchall()
    
    def query_latest_price(self, symbol: str) -> Optional[Dict]:
        """Get latest price for symbol"""
        with self._get_cursor(dict_cursor=True) as cursor:
            cursor.execute("""
                SELECT time, symbol, close, volume
                FROM stock_prices
                WHERE symbol = %s
                ORDER BY time DESC
                LIMIT 1;
            """, (symbol,))
            
            return cursor.fetchone()
    
    def query_options_chain(self, symbol: str, min_expiration: datetime) -> List[Dict]:
        """Get options chain for symbol"""
        with self._get_cursor(dict_cursor=True) as cursor:
            cursor.execute("""
                SELECT DISTINCT ON (options_ticker) *
                FROM option_quotes
                WHERE symbol = %s AND expiration >= %s
                ORDER BY options_ticker, time DESC;
            """, (symbol, min_expiration))
            
            return cursor.fetchall()
    
    def insert_prediction(self, prediction_time: datetime, symbol: str,
                         model_name: str, horizon: str, target_time: datetime,
                         predicted_price: float, confidence: float,
                         lower_bound: Optional[float] = None,
                         upper_bound: Optional[float] = None):
        """Insert model prediction"""
        with self._get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO predictions (prediction_time, symbol, model_name, horizon,
                                        target_time, predicted_price, confidence,
                                        lower_bound, upper_bound)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (prediction_time, symbol, model_name, horizon) DO UPDATE SET
                    predicted_price = EXCLUDED.predicted_price,
                    confidence = EXCLUDED.confidence,
                    lower_bound = EXCLUDED.lower_bound,
                    upper_bound = EXCLUDED.upper_bound;
            """, (prediction_time, symbol, model_name, horizon, target_time,
                  predicted_price, confidence, lower_bound, upper_bound))
    
    def close(self):
        """Close connection pool"""
        self.logger.info("TimescaleDB connection closed")
