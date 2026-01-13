"""
Polygon.io API Client for options and stock data
Documentation: https://polygon.io/docs/options/getting-started
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from .base_client import BaseAPIClient


class PolygonClient(BaseAPIClient):
    """Client for Polygon.io options and stock market data"""
    
    def __init__(self, api_key: str, rate_limit: int = 5, timeout: int = 30):
        """
        Initialize Polygon.io client
        
        Args:
            api_key: Polygon.io API key
            rate_limit: Requests per second (depends on subscription tier)
            timeout: Request timeout in seconds
        """
        super().__init__(
            api_key=api_key,
            base_url="https://api.polygon.io",
            rate_limit=rate_limit,
            timeout=timeout
        )
        self.logger = logging.getLogger(__name__)
    
    def get_options_chain(self, underlying_symbol: str, 
                          expiration_date: Optional[str] = None,
                          strike_price: Optional[float] = None,
                          contract_type: Optional[str] = None,
                          limit: int = 250) -> Dict[str, Any]:
        """
        Get options chain for a stock
        
        Args:
            underlying_symbol: Stock ticker (e.g., 'AAPL')
            expiration_date: Filter by expiration date (YYYY-MM-DD)
            strike_price: Filter by strike price
            contract_type: 'call' or 'put'
            limit: Maximum number of results (max 1000)
            
        Returns:
            Dictionary with options chain data
        """
        params = {
            'apiKey': self.api_key,
            'underlying_asset': underlying_symbol,
            'limit': limit
        }
        
        if expiration_date:
            params['expiration_date'] = expiration_date
        if strike_price:
            params['strike_price'] = strike_price
        if contract_type:
            params['contract_type'] = contract_type
        
        try:
            data = self._make_request('v3/reference/options/contracts', params)
            self.logger.info(f"Retrieved options chain for {underlying_symbol}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to get options chain for {underlying_symbol}: {e}")
            raise
    
    def get_option_contract(self, options_ticker: str) -> Dict[str, Any]:
        """
        Get details for a specific options contract
        
        Args:
            options_ticker: Options ticker (e.g., 'O:AAPL230616C00150000')
            
        Returns:
            Contract details
        """
        params = {'apiKey': self.api_key}
        
        try:
            data = self._make_request(
                f'v3/reference/options/contracts/{options_ticker}',
                params
            )
            return data
        except Exception as e:
            self.logger.error(f"Failed to get option contract {options_ticker}: {e}")
            raise
    
    def get_option_quotes(self, options_ticker: str, 
                         timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get real-time or historical option quotes
        
        Args:
            options_ticker: Options ticker
            timestamp: Specific timestamp for historical data
            
        Returns:
            Quote data with bid/ask/last
        """
        params = {'apiKey': self.api_key}
        
        if timestamp:
            date_str = timestamp.strftime('%Y-%m-%d')
            endpoint = f'v2/aggs/ticker/{options_ticker}/range/1/minute/{date_str}/{date_str}'
        else:
            endpoint = f'v2/last/trade/{options_ticker}'
        
        try:
            data = self._make_request(endpoint, params)
            return data
        except Exception as e:
            self.logger.error(f"Failed to get quotes for {options_ticker}: {e}")
            raise
    
    def get_stock_quotes(self, symbol: str, 
                        start_date: Optional[date] = None,
                        end_date: Optional[date] = None,
                        timespan: str = "minute",
                        multiplier: int = 1) -> Dict[str, Any]:
        """
        Get stock price data (OHLCV)
        
        Args:
            symbol: Stock ticker
            start_date: Start date for historical data
            end_date: End date for historical data
            timespan: Time span (minute, hour, day, week, month)
            multiplier: Size of timespan (e.g., 5 for 5-minute bars)
            
        Returns:
            OHLCV data
        """
        params = {'apiKey': self.api_key}
        
        if start_date and end_date:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            endpoint = f'v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_str}/{end_str}'
        else:
            # Real-time snapshot
            endpoint = f'v2/snapshot/locale/us/markets/stocks/tickers/{symbol}'
        
        try:
            data = self._make_request(endpoint, params)
            self.logger.info(f"Retrieved stock quotes for {symbol}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to get stock quotes for {symbol}: {e}")
            raise
    
    def get_option_greeks(self, options_ticker: str) -> Dict[str, Any]:
        """
        Get Greeks for an option contract (requires premium subscription)
        
        Args:
            options_ticker: Options ticker
            
        Returns:
            Greeks (delta, gamma, theta, vega, rho)
        """
        params = {'apiKey': self.api_key}
        
        try:
            # Note: This endpoint may require premium access
            data = self._make_request(
                f'v3/snapshot/options/{options_ticker}',
                params
            )
            
            if 'results' in data and 'greeks' in data['results']:
                return data['results']['greeks']
            else:
                self.logger.warning(f"Greeks not available for {options_ticker}")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to get Greeks for {options_ticker}: {e}")
            raise
    
    def get_market_status(self) -> Dict[str, Any]:
        """
        Get current market status
        
        Returns:
            Market status (open/closed) and hours
        """
        params = {'apiKey': self.api_key}
        
        try:
            data = self._make_request('v1/marketstatus/now', params)
            return data
        except Exception as e:
            self.logger.error(f"Failed to get market status: {e}")
            raise
    
    def stream_options_trades(self, symbols: List[str], callback):
        """
        Stream real-time options trades (WebSocket)
        
        Args:
            symbols: List of options tickers to stream
            callback: Function to handle incoming data
        """
        # WebSocket implementation would go here
        # This requires the polygon-client-python library
        self.logger.info(f"Streaming not yet implemented. Symbols: {symbols}")
        raise NotImplementedError("WebSocket streaming requires additional setup")
