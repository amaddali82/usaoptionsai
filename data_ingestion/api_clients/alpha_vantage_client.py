"""
Alpha Vantage API Client for options, stocks, and fundamental data
Documentation: https://www.alphavantage.co/documentation/
"""
import logging
from typing import Dict, Any, Optional
from .base_client import BaseAPIClient


class AlphaVantageClient(BaseAPIClient):
    """Client for Alpha Vantage API"""
    
    def __init__(self, api_key: str, rate_limit: int = 5, timeout: int = 30):
        """
        Initialize Alpha Vantage client
        
        Args:
            api_key: Alpha Vantage API key
            rate_limit: Requests per minute (5 for free tier, 75 for premium)
            timeout: Request timeout in seconds
        """
        super().__init__(
            api_key=api_key,
            base_url="https://www.alphavantage.co",
            rate_limit=rate_limit / 60,  # Convert to per-second
            timeout=timeout
        )
        self.logger = logging.getLogger(__name__)
    
    def get_option_chain(self, symbol: str) -> Dict[str, Any]:
        """
        Get options chain for a symbol
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            
        Returns:
            Options chain data with all expirations and strikes
        """
        params = {
            'function': 'HISTORICAL_OPTIONS',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            data = self._make_request('query', params)
            self.logger.info(f"Retrieved options chain for {symbol}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to get options chain for {symbol}: {e}")
            raise
    
    def get_realtime_options(self, symbol: str, contract: str) -> Dict[str, Any]:
        """
        Get real-time option data
        
        Args:
            symbol: Stock ticker
            contract: Contract identifier
            
        Returns:
            Real-time option data
        """
        params = {
            'function': 'REALTIME_OPTIONS',
            'symbol': symbol,
            'contract': contract,
            'apikey': self.api_key
        }
        
        try:
            data = self._make_request('query', params)
            return data
        except Exception as e:
            self.logger.error(f"Failed to get realtime options for {symbol}: {e}")
            raise
    
    def get_intraday_prices(self, symbol: str, interval: str = '5min',
                           adjusted: bool = True, outputsize: str = 'compact') -> Dict[str, Any]:
        """
        Get intraday stock prices
        
        Args:
            symbol: Stock ticker
            interval: Time interval (1min, 5min, 15min, 30min, 60min)
            adjusted: Adjusted for splits/dividends
            outputsize: 'compact' (100 points) or 'full' (all available)
            
        Returns:
            Intraday time series data
        """
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'adjusted': str(adjusted).lower(),
            'outputsize': outputsize,
            'apikey': self.api_key
        }
        
        try:
            data = self._make_request('query', params)
            self.logger.info(f"Retrieved intraday data for {symbol}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to get intraday prices for {symbol}: {e}")
            raise
    
    def get_daily_prices(self, symbol: str, adjusted: bool = True,
                        outputsize: str = 'compact') -> Dict[str, Any]:
        """
        Get daily stock prices
        
        Args:
            symbol: Stock ticker
            adjusted: Adjusted for splits/dividends
            outputsize: 'compact' (100 days) or 'full' (20+ years)
            
        Returns:
            Daily time series data
        """
        function = 'TIME_SERIES_DAILY_ADJUSTED' if adjusted else 'TIME_SERIES_DAILY'
        params = {
            'function': function,
            'symbol': symbol,
            'outputsize': outputsize,
            'apikey': self.api_key
        }
        
        try:
            data = self._make_request('query', params)
            self.logger.info(f"Retrieved daily data for {symbol}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to get daily prices for {symbol}: {e}")
            raise
    
    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Get fundamental data and company overview
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Company fundamentals (P/E, EPS, market cap, etc.)
        """
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            data = self._make_request('query', params)
            self.logger.info(f"Retrieved company overview for {symbol}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to get company overview for {symbol}: {e}")
            raise
    
    def get_earnings(self, symbol: str) -> Dict[str, Any]:
        """
        Get quarterly and annual earnings
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Historical earnings data
        """
        params = {
            'function': 'EARNINGS',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            data = self._make_request('query', params)
            self.logger.info(f"Retrieved earnings for {symbol}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to get earnings for {symbol}: {e}")
            raise
    
    def get_technical_indicator(self, symbol: str, indicator: str,
                               interval: str = 'daily', time_period: int = 14,
                               **kwargs) -> Dict[str, Any]:
        """
        Get technical indicator data
        
        Args:
            symbol: Stock ticker
            indicator: Indicator name (RSI, MACD, SMA, EMA, BBANDS, etc.)
            interval: Time interval
            time_period: Number of periods for indicator
            **kwargs: Additional indicator-specific parameters
            
        Returns:
            Technical indicator time series
        """
        params = {
            'function': indicator.upper(),
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            'apikey': self.api_key
        }
        params.update(kwargs)
        
        try:
            data = self._make_request('query', params)
            self.logger.info(f"Retrieved {indicator} for {symbol}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to get {indicator} for {symbol}: {e}")
            raise
    
    def get_news_sentiment(self, tickers: Optional[str] = None,
                          topics: Optional[str] = None,
                          time_from: Optional[str] = None,
                          time_to: Optional[str] = None,
                          limit: int = 50) -> Dict[str, Any]:
        """
        Get news and sentiment data
        
        Args:
            tickers: Comma-separated list of tickers
            topics: Topic categories
            time_from: Start time (YYYYMMDDTHHMM)
            time_to: End time (YYYYMMDDTHHMM)
            limit: Maximum number of results
            
        Returns:
            News articles with sentiment scores
        """
        params = {
            'function': 'NEWS_SENTIMENT',
            'apikey': self.api_key,
            'limit': limit
        }
        
        if tickers:
            params['tickers'] = tickers
        if topics:
            params['topics'] = topics
        if time_from:
            params['time_from'] = time_from
        if time_to:
            params['time_to'] = time_to
        
        try:
            data = self._make_request('query', params)
            self.logger.info(f"Retrieved news sentiment")
            return data
        except Exception as e:
            self.logger.error(f"Failed to get news sentiment: {e}")
            raise
    
    def get_market_movers(self) -> Dict[str, Any]:
        """
        Get top gainers, losers, and most active stocks
        
        Returns:
            Market movers data
        """
        params = {
            'function': 'TOP_GAINERS_LOSERS',
            'apikey': self.api_key
        }
        
        try:
            data = self._make_request('query', params)
            self.logger.info("Retrieved market movers")
            return data
        except Exception as e:
            self.logger.error(f"Failed to get market movers: {e}")
            raise
