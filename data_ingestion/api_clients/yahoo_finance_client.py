"""
Yahoo Finance Client using yfinance library
Free and reliable for historical and basic real-time data
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd


class YahooFinanceClient:
    """Client for Yahoo Finance data via yfinance library"""
    
    def __init__(self):
        """Initialize Yahoo Finance client"""
        self.logger = logging.getLogger(__name__)
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive stock information
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Dictionary with company info, fundamentals, etc.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            self.logger.info(f"Retrieved info for {symbol}")
            return info
        except Exception as e:
            self.logger.error(f"Failed to get info for {symbol}: {e}")
            raise
    
    def get_historical_prices(self, symbol: str, 
                             period: str = "1mo",
                             interval: str = "1d",
                             start: Optional[datetime] = None,
                             end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get historical price data
        
        Args:
            symbol: Stock ticker
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            start: Start date (if not using period)
            end: End date (if not using period)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            if start and end:
                data = ticker.history(start=start, end=end, interval=interval)
            else:
                data = ticker.history(period=period, interval=interval)
            
            self.logger.info(f"Retrieved historical data for {symbol}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to get historical prices for {symbol}: {e}")
            raise
    
    def get_options_chain(self, symbol: str, 
                         expiration: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get options chain
        
        Args:
            symbol: Stock ticker
            expiration: Specific expiration date (YYYY-MM-DD), None for nearest
            
        Returns:
            Dictionary with 'calls' and 'puts' DataFrames
        """
        try:
            ticker = yf.Ticker(symbol)
            
            if expiration:
                options = ticker.option_chain(expiration)
            else:
                # Get nearest expiration
                expirations = ticker.options
                if not expirations:
                    raise ValueError(f"No options available for {symbol}")
                options = ticker.option_chain(expirations[0])
            
            result = {
                'calls': options.calls,
                'puts': options.puts,
                'expiration': expiration or expirations[0]
            }
            
            self.logger.info(f"Retrieved options chain for {symbol}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to get options chain for {symbol}: {e}")
            raise
    
    def get_all_options_expirations(self, symbol: str) -> List[str]:
        """
        Get all available option expiration dates
        
        Args:
            symbol: Stock ticker
            
        Returns:
            List of expiration dates
        """
        try:
            ticker = yf.Ticker(symbol)
            expirations = list(ticker.options)
            self.logger.info(f"Retrieved {len(expirations)} expirations for {symbol}")
            return expirations
        except Exception as e:
            self.logger.error(f"Failed to get expirations for {symbol}: {e}")
            raise
    
    def get_dividends(self, symbol: str) -> pd.Series:
        """
        Get dividend history
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Series with dividend dates and amounts
        """
        try:
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends
            self.logger.info(f"Retrieved dividends for {symbol}")
            return dividends
        except Exception as e:
            self.logger.error(f"Failed to get dividends for {symbol}: {e}")
            raise
    
    def get_splits(self, symbol: str) -> pd.Series:
        """
        Get stock split history
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Series with split dates and ratios
        """
        try:
            ticker = yf.Ticker(symbol)
            splits = ticker.splits
            self.logger.info(f"Retrieved splits for {symbol}")
            return splits
        except Exception as e:
            self.logger.error(f"Failed to get splits for {symbol}: {e}")
            raise
    
    def get_financials(self, symbol: str, freq: str = "yearly") -> pd.DataFrame:
        """
        Get financial statements
        
        Args:
            symbol: Stock ticker
            freq: 'yearly' or 'quarterly'
            
        Returns:
            DataFrame with financial data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            if freq == "yearly":
                financials = ticker.financials
            else:
                financials = ticker.quarterly_financials
            
            self.logger.info(f"Retrieved {freq} financials for {symbol}")
            return financials
        except Exception as e:
            self.logger.error(f"Failed to get financials for {symbol}: {e}")
            raise
    
    def get_balance_sheet(self, symbol: str, freq: str = "yearly") -> pd.DataFrame:
        """Get balance sheet"""
        try:
            ticker = yf.Ticker(symbol)
            
            if freq == "yearly":
                bs = ticker.balance_sheet
            else:
                bs = ticker.quarterly_balance_sheet
            
            self.logger.info(f"Retrieved {freq} balance sheet for {symbol}")
            return bs
        except Exception as e:
            self.logger.error(f"Failed to get balance sheet for {symbol}: {e}")
            raise
    
    def get_cashflow(self, symbol: str, freq: str = "yearly") -> pd.DataFrame:
        """Get cash flow statement"""
        try:
            ticker = yf.Ticker(symbol)
            
            if freq == "yearly":
                cf = ticker.cashflow
            else:
                cf = ticker.quarterly_cashflow
            
            self.logger.info(f"Retrieved {freq} cash flow for {symbol}")
            return cf
        except Exception as e:
            self.logger.error(f"Failed to get cash flow for {symbol}: {e}")
            raise
    
    def get_earnings(self, symbol: str, freq: str = "yearly") -> pd.DataFrame:
        """Get earnings data"""
        try:
            ticker = yf.Ticker(symbol)
            
            if freq == "yearly":
                earnings = ticker.earnings
            else:
                earnings = ticker.quarterly_earnings
            
            self.logger.info(f"Retrieved {freq} earnings for {symbol}")
            return earnings
        except Exception as e:
            self.logger.error(f"Failed to get earnings for {symbol}: {e}")
            raise
    
    def get_recommendations(self, symbol: str) -> pd.DataFrame:
        """
        Get analyst recommendations
        
        Args:
            symbol: Stock ticker
            
        Returns:
            DataFrame with analyst ratings
        """
        try:
            ticker = yf.Ticker(symbol)
            recommendations = ticker.recommendations
            self.logger.info(f"Retrieved recommendations for {symbol}")
            return recommendations
        except Exception as e:
            self.logger.error(f"Failed to get recommendations for {symbol}: {e}")
            raise
    
    def get_institutional_holders(self, symbol: str) -> pd.DataFrame:
        """Get institutional holders"""
        try:
            ticker = yf.Ticker(symbol)
            holders = ticker.institutional_holders
            self.logger.info(f"Retrieved institutional holders for {symbol}")
            return holders
        except Exception as e:
            self.logger.error(f"Failed to get institutional holders for {symbol}: {e}")
            raise
    
    def download_multiple_symbols(self, symbols: List[str], 
                                  period: str = "1mo",
                                  interval: str = "1d") -> pd.DataFrame:
        """
        Download data for multiple symbols at once (more efficient)
        
        Args:
            symbols: List of stock tickers
            period: Time period
            interval: Data interval
            
        Returns:
            DataFrame with data for all symbols
        """
        try:
            data = yf.download(
                tickers=" ".join(symbols),
                period=period,
                interval=interval,
                group_by='ticker',
                auto_adjust=True,
                threads=True
            )
            self.logger.info(f"Downloaded data for {len(symbols)} symbols")
            return data
        except Exception as e:
            self.logger.error(f"Failed to download multiple symbols: {e}")
            raise
