"""
Simplified Data Ingestion - Uses Yahoo Finance (No API Key Needed)
Stores data directly to files for model training
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import json
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

class SimpleDataIngestion:
    """Simplified data ingestion using Yahoo Finance"""
    
    def __init__(self, symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']):
        self.symbols = symbols
        self.logger = logging.getLogger(__name__)
    
    def fetch_historical_data(self, symbol, period='3mo', interval='1h'):
        """Fetch historical data for a symbol"""
        try:
            self.logger.info(f"Fetching data for {symbol}...")
            ticker = yf.Ticker(symbol)
            
            # Get historical price data
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                self.logger.warning(f"No data returned for {symbol}")
                return None
            
            # Add symbol column
            df['symbol'] = symbol
            df['timestamp'] = df.index
            
            # Calculate technical indicators
            df = self.calculate_indicators(df)
            
            self.logger.info(f"Fetched {len(df)} records for {symbol}")
            return df
        
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
            # Moving averages
            df['sma_5'] = df['Close'].rolling(window=5).mean()
            df['sma_10'] = df['Close'].rolling(window=10).mean()
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['sma_50'] = df['Close'].rolling(window=50).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            df['bb_middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Volatility
            df['volatility'] = df['Close'].pct_change().rolling(window=20).std()
            
            # Returns
            df['return_1d'] = df['Close'].pct_change()
            df['return_5d'] = df['Close'].pct_change(5)
            
            # Volume indicators
            df['volume_sma'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return df
    
    def save_data(self, df, symbol):
        """Save data to CSV file"""
        try:
            filename = DATA_DIR / f"{symbol}_data.csv"
            df.to_csv(filename, index=False)
            self.logger.info(f"Saved data to {filename}")
            return str(filename)
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            return None
    
    def fetch_all_symbols(self):
        """Fetch data for all symbols"""
        results = {}
        
        for symbol in self.symbols:
            df = self.fetch_historical_data(symbol)
            if df is not None:
                filepath = self.save_data(df, symbol)
                results[symbol] = {
                    'records': len(df),
                    'file': filepath,
                    'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}"
                }
            time.sleep(1)  # Rate limiting
        
        return results
    
    def get_latest_prices(self):
        """Get latest prices for all symbols"""
        prices = {}
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                prices[symbol] = {
                    'price': info.get('currentPrice', 0),
                    'change': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('volume', 0)
                }
            except Exception as e:
                self.logger.error(f"Error fetching price for {symbol}: {str(e)}")
        
        return prices


def main():
    """Main function"""
    print("\n" + "="*70)
    print("🚀 USA Options AI - Data Ingestion Starting")
    print("="*70)
    print("\n📊 Using Yahoo Finance (Free, No API Key Required)")
    print("📈 Fetching historical data for training...\n")
    
    # Initialize ingestion
    ingestion = SimpleDataIngestion()
    
    # Fetch all data
    print("⏳ Downloading data (this may take a minute)...\n")
    results = ingestion.fetch_all_symbols()
    
    # Display results
    print("\n" + "="*70)
    print("✅ DATA INGESTION COMPLETE")
    print("="*70 + "\n")
    
    total_records = 0
    for symbol, info in results.items():
        print(f"📊 {symbol}:")
        print(f"   Records: {info['records']}")
        print(f"   File: {info['file']}")
        print(f"   Range: {info['date_range']}")
        print()
        total_records += info['records']
    
    print(f"🎯 Total Records Downloaded: {total_records:,}")
    print(f"💾 Data saved to: {DATA_DIR.absolute()}")
    
    # Get latest prices
    print("\n📈 Latest Prices:")
    prices = ingestion.get_latest_prices()
    for symbol, data in prices.items():
        change_symbol = "📈" if data['change'] > 0 else "📉"
        print(f"   {change_symbol} {symbol}: ${data['price']:.2f} ({data['change']:+.2f}%)")
    
    print("\n✅ Ready for model training!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
