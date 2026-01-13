"""
Enhanced Data Ingestion for Multiple Stocks
Fetches data for top 50 liquid stocks across all sectors
"""
import yfinance as yf
import pandas as pd
import numpy as np
import time
import logging
import sys
import os
from datetime import datetime

# Add config directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.top_stocks import TOP_50_LIQUID, ALL_SECTORS, get_top_n_per_sector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

class MultiStockIngestion:
    """Download and process data for multiple stocks"""
    
    def __init__(self, symbols, period='3mo', interval='1h'):
        self.symbols = symbols
        self.period = period
        self.interval = interval
        self.success_count = 0
        self.fail_count = 0
        self.failed_symbols = []
    
    def fetch_historical_data(self, symbol):
        """Fetch historical price data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=self.period, interval=self.interval)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Add symbol column
            df['symbol'] = symbol
            df.reset_index(inplace=True)
            df['timestamp'] = df['Datetime'] if 'Datetime' in df.columns else df['Date']
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
            # Simple Moving Averages
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
            df['volatility'] = df['Close'].rolling(window=20).std()
            
            # Returns
            df['return_1d'] = df['Close'].pct_change()
            df['return_5d'] = df['Close'].pct_change(periods=5)
            
            # Volume indicators
            df['volume_sma'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def save_data(self, symbol, df):
        """Save data to CSV"""
        try:
            filepath = os.path.join(DATA_DIR, f'{symbol}_data.csv')
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(df)} records to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {e}")
            return False
    
    def process_symbol(self, symbol):
        """Fetch, process and save data for a single symbol"""
        logger.info(f"Processing {symbol}...")
        
        # Fetch data
        df = self.fetch_historical_data(symbol)
        if df is None:
            self.fail_count += 1
            self.failed_symbols.append(symbol)
            return False
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Save data
        if self.save_data(symbol, df):
            self.success_count += 1
            return True
        else:
            self.fail_count += 1
            self.failed_symbols.append(symbol)
            return False
    
    def run(self, delay=0.5):
        """Process all symbols with rate limiting"""
        total = len(self.symbols)
        logger.info(f"Starting data ingestion for {total} symbols...")
        
        start_time = time.time()
        
        for i, symbol in enumerate(self.symbols, 1):
            try:
                logger.info(f"\n[{i}/{total}] {symbol}")
                self.process_symbol(symbol)
                
                # Rate limiting
                if i < total:
                    time.sleep(delay)
                
            except KeyboardInterrupt:
                logger.warning("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error processing {symbol}: {e}")
                self.fail_count += 1
                self.failed_symbols.append(symbol)
        
        elapsed = time.time() - start_time
        
        # Print summary
        print("\n" + "="*70)
        print("✅ DATA INGESTION COMPLETE")
        print("="*70)
        print(f"Total Symbols: {total}")
        print(f"✓ Successful: {self.success_count}")
        print(f"✗ Failed: {self.fail_count}")
        print(f"⏱ Time Elapsed: {elapsed:.1f} seconds")
        print(f"📊 Success Rate: {self.success_count/total*100:.1f}%")
        
        if self.failed_symbols:
            print(f"\n⚠️  Failed Symbols ({len(self.failed_symbols)}):")
            for symbol in self.failed_symbols[:10]:
                print(f"   - {symbol}")
            if len(self.failed_symbols) > 10:
                print(f"   ... and {len(self.failed_symbols) - 10} more")
        
        print("="*70)
        
        return self.success_count, self.fail_count

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("🚀 USA OPTIONS AI - MULTI-STOCK DATA INGESTION")
    print("="*70)
    
    # Choose data set
    print("\nSelect dataset:")
    print("1. Top 50 Most Liquid Stocks (Recommended for testing)")
    print("2. Top 10 per Sector (110 stocks)")
    print("3. All Stocks (440+ stocks, ~2 hours)")
    
    choice = input("\nEnter choice (1/2/3) [1]: ").strip() or "1"
    
    if choice == "1":
        symbols = TOP_50_LIQUID
        print(f"\n📊 Fetching data for Top 50 Liquid Stocks")
    elif choice == "2":
        symbols = []
        top_per_sector = get_top_n_per_sector(10)
        for sector_symbols in top_per_sector.values():
            symbols.extend(sector_symbols)
        symbols = list(set(symbols))  # Remove duplicates
        print(f"\n📊 Fetching data for Top 10 per Sector ({len(symbols)} stocks)")
    elif choice == "3":
        from config.top_stocks import ALL_SYMBOLS
        symbols = ALL_SYMBOLS
        print(f"\n📊 Fetching data for ALL Stocks ({len(symbols)} stocks)")
        confirm = input("This will take ~2 hours. Continue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Cancelled.")
            return
    else:
        print("Invalid choice. Using Top 50.")
        symbols = TOP_50_LIQUID
    
    print(f"\n⏳ Period: 3 months | Interval: 1 hour")
    print(f"📁 Data Directory: {os.path.abspath(DATA_DIR)}")
    print(f"🔄 Rate Limit: 0.5 seconds between requests\n")
    
    # Run ingestion
    ingestion = MultiStockIngestion(symbols, period='3mo', interval='1h')
    success, failed = ingestion.run(delay=0.5)
    
    print(f"\n✅ Ingestion complete! {success} files ready for model training.")

if __name__ == '__main__':
    main()
