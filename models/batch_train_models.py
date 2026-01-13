"""
Batch Model Training for Multiple Stocks
Trains neural network models for all available data files
"""
import os
import glob
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from tensorflow import keras
    from sklearn.preprocessing import StandardScaler
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available. Install: pip install tensorflow scikit-learn")
    exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = 'data'
MODELS_DIR = 'saved_models'
os.makedirs(MODELS_DIR, exist_ok=True)

class BatchModelTrainer:
    """Train models for multiple stocks in batch"""
    
    def __init__(self, epochs=20, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.success_count = 0
        self.fail_count = 0
        self.results = []
    
    def load_and_prepare_data(self, filepath):
        """Load data and prepare features"""
        try:
            df = pd.read_csv(filepath)
            
            # Feature columns (lowercase)
            feature_cols = ['open', 'high', 'low', 'volume', 'sma_5', 'sma_10', 
                           'sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal',
                           'bb_upper', 'bb_lower', 'volatility']
            
            # Lowercase all columns
            df.columns = df.columns.str.lower()
            
            # Check if required columns exist
            available_cols = [col for col in feature_cols if col in df.columns]
            if len(available_cols) < 10:
                logger.warning(f"Insufficient features: {len(available_cols)}")
                return None, None, None, None
            
            # Prepare features and target
            X = df[available_cols].values
            y = df['close'].values
            
            # Remove NaN rows
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            if len(X) < 100:
                logger.warning(f"Insufficient data: {len(X)} samples")
                return None, None, None, None
            
            # Normalize features
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            
            # Train/test split
            train_size = int(len(X_scaled) * 0.8)
            X_train = X_scaled[:train_size]
            X_test = X_scaled[train_size:]
            y_train = y_scaled[:train_size]
            y_test = y_scaled[train_size:]
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None, None, None, None
    
    def build_model(self, input_shape):
        """Build neural network model"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model_for_symbol(self, symbol, data_file):
        """Train model for a single symbol"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Training model for {symbol}")
        logger.info(f"{'='*50}")
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_and_prepare_data(data_file)
        
        if X_train is None:
            logger.error(f"Failed to load data for {symbol}")
            self.fail_count += 1
            return None
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Features: {X_train.shape[1]}")
        
        # Build model
        model = self.build_model(X_train.shape[1])
        
        # Train model
        logger.info("Training model...")
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        
        training_time = time.time() - start_time
        
        # Evaluate
        train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        
        logger.info(f"Training MAE: {train_mae:.4f}")
        logger.info(f"Test MAE: {test_mae:.4f}")
        logger.info(f"Training time: {training_time:.1f}s")
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f'{symbol}_model.h5')
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Store results
        self.results.append({
            'symbol': symbol,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X_train.shape[1],
            'train_mae': train_mae,
            'test_mae': test_mae,
            'training_time': training_time,
            'model_path': model_path
        })
        
        self.success_count += 1
        return model
    
    def train_all(self):
        """Train models for all data files"""
        # Find all data files
        data_files = glob.glob(os.path.join(DATA_DIR, '*_data.csv'))
        
        if not data_files:
            logger.error(f"No data files found in {DATA_DIR}")
            return
        
        total = len(data_files)
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 BATCH MODEL TRAINING - {total} STOCKS")
        logger.info(f"{'='*70}\n")
        
        start_time = time.time()
        
        for i, data_file in enumerate(data_files, 1):
            symbol = os.path.basename(data_file).replace('_data.csv', '')
            
            try:
                logger.info(f"\n[{i}/{total}] Processing {symbol}")
                self.train_model_for_symbol(symbol, data_file)
                
            except KeyboardInterrupt:
                logger.warning("\n⚠️  Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error training {symbol}: {e}")
                self.fail_count += 1
        
        elapsed = time.time() - start_time
        
        # Print summary
        self.print_summary(total, elapsed)
    
    def print_summary(self, total, elapsed):
        """Print training summary"""
        print("\n" + "="*70)
        print("✅ BATCH MODEL TRAINING COMPLETE")
        print("="*70)
        print(f"Total Stocks: {total}")
        print(f"✓ Successful: {self.success_count}")
        print(f"✗ Failed: {self.fail_count}")
        print(f"⏱ Total Time: {elapsed/60:.1f} minutes")
        print(f"📊 Success Rate: {self.success_count/total*100:.1f}%")
        
        if self.results:
            # Calculate statistics
            train_maes = [r['train_mae'] for r in self.results]
            test_maes = [r['test_mae'] for r in self.results]
            
            print(f"\n📈 Performance Statistics:")
            print(f"   Average Train MAE: {np.mean(train_maes):.4f}")
            print(f"   Average Test MAE: {np.mean(test_maes):.4f}")
            print(f"   Best Test MAE: {np.min(test_maes):.4f}")
            print(f"   Worst Test MAE: {np.max(test_maes):.4f}")
            
            # Top 10 best models
            sorted_results = sorted(self.results, key=lambda x: x['test_mae'])
            print(f"\n🏆 Top 10 Best Models (by Test MAE):")
            for i, result in enumerate(sorted_results[:10], 1):
                print(f"   {i}. {result['symbol']}: {result['test_mae']:.4f}")
            
            # Save results to CSV
            results_df = pd.DataFrame(self.results)
            results_file = os.path.join(MODELS_DIR, 'training_results.csv')
            results_df.to_csv(results_file, index=False)
            print(f"\n💾 Detailed results saved to {results_file}")
        
        print("="*70)

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("🧠 USA OPTIONS AI - BATCH MODEL TRAINING")
    print("="*70)
    
    # Check for data files
    data_files = glob.glob(os.path.join(DATA_DIR, '*_data.csv'))
    print(f"\n📊 Found {len(data_files)} data files")
    
    if not data_files:
        print(f"\n⚠️  No data files found in {DATA_DIR}")
        print("Please run data ingestion first:")
        print("   python data_ingestion/multi_stock_ingestion.py")
        return
    
    print(f"🧠 Models will be saved to: {os.path.abspath(MODELS_DIR)}")
    print(f"⚙️  Configuration: 20 epochs, batch size 32")
    print(f"⏱ Estimated time: ~{len(data_files) * 0.15:.0f} minutes\n")
    
    proceed = input("Start training? (yes/no) [yes]: ").strip().lower() or 'yes'
    
    if proceed == 'yes':
        trainer = BatchModelTrainer(epochs=20, batch_size=32)
        trainer.train_all()
    else:
        print("Training cancelled.")

if __name__ == '__main__':
    main()
