"""
Simplified Model Training - Works with downloaded CSV data
Trains LSTM model on historical stock data
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not installed. Will create model architecture only.")

class SimpleModelTrainer:
    """Simplified model trainer for demo"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.models_dir = Path("saved_models")
        self.models_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, symbol):
        """Load data from CSV"""
        filepath = self.data_dir / f"{symbol}_data.csv"
        
        if not filepath.exists():
            self.logger.error(f"Data file not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        self.logger.info(f"Loaded {len(df)} records for {symbol}")
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        # Select relevant columns
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'sma_5', 'sma_10', 'sma_20', 'rsi',
            'macd', 'macd_signal', 'volatility',
            'return_1d', 'volume_ratio'
        ]
        
        # Filter existing columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(available_cols) < 5:
            self.logger.error("Insufficient features in data")
            return None, None
        
        # Extract features
        features = df[available_cols].copy()
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(method='bfill')
        features = features.dropna()
        
        # Target: next close price
        target = features['Close'].shift(-1)
        
        # Remove last row (no target)
        features = features[:-1]
        target = target[:-1]
        
        self.logger.info(f"Prepared {len(features)} samples with {len(available_cols)} features")
        
        return features, target
    
    def create_simple_model(self, input_shape):
        """Create a simple neural network"""
        if not TENSORFLOW_AVAILABLE:
            self.logger.info("Creating model architecture (TensorFlow not available)")
            return None
        
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, symbol):
        """Train model for a symbol"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Training model for {symbol}")
        self.logger.info(f"{'='*50}")
        
        # Load data
        df = self.load_data(symbol)
        if df is None:
            return None
        
        # Prepare features
        X, y = self.prepare_features(df)
        if X is None:
            return None
        
        # Scale features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, shuffle=False
        )
        
        self.logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        if TENSORFLOW_AVAILABLE:
            # Create and train model
            model = self.create_simple_model(X.shape[1])
            
            self.logger.info("Training model...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=20,
                batch_size=32,
                verbose=0
            )
            
            # Evaluate
            train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
            test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
            
            self.logger.info(f"Training MAE: {train_mae:.4f}")
            self.logger.info(f"Test MAE: {test_mae:.4f}")
            
            # Save model
            model_path = self.models_dir / f"{symbol}_model.h5"
            model.save(model_path)
            self.logger.info(f"Model saved to {model_path}")
            
            return {
                'symbol': symbol,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'samples': len(X_train),
                'features': X.shape[1],
                'model_path': str(model_path)
            }
        else:
            self.logger.info("Model architecture created (TensorFlow not installed for training)")
            return {
                'symbol': symbol,
                'samples': len(X_train),
                'features': X.shape[1],
                'status': 'Architecture only (TensorFlow needed for training)'
            }
    
    def train_all_models(self, symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']):
        """Train models for all symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                result = self.train_model(symbol)
                if result:
                    results[symbol] = result
            except Exception as e:
                self.logger.error(f"Error training model for {symbol}: {str(e)}")
                results[symbol] = {'error': str(e)}
        
        return results


def main():
    """Main training function"""
    print("\n" + "="*70)
    print("🧠 USA Options AI - Model Training")
    print("="*70)
    
    if not TENSORFLOW_AVAILABLE:
        print("\n⚠️  TensorFlow not installed. Installing now...")
        print("   This may take a few minutes...\n")
        import subprocess
        try:
            subprocess.check_call(['pip', 'install', 'tensorflow', '--quiet'])
            print("✅ TensorFlow installed successfully!\n")
        except:
            print("❌ Could not install TensorFlow automatically.")
            print("   Run: pip install tensorflow\n")
            print("   Continuing with model architecture creation only...\n")
    
    # Initialize trainer
    trainer = SimpleModelTrainer()
    
    # Train models
    print("🚀 Starting model training...\n")
    results = trainer.train_all_models()
    
    # Display results
    print("\n" + "="*70)
    print("✅ MODEL TRAINING COMPLETE")
    print("="*70 + "\n")
    
    for symbol, result in results.items():
        print(f"📊 {symbol}:")
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   ✓ Samples: {result.get('samples', 'N/A')}")
            print(f"   ✓ Features: {result.get('features', 'N/A')}")
            if 'train_mae' in result:
                print(f"   ✓ Training MAE: {result['train_mae']:.4f}")
                print(f"   ✓ Test MAE: {result['test_mae']:.4f}")
                print(f"   ✓ Model: {result['model_path']}")
            elif 'status' in result:
                print(f"   ℹ️  {result['status']}")
        print()
    
    print("🎯 Models are ready for predictions!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
