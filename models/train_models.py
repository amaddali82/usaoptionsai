"""
Model Training Orchestrator
Coordinates training of all ML models with data from database
"""
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from storage.timescaledb_client import TimescaleDBManager
from storage.influxdb_client import InfluxDBManager
from models.short_term.lstm_model import LSTMShortTermModel, CNNLSTMModel
from models.medium_term.transformer_model import TransformerMediumTermModel, ARIMAModel
from sklearn.model_selection import train_test_split


class ModelTrainingOrchestrator:
    """Orchestrates training of all ML models"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize training orchestrator
        
        Args:
            config_path: Path to model configuration file
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize database clients
        self.timescale_db = TimescaleDBManager()
        self.influx_db = InfluxDBManager()
        
        # Model storage directory
        self.model_dir = Path("saved_models")
        self.model_dir.mkdir(exist_ok=True)
        
        self.logger.info("Model Training Orchestrator initialized")
    
    def load_training_data(self, 
                          symbol: str,
                          start_date: datetime,
                          end_date: datetime,
                          data_type: str = "stock") -> pd.DataFrame:
        """
        Load training data from database
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            data_type: 'stock' or 'option'
            
        Returns:
            DataFrame with training data
        """
        self.logger.info(f"Loading training data for {symbol} from {start_date} to {end_date}")
        
        if data_type == "stock":
            # Load from TimescaleDB
            df_prices = self.timescale_db.query_price_history(
                symbol=symbol,
                start_time=start_date,
                end_time=end_date
            )
            
            # Load technical indicators
            df_indicators = self.timescale_db.query_technical_indicators(
                symbol=symbol,
                start_time=start_date,
                end_time=end_date
            )
            
            # Merge data
            if not df_prices.empty and not df_indicators.empty:
                df = pd.merge(df_prices, df_indicators, on=['timestamp', 'symbol'], how='inner')
            else:
                df = df_prices
            
        elif data_type == "option":
            # Load option data
            df = self.timescale_db.query_options_chain(
                symbol=symbol,
                date=end_date
            )
        
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        self.logger.info(f"Loaded {len(df)} records")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, 
                        feature_columns: List[str]) -> pd.DataFrame:
        """
        Prepare features for training
        
        Args:
            df: Input dataframe
            feature_columns: List of feature column names
            
        Returns:
            DataFrame with selected features
        """
        # Select feature columns that exist in the dataframe
        available_features = [col for col in feature_columns if col in df.columns]
        
        if len(available_features) < len(feature_columns):
            missing = set(feature_columns) - set(available_features)
            self.logger.warning(f"Missing features: {missing}")
        
        # Extract features
        features_df = df[available_features].copy()
        
        # Handle missing values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining NaN
        features_df = features_df.dropna()
        
        return features_df
    
    def train_lstm_short_term(self, symbol: str, 
                             lookback_days: int = 90,
                             test_size: float = 0.2) -> Dict[str, any]:
        """
        Train short-term LSTM model
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days of historical data
            test_size: Fraction of data for testing
            
        Returns:
            Training results and metrics
        """
        self.logger.info(f"Training short-term LSTM model for {symbol}")
        
        # Load data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        
        df = self.load_training_data(symbol, start_date, end_date, data_type="stock")
        
        if df.empty:
            self.logger.error(f"No data available for {symbol}")
            return {'error': 'No data available'}
        
        # Prepare features
        feature_cols = self.config['models']['lstm_sentiment']['features']
        features_df = self.prepare_features(df, feature_cols)
        
        # Initialize model
        model_params = self.config['models']['lstm_sentiment']
        lstm_model = LSTMShortTermModel(
            sequence_length=model_params['sequence_length'],
            n_features=len(features_df.columns),
            lstm_units=model_params['lstm_units'],
            dropout_rate=model_params['dropout'],
            learning_rate=model_params.get('learning_rate', 0.001)
        )
        
        # Prepare sequences
        X, y = lstm_model.prepare_sequences(features_df, target_col='close', scale=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, shuffle=False
        )
        
        # Train model
        history = lstm_model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=model_params.get('epochs', 50),
            batch_size=model_params.get('batch_size', 32)
        )
        
        # Evaluate
        metrics = lstm_model.evaluate(X_test, y_test)
        
        # Save model
        model_path = self.model_dir / f"lstm_short_term_{symbol}.h5"
        scaler_path = self.model_dir / f"lstm_short_term_{symbol}_scaler.pkl"
        lstm_model.save(str(model_path), str(scaler_path))
        
        self.logger.info(f"LSTM model saved to {model_path}")
        self.logger.info(f"Test metrics: {metrics}")
        
        return {
            'symbol': symbol,
            'model_type': 'lstm_short_term',
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'metrics': metrics,
            'history': history,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def train_transformer_medium_term(self, symbol: str,
                                     lookback_days: int = 180,
                                     test_size: float = 0.2) -> Dict[str, any]:
        """
        Train medium-term Transformer model
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days of historical data
            test_size: Fraction of data for testing
            
        Returns:
            Training results and metrics
        """
        self.logger.info(f"Training medium-term Transformer model for {symbol}")
        
        # Load data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        
        df = self.load_training_data(symbol, start_date, end_date, data_type="stock")
        
        if df.empty:
            self.logger.error(f"No data available for {symbol}")
            return {'error': 'No data available'}
        
        # Prepare features (more comprehensive for medium-term)
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
            'volatility', 'return_1d', 'return_5d', 'volume_sma'
        ]
        
        features_df = self.prepare_features(df, feature_cols)
        
        # Initialize model
        model_params = self.config['models']['transformer']
        transformer_model = TransformerMediumTermModel(
            sequence_length=model_params['sequence_length'],
            n_features=len(features_df.columns),
            d_model=model_params['d_model'],
            num_heads=model_params['num_heads'],
            num_transformer_blocks=model_params['num_encoder_layers'],
            dropout_rate=model_params['dropout']
        )
        
        # Prepare sequences (reuse LSTM's method)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(features_df.values)
        
        sequence_length = model_params['sequence_length']
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i - sequence_length:i])
            y.append(scaled_data[i, features_df.columns.get_loc('close')])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, shuffle=False
        )
        
        # Train model
        history = transformer_model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=model_params.get('epochs', 100),
            batch_size=model_params.get('batch_size', 16)
        )
        
        # Evaluate
        test_loss = transformer_model.model.evaluate(X_test, y_test, verbose=0)
        
        # Save model
        model_path = self.model_dir / f"transformer_medium_term_{symbol}.h5"
        transformer_model.save(str(model_path))
        
        self.logger.info(f"Transformer model saved to {model_path}")
        self.logger.info(f"Test loss: {test_loss}")
        
        return {
            'symbol': symbol,
            'model_type': 'transformer_medium_term',
            'model_path': str(model_path),
            'metrics': {'loss': test_loss[0], 'mae': test_loss[1], 'mse': test_loss[2]},
            'history': history,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def train_arima_medium_term(self, symbol: str,
                               lookback_days: int = 365) -> Dict[str, any]:
        """
        Train medium-term ARIMA model
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days of historical data
            
        Returns:
            Training results and metrics
        """
        self.logger.info(f"Training medium-term ARIMA model for {symbol}")
        
        # Load data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        
        df = self.load_training_data(symbol, start_date, end_date, data_type="stock")
        
        if df.empty:
            self.logger.error(f"No data available for {symbol}")
            return {'error': 'No data available'}
        
        # Extract price series
        df = df.set_index('timestamp').sort_index()
        price_series = df['close']
        
        # Split for testing
        test_size = int(len(price_series) * 0.2)
        train_series = price_series[:-test_size]
        test_series = price_series[-test_size:]
        
        # Initialize and train ARIMA model
        arima_model = ARIMAModel()
        arima_model.train(train_series, auto_tune=True)
        
        # Evaluate
        metrics = arima_model.evaluate(test_series)
        
        self.logger.info(f"ARIMA model trained successfully")
        self.logger.info(f"Test metrics: {metrics}")
        
        return {
            'symbol': symbol,
            'model_type': 'arima_medium_term',
            'order': arima_model.order,
            'seasonal_order': arima_model.seasonal_order,
            'metrics': metrics,
            'training_samples': len(train_series),
            'test_samples': len(test_series)
        }
    
    def train_all_models(self, symbols: List[str]) -> Dict[str, List[Dict]]:
        """
        Train all models for given symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with training results for each model type
        """
        results = {
            'lstm_short_term': [],
            'transformer_medium_term': [],
            'arima_medium_term': []
        }
        
        for symbol in symbols:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Training models for {symbol}")
            self.logger.info(f"{'='*50}\n")
            
            try:
                # Train LSTM
                lstm_result = self.train_lstm_short_term(symbol)
                results['lstm_short_term'].append(lstm_result)
                
                # Train Transformer
                transformer_result = self.train_transformer_medium_term(symbol)
                results['transformer_medium_term'].append(transformer_result)
                
                # Train ARIMA
                arima_result = self.train_arima_medium_term(symbol)
                results['arima_medium_term'].append(arima_result)
                
            except Exception as e:
                self.logger.error(f"Error training models for {symbol}: {str(e)}")
                continue
        
        return results


def main():
    """Main training script"""
    
    # Initialize orchestrator
    orchestrator = ModelTrainingOrchestrator()
    
    # Define symbols to train on
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Train all models
    results = orchestrator.train_all_models(symbols)
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    
    for model_type, model_results in results.items():
        print(f"\n{model_type}:")
        for result in model_results:
            if 'error' not in result:
                print(f"  {result['symbol']}: {result.get('metrics', {})}")
            else:
                print(f"  {result.get('symbol', 'Unknown')}: {result['error']}")


if __name__ == "__main__":
    main()
