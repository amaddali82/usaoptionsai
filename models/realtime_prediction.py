"""
Real-time Prediction Service
Loads trained models and generates real-time predictions from streaming data
"""
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml
import asyncio
from kafka import KafkaConsumer
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from storage.timescaledb_client import TimescaleDBManager
from storage.influxdb_client import InfluxDBManager
from models.short_term.lstm_model import LSTMShortTermModel
from models.medium_term.transformer_model import TransformerMediumTermModel
from recommendation_engine.signal_generator import TradingSignalGenerator


class RealtimePredictionService:
    """Real-time prediction service using trained models"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize prediction service
        
        Args:
            config_path: Path to model configuration
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
        
        # Initialize signal generator
        self.signal_generator = TradingSignalGenerator()
        
        # Model storage
        self.models = {
            'short_term': {},   # symbol -> LSTMShortTermModel
            'medium_term': {}   # symbol -> TransformerMediumTermModel
        }
        
        self.model_dir = Path("saved_models")
        
        self.logger.info("Real-time Prediction Service initialized")
    
    def load_model(self, symbol: str, model_type: str = "short_term"):
        """
        Load trained model for symbol
        
        Args:
            symbol: Stock symbol
            model_type: 'short_term' or 'medium_term'
        """
        if symbol in self.models[model_type]:
            self.logger.info(f"Model already loaded for {symbol}")
            return
        
        try:
            if model_type == "short_term":
                model_path = self.model_dir / f"lstm_short_term_{symbol}.h5"
                scaler_path = self.model_dir / f"lstm_short_term_{symbol}_scaler.pkl"
                
                if model_path.exists() and scaler_path.exists():
                    model = LSTMShortTermModel.load(str(model_path), str(scaler_path))
                    self.models[model_type][symbol] = model
                    self.logger.info(f"Loaded short-term model for {symbol}")
                else:
                    self.logger.warning(f"Model files not found for {symbol}")
            
            elif model_type == "medium_term":
                model_path = self.model_dir / f"transformer_medium_term_{symbol}.h5"
                
                if model_path.exists():
                    model = TransformerMediumTermModel.load(str(model_path))
                    self.models[model_type][symbol] = model
                    self.logger.info(f"Loaded medium-term model for {symbol}")
                else:
                    self.logger.warning(f"Model files not found for {symbol}")
        
        except Exception as e:
            self.logger.error(f"Error loading model for {symbol}: {str(e)}")
    
    def get_recent_data(self, symbol: str, sequence_length: int = 60) -> Optional[pd.DataFrame]:
        """
        Get recent data for prediction
        
        Args:
            symbol: Stock symbol
            sequence_length: Number of recent data points needed
            
        Returns:
            DataFrame with recent data
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=sequence_length * 2)  # Buffer for missing data
        
        # Query from TimescaleDB
        df = self.timescale_db.query_price_history(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        if df.empty:
            self.logger.warning(f"No recent data found for {symbol}")
            return None
        
        # Load technical indicators
        df_indicators = self.timescale_db.query_technical_indicators(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        # Merge if indicators available
        if not df_indicators.empty:
            df = pd.merge(df, df_indicators, on=['timestamp', 'symbol'], how='left')
        
        # Sort by time and get last N rows
        df = df.sort_values('timestamp').tail(sequence_length)
        
        return df
    
    def predict_short_term(self, symbol: str, steps_ahead: int = 1) -> Optional[Dict]:
        """
        Generate short-term prediction
        
        Args:
            symbol: Stock symbol
            steps_ahead: Number of steps to predict ahead
            
        Returns:
            Prediction dictionary
        """
        # Load model if not loaded
        if symbol not in self.models['short_term']:
            self.load_model(symbol, 'short_term')
        
        model = self.models['short_term'].get(symbol)
        if model is None:
            self.logger.error(f"Model not available for {symbol}")
            return None
        
        # Get recent data
        df = self.get_recent_data(symbol, model.sequence_length)
        if df is None or len(df) < model.sequence_length:
            self.logger.error(f"Insufficient data for prediction")
            return None
        
        # Prepare features
        feature_cols = model.feature_names
        features_df = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Make prediction
        try:
            prediction = model.predict_next(features_df, steps_ahead=steps_ahead)
            
            # Add metadata
            prediction['symbol'] = symbol
            prediction['timestamp'] = datetime.utcnow()
            prediction['model_type'] = 'short_term'
            prediction['current_price'] = df['close'].iloc[-1]
            
            self.logger.info(f"Short-term prediction for {symbol}: {prediction['predicted_price']:.2f}")
            
            return prediction
        
        except Exception as e:
            self.logger.error(f"Error making prediction for {symbol}: {str(e)}")
            return None
    
    def predict_medium_term(self, symbol: str) -> Optional[Dict]:
        """
        Generate medium-term prediction
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Prediction dictionary
        """
        # Load model if not loaded
        if symbol not in self.models['medium_term']:
            self.load_model(symbol, 'medium_term')
        
        model = self.models['medium_term'].get(symbol)
        if model is None:
            self.logger.error(f"Model not available for {symbol}")
            return None
        
        # Get recent data
        df = self.get_recent_data(symbol, model.sequence_length)
        if df is None or len(df) < model.sequence_length:
            self.logger.error(f"Insufficient data for prediction")
            return None
        
        # Prepare features
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
            'volatility', 'return_1d', 'return_5d', 'volume_sma'
        ]
        
        available_features = [col for col in feature_cols if col in df.columns]
        features_df = df[available_features].fillna(method='ffill').fillna(method='bfill')
        
        # Scale and prepare
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(features_df.values)
        X = scaled_data.reshape(1, model.sequence_length, len(available_features))
        
        # Make prediction
        try:
            pred_result = model.predict(X)
            
            # Inverse transform prediction
            pred_scaled = np.zeros((1, len(available_features)))
            pred_scaled[0, available_features.index('close')] = pred_result['predictions'][0]
            pred_price = scaler.inverse_transform(pred_scaled)[0, available_features.index('close')]
            
            prediction = {
                'symbol': symbol,
                'timestamp': datetime.utcnow(),
                'model_type': 'medium_term',
                'current_price': df['close'].iloc[-1],
                'predicted_price': pred_price,
                'confidence': 0.75  # Default confidence
            }
            
            self.logger.info(f"Medium-term prediction for {symbol}: {pred_price:.2f}")
            
            return prediction
        
        except Exception as e:
            self.logger.error(f"Error making prediction for {symbol}: {str(e)}")
            return None
    
    def generate_trading_signal(self, symbol: str, 
                                prediction: Dict,
                                technical_indicators: Dict) -> Optional[Dict]:
        """
        Generate trading signal from prediction
        
        Args:
            symbol: Stock symbol
            prediction: Prediction dictionary
            technical_indicators: Technical indicator values
            
        Returns:
            Trading signal dictionary
        """
        try:
            signal = self.signal_generator.generate_signal(
                current_price=prediction['current_price'],
                predicted_price=prediction['predicted_price'],
                prediction_confidence=prediction.get('confidence', 0.5),
                technical_indicators=technical_indicators,
                volatility=technical_indicators.get('volatility', 0.2),
                time_horizon='short' if prediction['model_type'] == 'short_term' else 'medium'
            )
            
            signal['symbol'] = symbol
            
            self.logger.info(f"Generated {signal['signal']} signal for {symbol}")
            
            return signal
        
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {str(e)}")
            return None
    
    def save_prediction(self, prediction: Dict):
        """Save prediction to database"""
        try:
            self.timescale_db.insert_prediction(
                symbol=prediction['symbol'],
                timestamp=prediction['timestamp'],
                model_type=prediction['model_type'],
                predicted_price=prediction['predicted_price'],
                confidence=prediction.get('confidence', 0.5),
                actual_price=None
            )
            
            # Also save to InfluxDB for real-time queries
            self.influx_db.write_prediction(
                symbol=prediction['symbol'],
                model_type=prediction['model_type'],
                predicted_price=prediction['predicted_price'],
                confidence=prediction.get('confidence', 0.5)
            )
            
            self.logger.info(f"Saved prediction for {prediction['symbol']}")
        
        except Exception as e:
            self.logger.error(f"Error saving prediction: {str(e)}")
    
    def save_signal(self, signal: Dict):
        """Save trading signal to database"""
        try:
            self.timescale_db.insert_trading_signal(
                symbol=signal['symbol'],
                timestamp=signal['timestamp'],
                signal_type=signal['signal'],
                confidence=signal['confidence'],
                target_price=signal['target_1'],
                stop_loss=signal['stop_loss'],
                metadata=json.dumps({
                    'expected_return': signal['expected_return'],
                    'position_size': signal['position_size'],
                    'risk_reward_ratio': signal['risk_reward_ratio']
                })
            )
            
            self.logger.info(f"Saved trading signal for {signal['symbol']}")
        
        except Exception as e:
            self.logger.error(f"Error saving signal: {str(e)}")
    
    async def process_symbol(self, symbol: str):
        """
        Process predictions and signals for a symbol
        
        Args:
            symbol: Stock symbol
        """
        # Generate short-term prediction
        short_pred = self.predict_short_term(symbol, steps_ahead=1)
        
        if short_pred:
            self.save_prediction(short_pred)
            
            # Get technical indicators from recent data
            df = self.get_recent_data(symbol, 1)
            if df is not None and len(df) > 0:
                indicators = df.iloc[-1].to_dict()
                
                # Generate signal
                signal = self.generate_trading_signal(symbol, short_pred, indicators)
                if signal:
                    self.save_signal(signal)
        
        # Generate medium-term prediction (less frequently)
        medium_pred = self.predict_medium_term(symbol)
        if medium_pred:
            self.save_prediction(medium_pred)
    
    async def run(self, symbols: List[str], interval: int = 60):
        """
        Run prediction service continuously
        
        Args:
            symbols: List of symbols to process
            interval: Update interval in seconds
        """
        self.logger.info(f"Starting prediction service for {len(symbols)} symbols")
        
        while True:
            try:
                # Process all symbols
                tasks = [self.process_symbol(symbol) for symbol in symbols]
                await asyncio.gather(*tasks)
                
                self.logger.info(f"Completed prediction cycle, sleeping for {interval}s")
                await asyncio.sleep(interval)
            
            except KeyboardInterrupt:
                self.logger.info("Shutting down prediction service")
                break
            
            except Exception as e:
                self.logger.error(f"Error in prediction loop: {str(e)}")
                await asyncio.sleep(interval)


def main():
    """Main entry point"""
    
    # Initialize service
    service = RealtimePredictionService()
    
    # Define symbols to track
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Run service
    asyncio.run(service.run(symbols, interval=60))


if __name__ == "__main__":
    main()
