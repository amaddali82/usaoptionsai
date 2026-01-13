"""
Short-term LSTM Model for Intraday to 1-Week Predictions
Incorporates price data, technical indicators, and sentiment
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Tuple, Optional
import logging
import joblib
from datetime import datetime, timedelta


class LSTMShortTermModel:
    """LSTM model for short-term options price prediction"""
    
    def __init__(self, sequence_length: int = 60, 
                 n_features: int = 20,
                 lstm_units: List[int] = [128, 64],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize LSTM model
        
        Args:
            sequence_length: Number of time steps to look back
            n_features: Number of input features
            lstm_units: List of LSTM layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.logger = logging.getLogger(__name__)
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        self._build_model()
    
    def _build_model(self):
        """Build LSTM architecture"""
        
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        
        # First LSTM layer
        x = layers.LSTM(
            self.lstm_units[0],
            return_sequences=True if len(self.lstm_units) > 1 else False,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate
        )(inputs)
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:]):
            return_seq = i < len(self.lstm_units) - 2
            x = layers.LSTM(
                units,
                return_sequences=return_seq,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            )(x)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer (predicting price change)
        outputs = layers.Dense(1, activation='linear')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='lstm_short_term')
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.logger.info("LSTM model built successfully")
        self.logger.info(f"Model summary:\n{self.model.summary()}")
    
    def prepare_sequences(self, data: pd.DataFrame, 
                         target_col: str = 'close',
                         scale: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for training/prediction
        
        Args:
            data: DataFrame with features
            target_col: Column name for target variable
            scale: Whether to scale features
            
        Returns:
            X, y arrays for training
        """
        if scale:
            from sklearn.preprocessing import StandardScaler
            
            if self.scaler is None:
                self.scaler = StandardScaler()
                scaled_data = self.scaler.fit_transform(data.values)
            else:
                scaled_data = self.scaler.transform(data.values)
        else:
            scaled_data = data.values
        
        # Store feature names
        if self.feature_names is None:
            self.feature_names = data.columns.tolist()
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i - self.sequence_length:i])
            
            # Target is next time step's price
            target_idx = data.columns.get_loc(target_col)
            y.append(scaled_data[i, target_idx])
        
        return np.array(X), np.array(y)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 50,
              batch_size: int = 32,
              callbacks: Optional[List] = None) -> Dict:
        """
        Train the model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if X_val is not None else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.logger.info(f"Training LSTM model for {epochs} epochs...")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.logger.info("Training completed")
        
        return history.history
    
    def predict(self, X: np.ndarray, 
                return_confidence: bool = True) -> Dict[str, np.ndarray]:
        """
        Make predictions
        
        Args:
            X: Input sequences
            return_confidence: Whether to return confidence intervals
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        predictions = self.model.predict(X, verbose=0)
        
        result = {'predictions': predictions.flatten()}
        
        if return_confidence:
            # Monte Carlo dropout for uncertainty estimation
            mc_predictions = []
            for _ in range(30):  # 30 forward passes
                mc_pred = self.model(X, training=True)
                mc_predictions.append(mc_pred.numpy())
            
            mc_predictions = np.array(mc_predictions)
            
            # Calculate confidence intervals
            result['mean'] = mc_predictions.mean(axis=0).flatten()
            result['std'] = mc_predictions.std(axis=0).flatten()
            result['lower_bound'] = np.percentile(mc_predictions, 5, axis=0).flatten()
            result['upper_bound'] = np.percentile(mc_predictions, 95, axis=0).flatten()
        
        return result
    
    def predict_next(self, recent_data: pd.DataFrame,
                    steps_ahead: int = 1) -> Dict[str, float]:
        """
        Predict next N steps
        
        Args:
            recent_data: Recent data (at least sequence_length rows)
            steps_ahead: Number of steps to predict ahead
            
        Returns:
            Dictionary with prediction details
        """
        if len(recent_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points")
        
        # Prepare sequence
        scaled_data = self.scaler.transform(recent_data.values)
        X = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, self.n_features)
        
        predictions = []
        
        for _ in range(steps_ahead):
            pred = self.predict(X, return_confidence=True)
            predictions.append(pred)
            
            # Update sequence for next prediction (simple approach)
            # In production, you'd update with actual new data
            new_step = X[0, -1, :].copy()
            new_step[0] = pred['mean'][0]  # Update price feature
            
            X = np.roll(X, -1, axis=1)
            X[0, -1, :] = new_step
        
        # Return last prediction
        return {
            'predicted_price': predictions[-1]['mean'][0],
            'confidence': 1 - predictions[-1]['std'][0],  # Higher std = lower confidence
            'lower_bound': predictions[-1]['lower_bound'][0],
            'upper_bound': predictions[-1]['upper_bound'][0],
            'steps_ahead': steps_ahead
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'loss': results[0],
            'mae': results[1],
            'mse': results[2],
            'rmse': np.sqrt(results[2])
        }
        
        self.logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def save(self, model_path: str, scaler_path: str):
        """Save model and scaler"""
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_names, scaler_path.replace('.pkl', '_features.pkl'))
        self.logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load(cls, model_path: str, scaler_path: str):
        """Load saved model"""
        instance = cls()
        instance.model = keras.models.load_model(model_path)
        instance.scaler = joblib.load(scaler_path)
        instance.feature_names = joblib.load(scaler_path.replace('.pkl', '_features.pkl'))
        return instance


class CNNLSTMModel:
    """CNN-LSTM hybrid for limit order book imaging"""
    
    def __init__(self, image_shape: Tuple[int, int] = (100, 40),
                 conv_filters: List[int] = [32, 64],
                 lstm_units: int = 128,
                 dropout_rate: float = 0.3):
        """
        Initialize CNN-LSTM model
        
        Args:
            image_shape: Shape of LOB image (time_steps, features)
            conv_filters: List of conv layer filter counts
            lstm_units: LSTM layer size
            dropout_rate: Dropout rate
        """
        self.image_shape = image_shape
        self.conv_filters = conv_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.logger = logging.getLogger(__name__)
        
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Build CNN-LSTM architecture"""
        
        inputs = keras.Input(shape=(*self.image_shape, 1))
        
        # CNN layers
        x = inputs
        for filters in self.conv_filters:
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Reshape for LSTM
        x = layers.Reshape((x.shape[1], -1))(x)
        
        # LSTM layer
        x = layers.LSTM(self.lstm_units, dropout=self.dropout_rate)(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output (3 classes: buy, hold, sell)
        outputs = layers.Dense(3, activation='softmax')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='cnn_lstm')
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.logger.info("CNN-LSTM model built successfully")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 50,
              batch_size: int = 64) -> Dict:
        """Train the model"""
        
        # Ensure correct shape
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(*X_train.shape, 1)
        if X_val is not None and len(X_val.shape) == 3:
            X_val = X_val.reshape(*X_val.shape, 1)
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ],
            verbose=1
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions"""
        if len(X.shape) == 3:
            X = X.reshape(*X.shape, 1)
        
        predictions = self.model.predict(X, verbose=0)
        
        return {
            'probabilities': predictions,
            'classes': np.argmax(predictions, axis=1),
            'confidence': np.max(predictions, axis=1)
        }
    
    def save(self, model_path: str):
        """Save model"""
        self.model.save(model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load(cls, model_path: str):
        """Load saved model"""
        instance = cls()
        instance.model = keras.models.load_model(model_path)
        return instance
