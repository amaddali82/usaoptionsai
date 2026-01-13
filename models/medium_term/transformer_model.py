"""
Medium-term Transformer Model for Weekly to Monthly Predictions
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Tuple, Optional
import logging


class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention"""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerMediumTermModel:
    """Transformer model for medium-term predictions"""
    
    def __init__(self, sequence_length: int = 60,
                 n_features: int = 30,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_transformer_blocks: int = 4,
                 ff_dim: int = 512,
                 dropout_rate: float = 0.1,
                 learning_rate: float = 0.0001):
        """
        Initialize Transformer model
        
        Args:
            sequence_length: Input sequence length
            n_features: Number of input features
            d_model: Embedding dimension
            num_heads: Number of attention heads
            num_transformer_blocks: Number of transformer blocks
            ff_dim: Feed-forward network dimension
            dropout_rate: Dropout rate
            learning_rate: Learning rate
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.logger = logging.getLogger(__name__)
        
        self.model = None
        self.scaler = None
        
        self._build_model()
    
    def _build_model(self):
        """Build Transformer architecture"""
        
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        
        # Project features to d_model dimensions
        x = layers.Dense(self.d_model)(inputs)
        
        # Add positional encoding
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        position_embedding = layers.Embedding(
            input_dim=self.sequence_length,
            output_dim=self.d_model
        )(positions)
        
        x = x + position_embedding
        
        # Transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(self.d_model, self.num_heads, self.ff_dim, self.dropout_rate)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='linear')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='transformer_medium_term')
        
        # Compile with custom learning rate schedule
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            alpha=0.1
        )
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.logger.info("Transformer model built successfully")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 16) -> Dict:
        """Train the model"""
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.logger.info(f"Training Transformer model for {epochs} epochs...")
        
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
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions"""
        predictions = self.model.predict(X, verbose=0)
        
        return {
            'predictions': predictions.flatten()
        }
    
    def save(self, model_path: str):
        """Save model"""
        self.model.save(model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load(cls, model_path: str):
        """Load saved model"""
        instance = cls()
        instance.model = keras.models.load_model(
            model_path,
            custom_objects={'TransformerBlock': TransformerBlock}
        )
        return instance


class ARIMAModel:
    """ARIMA model for medium-term predictions"""
    
    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)):
        """
        Initialize ARIMA model
        
        Args:
            order: (p, d, q) ARIMA order
            seasonal_order: (P, D, Q, s) seasonal order
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.logger = logging.getLogger(__name__)
        
        self.model = None
        self.fitted_model = None
    
    def train(self, data: pd.Series, auto_tune: bool = True):
        """
        Train ARIMA model
        
        Args:
            data: Time series data
            auto_tune: Use auto ARIMA to find best parameters
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        if auto_tune:
            try:
                import pmdarima as pm
                
                self.logger.info("Auto-tuning ARIMA parameters...")
                
                auto_model = pm.auto_arima(
                    data,
                    seasonal=True,
                    m=12,
                    information_criterion='aic',
                    max_p=10,
                    max_q=10,
                    max_P=5,
                    max_Q=5,
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
                
                self.order = auto_model.order
                self.seasonal_order = auto_model.seasonal_order
                
                self.logger.info(f"Best ARIMA order: {self.order}")
                self.logger.info(f"Best seasonal order: {self.seasonal_order}")
                
            except ImportError:
                self.logger.warning("pmdarima not installed, using default parameters")
        
        # Fit SARIMAX model
        self.model = SARIMAX(
            data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.fitted_model = self.model.fit(disp=False)
        
        self.logger.info("ARIMA model trained successfully")
        self.logger.info(f"AIC: {self.fitted_model.aic:.2f}")
        self.logger.info(f"BIC: {self.fitted_model.bic:.2f}")
    
    def predict(self, steps: int = 1) -> Dict[str, np.ndarray]:
        """
        Make predictions
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        forecast = self.fitted_model.get_forecast(steps=steps)
        
        return {
            'predictions': forecast.predicted_mean.values,
            'lower_bound': forecast.conf_int().iloc[:, 0].values,
            'upper_bound': forecast.conf_int().iloc[:, 1].values
        }
    
    def evaluate(self, test_data: pd.Series) -> Dict[str, float]:
        """Evaluate model on test data"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        predictions = self.predict(steps=len(test_data))['predictions']
        
        mse = mean_squared_error(test_data, predictions)
        mae = mean_absolute_error(test_data, predictions)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae
        }
