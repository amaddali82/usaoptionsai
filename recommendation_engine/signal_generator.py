"""
Recommendation Engine for Trading Signal Generation
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from enum import Enum


class SignalType(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class ConfidenceLevel(Enum):
    """Confidence levels for signals"""
    HIGH = (0.7, 0.8)      # 70-80%
    MEDIUM = (0.5, 0.6)    # 50-60%
    LOW = (0.3, 0.4)       # 30-40%


class TradingSignalGenerator:
    """Generate trading signals based on predictions and technical analysis"""
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize signal generator
        
        Args:
            risk_free_rate: Annual risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
    
    def generate_signal(self, 
                       current_price: float,
                       predicted_price: float,
                       prediction_confidence: float,
                       technical_indicators: Dict[str, float],
                       volatility: float,
                       time_horizon: str = "short") -> Dict[str, any]:
        """
        Generate trading signal
        
        Args:
            current_price: Current stock price
            predicted_price: Predicted price
            prediction_confidence: Model confidence (0-1)
            technical_indicators: Dict of technical indicators (RSI, MACD, etc.)
            volatility: Historical or implied volatility
            time_horizon: 'short', 'medium', or 'long'
            
        Returns:
            Dictionary with signal details
        """
        # Calculate expected return
        expected_return = (predicted_price - current_price) / current_price
        
        # Determine signal direction
        if expected_return > 0.02:  # >2% expected gain
            base_signal = SignalType.BUY
        elif expected_return < -0.02:  # >2% expected loss
            base_signal = SignalType.SELL
        else:
            base_signal = SignalType.HOLD
        
        # Adjust signal strength based on confidence and technical indicators
        signal_strength = self._calculate_signal_strength(
            expected_return,
            prediction_confidence,
            technical_indicators,
            volatility
        )
        
        # Determine final signal
        if signal_strength > 0.7:
            if base_signal == SignalType.BUY:
                signal = SignalType.STRONG_BUY
            elif base_signal == SignalType.SELL:
                signal = SignalType.STRONG_SELL
            else:
                signal = base_signal
        else:
            signal = base_signal
        
        # Calculate targets and stops
        targets = self._calculate_targets(
            current_price,
            predicted_price,
            volatility,
            signal,
            time_horizon
        )
        
        # Calculate position size
        position_size = self._calculate_position_size(
            signal_strength,
            volatility
        )
        
        return {
            'symbol': None,  # To be filled by caller
            'timestamp': datetime.utcnow(),
            'signal': signal.value,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'expected_return': expected_return,
            'confidence': signal_strength,
            'target_1': targets['target_1'],
            'target_2': targets['target_2'],
            'target_3': targets['target_3'],
            'stop_loss': targets['stop_loss'],
            'position_size': position_size,
            'time_horizon': time_horizon,
            'technical_score': self._score_technicals(technical_indicators),
            'risk_reward_ratio': self._calculate_risk_reward(
                current_price,
                targets['target_1'],
                targets['stop_loss']
            )
        }
    
    def _calculate_signal_strength(self,
                                   expected_return: float,
                                   prediction_confidence: float,
                                   technical_indicators: Dict[str, float],
                                   volatility: float) -> float:
        """Calculate overall signal strength (0-1)"""
        
        # Weight different factors
        weights = {
            'prediction': 0.4,
            'technical': 0.3,
            'momentum': 0.2,
            'volatility': 0.1
        }
        
        # Prediction strength (based on expected return and confidence)
        prediction_strength = abs(expected_return) * prediction_confidence
        prediction_strength = min(prediction_strength / 0.1, 1.0)  # Normalize
        
        # Technical strength
        technical_strength = self._score_technicals(technical_indicators)
        
        # Momentum strength (from indicators)
        momentum_strength = self._score_momentum(technical_indicators)
        
        # Volatility factor (lower volatility = higher confidence)
        volatility_factor = 1 - min(volatility, 1.0)
        
        # Combine scores
        overall_strength = (
            weights['prediction'] * prediction_strength +
            weights['technical'] * technical_strength +
            weights['momentum'] * momentum_strength +
            weights['volatility'] * volatility_factor
        )
        
        return np.clip(overall_strength, 0, 1)
    
    def _score_technicals(self, indicators: Dict[str, float]) -> float:
        """Score technical indicators (0-1)"""
        
        score = 0.5  # Neutral start
        count = 0
        
        # RSI
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi < 30:  # Oversold
                score += 0.2
            elif rsi > 70:  # Overbought
                score -= 0.2
            count += 1
        
        # MACD
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd']
            signal = indicators['macd_signal']
            if macd > signal:  # Bullish
                score += 0.15
            else:  # Bearish
                score -= 0.15
            count += 1
        
        # Moving averages
        if 'sma_20' in indicators and 'sma_50' in indicators:
            if indicators['sma_20'] > indicators['sma_50']:  # Golden cross
                score += 0.15
            else:  # Death cross
                score -= 0.15
            count += 1
        
        return np.clip(score, 0, 1)
    
    def _score_momentum(self, indicators: Dict[str, float]) -> float:
        """Score momentum indicators (0-1)"""
        
        score = 0.5
        
        if 'return_5d' in indicators:
            ret = indicators['return_5d']
            score += np.clip(ret * 5, -0.3, 0.3)  # Scale to impact
        
        if 'volume_ratio' in indicators:
            vol_ratio = indicators['volume_ratio']
            if vol_ratio > 1.5:  # High volume
                score += 0.1
            elif vol_ratio < 0.5:  # Low volume
                score -= 0.1
        
        return np.clip(score, 0, 1)
    
    def _calculate_targets(self,
                          current_price: float,
                          predicted_price: float,
                          volatility: float,
                          signal: SignalType,
                          time_horizon: str) -> Dict[str, float]:
        """Calculate profit targets and stop loss"""
        
        # Expected move based on volatility
        expected_move = current_price * volatility * self._horizon_factor(time_horizon)
        
        if signal in [SignalType.BUY, SignalType.STRONG_BUY]:
            # Bullish targets
            target_1 = current_price + (expected_move * 0.5)  # Conservative
            target_2 = current_price + (expected_move * 1.0)  # Moderate
            target_3 = current_price + (expected_move * 1.5)  # Aggressive
            stop_loss = current_price - (expected_move * 0.5)  # 50% of expected move
            
        elif signal in [SignalType.SELL, SignalType.STRONG_SELL]:
            # Bearish targets
            target_1 = current_price - (expected_move * 0.5)
            target_2 = current_price - (expected_move * 1.0)
            target_3 = current_price - (expected_move * 1.5)
            stop_loss = current_price + (expected_move * 0.5)
            
        else:  # HOLD
            target_1 = current_price
            target_2 = current_price
            target_3 = current_price
            stop_loss = current_price - (expected_move * 0.3)
        
        return {
            'target_1': target_1,
            'target_2': target_2,
            'target_3': target_3,
            'stop_loss': stop_loss
        }
    
    def _horizon_factor(self, time_horizon: str) -> float:
        """Get time horizon adjustment factor"""
        factors = {
            'short': np.sqrt(1/252),    # Daily
            'medium': np.sqrt(5/252),   # Weekly
            'long': np.sqrt(21/252)     # Monthly
        }
        return factors.get(time_horizon, factors['short'])
    
    def _calculate_position_size(self, signal_strength: float, volatility: float) -> float:
        """
        Calculate recommended position size (0-1, fraction of capital)
        
        Uses Kelly criterion-inspired approach
        """
        # Base position size on signal strength
        base_size = signal_strength * 0.5  # Max 50% of capital
        
        # Adjust for volatility (lower vol = larger position)
        vol_adjustment = 1 - min(volatility, 0.5)
        
        position_size = base_size * vol_adjustment
        
        # Ensure reasonable bounds
        return np.clip(position_size, 0.05, 0.25)  # 5% to 25%
    
    def _calculate_risk_reward(self, entry: float, target: float, stop: float) -> float:
        """Calculate risk-reward ratio"""
        potential_reward = abs(target - entry)
        potential_risk = abs(entry - stop)
        
        if potential_risk == 0:
            return 0
        
        return potential_reward / potential_risk


class OptionsStrategyRecommender:
    """Recommend options strategies based on market outlook"""
    
    def __init__(self):
        """Initialize strategy recommender"""
        self.logger = logging.getLogger(__name__)
    
    def recommend_strategy(self,
                          market_outlook: str,
                          implied_volatility: float,
                          time_to_expiration: int,
                          current_price: float,
                          risk_tolerance: str = "moderate") -> Dict[str, any]:
        """
        Recommend options strategy
        
        Args:
            market_outlook: 'bullish', 'bearish', 'neutral', 'volatile'
            implied_volatility: Current IV level
            time_to_expiration: Days to expiration
            current_price: Current stock price
            risk_tolerance: 'conservative', 'moderate', 'aggressive'
            
        Returns:
            Strategy recommendation with parameters
        """
        
        if market_outlook == 'bullish':
            if risk_tolerance == 'aggressive':
                strategy = self._long_call_strategy(current_price, time_to_expiration)
            else:
                strategy = self._bull_call_spread_strategy(current_price, time_to_expiration)
        
        elif market_outlook == 'bearish':
            if risk_tolerance == 'aggressive':
                strategy = self._long_put_strategy(current_price, time_to_expiration)
            else:
                strategy = self._bear_put_spread_strategy(current_price, time_to_expiration)
        
        elif market_outlook == 'neutral':
            if implied_volatility > 0.3:  # High IV
                strategy = self._iron_condor_strategy(current_price, time_to_expiration)
            else:
                strategy = self._covered_call_strategy(current_price, time_to_expiration)
        
        elif market_outlook == 'volatile':
            strategy = self._straddle_strategy(current_price, time_to_expiration)
        
        else:
            raise ValueError(f"Unknown market outlook: {market_outlook}")
        
        strategy['implied_volatility'] = implied_volatility
        strategy['time_to_expiration'] = time_to_expiration
        
        return strategy
    
    def _long_call_strategy(self, current_price: float, dte: int) -> Dict:
        """Long call strategy"""
        strike = current_price * 1.05  # 5% OTM
        
        return {
            'strategy': 'Long Call',
            'legs': [
                {'action': 'buy', 'option_type': 'call', 'strike': strike, 'quantity': 1}
            ],
            'max_profit': 'Unlimited',
            'max_loss': 'Premium paid',
            'breakeven': 'Strike + Premium',
            'recommended_dte': min(dte, 60)
        }
    
    def _bull_call_spread_strategy(self, current_price: float, dte: int) -> Dict:
        """Bull call spread"""
        lower_strike = current_price * 1.02  # 2% OTM
        upper_strike = current_price * 1.08  # 8% OTM
        
        return {
            'strategy': 'Bull Call Spread',
            'legs': [
                {'action': 'buy', 'option_type': 'call', 'strike': lower_strike, 'quantity': 1},
                {'action': 'sell', 'option_type': 'call', 'strike': upper_strike, 'quantity': 1}
            ],
            'max_profit': f'${upper_strike - lower_strike} - Net Premium',
            'max_loss': 'Net premium paid',
            'breakeven': 'Lower strike + Net premium',
            'recommended_dte': min(dte, 45)
        }
    
    def _long_put_strategy(self, current_price: float, dte: int) -> Dict:
        """Long put strategy"""
        strike = current_price * 0.95  # 5% OTM
        
        return {
            'strategy': 'Long Put',
            'legs': [
                {'action': 'buy', 'option_type': 'put', 'strike': strike, 'quantity': 1}
            ],
            'max_profit': 'Strike - Premium',
            'max_loss': 'Premium paid',
            'breakeven': 'Strike - Premium',
            'recommended_dte': min(dte, 60)
        }
    
    def _bear_put_spread_strategy(self, current_price: float, dte: int) -> Dict:
        """Bear put spread"""
        upper_strike = current_price * 0.98  # 2% OTM
        lower_strike = current_price * 0.92  # 8% OTM
        
        return {
            'strategy': 'Bear Put Spread',
            'legs': [
                {'action': 'buy', 'option_type': 'put', 'strike': upper_strike, 'quantity': 1},
                {'action': 'sell', 'option_type': 'put', 'strike': lower_strike, 'quantity': 1}
            ],
            'max_profit': f'${upper_strike - lower_strike} - Net Premium',
            'max_loss': 'Net premium paid',
            'breakeven': 'Upper strike - Net premium',
            'recommended_dte': min(dte, 45)
        }
    
    def _iron_condor_strategy(self, current_price: float, dte: int) -> Dict:
        """Iron condor for neutral market with high IV"""
        put_lower = current_price * 0.90
        put_upper = current_price * 0.95
        call_lower = current_price * 1.05
        call_upper = current_price * 1.10
        
        return {
            'strategy': 'Iron Condor',
            'legs': [
                {'action': 'sell', 'option_type': 'put', 'strike': put_upper, 'quantity': 1},
                {'action': 'buy', 'option_type': 'put', 'strike': put_lower, 'quantity': 1},
                {'action': 'sell', 'option_type': 'call', 'strike': call_lower, 'quantity': 1},
                {'action': 'buy', 'option_type': 'call', 'strike': call_upper, 'quantity': 1}
            ],
            'max_profit': 'Net premium received',
            'max_loss': 'Width of spread - Net premium',
            'breakeven': 'Two breakevens',
            'recommended_dte': min(dte, 30)
        }
    
    def _covered_call_strategy(self, current_price: float, dte: int) -> Dict:
        """Covered call for income generation"""
        strike = current_price * 1.05  # 5% OTM
        
        return {
            'strategy': 'Covered Call',
            'legs': [
                {'action': 'own', 'type': 'stock', 'quantity': 100},
                {'action': 'sell', 'option_type': 'call', 'strike': strike, 'quantity': 1}
            ],
            'max_profit': 'Strike - Stock cost + Premium',
            'max_loss': 'Stock cost - Premium',
            'breakeven': 'Stock cost - Premium',
            'recommended_dte': min(dte, 30)
        }
    
    def _straddle_strategy(self, current_price: float, dte: int) -> Dict:
        """Straddle for volatile markets"""
        strike = current_price  # ATM
        
        return {
            'strategy': 'Long Straddle',
            'legs': [
                {'action': 'buy', 'option_type': 'call', 'strike': strike, 'quantity': 1},
                {'action': 'buy', 'option_type': 'put', 'strike': strike, 'quantity': 1}
            ],
            'max_profit': 'Unlimited',
            'max_loss': 'Total premium paid',
            'breakeven': 'Strike ± Total premium',
            'recommended_dte': min(dte, 45)
        }
