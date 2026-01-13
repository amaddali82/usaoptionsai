"""
Monte Carlo Simulation for Options Pricing
Handles European and exotic options
"""
import numpy as np
import logging
from typing import Dict, Callable, Optional


class MonteCarloSimulator:
    """Monte Carlo simulation for option pricing"""
    
    def __init__(self, num_simulations: int = 10000, seed: Optional[int] = 42):
        """
        Initialize Monte Carlo simulator
        
        Args:
            num_simulations: Number of price paths to simulate
            seed: Random seed for reproducibility
        """
        self.num_simulations = num_simulations
        self.logger = logging.getLogger(__name__)
        
        if seed is not None:
            np.random.seed(seed)
    
    def simulate_gbm_paths(self, S0: float, r: float, sigma: float, T: float,
                          steps: int, q: float = 0, 
                          antithetic: bool = True) -> np.ndarray:
        """
        Simulate stock price paths using Geometric Brownian Motion
        
        Args:
            S0: Initial stock price
            r: Risk-free rate
            sigma: Volatility
            T: Time horizon (years)
            steps: Number of time steps
            q: Dividend yield
            antithetic: Use antithetic variates for variance reduction
            
        Returns:
            Array of shape (num_simulations, steps+1) with price paths
        """
        dt = T / steps
        num_sims = self.num_simulations // 2 if antithetic else self.num_simulations
        
        # Generate random normal variables
        Z = np.random.standard_normal((num_sims, steps))
        
        if antithetic:
            # Create antithetic paths
            Z = np.concatenate([Z, -Z], axis=0)
        
        # Pre-compute drift and diffusion
        drift = (r - q - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Generate paths
        paths = np.zeros((self.num_simulations, steps + 1))
        paths[:, 0] = S0
        
        for t in range(1, steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion * Z[:, t-1])
        
        return paths
    
    def european_option_price(self, S0: float, K: float, T: float, r: float,
                             sigma: float, option_type: str = 'call',
                             q: float = 0) -> Dict[str, float]:
        """
        Price European option using Monte Carlo
        
        Returns:
            Dictionary with price, standard error, and confidence interval
        """
        paths = self.simulate_gbm_paths(S0, r, sigma, T, steps=1, q=q)
        ST = paths[:, -1]  # Terminal prices
        
        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # Discount to present value
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        # Calculate statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.num_simulations)
        
        # 95% confidence interval
        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error
        
        return {
            'price': price,
            'std_error': std_error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    def asian_option_price(self, S0: float, K: float, T: float, r: float,
                          sigma: float, option_type: str = 'call',
                          q: float = 0, steps: int = 252) -> Dict[str, float]:
        """
        Price Asian (average price) option
        
        Args:
            steps: Number of averaging points
        """
        paths = self.simulate_gbm_paths(S0, r, sigma, T, steps, q)
        
        # Calculate arithmetic average
        avg_prices = np.mean(paths, axis=1)
        
        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(avg_prices - K, 0)
        else:
            payoffs = np.maximum(K - avg_prices, 0)
        
        # Discount to present value
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.num_simulations)
        
        return {
            'price': price,
            'std_error': std_error,
            'ci_lower': price - 1.96 * std_error,
            'ci_upper': price + 1.96 * std_error
        }
    
    def barrier_option_price(self, S0: float, K: float, T: float, r: float,
                            sigma: float, barrier: float, barrier_type: str = 'down-out',
                            option_type: str = 'call', q: float = 0,
                            steps: int = 252) -> Dict[str, float]:
        """
        Price barrier option
        
        Args:
            barrier: Barrier level
            barrier_type: 'down-out', 'down-in', 'up-out', 'up-in'
        """
        paths = self.simulate_gbm_paths(S0, r, sigma, T, steps, q)
        ST = paths[:, -1]
        
        # Check barrier conditions
        if barrier_type == 'down-out':
            # Knocked out if price goes below barrier
            barrier_hit = np.any(paths <= barrier, axis=1)
        elif barrier_type == 'down-in':
            # Activated if price goes below barrier
            barrier_hit = np.any(paths <= barrier, axis=1)
        elif barrier_type == 'up-out':
            # Knocked out if price goes above barrier
            barrier_hit = np.any(paths >= barrier, axis=1)
        elif barrier_type == 'up-in':
            # Activated if price goes above barrier
            barrier_hit = np.any(paths >= barrier, axis=1)
        else:
            raise ValueError(f"Unknown barrier type: {barrier_type}")
        
        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # Apply barrier conditions
        if 'out' in barrier_type:
            # Knock-out: payoff is zero if barrier hit
            payoffs = np.where(barrier_hit, 0, payoffs)
        else:
            # Knock-in: payoff is zero if barrier NOT hit
            payoffs = np.where(barrier_hit, payoffs, 0)
        
        # Discount to present value
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.num_simulations)
        
        return {
            'price': price,
            'std_error': std_error,
            'ci_lower': price - 1.96 * std_error,
            'ci_upper': price + 1.96 * std_error
        }
    
    def digital_option_price(self, S0: float, K: float, T: float, r: float,
                            sigma: float, payout: float = 1.0,
                            option_type: str = 'call', q: float = 0) -> Dict[str, float]:
        """
        Price digital (binary) option with fixed payout
        """
        paths = self.simulate_gbm_paths(S0, r, sigma, T, steps=1, q=q)
        ST = paths[:, -1]
        
        # Binary payoff
        if option_type == 'call':
            payoffs = np.where(ST > K, payout, 0)
        else:
            payoffs = np.where(ST < K, payout, 0)
        
        # Discount to present value
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.num_simulations)
        
        return {
            'price': price,
            'std_error': std_error,
            'ci_lower': price - 1.96 * std_error,
            'ci_upper': price + 1.96 * std_error
        }
    
    def expected_move(self, S0: float, T: float, sigma: float,
                     confidence_level: float = 0.68) -> Dict[str, float]:
        """
        Calculate expected move based on implied volatility
        
        Args:
            S0: Current price
            T: Time to expiration (years)
            sigma: Implied volatility
            confidence_level: Confidence level (0.68 = 1 std dev, 0.95 = 2 std dev)
            
        Returns:
            Expected move range
        """
        # Standard deviation of returns
        std_dev = sigma * np.sqrt(T)
        
        # Z-score for confidence level
        from scipy.stats import norm
        z_score = norm.ppf((1 + confidence_level) / 2)
        
        move = S0 * std_dev * z_score
        
        return {
            'current_price': S0,
            'upper_bound': S0 + move,
            'lower_bound': S0 - move,
            'move_dollars': move,
            'move_percent': (move / S0) * 100,
            'confidence_level': confidence_level
        }
    
    def value_at_risk(self, S0: float, T: float, r: float, sigma: float,
                     confidence_level: float = 0.95, position_size: float = 1.0,
                     q: float = 0) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) using Monte Carlo
        
        Args:
            position_size: Number of shares/contracts
            confidence_level: VaR confidence level (e.g., 0.95 = 95%)
            
        Returns:
            VaR statistics
        """
        paths = self.simulate_gbm_paths(S0, r, sigma, T, steps=1, q=q)
        ST = paths[:, -1]
        
        # Calculate portfolio values
        portfolio_values = position_size * ST
        initial_value = position_size * S0
        
        # Calculate returns
        returns = portfolio_values - initial_value
        
        # Sort returns
        sorted_returns = np.sort(returns)
        
        # Find VaR at confidence level
        var_index = int((1 - confidence_level) * self.num_simulations)
        var = -sorted_returns[var_index]
        
        # Calculate Conditional VaR (CVaR or Expected Shortfall)
        cvar = -np.mean(sorted_returns[:var_index])
        
        return {
            'var': var,
            'cvar': cvar,
            'confidence_level': confidence_level,
            'var_percent': (var / initial_value) * 100,
            'cvar_percent': (cvar / initial_value) * 100
        }
