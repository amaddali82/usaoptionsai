"""
Black-Scholes Option Pricing Model
Analytical pricing for European options
"""
import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple
import logging


class BlackScholesModel:
    """Black-Scholes option pricing and Greeks calculation"""
    
    def __init__(self):
        """Initialize Black-Scholes model"""
        self.logger = logging.getLogger(__name__)
    
    def _d1(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate d1 parameter"""
        return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    def _d2(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate d2 parameter"""
        return self._d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)
    
    def call_price(self, S: float, K: float, T: float, r: float, 
                   sigma: float, q: float = 0) -> float:
        """
        Calculate European call option price
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            q: Dividend yield
            
        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)
        
        d1 = self._d1(S, K, T, r, sigma, q)
        d2 = self._d2(S, K, T, r, sigma, q)
        
        call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call
    
    def put_price(self, S: float, K: float, T: float, r: float,
                  sigma: float, q: float = 0) -> float:
        """
        Calculate European put option price
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            q: Dividend yield
            
        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S, 0)
        
        d1 = self._d1(S, K, T, r, sigma, q)
        d2 = self._d2(S, K, T, r, sigma, q)
        
        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        return put
    
    def delta(self, S: float, K: float, T: float, r: float,
              sigma: float, option_type: str = 'call', q: float = 0) -> float:
        """
        Calculate option delta (rate of change with respect to underlying price)
        
        Returns:
            Delta (between 0 and 1 for calls, -1 and 0 for puts)
        """
        if T <= 0:
            return 1.0 if S > K and option_type == 'call' else 0.0
        
        d1 = self._d1(S, K, T, r, sigma, q)
        
        if option_type == 'call':
            return np.exp(-q * T) * norm.cdf(d1)
        else:
            return -np.exp(-q * T) * norm.cdf(-d1)
    
    def gamma(self, S: float, K: float, T: float, r: float,
              sigma: float, q: float = 0) -> float:
        """
        Calculate option gamma (rate of change of delta)
        
        Returns:
            Gamma (always positive)
        """
        if T <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma, q)
        
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    
    def vega(self, S: float, K: float, T: float, r: float,
             sigma: float, q: float = 0) -> float:
        """
        Calculate option vega (sensitivity to volatility)
        
        Returns:
            Vega (per 1% change in volatility)
        """
        if T <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma, q)
        
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
        return vega
    
    def theta(self, S: float, K: float, T: float, r: float,
              sigma: float, option_type: str = 'call', q: float = 0) -> float:
        """
        Calculate option theta (time decay)
        
        Returns:
            Theta (per day)
        """
        if T <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma, q)
        d2 = self._d2(S, K, T, r, sigma, q)
        
        if option_type == 'call':
            theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     - r * K * np.exp(-r * T) * norm.cdf(d2)
                     + q * S * np.exp(-q * T) * norm.cdf(d1))
        else:
            theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)
                     - q * S * np.exp(-q * T) * norm.cdf(-d1))
        
        # Convert to per-day
        return theta / 365
    
    def rho(self, S: float, K: float, T: float, r: float,
            sigma: float, option_type: str = 'call', q: float = 0) -> float:
        """
        Calculate option rho (sensitivity to interest rate)
        
        Returns:
            Rho (per 1% change in interest rate)
        """
        if T <= 0:
            return 0.0
        
        d2 = self._d2(S, K, T, r, sigma, q)
        
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return rho
    
    def all_greeks(self, S: float, K: float, T: float, r: float,
                   sigma: float, option_type: str = 'call', q: float = 0) -> Dict[str, float]:
        """
        Calculate all Greeks at once
        
        Returns:
            Dictionary with all Greeks
        """
        return {
            'delta': self.delta(S, K, T, r, sigma, option_type, q),
            'gamma': self.gamma(S, K, T, r, sigma, q),
            'vega': self.vega(S, K, T, r, sigma, q),
            'theta': self.theta(S, K, T, r, sigma, option_type, q),
            'rho': self.rho(S, K, T, r, sigma, option_type, q)
        }
    
    def implied_volatility(self, market_price: float, S: float, K: float,
                          T: float, r: float, option_type: str = 'call',
                          q: float = 0, tol: float = 1e-6, max_iter: int = 100) -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Args:
            market_price: Observed market price of option
            S, K, T, r, q: Black-Scholes parameters
            option_type: 'call' or 'put'
            tol: Convergence tolerance
            max_iter: Maximum iterations
            
        Returns:
            Implied volatility
        """
        # Initial guess
        sigma = 0.3
        
        for i in range(max_iter):
            if option_type == 'call':
                price = self.call_price(S, K, T, r, sigma, q)
            else:
                price = self.put_price(S, K, T, r, sigma, q)
            
            vega_val = self.vega(S, K, T, r, sigma, q) * 100  # Convert back
            
            diff = market_price - price
            
            if abs(diff) < tol:
                return sigma
            
            if vega_val == 0:
                break
            
            sigma = sigma + diff / vega_val
            
            # Keep sigma positive and reasonable
            sigma = max(0.01, min(sigma, 5.0))
        
        self.logger.warning(f"IV calculation did not converge after {max_iter} iterations")
        return sigma
