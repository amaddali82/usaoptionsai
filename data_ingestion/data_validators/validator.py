"""
Data Validators for quality checks
"""
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime


class DataValidator:
    """Validates data quality before processing"""
    
    def __init__(self):
        """Initialize validator"""
        self.logger = logging.getLogger(__name__)
    
    def validate_options_chain(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate options chain data
        
        Args:
            data: Options chain data
            
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        required_fields = ['symbol', 'data']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if 'data' in data:
            chain_data = data['data']
            
            # Validate option contracts
            if 'results' in chain_data:
                for contract in chain_data['results']:
                    # Check strike price
                    if 'strike_price' not in contract or contract['strike_price'] <= 0:
                        errors.append(f"Invalid strike price in contract")
                    
                    # Check expiration date
                    if 'expiration_date' not in contract:
                        errors.append(f"Missing expiration date")
                    
                    # Check contract type
                    if 'contract_type' not in contract or contract['contract_type'] not in ['call', 'put']:
                        errors.append(f"Invalid contract type")
        
        return (len(errors) == 0, errors)
    
    def validate_option_quote(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate option quote data
        
        Args:
            data: Option quote data
            
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check bid/ask spread
        if 'bid' in data and 'ask' in data:
            bid = data['bid']
            ask = data['ask']
            
            if bid < 0 or ask < 0:
                errors.append("Negative bid or ask price")
            
            if bid > ask:
                errors.append(f"Bid ({bid}) > Ask ({ask})")
            
            # Check for suspicious spreads (>50% of mid-price)
            if ask > 0:
                spread_pct = (ask - bid) / ((ask + bid) / 2) * 100
                if spread_pct > 50:
                    errors.append(f"Suspicious spread: {spread_pct:.2f}%")
        
        # Check volume
        if 'volume' in data and data['volume'] < 0:
            errors.append("Negative volume")
        
        # Check implied volatility
        if 'implied_volatility' in data:
            iv = data['implied_volatility']
            if iv < 0 or iv > 5:  # IV > 500% is suspicious
                errors.append(f"Suspicious implied volatility: {iv}")
        
        return (len(errors) == 0, errors)
    
    def validate_stock_price(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate stock price data
        
        Args:
            data: Stock price data
            
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check OHLC relationship
        if all(k in data for k in ['open', 'high', 'low', 'close']):
            o, h, l, c = data['open'], data['high'], data['low'], data['close']
            
            if any(v <= 0 for v in [o, h, l, c]):
                errors.append("Non-positive OHLC values")
            
            if l > h:
                errors.append(f"Low ({l}) > High ({h})")
            
            if not (l <= o <= h):
                errors.append(f"Open ({o}) not between Low ({l}) and High ({h})")
            
            if not (l <= c <= h):
                errors.append(f"Close ({c}) not between Low ({l}) and High ({h})")
        
        # Check volume
        if 'volume' in data:
            vol = data['volume']
            if vol < 0:
                errors.append("Negative volume")
            
            # Warn on zero volume
            if vol == 0:
                errors.append("Zero volume (may be non-trading period)")
        
        return (len(errors) == 0, errors)
    
    def validate_greeks(self, greeks: Dict[str, float]) -> tuple[bool, List[str]]:
        """
        Validate option Greeks
        
        Args:
            greeks: Greeks data
            
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Delta should be between -1 and 1 (0-1 for calls, -1-0 for puts)
        if 'delta' in greeks:
            delta = greeks['delta']
            if not (-1 <= delta <= 1):
                errors.append(f"Delta ({delta}) out of range [-1, 1]")
        
        # Gamma should be non-negative
        if 'gamma' in greeks:
            gamma = greeks['gamma']
            if gamma < 0:
                errors.append(f"Negative gamma ({gamma})")
        
        # Vega should be non-negative
        if 'vega' in greeks:
            vega = greeks['vega']
            if vega < 0:
                errors.append(f"Negative vega ({vega})")
        
        # Theta is typically negative (time decay)
        if 'theta' in greeks:
            theta = greeks['theta']
            if theta > 1:  # Some positive theta is possible but rare
                errors.append(f"Unusually high positive theta ({theta})")
        
        return (len(errors) == 0, errors)
    
    def check_for_outliers(self, df: pd.DataFrame, column: str,
                          method: str = 'iqr', threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers in a DataFrame column
        
        Args:
            df: DataFrame
            column: Column name
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean series indicating outliers
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = z_scores > threshold
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if outliers.any():
            self.logger.warning(f"Found {outliers.sum()} outliers in {column}")
        
        return outliers
    
    def validate_timestamp(self, timestamp_str: str) -> tuple[bool, Optional[str]]:
        """
        Validate timestamp format and recency
        
        Args:
            timestamp_str: ISO format timestamp string
            
        Returns:
            (is_valid, error_message)
        """
        try:
            ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            # Check if timestamp is too old (> 1 day for real-time data)
            age = (datetime.utcnow() - ts.replace(tzinfo=None)).total_seconds()
            if age > 86400:  # 24 hours
                return (False, f"Timestamp too old: {age/3600:.1f} hours")
            
            # Check if timestamp is in the future
            if age < 0:
                return (False, f"Timestamp in future: {-age/60:.1f} minutes")
            
            return (True, None)
            
        except Exception as e:
            return (False, f"Invalid timestamp format: {e}")
    
    def validate_batch(self, data_list: List[Dict[str, Any]],
                      data_type: str) -> Dict[str, Any]:
        """
        Validate a batch of data
        
        Args:
            data_list: List of data items
            data_type: Type of data ('options_chain', 'quote', 'price', 'greeks')
            
        Returns:
            Dictionary with validation summary
        """
        validation_methods = {
            'options_chain': self.validate_options_chain,
            'quote': self.validate_option_quote,
            'price': self.validate_stock_price,
            'greeks': self.validate_greeks
        }
        
        if data_type not in validation_methods:
            raise ValueError(f"Unknown data type: {data_type}")
        
        validate_func = validation_methods[data_type]
        
        total = len(data_list)
        valid_count = 0
        all_errors = []
        
        for i, data in enumerate(data_list):
            is_valid, errors = validate_func(data)
            if is_valid:
                valid_count += 1
            else:
                all_errors.append({
                    'index': i,
                    'errors': errors,
                    'data_sample': str(data)[:100]
                })
        
        invalid_count = total - valid_count
        
        summary = {
            'total': total,
            'valid': valid_count,
            'invalid': invalid_count,
            'success_rate': valid_count / total * 100 if total > 0 else 0,
            'errors': all_errors
        }
        
        self.logger.info(
            f"Validated {total} {data_type} records: "
            f"{valid_count} valid, {invalid_count} invalid "
            f"({summary['success_rate']:.1f}% success rate)"
        )
        
        return summary
