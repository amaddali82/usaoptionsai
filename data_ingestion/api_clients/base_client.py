"""
Base API Client with common functionality
"""
import time
import logging
from typing import Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from cachetools import TTLCache


class BaseAPIClient:
    """Base class for all API clients with retry logic and caching"""
    
    def __init__(self, api_key: str, base_url: str, rate_limit: int = 5, 
                 timeout: int = 30, cache_enabled: bool = True, cache_ttl: int = 300):
        """
        Initialize base API client
        
        Args:
            api_key: API authentication key
            base_url: Base URL for API endpoints
            rate_limit: Maximum requests per second
            timeout: Request timeout in seconds
            cache_enabled: Enable response caching
            cache_ttl: Cache time-to-live in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup session with retry logic
        self.session = self._create_session()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0 / rate_limit if rate_limit > 0 else 0
        
        # Caching
        self.cache_enabled = cache_enabled
        if cache_enabled:
            self.cache = TTLCache(maxsize=1000, ttl=cache_ttl)
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry logic"""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        if self.min_request_interval > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
            self.last_request_time = time.time()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if available"""
        if self.cache_enabled and cache_key in self.cache:
            self.logger.debug(f"Cache hit for key: {cache_key}")
            return self.cache[cache_key]
        return None
    
    def _save_to_cache(self, cache_key: str, data: Any):
        """Save data to cache"""
        if self.cache_enabled:
            self.cache[cache_key] = data
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None,
                     method: str = "GET", use_cache: bool = True) -> Dict[str, Any]:
        """
        Make HTTP request with rate limiting and caching
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            method: HTTP method
            use_cache: Whether to use cached response
            
        Returns:
            Response data as dictionary
            
        Raises:
            requests.RequestException: On request failure
        """
        # Prepare request
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        params = params or {}
        
        # Check cache
        cache_key = f"{method}:{url}:{str(sorted(params.items()))}"
        if use_cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Rate limiting
        self._rate_limit()
        
        # Make request
        try:
            self.logger.debug(f"Making {method} request to {url}")
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Cache response
            if use_cache:
                self._save_to_cache(cache_key, data)
            
            return data
            
        except requests.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if API is accessible"""
        try:
            # Override in subclass with actual health check endpoint
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
