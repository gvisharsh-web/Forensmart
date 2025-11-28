"""
UTILS MODULE - Shared Utilities with Error Handling Loopholes
Provides common utilities with automatic error handling

This module provides:
- Error handling loopholes (automatic error recovery)
- Input validation
- Caching utilities
- Retry mechanisms
- Common helper functions
"""

import os
import json
import logging
import time
from typing import Optional, Dict, Any, Callable, List
from functools import wraps
from datetime import datetime, timedelta

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# ERROR HANDLING LOOPHOLES
# ============================================================================

class ErrorHandlingLoopholes:
    """Automatic error handling and recovery"""

    @staticmethod
    def is_error_handling_enabled() -> bool:
        """Check if error handling loopholes are enabled"""
        return os.getenv('ERROR_HANDLING_ENABLED', 'true').lower() == 'true'

    @staticmethod
    def auto_retry_on_error(
        func: Callable,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        *args,
        **kwargs
    ) -> Any:
        """
        Automatically retry function on error with exponential backoff
        
        Args:
            func: Function to retry
            max_attempts: Maximum retry attempts
            delay: Initial delay between retries
            backoff: Backoff multiplier
            *args, **kwargs: Function arguments
        
        Returns:
            Function result or None on failure
        """
        
        if not ErrorHandlingLoopholes.is_error_handling_enabled():
            return func(*args, **kwargs)
        
        current_delay = delay
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_attempts}: {func.__name__}")
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Success on attempt {attempt + 1}")
                return result
            
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_attempts - 1:
                    logger.info(f"Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
        
        logger.error(f"All {max_attempts} attempts failed: {str(last_error)}")
        return None

    @staticmethod
    def safe_execute(
        func: Callable,
        default_return: Any = None,
        log_error: bool = True,
        *args,
        **kwargs
    ) -> Any:
        """
        Safely execute function with automatic error handling
        
        Args:
            func: Function to execute
            default_return: Default return value on error
            log_error: Whether to log errors
            *args, **kwargs: Function arguments
        
        Returns:
            Function result or default_return on error
        """
        
        if not ErrorHandlingLoopholes.is_error_handling_enabled():
            return func(*args, **kwargs)
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_error:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            return default_return

    @staticmethod
    def handle_missing_data(
        data: Any,
        default: Any = None,
        key: Optional[str] = None
    ) -> Any:
        """
        Handle missing data gracefully
        
        Args:
            data: Data to check
            default: Default value if missing
            key: Dictionary key to check
        
        Returns:
            Data value or default
        """
        
        try:
            if data is None:
                return default
            
            if key and isinstance(data, dict):
                return data.get(key, default)
            
            return data
        except Exception as e:
            logger.warning(f"Error handling missing data: {str(e)}")
            return default

    @staticmethod
    def validate_input(
        value: Any,
        expected_type: type,
        allow_none: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> bool:
        """
        Validate input with automatic error handling
        
        Args:
            value: Value to validate
            expected_type: Expected type
            allow_none: Allow None values
            min_length: Minimum length (for strings/lists)
            max_length: Maximum length (for strings/lists)
        
        Returns:
            True if valid, False otherwise
        """
        
        try:
            # Check None
            if value is None:
                return allow_none
            
            # Check type
            if not isinstance(value, expected_type):
                logger.warning(f"Type mismatch: expected {expected_type}, got {type(value)}")
                return False
            
            # Check length
            if hasattr(value, '__len__'):
                if min_length and len(value) < min_length:
                    logger.warning(f"Value too short: {len(value)} < {min_length}")
                    return False
                
                if max_length and len(value) > max_length:
                    logger.warning(f"Value too long: {len(value)} > {max_length}")
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False


# ============================================================================
# CACHING UTILITIES
# ============================================================================

class CacheManager:
    """Simple caching with automatic error handling"""

    def __init__(self, cache_dir: str = "cache", ttl_seconds: int = 3600):
        """Initialize cache manager"""
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        try:
            # Check memory cache first
            if key in self.memory_cache:
                cache_entry = self.memory_cache[key]
                if datetime.now() < cache_entry['expiry']:
                    logger.debug(f"Cache hit (memory): {key}")
                    return cache_entry['value']
                else:
                    del self.memory_cache[key]
            
            # Check file cache
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache_entry = json.load(f)
                
                if datetime.fromisoformat(cache_entry['expiry']) > datetime.now():
                    logger.debug(f"Cache hit (file): {key}")
                    return cache_entry['value']
                else:
                    os.remove(cache_file)
            
            logger.debug(f"Cache miss: {key}")
            return None
        
        except Exception as e:
            logger.warning(f"Cache get error: {str(e)}")
            return None

    def set(self, key: str, value: Any) -> bool:
        """Set cached value"""
        try:
            # Store in memory cache
            self.memory_cache[key] = {
                'value': value,
                'expiry': datetime.now() + timedelta(seconds=self.ttl_seconds)
            }
            
            # Store in file cache
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            cache_entry = {
                'value': value,
                'expiry': (datetime.now() + timedelta(seconds=self.ttl_seconds)).isoformat(),
                'created': datetime.now().isoformat()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_entry, f)
            
            logger.debug(f"Cache set: {key}")
            return True
        
        except Exception as e:
            logger.warning(f"Cache set error: {str(e)}")
            return False

    def clear(self, key: Optional[str] = None) -> bool:
        """Clear cache"""
        try:
            if key:
                # Clear specific key
                if key in self.memory_cache:
                    del self.memory_cache[key]
                
                cache_file = os.path.join(self.cache_dir, f"{key}.json")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                
                logger.debug(f"Cache cleared: {key}")
            else:
                # Clear all cache
                self.memory_cache.clear()
                
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.json'):
                        os.remove(os.path.join(self.cache_dir, file))
                
                logger.debug("All cache cleared")
            
            return True
        
        except Exception as e:
            logger.warning(f"Cache clear error: {str(e)}")
            return False


# ============================================================================
# ARTIFACT PATH BUILDER
# ============================================================================

class ArtifactPathBuilder:
    """Build artifact paths with error handling"""

    BASE_DIR = "artifacts"

    @classmethod
    def resolve(
        cls,
        case_id: Optional[str],
        *segments: str,
        ensure_dir: bool = False,
        ensure_parent: bool = False
    ) -> str:
        """Resolve artifact path"""
        try:
            safe_case = (case_id or "default_case").strip() or "default_case"
            path = os.path.join(cls.BASE_DIR, safe_case, *segments)
            
            if ensure_dir:
                os.makedirs(path, exist_ok=True)
            elif ensure_parent:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            
            return path
        
        except Exception as e:
            logger.error(f"Error resolving artifact path: {str(e)}")
            return cls.BASE_DIR


# ============================================================================
# RESULTS REPOSITORY
# ============================================================================

class ResultsRepository:
    """Manage extraction results with error handling"""

    @staticmethod
    def save(case_id: str, results: Dict[str, Any]) -> bool:
        """Save results"""
        try:
            path = ArtifactPathBuilder.resolve(case_id, ensure_dir=True)
            results_file = os.path.join(path, "results.json")
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved: {case_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False

    @staticmethod
    def load(case_id: str) -> Optional[Dict[str, Any]]:
        """Load results"""
        try:
            path = ArtifactPathBuilder.resolve(case_id)
            results_file = os.path.join(path, "results.json")
            
            if not os.path.exists(results_file):
                logger.warning(f"Results file not found: {case_id}")
                return None
            
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Results loaded: {case_id}")
            return results
        
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            return None

    @staticmethod
    def delete(case_id: str) -> bool:
        """Delete results"""
        try:
            path = ArtifactPathBuilder.resolve(case_id)
            results_file = os.path.join(path, "results.json")
            
            if os.path.exists(results_file):
                os.remove(results_file)
                logger.info(f"Results deleted: {case_id}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error deleting results: {str(e)}")
            return False


# ============================================================================
# GLOBAL CACHE INSTANCE
# ============================================================================

_cache_manager: Optional[CacheManager] = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
