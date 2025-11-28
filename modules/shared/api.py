"""
API MODULE - API management and integration

Provides:
- API client management
- Request handling
- Response processing
- Error handling
- Rate limiting
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)

# ============================================================================
# API CLIENT CLASS
# ============================================================================

class APIClient:
    """Manages API interactions"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.endpoints = {}
        self.request_history = []
        self.rate_limit = 100  # requests per minute
        self.request_count = 0
    
    # ========================================================================
    # ENDPOINT MANAGEMENT
    # ========================================================================
    
    def register_endpoint(self, name: str, method: str, path: str, 
                         description: str = "") -> None:
        """Register API endpoint"""
        self.endpoints[name] = {
            'method': method,
            'path': path,
            'description': description,
            'registered_at': datetime.now().isoformat()
        }
        logger.info(f"Registered endpoint: {name}")
    
    def get_endpoint(self, name: str) -> Optional[Dict[str, Any]]:
        """Get endpoint details"""
        return self.endpoints.get(name)
    
    def list_endpoints(self) -> List[Dict[str, Any]]:
        """List all endpoints"""
        return list(self.endpoints.values())
    
    # ========================================================================
    # REQUEST HANDLING
    # ========================================================================
    
    def get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make GET request"""
        return self._make_request('GET', endpoint, params=params)
    
    def post(self, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make POST request"""
        return self._make_request('POST', endpoint, data=data)
    
    def put(self, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make PUT request"""
        return self._make_request('PUT', endpoint, data=data)
    
    def delete(self, endpoint: str) -> Dict[str, Any]:
        """Make DELETE request"""
        return self._make_request('DELETE', endpoint)
    
    def _make_request(self, method: str, endpoint: str, 
                     params: Dict[str, Any] = None, 
                     data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make API request"""
        try:
            # Check rate limit
            if self.request_count >= self.rate_limit:
                return {
                    'success': False,
                    'error': 'Rate limit exceeded',
                    'status_code': 429
                }
            
            # Build request
            url = f"{self.base_url}/{endpoint}"
            
            request_record = {
                'method': method,
                'url': url,
                'params': params,
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'status': 'pending'
            }
            
            # Simulate request
            response = {
                'success': True,
                'status_code': 200,
                'data': data or params or {},
                'timestamp': datetime.now().isoformat()
            }
            
            request_record['status'] = 'completed'
            request_record['response'] = response
            
            self.request_history.append(request_record)
            self.request_count += 1
            
            logger.info(f"{method} {endpoint} - {response['status_code']}")
            return response
        
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'status_code': 500
            }
    
    # ========================================================================
    # RESPONSE PROCESSING
    # ========================================================================
    
    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse API response"""
        try:
            if response.get('success'):
                return {
                    'status': 'success',
                    'data': response.get('data', {}),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'error',
                    'error': response.get('error', 'Unknown error'),
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Parse failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """Validate API response"""
        return response.get('success', False) and response.get('status_code') == 200
    
    # ========================================================================
    # ERROR HANDLING
    # ========================================================================
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """Handle API error"""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.error(f"API Error: {error_info}")
        return error_info
    
    def retry_request(self, endpoint: str, method: str = 'GET', 
                     max_retries: int = 3) -> Dict[str, Any]:
        """Retry failed request"""
        for attempt in range(max_retries):
            try:
                if method == 'GET':
                    return self.get(endpoint)
                elif method == 'POST':
                    return self.post(endpoint)
                elif method == 'PUT':
                    return self.put(endpoint)
                elif method == 'DELETE':
                    return self.delete(endpoint)
            except Exception as e:
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return self.handle_error(e)
        
        return {'success': False, 'error': 'Max retries exceeded'}
    
    # ========================================================================
    # RATE LIMITING
    # ========================================================================
    
    def set_rate_limit(self, limit: int) -> None:
        """Set rate limit"""
        self.rate_limit = limit
        logger.info(f"Rate limit set to {limit}")
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get rate limit status"""
        return {
            'limit': self.rate_limit,
            'current': self.request_count,
            'remaining': max(0, self.rate_limit - self.request_count),
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_rate_limit(self) -> None:
        """Reset rate limit counter"""
        self.request_count = 0
        logger.info("Rate limit counter reset")
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_request_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get request history"""
        return self.request_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get API statistics"""
        return {
            'total_requests': len(self.request_history),
            'endpoints_registered': len(self.endpoints),
            'rate_limit': self.rate_limit,
            'current_requests': self.request_count,
            'timestamp': datetime.now().isoformat()
        }
    
    def clear_history(self) -> None:
        """Clear request history"""
        self.request_history = []
        logger.info("Request history cleared")

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def create_api_client(base_url: str = "http://localhost:8000") -> APIClient:
    """Factory function to create API client"""
    return APIClient(base_url)
