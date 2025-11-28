"""
ERROR HANDLING WRAPPER FOR ANALYSIS MODULES

Integrates error handling with:
- Communications Analyzer
- Location Intelligence
- Media Viewer
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)

# ============================================================================
# ERROR HANDLING WRAPPER CLASS
# ============================================================================

class AnalysisErrorHandler:
    """Wraps analysis operations with error handling"""
    
    def __init__(self):
        try:
            from modules.error_handling import ErrorHandlingSystem
            self.error_system = ErrorHandlingSystem()
            self.available = True
        except ImportError:
            logger.warning("Error handling system not available")
            self.available = False
        
        self.error_log = []
    
    # ========================================================================
    # DECORATOR FOR ERROR HANDLING
    # ========================================================================
    
    def handle_analysis_errors(self, analysis_type: str):
        """Decorator to handle errors in analysis operations"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Dict[str, Any]:
                try:
                    logger.info(f"Starting {analysis_type} analysis")
                    result = func(*args, **kwargs)
                    
                    # Log successful analysis
                    self._log_analysis(analysis_type, 'success', result)
                    
                    return result
                
                except Exception as e:
                    logger.error(f"Error in {analysis_type}: {str(e)}")
                    
                    # Handle error
                    if self.available:
                        error_result = self.error_system.handle_error(error=e)
                        
                        # Log error
                        self._log_analysis(analysis_type, 'error', {
                            'error_type': error_result['error_info']['type'],
                            'message': error_result['error_info']['message'],
                            'rectification': error_result['rectification']
                        })
                        
                        # Return error response
                        return {
                            'success': False,
                            'error': error_result['error_info']['message'],
                            'error_type': error_result['error_info']['type'],
                            'rectification': error_result['rectification'],
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        return {
                            'success': False,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        }
            
            return wrapper
        return decorator
    
    # ========================================================================
    # COMMS ANALYZER ERROR HANDLING
    # ========================================================================
    
    def handle_comms_analysis(self, analysis_func: Callable) -> Callable:
        """Handle communications analysis errors"""
        return self.handle_analysis_errors('communications_analysis')(analysis_func)
    
    # ========================================================================
    # LOCATION INTELLIGENCE ERROR HANDLING
    # ========================================================================
    
    def handle_location_analysis(self, analysis_func: Callable) -> Callable:
        """Handle location intelligence errors"""
        return self.handle_analysis_errors('location_intelligence')(analysis_func)
    
    # ========================================================================
    # MEDIA VIEWER ERROR HANDLING
    # ========================================================================
    
    def handle_media_analysis(self, analysis_func: Callable) -> Callable:
        """Handle media viewer errors"""
        return self.handle_analysis_errors('media_analysis')(analysis_func)
    
    # ========================================================================
    # VALIDATION & RECOVERY
    # ========================================================================
    
    def validate_analysis_input(self, data: Dict[str, Any], 
                               analysis_type: str) -> Dict[str, Any]:
        """Validate analysis input"""
        try:
            if not data:
                raise ValueError("No data provided for analysis")
            
            if self.available:
                # Use error system to validate
                validation_result = self.error_system.validate_input(
                    data,
                    {'type': dict, 'required_fields': ['case_id']}
                )
                
                if not validation_result['valid']:
                    return {
                        'valid': False,
                        'errors': validation_result['errors'],
                        'timestamp': datetime.now().isoformat()
                    }
            
            return {
                'valid': True,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                'valid': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def recover_from_analysis_error(self, error: Exception, 
                                   analysis_type: str) -> Dict[str, Any]:
        """Recover from analysis error"""
        try:
            if self.available:
                error_info = {
                    'type': type(error).__name__,
                    'message': str(error),
                    'analysis_type': analysis_type
                }
                
                recovery = self.error_system.rectifier.rectify_error(error_info)
                
                return {
                    'recovered': recovery.get('success', False),
                    'recovery_strategy': recovery.get('fix_type'),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'recovered': False,
                    'error': 'Error handling system not available',
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return {
                'recovered': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # ========================================================================
    # LOGGING & MONITORING
    # ========================================================================
    
    def _log_analysis(self, analysis_type: str, status: str, 
                     details: Dict[str, Any]) -> None:
        """Log analysis operation"""
        log_entry = {
            'analysis_type': analysis_type,
            'status': status,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        self.error_log.append(log_entry)
        logger.info(f"Analysis logged: {analysis_type} - {status}")
    
    def get_analysis_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get analysis log"""
        return self.error_log[-limit:]
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        if not self.error_log:
            return {}
        
        stats = {
            'total_analyses': len(self.error_log),
            'successful': sum(1 for e in self.error_log if e['status'] == 'success'),
            'failed': sum(1 for e in self.error_log if e['status'] == 'error'),
            'timestamp': datetime.now().isoformat()
        }
        
        return stats

# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

analysis_error_handler = AnalysisErrorHandler()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def wrap_comms_analysis(func: Callable) -> Callable:
    """Wrap communications analysis with error handling"""
    return analysis_error_handler.handle_comms_analysis(func)

def wrap_location_analysis(func: Callable) -> Callable:
    """Wrap location intelligence with error handling"""
    return analysis_error_handler.handle_location_analysis(func)

def wrap_media_analysis(func: Callable) -> Callable:
    """Wrap media viewer with error handling"""
    return analysis_error_handler.handle_media_analysis(func)

def get_analysis_error_handler() -> AnalysisErrorHandler:
    """Get analysis error handler instance"""
    return analysis_error_handler
