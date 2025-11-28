"""
EXTRACTION ERROR HANDLER - Error handling for extraction operations

Integrates error handling with:
- Device connection
- Module extraction
- Extraction progress
- Extraction results
- Extraction validation
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)

# ============================================================================
# EXTRACTION ERROR HANDLER CLASS
# ============================================================================

class ExtractionErrorHandler:
    """Handles errors in extraction operations"""
    
    def __init__(self):
        try:
            from modules.error_handling import ErrorHandlingSystem
            self.error_system = ErrorHandlingSystem()
            self.available = True
        except ImportError:
            logger.warning("Error handling system not available")
            self.available = False
        
        self.extraction_errors = []
    
    # ========================================================================
    # DEVICE CONNECTION ERRORS
    # ========================================================================
    
    def handle_device_connection_error(self, device_id: str, 
                                      error: Exception) -> Dict[str, Any]:
        """Handle device connection errors"""
        try:
            logger.error(f"Device connection error for {device_id}: {str(error)}")
            
            if self.available:
                error_result = self.error_system.handle_error(error=error)
                
                self.extraction_errors.append({
                    'device_id': device_id,
                    'error_type': 'DeviceConnectionError',
                    'error_class': error_result['error_info']['type'],
                    'message': error_result['error_info']['message'],
                    'status': 'handled',
                    'timestamp': datetime.now().isoformat()
                })
                
                return {
                    'success': False,
                    'device_id': device_id,
                    'error': 'Device connection failed',
                    'error_type': error_result['error_info']['type'],
                    'recovery': error_result['rectification'],
                    'recommendations': self._get_device_connection_recommendations(),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'device_id': device_id,
                    'error': 'Device connection failed'
                }
        except Exception as e:
            logger.error(f"Device connection error handling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_device_connection_recommendations(self) -> List[str]:
        """Get device connection recommendations"""
        return [
            'Check USB cable connection',
            'Restart the device',
            'Try different USB port',
            'Update ADB drivers',
            'Enable USB debugging on device',
            'Check device authorization',
            'Verify device is recognized by system'
        ]
    
    # ========================================================================
    # MODULE EXTRACTION ERRORS
    # ========================================================================
    
    def handle_module_extraction_error(self, case_id: str, device_id: str,
                                      module_name: str, 
                                      error: Exception) -> Dict[str, Any]:
        """Handle module extraction errors"""
        try:
            logger.error(f"Module extraction error for {module_name}: {str(error)}")
            
            if self.available:
                error_result = self.error_system.handle_error(error=error)
                
                self.extraction_errors.append({
                    'case_id': case_id,
                    'device_id': device_id,
                    'module': module_name,
                    'error_type': 'ModuleExtractionError',
                    'error_class': error_result['error_info']['type'],
                    'message': error_result['error_info']['message'],
                    'status': 'handled',
                    'timestamp': datetime.now().isoformat()
                })
                
                return {
                    'success': False,
                    'case_id': case_id,
                    'device_id': device_id,
                    'module': module_name,
                    'error': f'Failed to extract {module_name}',
                    'error_type': error_result['error_info']['type'],
                    'recovery': error_result['rectification'],
                    'recommendations': self._get_module_extraction_recommendations(module_name),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'module': module_name,
                    'error': f'Failed to extract {module_name}'
                }
        except Exception as e:
            logger.error(f"Module extraction error handling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_module_extraction_recommendations(self, module_name: str) -> List[str]:
        """Get module extraction recommendations"""
        recommendations = [
            f'Retry extraction for {module_name}',
            'Check device storage space',
            'Verify module is available on device',
            'Check device permissions',
            'Try extracting other modules first'
        ]
        
        if module_name == 'communications':
            recommendations.extend([
                'Check if communications data is available',
                'Verify messaging apps are installed',
                'Check call logs availability'
            ])
        elif module_name == 'location':
            recommendations.extend([
                'Check location services are enabled',
                'Verify GPS data is available',
                'Check location history settings'
            ])
        elif module_name == 'media':
            recommendations.extend([
                'Check media storage availability',
                'Verify media files are not corrupted',
                'Check file permissions'
            ])
        
        return recommendations
    
    # ========================================================================
    # EXTRACTION PROGRESS ERRORS
    # ========================================================================
    
    def handle_extraction_progress_error(self, extraction_id: str,
                                        progress: float,
                                        error: Exception) -> Dict[str, Any]:
        """Handle extraction progress errors"""
        try:
            logger.error(f"Extraction progress error at {progress}%: {str(error)}")
            
            if self.available:
                error_result = self.error_system.handle_error(error=error)
                
                self.extraction_errors.append({
                    'extraction_id': extraction_id,
                    'progress': progress,
                    'error_type': 'ExtractionProgressError',
                    'error_class': error_result['error_info']['type'],
                    'message': error_result['error_info']['message'],
                    'status': 'handled',
                    'timestamp': datetime.now().isoformat()
                })
                
                return {
                    'success': False,
                    'extraction_id': extraction_id,
                    'progress': progress,
                    'error': 'Extraction interrupted',
                    'error_type': error_result['error_info']['type'],
                    'recovery': error_result['rectification'],
                    'recommendations': [
                        'Retry extraction from current progress',
                        'Check device connection stability',
                        'Reduce extraction scope',
                        'Try extracting in smaller batches'
                    ],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'extraction_id': extraction_id,
                    'error': 'Extraction interrupted'
                }
        except Exception as e:
            logger.error(f"Extraction progress error handling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # ========================================================================
    # EXTRACTION VALIDATION ERRORS
    # ========================================================================
    
    def validate_extraction_input(self, case_id: str, device_id: str,
                                 modules: List[str]) -> Dict[str, Any]:
        """Validate extraction input"""
        try:
            if not case_id:
                raise ValueError("Case ID is required")
            if not device_id:
                raise ValueError("Device ID is required")
            if not modules or len(modules) == 0:
                raise ValueError("At least one module must be selected")
            
            if self.available:
                validation = self.error_system.validate_input(
                    {
                        'case_id': case_id,
                        'device_id': device_id,
                        'modules': modules
                    },
                    {
                        'type': dict,
                        'required_fields': ['case_id', 'device_id', 'modules']
                    }
                )
                
                if not validation['valid']:
                    return {
                        'valid': False,
                        'errors': validation['errors'],
                        'timestamp': datetime.now().isoformat()
                    }
            
            return {
                'valid': True,
                'case_id': case_id,
                'device_id': device_id,
                'modules': modules,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Extraction validation failed: {e}")
            return {
                'valid': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # ========================================================================
    # EXTRACTION RECOVERY
    # ========================================================================
    
    def recover_extraction(self, extraction_id: str, 
                          error: Exception) -> Dict[str, Any]:
        """Recover from extraction error"""
        try:
            if self.available:
                error_info = {
                    'type': 'ExtractionError',
                    'extraction_id': extraction_id
                }
                
                recovery = self.error_system.rectifier.rectify_error(error_info)
                
                return {
                    'recovered': recovery.get('success', False),
                    'recovery_strategy': recovery.get('fix_type'),
                    'extraction_id': extraction_id,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'recovered': False,
                    'error': 'Error handling system not available',
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Extraction recovery failed: {e}")
            return {
                'recovered': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # ========================================================================
    # LOGGING & STATISTICS
    # ========================================================================
    
    def get_extraction_error_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get extraction error log"""
        return self.extraction_errors[-limit:]
    
    def get_extraction_error_statistics(self) -> Dict[str, Any]:
        """Get extraction error statistics"""
        if not self.extraction_errors:
            return {}
        
        stats = {
            'total_errors': len(self.extraction_errors),
            'by_type': {},
            'by_module': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for error in self.extraction_errors:
            error_type = error.get('error_type', 'unknown')
            module = error.get('module', 'unknown')
            
            stats['by_type'][error_type] = stats['by_type'].get(error_type, 0) + 1
            if module != 'unknown':
                stats['by_module'][module] = stats['by_module'].get(module, 0) + 1
        
        return stats

# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

extraction_error_handler = ExtractionErrorHandler()

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def get_extraction_error_handler() -> ExtractionErrorHandler:
    """Get extraction error handler instance"""
    return extraction_error_handler
