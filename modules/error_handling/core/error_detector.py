"""
ERROR DETECTOR - Detects all error types in real-time

Detects:
- Code errors (syntax, indentation, runtime)
- Logic errors (business logic, state, boundaries)
- Silent errors (incomplete extraction, missing validation)
- Extraction errors (device, ADB, USB, timeout)
- Consent errors (approval, verification)
- System errors (storage, memory, network)
- Future errors (predictive)
"""

import logging
import traceback
import sys
import ast
import inspect
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================================
# ERROR CATEGORIES & SEVERITY
# ============================================================================

class ErrorCategory(Enum):
    """Error categories"""
    CODE = "code"
    LOGIC = "logic"
    SILENT = "silent"
    EXTRACTION = "extraction"
    CONSENT = "consent"
    SYSTEM = "system"
    FUTURE = "future"

class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = 5  # System breaking
    HIGH = 4      # Feature breaking
    MEDIUM = 3    # Partial failure
    LOW = 2       # Minor issue
    INFO = 1      # Informational

# ============================================================================
# ERROR DETECTOR CLASS
# ============================================================================

class ErrorDetector:
    """Detects all types of errors in real-time"""
    
    def __init__(self):
        self.error_history = []
        self.error_patterns = {}
        self.max_history = 1000
        self.detectors = {
            'code': self.detect_code_errors,
            'logic': self.detect_logic_errors,
            'silent': self.detect_silent_errors,
            'extraction': self.detect_extraction_errors,
            'consent': self.detect_consent_errors,
            'system': self.detect_system_errors,
        }
    
    # ========================================================================
    # CODE ERROR DETECTION
    # ========================================================================
    
    def detect_code_errors(self, code: str = None, exception: Exception = None) -> Optional[Dict[str, Any]]:
        """
        Detect syntax and runtime errors
        
        Args:
            code: Python code to check for syntax errors
            exception: Exception object to analyze
            
        Returns:
            Error info dict or None
        """
        try:
            # Check syntax errors
            if code:
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    error_info = {
                        'type': 'SyntaxError',
                        'category': ErrorCategory.CODE,
                        'severity': ErrorSeverity.HIGH,
                        'message': str(e),
                        'line': e.lineno,
                        'offset': e.offset,
                        'text': e.text,
                        'timestamp': datetime.now(),
                        'auto_fixable': True,
                        'fix_type': 'syntax_fix'
                    }
                    self._log_error(error_info)
                    return error_info
            
            # Check runtime errors
            if exception:
                error_type = type(exception).__name__
                error_info = {
                    'type': error_type,
                    'category': ErrorCategory.CODE,
                    'severity': self._assess_code_error_severity(error_type),
                    'message': str(exception),
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.now(),
                    'auto_fixable': self._is_code_error_fixable(error_type),
                    'fix_type': self._get_code_error_fix_type(error_type)
                }
                self._log_error(error_info)
                return error_info
        
        except Exception as e:
            logger.error(f"Error in code error detection: {e}")
        
        return None
    
    def _assess_code_error_severity(self, error_type: str) -> ErrorSeverity:
        """Assess severity of code error"""
        critical_errors = ['SyntaxError', 'IndentationError', 'NameError', 'TypeError']
        high_errors = ['ValueError', 'KeyError', 'IndexError', 'AttributeError']
        
        if error_type in critical_errors:
            return ErrorSeverity.CRITICAL
        elif error_type in high_errors:
            return ErrorSeverity.HIGH
        else:
            return ErrorSeverity.MEDIUM
    
    def _is_code_error_fixable(self, error_type: str) -> bool:
        """Check if code error is auto-fixable"""
        fixable = ['IndentationError', 'NameError', 'TypeError', 'ValueError', 'KeyError']
        return error_type in fixable
    
    def _get_code_error_fix_type(self, error_type: str) -> str:
        """Get fix type for code error"""
        fixes = {
            'IndentationError': 'fix_indentation',
            'SyntaxError': 'fix_syntax',
            'NameError': 'fix_undefined_variable',
            'TypeError': 'fix_type_mismatch',
            'ValueError': 'fix_invalid_value',
            'KeyError': 'fix_missing_key',
            'IndexError': 'fix_index_error',
            'AttributeError': 'fix_missing_attribute'
        }
        return fixes.get(error_type, 'manual_fix')
    
    # ========================================================================
    # LOGIC ERROR DETECTION
    # ========================================================================
    
    def detect_logic_errors(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect business logic errors
        
        Args:
            context: Operation context with parameters
            
        Returns:
            Error info dict or None
        """
        try:
            # Check invalid extraction parameters
            if 'extraction_params' in context:
                if not self._validate_extraction_params(context['extraction_params']):
                    error_info = {
                        'type': 'InvalidExtractionParams',
                        'category': ErrorCategory.LOGIC,
                        'severity': ErrorSeverity.HIGH,
                        'message': 'Invalid extraction parameters',
                        'context': context,
                        'timestamp': datetime.now(),
                        'auto_fixable': True,
                        'fix_type': 'validate_and_fix_params'
                    }
                    self._log_error(error_info)
                    return error_info
            
            # Check state transition validity
            if 'state_transition' in context:
                from_state = context['state_transition'].get('from')
                to_state = context['state_transition'].get('to')
                if not self._validate_state_transition(from_state, to_state):
                    error_info = {
                        'type': 'InvalidStateTransition',
                        'category': ErrorCategory.LOGIC,
                        'severity': ErrorSeverity.HIGH,
                        'message': f'Invalid transition from {from_state} to {to_state}',
                        'context': context,
                        'timestamp': datetime.now(),
                        'auto_fixable': True,
                        'fix_type': 'restore_valid_state'
                    }
                    self._log_error(error_info)
                    return error_info
            
            # Check boundary violations
            if 'boundary_check' in context:
                if not self._check_boundaries(context['boundary_check']):
                    error_info = {
                        'type': 'BoundaryViolation',
                        'category': ErrorCategory.LOGIC,
                        'severity': ErrorSeverity.MEDIUM,
                        'message': 'Boundary violation detected',
                        'context': context,
                        'timestamp': datetime.now(),
                        'auto_fixable': True,
                        'fix_type': 'enforce_boundaries'
                    }
                    self._log_error(error_info)
                    return error_info
        
        except Exception as e:
            logger.error(f"Error in logic error detection: {e}")
        
        return None
    
    def _validate_extraction_params(self, params: Dict[str, Any]) -> bool:
        """Validate extraction parameters"""
        required = ['case_id', 'device_id']
        return all(param in params for param in required)
    
    def _validate_state_transition(self, from_state: str, to_state: str) -> bool:
        """Validate state transition"""
        valid_transitions = {
            'idle': ['extracting', 'analyzing'],
            'extracting': ['idle', 'error', 'complete'],
            'analyzing': ['idle', 'error', 'complete'],
            'error': ['idle', 'extracting'],
            'complete': ['idle', 'archiving'],
            'archiving': ['idle', 'complete']
        }
        return to_state in valid_transitions.get(from_state, [])
    
    def _check_boundaries(self, boundary_check: Dict[str, Any]) -> bool:
        """Check boundary conditions"""
        value = boundary_check.get('value')
        min_val = boundary_check.get('min')
        max_val = boundary_check.get('max')
        
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True
    
    # ========================================================================
    # SILENT ERROR DETECTION
    # ========================================================================
    
    def detect_silent_errors(self, operation_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect silent errors (no exception raised)
        
        Args:
            operation_result: Result of operation
            
        Returns:
            Error info dict or None
        """
        try:
            # Check incomplete extraction
            if 'extraction_result' in operation_result:
                if not self._check_extraction_completeness(operation_result['extraction_result']):
                    error_info = {
                        'type': 'IncompleteExtraction',
                        'category': ErrorCategory.SILENT,
                        'severity': ErrorSeverity.HIGH,
                        'message': 'Extraction incomplete - not all modules extracted',
                        'result': operation_result,
                        'timestamp': datetime.now(),
                        'auto_fixable': True,
                        'fix_type': 'retry_extraction'
                    }
                    self._log_error(error_info)
                    return error_info
            
            # Check missing validation
            if 'data' in operation_result:
                if not self._check_validation_completeness(operation_result['data']):
                    error_info = {
                        'type': 'MissingValidation',
                        'category': ErrorCategory.SILENT,
                        'severity': ErrorSeverity.MEDIUM,
                        'message': 'Data validation incomplete',
                        'result': operation_result,
                        'timestamp': datetime.now(),
                        'auto_fixable': True,
                        'fix_type': 'add_validation_checks'
                    }
                    self._log_error(error_info)
                    return error_info
            
            # Check for None/null values
            if 'data' in operation_result:
                if self._check_null_handling(operation_result['data']):
                    error_info = {
                        'type': 'NullHandlingIssue',
                        'category': ErrorCategory.SILENT,
                        'severity': ErrorSeverity.MEDIUM,
                        'message': 'Null/None values not properly handled',
                        'result': operation_result,
                        'timestamp': datetime.now(),
                        'auto_fixable': True,
                        'fix_type': 'add_null_checks'
                    }
                    self._log_error(error_info)
                    return error_info
        
        except Exception as e:
            logger.error(f"Error in silent error detection: {e}")
        
        return None
    
    def _check_extraction_completeness(self, extraction_result: Dict[str, Any]) -> bool:
        """Check if extraction is complete"""
        expected_modules = ['communications', 'location', 'media', 'device', 'security']
        extracted_modules = extraction_result.get('modules', [])
        return len(extracted_modules) == len(expected_modules)
    
    def _check_validation_completeness(self, data: Dict[str, Any]) -> bool:
        """Check if data validation is complete"""
        required_validations = ['type_check', 'range_check', 'format_check']
        validations = data.get('validations', [])
        return all(v in validations for v in required_validations)
    
    def _check_null_handling(self, data: Dict[str, Any]) -> bool:
        """Check for unhandled null values"""
        for key, value in data.items():
            if value is None:
                return True
        return False
    
    # ========================================================================
    # EXTRACTION ERROR DETECTION
    # ========================================================================
    
    def detect_extraction_errors(self, extraction_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect extraction-specific errors
        
        Args:
            extraction_context: Extraction operation context
            
        Returns:
            Error info dict or None
        """
        try:
            # Check device connectivity
            if 'device_status' in extraction_context:
                if extraction_context['device_status'] == 'offline':
                    error_info = {
                        'type': 'DeviceOffline',
                        'category': ErrorCategory.EXTRACTION,
                        'severity': ErrorSeverity.HIGH,
                        'message': 'Device is offline',
                        'context': extraction_context,
                        'timestamp': datetime.now(),
                        'auto_fixable': True,
                        'fix_type': 'reconnect_device'
                    }
                    self._log_error(error_info)
                    return error_info
            
            # Check extraction timeout
            if 'extraction_duration' in extraction_context:
                if extraction_context['extraction_duration'] > 3600:  # 1 hour
                    error_info = {
                        'type': 'ExtractionTimeout',
                        'category': ErrorCategory.EXTRACTION,
                        'severity': ErrorSeverity.HIGH,
                        'message': 'Extraction timeout exceeded',
                        'context': extraction_context,
                        'timestamp': datetime.now(),
                        'auto_fixable': True,
                        'fix_type': 'increase_timeout'
                    }
                    self._log_error(error_info)
                    return error_info
            
            # Check partial extraction
            if 'extraction_status' in extraction_context:
                if extraction_context['extraction_status'] == 'partial':
                    error_info = {
                        'type': 'PartialExtraction',
                        'category': ErrorCategory.EXTRACTION,
                        'severity': ErrorSeverity.MEDIUM,
                        'message': 'Extraction is incomplete',
                        'context': extraction_context,
                        'timestamp': datetime.now(),
                        'auto_fixable': True,
                        'fix_type': 'retry_extraction'
                    }
                    self._log_error(error_info)
                    return error_info
        
        except Exception as e:
            logger.error(f"Error in extraction error detection: {e}")
        
        return None
    
    # ========================================================================
    # CONSENT ERROR DETECTION
    # ========================================================================
    
    def detect_consent_errors(self, consent_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect consent/approval errors
        
        Args:
            consent_context: Consent operation context
            
        Returns:
            Error info dict or None
        """
        try:
            # Check if consent given
            if 'consent_status' in consent_context:
                if consent_context['consent_status'] == 'not_given':
                    error_info = {
                        'type': 'ConsentNotGiven',
                        'category': ErrorCategory.CONSENT,
                        'severity': ErrorSeverity.HIGH,
                        'message': 'Extraction requires consent',
                        'context': consent_context,
                        'timestamp': datetime.now(),
                        'auto_fixable': False,
                        'fix_type': 'request_consent'
                    }
                    self._log_error(error_info)
                    return error_info
            
            # Check approval status
            if 'approval_status' in consent_context:
                if consent_context['approval_status'] == 'pending':
                    error_info = {
                        'type': 'ApprovalPending',
                        'category': ErrorCategory.CONSENT,
                        'severity': ErrorSeverity.MEDIUM,
                        'message': 'Approval is still pending',
                        'context': consent_context,
                        'timestamp': datetime.now(),
                        'auto_fixable': False,
                        'fix_type': 'wait_for_approval'
                    }
                    self._log_error(error_info)
                    return error_info
            
            # Check consent expiration
            if 'consent_expiry' in consent_context:
                from datetime import datetime as dt
                if dt.now() > dt.fromisoformat(consent_context['consent_expiry']):
                    error_info = {
                        'type': 'ConsentExpired',
                        'category': ErrorCategory.CONSENT,
                        'severity': ErrorSeverity.HIGH,
                        'message': 'Consent has expired',
                        'context': consent_context,
                        'timestamp': datetime.now(),
                        'auto_fixable': False,
                        'fix_type': 'request_new_consent'
                    }
                    self._log_error(error_info)
                    return error_info
        
        except Exception as e:
            logger.error(f"Error in consent error detection: {e}")
        
        return None
    
    # ========================================================================
    # SYSTEM ERROR DETECTION
    # ========================================================================
    
    def detect_system_errors(self) -> Optional[Dict[str, Any]]:
        """
        Detect system infrastructure errors
        
        Returns:
            Error info dict or None
        """
        try:
            import shutil
            import psutil
            
            # Check storage
            disk = shutil.disk_usage('/')
            if disk.free < 1000000000:  # Less than 1GB
                error_info = {
                    'type': 'StorageFull',
                    'category': ErrorCategory.SYSTEM,
                    'severity': ErrorSeverity.CRITICAL,
                    'message': 'Storage space is running low',
                    'available_space': disk.free,
                    'timestamp': datetime.now(),
                    'auto_fixable': True,
                    'fix_type': 'cleanup_storage'
                }
                self._log_error(error_info)
                return error_info
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                error_info = {
                    'type': 'MemoryExhausted',
                    'category': ErrorCategory.SYSTEM,
                    'severity': ErrorSeverity.CRITICAL,
                    'message': 'Memory usage is critical',
                    'memory_percent': memory.percent,
                    'timestamp': datetime.now(),
                    'auto_fixable': True,
                    'fix_type': 'free_memory'
                }
                self._log_error(error_info)
                return error_info
        
        except Exception as e:
            logger.error(f"Error in system error detection: {e}")
        
        return None
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _log_error(self, error_info: Dict[str, Any]) -> None:
        """Log error to history"""
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        logger.warning(f"Error detected: {error_info['type']} - {error_info['message']}")
    
    def get_error_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent error history"""
        return self.error_history[-limit:]
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[Dict[str, Any]]:
        """Get errors by category"""
        return [e for e in self.error_history if e.get('category') == category]
    
    def clear_history(self) -> None:
        """Clear error history"""
        self.error_history = []

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def create_error_detector() -> ErrorDetector:
    """Factory function to create error detector"""
    return ErrorDetector()
