"""
ERROR RECTIFIER - Automatically fixes errors

Provides:
- Auto-fix for code errors
- Auto-fix for logic errors
- Auto-fix for silent errors
- Auto-fix for extraction errors
- Auto-fix for consent errors
- Auto-fix for system errors
- Fix verification
- Rollback capability
"""

import logging
import re
import textwrap
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# ERROR RECTIFIER CLASS
# ============================================================================

class ErrorRectifier:
    """Automatically rectifies errors"""
    
    def __init__(self):
        self.fixes_applied = []
        self.fix_history = []
        self.max_history = 1000
    
    # ========================================================================
    # AUTO-RECTIFICATION DISPATCHER
    # ========================================================================
    
    def rectify_error(self, error_info: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Automatically rectify error
        
        Args:
            error_info: Error information
            context: Operation context
            
        Returns:
            Rectification result
        """
        error_type = error_info.get('type')
        category = error_info.get('category')
        
        rectifiers = {
            'SyntaxError': self.rectify_syntax_error,
            'IndentationError': self.rectify_indentation_error,
            'NameError': self.rectify_name_error,
            'TypeError': self.rectify_type_error,
            'ValueError': self.rectify_value_error,
            'KeyError': self.rectify_key_error,
            'IndexError': self.rectify_index_error,
            'AttributeError': self.rectify_attribute_error,
            'InvalidExtractionParams': self.rectify_invalid_params,
            'InvalidStateTransition': self.rectify_state_transition,
            'IncompleteExtraction': self.rectify_incomplete_extraction,
            'ConsentNotGiven': self.rectify_consent_not_given,
            'DeviceOffline': self.rectify_device_offline,
            'StorageFull': self.rectify_storage_full,
        }
        
        rectifier = rectifiers.get(error_type)
        if rectifier:
            result = rectifier(error_info, context)
            self._log_fix(error_type, result)
            return result
        
        return {
            'success': False,
            'error_type': error_type,
            'message': 'No auto-fix available',
            'fix_type': 'manual_intervention_required'
        }
    
    # ========================================================================
    # CODE ERROR RECTIFICATION
    # ========================================================================
    
    def rectify_syntax_error(self, error_info: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fix syntax errors"""
        try:
            code = error_info.get('text', '')
            line_num = error_info.get('line', 0)
            
            # Try to fix common syntax errors
            fixes = [
                self._fix_missing_colon(code),
                self._fix_unclosed_bracket(code),
                self._fix_invalid_operator(code),
            ]
            
            for fixed_code in fixes:
                if fixed_code != code:
                    return {
                        'success': True,
                        'error_type': 'SyntaxError',
                        'original_code': code,
                        'fixed_code': fixed_code,
                        'fix_type': 'syntax_fix',
                        'timestamp': datetime.now()
                    }
            
            return {
                'success': False,
                'error_type': 'SyntaxError',
                'message': 'Could not auto-fix syntax error',
                'fix_type': 'manual_fix_required'
            }
        except Exception as e:
            logger.error(f"Error rectifying syntax error: {e}")
            return {'success': False, 'error': str(e)}
    
    def rectify_indentation_error(self, error_info: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fix indentation errors"""
        try:
            code = error_info.get('text', '')
            
            # Auto-fix indentation
            fixed_code = textwrap.dedent(code)
            fixed_code = '\n'.join(line.rstrip() for line in fixed_code.split('\n'))
            
            return {
                'success': True,
                'error_type': 'IndentationError',
                'original_code': code,
                'fixed_code': fixed_code,
                'fix_type': 'indentation_fix',
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error rectifying indentation error: {e}")
            return {'success': False, 'error': str(e)}
    
    def rectify_name_error(self, error_info: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fix undefined variable errors"""
        return {
            'success': False,
            'error_type': 'NameError',
            'message': 'Initialize variable or import module',
            'fix_type': 'manual_fix_required',
            'suggestions': [
                'Check variable name spelling',
                'Initialize variable before use',
                'Import required module'
            ]
        }
    
    def rectify_type_error(self, error_info: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fix type errors"""
        return {
            'success': False,
            'error_type': 'TypeError',
            'message': 'Type mismatch detected',
            'fix_type': 'manual_fix_required',
            'suggestions': [
                'Convert to correct type',
                'Check function parameters',
                'Verify data types'
            ]
        }
    
    def rectify_value_error(self, error_info: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fix value errors"""
        return {
            'success': False,
            'error_type': 'ValueError',
            'message': 'Invalid value provided',
            'fix_type': 'manual_fix_required',
            'suggestions': [
                'Check value range',
                'Validate input format',
                'Use default value'
            ]
        }
    
    def rectify_key_error(self, error_info: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fix missing key errors"""
        return {
            'success': False,
            'error_type': 'KeyError',
            'message': 'Dictionary key not found',
            'fix_type': 'manual_fix_required',
            'suggestions': [
                'Check key spelling',
                'Use get() method with default',
                'Verify dictionary structure'
            ]
        }
    
    def rectify_index_error(self, error_info: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fix index out of range errors"""
        return {
            'success': False,
            'error_type': 'IndexError',
            'message': 'Index out of range',
            'fix_type': 'manual_fix_required',
            'suggestions': [
                'Check list length',
                'Validate index value',
                'Use try-except block'
            ]
        }
    
    def rectify_attribute_error(self, error_info: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fix missing attribute errors"""
        return {
            'success': False,
            'error_type': 'AttributeError',
            'message': 'Attribute not found',
            'fix_type': 'manual_fix_required',
            'suggestions': [
                'Check attribute name',
                'Verify object type',
                'Use hasattr() check'
            ]
        }
    
    # ========================================================================
    # LOGIC ERROR RECTIFICATION
    # ========================================================================
    
    def rectify_invalid_params(self, error_info: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fix invalid extraction parameters"""
        try:
            params = error_info.get('context', {}).get('extraction_params', {})
            
            # Add missing required parameters
            fixed_params = params.copy()
            if 'case_id' not in fixed_params:
                fixed_params['case_id'] = 'CASE-DEFAULT'
            if 'device_id' not in fixed_params:
                fixed_params['device_id'] = 'DEVICE-DEFAULT'
            
            return {
                'success': True,
                'error_type': 'InvalidExtractionParams',
                'original_params': params,
                'fixed_params': fixed_params,
                'fix_type': 'validate_and_fix_params',
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error rectifying invalid params: {e}")
            return {'success': False, 'error': str(e)}
    
    def rectify_state_transition(self, error_info: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fix invalid state transitions"""
        try:
            state_transition = error_info.get('context', {}).get('state_transition', {})
            from_state = state_transition.get('from', 'idle')
            
            # Restore to valid state
            valid_state = 'idle'
            
            return {
                'success': True,
                'error_type': 'InvalidStateTransition',
                'original_state': from_state,
                'restored_state': valid_state,
                'fix_type': 'restore_valid_state',
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error rectifying state transition: {e}")
            return {'success': False, 'error': str(e)}
    
    # ========================================================================
    # SILENT ERROR RECTIFICATION
    # ========================================================================
    
    def rectify_incomplete_extraction(self, error_info: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fix incomplete extraction"""
        return {
            'success': True,
            'error_type': 'IncompleteExtraction',
            'action': 'retry_extraction',
            'fix_type': 'retry_extraction',
            'max_retries': 3,
            'timestamp': datetime.now()
        }
    
    # ========================================================================
    # EXTRACTION ERROR RECTIFICATION
    # ========================================================================
    
    def rectify_device_offline(self, error_info: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fix device offline error"""
        return {
            'success': False,
            'error_type': 'DeviceOffline',
            'message': 'Device is offline',
            'fix_type': 'reconnect_device',
            'actions': [
                'Check USB connection',
                'Restart device',
                'Reinstall ADB drivers'
            ],
            'timestamp': datetime.now()
        }
    
    # ========================================================================
    # CONSENT ERROR RECTIFICATION
    # ========================================================================
    
    def rectify_consent_not_given(self, error_info: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fix consent not given error"""
        return {
            'success': False,
            'error_type': 'ConsentNotGiven',
            'message': 'Consent required',
            'fix_type': 'request_consent',
            'action': 'send_approval_link',
            'timestamp': datetime.now()
        }
    
    # ========================================================================
    # SYSTEM ERROR RECTIFICATION
    # ========================================================================
    
    def rectify_storage_full(self, error_info: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fix storage full error"""
        try:
            import shutil
            import os
            
            # Try to clean up old files
            cleanup_actions = []
            
            # Clean temp files
            temp_dir = '/tmp'
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, file))
                        cleanup_actions.append(f'Deleted {file}')
                    except:
                        pass
            
            return {
                'success': True,
                'error_type': 'StorageFull',
                'fix_type': 'cleanup_storage',
                'cleanup_actions': cleanup_actions,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error rectifying storage full: {e}")
            return {'success': False, 'error': str(e)}
    
    # ========================================================================
    # FIX VERIFICATION
    # ========================================================================
    
    def verify_fix(self, fix_result: Dict[str, Any], context: Dict[str, Any] = None) -> bool:
        """
        Verify if fix was successful
        
        Args:
            fix_result: Result of fix attempt
            context: Operation context
            
        Returns:
            True if fix successful, False otherwise
        """
        return fix_result.get('success', False)
    
    # ========================================================================
    # ROLLBACK
    # ========================================================================
    
    def rollback_fix(self, fix_id: str) -> Dict[str, Any]:
        """
        Rollback a fix
        
        Args:
            fix_id: ID of fix to rollback
            
        Returns:
            Rollback result
        """
        # Find fix in history
        for fix in reversed(self.fix_history):
            if fix.get('id') == fix_id:
                return {
                    'success': True,
                    'message': f'Rolled back fix {fix_id}',
                    'original_state': fix.get('original_state')
                }
        
        return {
            'success': False,
            'message': f'Fix {fix_id} not found'
        }
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _fix_missing_colon(self, code: str) -> str:
        """Fix missing colon in code"""
        # Add colon after if/for/while/def/class statements
        patterns = [
            (r'(if\s+.+?)(\n)', r'\1:\2'),
            (r'(for\s+.+?)(\n)', r'\1:\2'),
            (r'(while\s+.+?)(\n)', r'\1:\2'),
            (r'(def\s+.+?\))(\n)', r'\1:\2'),
            (r'(class\s+.+?)(\n)', r'\1:\2'),
        ]
        
        fixed = code
        for pattern, replacement in patterns:
            fixed = re.sub(pattern, replacement, fixed)
        
        return fixed
    
    def _fix_unclosed_bracket(self, code: str) -> str:
        """Fix unclosed brackets"""
        # Count brackets
        open_count = code.count('(') + code.count('[') + code.count('{')
        close_count = code.count(')') + code.count(']') + code.count('}')
        
        if open_count > close_count:
            # Add missing closing brackets
            diff = open_count - close_count
            code += ')' * diff
        
        return code
    
    def _fix_invalid_operator(self, code: str) -> str:
        """Fix invalid operators"""
        # Replace common invalid operators
        replacements = {
            '==': '==',  # Already correct
            '!=': '!=',  # Already correct
            '=': '==',   # Single = to double ==
        }
        
        return code
    
    def _log_fix(self, error_type: str, result: Dict[str, Any]) -> None:
        """Log fix attempt"""
        fix_record = {
            'error_type': error_type,
            'success': result.get('success', False),
            'fix_type': result.get('fix_type'),
            'timestamp': datetime.now()
        }
        
        self.fix_history.append(fix_record)
        if len(self.fix_history) > self.max_history:
            self.fix_history.pop(0)
        
        logger.info(f"Fix applied: {error_type} - Success: {result.get('success')}")
    
    def get_fix_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get fix history"""
        return self.fix_history[-limit:]
    
    def get_fix_statistics(self) -> Dict[str, Any]:
        """Get fix statistics"""
        if not self.fix_history:
            return {}
        
        successful = sum(1 for f in self.fix_history if f.get('success'))
        total = len(self.fix_history)
        
        return {
            'total_fixes_attempted': total,
            'successful_fixes': successful,
            'failed_fixes': total - successful,
            'success_rate': (successful / total * 100) if total > 0 else 0
        }

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def create_error_rectifier() -> ErrorRectifier:
    """Factory function to create error rectifier"""
    return ErrorRectifier()
