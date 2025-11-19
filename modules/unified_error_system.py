"""
UNIFIED ERROR SYSTEM - Single Integration File

Combines error_handler, error_prevention, and error_monitoring.
This unified approach makes integration easier and side effects visible.

SIDE EFFECTS NOTIFICATION:
1. Creates error_patterns.json in current directory
2. Logs errors to configured logger
3. Stores error history in memory
4. Streamlit UI functions modify session state

INTEGRATION:
- Import: from modules.unified_error_system import handle_error, validate_before_operation
- Use: handle_error(exception, context="...", module="...", function="...")
- Validate: valid, issues = validate_before_operation("operation", case_id=case_id)
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from collections import defaultdict, Counter
import hashlib

logger = logging.getLogger(__name__)


# ============================================================================
# ERROR DEFINITIONS
# ============================================================================

class ErrorSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ErrorCategory(Enum):
    PATH_ERROR = "path_error"
    PERMISSION_ERROR = "permission_error"
    VALIDATION_ERROR = "validation_error"
    EXTRACTION_ERROR = "extraction_error"
    PARSING_ERROR = "parsing_error"
    NETWORK_ERROR = "network_error"
    CONSENT_ERROR = "consent_error"
    DEVICE_ERROR = "device_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"


# ============================================================================
# VALIDATION SYSTEM
# ============================================================================

class ValidationResult:
    def __init__(self, valid: bool, message: str = "", suggestions: List[str] = None):
        self.valid = valid
        self.message = message
        self.suggestions = suggestions or []
    
    def __bool__(self):
        return self.valid


class Validator:
    @staticmethod
    def validate_path(path: Any, must_exist: bool = False) -> ValidationResult:
        try:
            if path is None or (isinstance(path, str) and not path.strip()):
                return ValidationResult(False, "Path is None or empty", ["Provide a valid path"])
            
            path_str = str(path)
            invalid_chars = ['<', '>', '|', '?', '*']
            
            if any(char in path_str for char in invalid_chars):
                return ValidationResult(False, "Path contains invalid characters")
            
            if must_exist and not os.path.exists(path_str):
                return ValidationResult(False, f"Path does not exist: {path_str}")
            
            if os.path.exists(path_str) and os.path.isfile(path_str):
                try:
                    with open(path_str, 'rb') as f:
                        f.read(1)
                except PermissionError:
                    return ValidationResult(False, f"Permission denied: {path_str}")
                except Exception as e:
                    return ValidationResult(False, f"Cannot read file: {e}")
            
            return ValidationResult(True, f"Path is valid: {path_str}")
        except Exception as e:
            logger.error(f"Path validation error: {e}")
            return ValidationResult(False, f"Path validation failed: {e}")
    
    @staticmethod
    def validate_artifact_paths(artifact_paths: Any) -> ValidationResult:
        try:
            if artifact_paths is None:
                return ValidationResult(True, "artifact_paths is None")
            
            if not isinstance(artifact_paths, dict):
                return ValidationResult(False, f"artifact_paths must be dict, got {type(artifact_paths).__name__}")
            
            invalid_paths = []
            for key, paths in artifact_paths.items():
                if isinstance(paths, list):
                    for path in paths:
                        result = Validator.validate_path(path, must_exist=True)
                        if not result.valid:
                            invalid_paths.append((key, path))
            
            if invalid_paths:
                return ValidationResult(False, f"Invalid artifact paths: {len(invalid_paths)}")
            
            return ValidationResult(True, "artifact_paths is valid")
        except Exception as e:
            logger.error(f"Artifact validation error: {e}")
            return ValidationResult(False, f"Artifact validation failed: {e}")
    
    @staticmethod
    def validate_case_id(case_id: Any) -> ValidationResult:
        try:
            if case_id is None or (isinstance(case_id, str) and not case_id.strip()):
                return ValidationResult(False, "case_id is None or empty")
            
            case_id_str = str(case_id)
            if len(case_id_str) > 100:
                return ValidationResult(False, f"case_id too long: {len(case_id_str)} chars")
            
            invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            if any(char in case_id_str for char in invalid_chars):
                return ValidationResult(False, "case_id contains invalid characters")
            
            return ValidationResult(True, "case_id is valid")
        except Exception as e:
            logger.error(f"case_id validation error: {e}")
            return ValidationResult(False, f"case_id validation failed: {e}")
    
    @staticmethod
    def validate_before_operation(operation_name: str, **kwargs) -> Tuple[bool, List[str]]:
        issues = []
        
        if 'case_id' in kwargs:
            result = Validator.validate_case_id(kwargs['case_id'])
            if not result.valid:
                issues.append(f"case_id: {result.message}")
        
        if 'artifact_paths' in kwargs:
            result = Validator.validate_artifact_paths(kwargs['artifact_paths'])
            if not result.valid:
                issues.append(f"artifact_paths: {result.message}")
        
        if 'path' in kwargs:
            result = Validator.validate_path(kwargs['path'])
            if not result.valid:
                issues.append(f"path: {result.message}")
        
        return len(issues) == 0, issues


# ============================================================================
# ERROR PATTERN STORAGE
# ============================================================================

class ErrorPattern:
    def __init__(self, error_hash: str, category: ErrorCategory, message: str, severity: ErrorSeverity):
        self.error_hash = error_hash
        self.category = category
        self.message = message
        self.severity = severity
        self.occurrences = 0
        self.last_seen = None
        self.first_seen = None
        self.contexts = []
        self.solutions = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_hash': self.error_hash,
            'category': self.category.value,
            'message': self.message,
            'severity': self.severity.value,
            'occurrences': self.occurrences,
            'last_seen': self.last_seen,
            'first_seen': self.first_seen,
        }


# ============================================================================
# UNIFIED ERROR SYSTEM
# ============================================================================

class UnifiedErrorSystem:
    def __init__(self, error_db_path: str = 'error_patterns.json'):
        self.error_db_path = error_db_path
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.validator = Validator()
        
        self._load_error_patterns()
        logger.info("UnifiedErrorSystem initialized")
    
    def _load_error_patterns(self):
        if os.path.exists(self.error_db_path):
            try:
                with open(self.error_db_path, 'r') as f:
                    data = json.load(f)
                    for pattern_data in data.get('patterns', []):
                        pattern = ErrorPattern(
                            pattern_data['error_hash'],
                            ErrorCategory(pattern_data['category']),
                            pattern_data['message'],
                            ErrorSeverity(pattern_data['severity'])
                        )
                        pattern.occurrences = pattern_data.get('occurrences', 0)
                        self.error_patterns[pattern.error_hash] = pattern
                logger.info(f"Loaded {len(self.error_patterns)} error patterns")
            except Exception as e:
                logger.warning(f"Failed to load error patterns: {e}")
    
    def _save_error_patterns(self):
        try:
            patterns_data = {
                'patterns': [p.to_dict() for p in self.error_patterns.values()],
                'last_updated': datetime.now().isoformat(),
                'total_errors': len(self.error_history),
            }
            with open(self.error_db_path, 'w') as f:
                json.dump(patterns_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save error patterns: {e}")
    
    def categorize_error(self, error: Exception) -> Tuple[ErrorCategory, ErrorSeverity]:
        error_str = str(error).lower()
        
        if any(x in error_str for x in ['path', 'file', 'directory', 'not found']):
            return ErrorCategory.PATH_ERROR, ErrorSeverity.HIGH
        if any(x in error_str for x in ['permission', 'denied', 'access']):
            return ErrorCategory.PERMISSION_ERROR, ErrorSeverity.HIGH
        if any(x in error_str for x in ['validation', 'invalid', 'type error']):
            return ErrorCategory.VALIDATION_ERROR, ErrorSeverity.MEDIUM
        if any(x in error_str for x in ['extraction', 'extract', 'pull']):
            return ErrorCategory.EXTRACTION_ERROR, ErrorSeverity.HIGH
        if any(x in error_str for x in ['parse', 'json', 'decode']):
            return ErrorCategory.PARSING_ERROR, ErrorSeverity.MEDIUM
        if any(x in error_str for x in ['adb', 'device', 'connection']):
            return ErrorCategory.DEVICE_ERROR, ErrorSeverity.HIGH
        if any(x in error_str for x in ['consent', 'approval']):
            return ErrorCategory.CONSENT_ERROR, ErrorSeverity.HIGH
        
        return ErrorCategory.UNKNOWN_ERROR, ErrorSeverity.MEDIUM
    
    def handle_error(self, error: Exception, context: str = "", module: str = "", function: str = "") -> Dict[str, Any]:
        try:
            category, severity = self.categorize_error(error)
            error_hash = hashlib.md5(f"{category.value}:{str(error)[:100]}".encode()).hexdigest()[:16]
            
            if error_hash not in self.error_patterns:
                pattern = ErrorPattern(error_hash, category, str(error), severity)
                self.error_patterns[error_hash] = pattern
            else:
                pattern = self.error_patterns[error_hash]
            
            pattern.occurrences += 1
            pattern.last_seen = datetime.now().isoformat()
            if pattern.first_seen is None:
                pattern.first_seen = datetime.now().isoformat()
            
            logger.error(f"Error in {module}.{function}: {error}", extra={'category': category.value, 'severity': severity.value})
            
            suggestions = self._generate_suggestions(category)
            
            error_record = {
                'timestamp': datetime.now().isoformat(),
                'error_hash': error_hash,
                'category': category.value,
                'severity': severity.value,
                'message': str(error),
                'context': context,
                'module': module,
                'function': function,
                'suggestions': suggestions,
            }
            self.error_history.append(error_record)
            
            if len(self.error_history) % 10 == 0:
                self._save_error_patterns()
            
            return {
                'handled': True,
                'category': category,
                'severity': severity,
                'message': str(error),
                'suggestions': suggestions,
                'error_hash': error_hash,
            }
        except Exception as handler_error:
            logger.error(f"Error handler failed: {handler_error}")
            return {
                'handled': False,
                'category': ErrorCategory.UNKNOWN_ERROR,
                'severity': ErrorSeverity.CRITICAL,
                'message': str(error),
                'suggestions': ['Check logs for details'],
                'error_hash': None,
            }
    
    def _generate_suggestions(self, category: ErrorCategory) -> List[str]:
        """Generate actionable suggestions based on error category"""
        suggestions = []
        
        if category == ErrorCategory.PATH_ERROR:
            suggestions = [
                "Verify the file or directory path exists",
                "Check that the path is correctly formatted",
                "Ensure you have read permissions for the path",
                "Try using an absolute path instead of relative",
                "Check if the file was recently moved or deleted"
            ]
        elif category == ErrorCategory.PERMISSION_ERROR:
            suggestions = [
                "Check file/directory permissions",
                "Run the application with elevated privileges if needed",
                "Verify the user has access to the resource",
                "Check if the resource is locked by another process",
                "Try running as administrator (Windows) or with sudo (Linux/Mac)"
            ]
        elif category == ErrorCategory.VALIDATION_ERROR:
            suggestions = [
                "Verify input data format matches requirements",
                "Check data types (string, number, boolean, etc.)",
                "Ensure required fields are not empty",
                "Validate against expected schema or format",
                "Check for special characters that might cause issues"
            ]
        elif category == ErrorCategory.EXTRACTION_ERROR:
            suggestions = [
                "Verify device is connected and authorized",
                "Check ADB connection status",
                "Ensure extraction modules are properly configured",
                "Try restarting the extraction process",
                "Check device storage space is available"
            ]
        elif category == ErrorCategory.PARSING_ERROR:
            suggestions = [
                "Verify JSON/XML structure is valid",
                "Check file encoding (UTF-8 recommended)",
                "Ensure no special characters are breaking the parser",
                "Validate against schema if available",
                "Try opening file in text editor to check format"
            ]
        elif category == ErrorCategory.NETWORK_ERROR:
            suggestions = [
                "Check internet/network connection",
                "Verify ADB daemon is running (adb start-server)",
                "Check if device is properly connected via USB",
                "Try reconnecting the device",
                "Verify firewall isn't blocking the connection"
            ]
        elif category == ErrorCategory.CONSENT_ERROR:
            suggestions = [
                "Ensure consent has been obtained from the nominee",
                "Generate a new approval link in the Consent tab",
                "Verify the nominee has approved the request",
                "Check if approval has expired",
                "Try refreshing the approval status"
            ]
        elif category == ErrorCategory.DEVICE_ERROR:
            suggestions = [
                "Check device is connected via USB",
                "Verify device is authorized (check phone screen)",
                "Ensure ADB is installed and in PATH",
                "Try 'adb devices' to see connected devices",
                "Restart ADB daemon: adb kill-server && adb start-server"
            ]
        elif category == ErrorCategory.CONFIGURATION_ERROR:
            suggestions = [
                "Check configuration file exists and is valid",
                "Verify all required settings are present",
                "Check for typos in configuration keys",
                "Ensure configuration file has correct permissions",
                "Validate configuration against schema"
            ]
        elif category == ErrorCategory.UNKNOWN_ERROR:
            suggestions = [
                "Check the error message and logs for details",
                "Try reproducing the error with different inputs",
                "Check if all dependencies are installed",
                "Verify system resources (disk space, memory)",
                "Contact support with error details and logs"
            ]
        
        return suggestions if suggestions else ["Check logs for more details"]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        if not self.error_history:
            return {'total_errors': 0, 'categories': {}, 'severities': {}}
        
        categories = Counter(e['category'] for e in self.error_history)
        severities = Counter(e['severity'] for e in self.error_history)
        
        return {
            'total_errors': len(self.error_history),
            'categories': dict(categories),
            'severities': dict(severities),
            'unique_patterns': len(self.error_patterns),
        }


# ============================================================================
# GLOBAL INSTANCE & CONVENIENCE FUNCTIONS
# ============================================================================

_unified_system = None


def get_unified_error_system() -> UnifiedErrorSystem:
    global _unified_system
    if _unified_system is None:
        _unified_system = UnifiedErrorSystem()
    return _unified_system


def handle_error(error: Exception, context: str = "", module: str = "", function: str = "") -> Dict[str, Any]:
    """Handle error with automatic categorization and suggestions"""
    system = get_unified_error_system()
    return system.handle_error(error, context, module, function)


def get_error_stats() -> Dict[str, Any]:
    """Get error statistics"""
    system = get_unified_error_system()
    return system.get_error_statistics()


def validate_before_operation(operation_name: str, **kwargs) -> Tuple[bool, List[str]]:
    """Validate before operation - returns (valid, issues_list)"""
    return Validator.validate_before_operation(operation_name, **kwargs)


def validate_path(path: Any, must_exist: bool = False) -> ValidationResult:
    """Validate a path"""
    return Validator.validate_path(path, must_exist=must_exist)


def validate_artifact_paths(artifact_paths: Any) -> ValidationResult:
    """Validate artifact_paths"""
    return Validator.validate_artifact_paths(artifact_paths)


def validate_case_id(case_id: Any) -> ValidationResult:
    """Validate case_id"""
    return Validator.validate_case_id(case_id)
