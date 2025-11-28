"""
ERROR PREVENTER - Prevents errors before they happen

Provides:
- Input validation
- Type checking
- Boundary validation
- State verification
- Resource monitoring
- Anomaly detection
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)

# ============================================================================
# ERROR PREVENTER CLASS
# ============================================================================

class ErrorPreventer:
    """Prevents errors before they occur"""
    
    def __init__(self):
        self.validation_rules = {}
        self.prevention_strategies = []
        self.monitored_operations = {}
        self.resource_limits = {
            'max_memory_percent': 90,
            'max_storage_percent': 95,
            'max_cpu_percent': 95,
            'max_extraction_time': 3600,  # 1 hour
        }
    
    # ========================================================================
    # INPUT VALIDATION
    # ========================================================================
    
    def validate_input(self, input_data: Any, validation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data against rules
        
        Args:
            input_data: Data to validate
            validation_rules: Validation rules to apply
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        # Type checking
        if 'type' in validation_rules:
            expected_type = validation_rules['type']
            if not isinstance(input_data, expected_type):
                errors.append(f"Expected type {expected_type}, got {type(input_data)}")
        
        # Required fields
        if isinstance(input_data, dict) and 'required_fields' in validation_rules:
            required = validation_rules['required_fields']
            missing = [f for f in required if f not in input_data]
            if missing:
                errors.append(f"Missing required fields: {missing}")
        
        # Value range
        if 'min' in validation_rules or 'max' in validation_rules:
            if isinstance(input_data, (int, float)):
                min_val = validation_rules.get('min')
                max_val = validation_rules.get('max')
                if min_val is not None and input_data < min_val:
                    errors.append(f"Value {input_data} is below minimum {min_val}")
                if max_val is not None and input_data > max_val:
                    errors.append(f"Value {input_data} is above maximum {max_val}")
        
        # String length
        if isinstance(input_data, str) and 'length' in validation_rules:
            length_rule = validation_rules['length']
            if 'min' in length_rule and len(input_data) < length_rule['min']:
                errors.append(f"String too short (min: {length_rule['min']})")
            if 'max' in length_rule and len(input_data) > length_rule['max']:
                errors.append(f"String too long (max: {length_rule['max']})")
        
        # Pattern matching
        if 'pattern' in validation_rules:
            import re
            pattern = validation_rules['pattern']
            if isinstance(input_data, str) and not re.match(pattern, input_data):
                errors.append(f"Value does not match pattern {pattern}")
        
        # Custom validation function
        if 'custom' in validation_rules:
            custom_validator = validation_rules['custom']
            try:
                if not custom_validator(input_data):
                    errors.append("Custom validation failed")
            except Exception as e:
                errors.append(f"Custom validation error: {str(e)}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'timestamp': datetime.now()
        }
    
    # ========================================================================
    # TYPE CHECKING
    # ========================================================================
    
    def add_type_checking(self, function: Callable) -> Callable:
        """
        Decorator to add type checking to function
        
        Args:
            function: Function to decorate
            
        Returns:
            Decorated function
        """
        @wraps(function)
        def wrapper(*args, **kwargs):
            # Get function annotations
            annotations = function.__annotations__
            
            # Check argument types
            for i, (arg_name, arg_type) in enumerate(annotations.items()):
                if arg_name == 'return':
                    continue
                
                if i < len(args):
                    arg_value = args[i]
                    if not isinstance(arg_value, arg_type):
                        raise TypeError(
                            f"Argument {arg_name} must be {arg_type}, got {type(arg_value)}"
                        )
            
            # Call original function
            result = function(*args, **kwargs)
            
            # Check return type
            if 'return' in annotations:
                return_type = annotations['return']
                if not isinstance(result, return_type):
                    logger.warning(
                        f"Function {function.__name__} returned {type(result)}, "
                        f"expected {return_type}"
                    )
            
            return result
        
        return wrapper
    
    # ========================================================================
    # BOUNDARY CHECKING
    # ========================================================================
    
    def add_boundary_checks(self, function: Callable, boundaries: Dict[str, Any]) -> Callable:
        """
        Decorator to add boundary checks to function
        
        Args:
            function: Function to decorate
            boundaries: Boundary definitions
            
        Returns:
            Decorated function
        """
        @wraps(function)
        def wrapper(*args, **kwargs):
            # Check boundaries before execution
            for param_name, boundary in boundaries.items():
                if param_name in kwargs:
                    value = kwargs[param_name]
                    
                    if 'min' in boundary and value < boundary['min']:
                        raise ValueError(
                            f"{param_name} must be >= {boundary['min']}, got {value}"
                        )
                    
                    if 'max' in boundary and value > boundary['max']:
                        raise ValueError(
                            f"{param_name} must be <= {boundary['max']}, got {value}"
                        )
            
            return function(*args, **kwargs)
        
        return wrapper
    
    # ========================================================================
    # STATE VERIFICATION
    # ========================================================================
    
    def add_state_verification(self, function: Callable, required_state: str) -> Callable:
        """
        Decorator to verify system state before function execution
        
        Args:
            function: Function to decorate
            required_state: Required system state
            
        Returns:
            Decorated function
        """
        @wraps(function)
        def wrapper(*args, **kwargs):
            # Check if system is in required state
            current_state = self._get_current_state()
            
            if current_state != required_state:
                raise RuntimeError(
                    f"Operation requires state '{required_state}', "
                    f"current state is '{current_state}'"
                )
            
            return function(*args, **kwargs)
        
        return wrapper
    
    def _get_current_state(self) -> str:
        """Get current system state"""
        # This would be implemented to get actual system state
        return 'idle'
    
    # ========================================================================
    # TIMEOUT PROTECTION
    # ========================================================================
    
    def add_timeout_protection(self, function: Callable, timeout_seconds: int) -> Callable:
        """
        Decorator to add timeout protection to function
        
        Args:
            function: Function to decorate
            timeout_seconds: Timeout in seconds
            
        Returns:
            Decorated function
        """
        @wraps(function)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(
                    f"Operation exceeded timeout of {timeout_seconds} seconds"
                )
            
            # Set signal handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = function(*args, **kwargs)
            finally:
                signal.alarm(0)  # Cancel alarm
            
            return result
        
        return wrapper
    
    # ========================================================================
    # RESOURCE MONITORING
    # ========================================================================
    
    def monitor_resource_usage(self) -> Dict[str, Any]:
        """
        Monitor system resource usage
        
        Returns:
            Resource usage metrics
        """
        try:
            import psutil
            import shutil
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_warning = memory.percent > self.resource_limits['max_memory_percent']
            
            # CPU usage
            cpu = psutil.cpu_percent(interval=1)
            cpu_warning = cpu > self.resource_limits['max_cpu_percent']
            
            # Storage usage
            disk = shutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            storage_warning = disk_percent > self.resource_limits['max_storage_percent']
            
            return {
                'memory': {
                    'percent': memory.percent,
                    'available_gb': memory.available / (1024**3),
                    'warning': memory_warning
                },
                'cpu': {
                    'percent': cpu,
                    'warning': cpu_warning
                },
                'storage': {
                    'percent': disk_percent,
                    'available_gb': disk.free / (1024**3),
                    'warning': storage_warning
                },
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error monitoring resources: {e}")
            return {}
    
    def detect_resource_exhaustion(self) -> Optional[Dict[str, Any]]:
        """
        Detect resource exhaustion
        
        Returns:
            Resource exhaustion alert or None
        """
        resources = self.monitor_resource_usage()
        
        if resources.get('memory', {}).get('warning'):
            return {
                'type': 'MemoryExhausted',
                'severity': 'CRITICAL',
                'message': 'Memory usage is critical',
                'available_gb': resources['memory']['available_gb'],
                'timestamp': datetime.now()
            }
        
        if resources.get('storage', {}).get('warning'):
            return {
                'type': 'StorageFull',
                'severity': 'CRITICAL',
                'message': 'Storage usage is critical',
                'available_gb': resources['storage']['available_gb'],
                'timestamp': datetime.now()
            }
        
        if resources.get('cpu', {}).get('warning'):
            return {
                'type': 'CPUOverload',
                'severity': 'HIGH',
                'message': 'CPU usage is high',
                'cpu_percent': resources['cpu']['percent'],
                'timestamp': datetime.now()
            }
        
        return None
    
    # ========================================================================
    # ANOMALY DETECTION
    # ========================================================================
    
    def detect_anomalies(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in metrics
        
        Args:
            metrics: System metrics
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Check for unusual patterns
        if 'error_rate' in metrics:
            if metrics['error_rate'] > 0.1:  # > 10% error rate
                anomalies.append({
                    'type': 'HighErrorRate',
                    'severity': 'HIGH',
                    'message': f"Error rate is {metrics['error_rate']*100:.1f}%",
                    'value': metrics['error_rate']
                })
        
        if 'response_time' in metrics:
            if metrics['response_time'] > 5000:  # > 5 seconds
                anomalies.append({
                    'type': 'SlowResponse',
                    'severity': 'MEDIUM',
                    'message': f"Response time is {metrics['response_time']}ms",
                    'value': metrics['response_time']
                })
        
        if 'extraction_duration' in metrics:
            if metrics['extraction_duration'] > self.resource_limits['max_extraction_time']:
                anomalies.append({
                    'type': 'ExtractionTimeout',
                    'severity': 'HIGH',
                    'message': f"Extraction duration exceeded {self.resource_limits['max_extraction_time']}s",
                    'value': metrics['extraction_duration']
                })
        
        return anomalies
    
    # ========================================================================
    # PREVENTION RULES
    # ========================================================================
    
    def add_validation_rule(self, rule_name: str, rule_func: Callable) -> None:
        """
        Add custom validation rule
        
        Args:
            rule_name: Name of rule
            rule_func: Validation function
        """
        self.validation_rules[rule_name] = rule_func
        logger.info(f"Added validation rule: {rule_name}")
    
    def apply_validation_rules(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all validation rules to data
        
        Args:
            data: Data to validate
            
        Returns:
            Validation result
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'rules_applied': []
        }
        
        for rule_name, rule_func in self.validation_rules.items():
            try:
                if not rule_func(data):
                    results['valid'] = False
                    results['errors'].append(f"Rule '{rule_name}' failed")
                results['rules_applied'].append(rule_name)
            except Exception as e:
                results['errors'].append(f"Rule '{rule_name}' error: {str(e)}")
        
        return results
    
    # ========================================================================
    # CONSISTENCY CHECKING
    # ========================================================================
    
    def verify_consistency(self, data: Dict[str, Any]) -> bool:
        """
        Verify data consistency
        
        Args:
            data: Data to verify
            
        Returns:
            True if consistent, False otherwise
        """
        # Check for required fields
        required_fields = ['case_id', 'device_id', 'timestamp']
        if not all(field in data for field in required_fields):
            return False
        
        # Check for data type consistency
        if 'case_id' in data and not isinstance(data['case_id'], str):
            return False
        
        if 'device_id' in data and not isinstance(data['device_id'], str):
            return False
        
        # Check for value consistency
        if 'status' in data:
            valid_statuses = ['idle', 'extracting', 'analyzing', 'complete', 'error']
            if data['status'] not in valid_statuses:
                return False
        
        return True
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def set_resource_limit(self, resource: str, limit: float) -> None:
        """Set resource limit"""
        if resource in self.resource_limits:
            self.resource_limits[resource] = limit
            logger.info(f"Set {resource} limit to {limit}")
    
    def get_resource_limits(self) -> Dict[str, float]:
        """Get current resource limits"""
        return self.resource_limits.copy()

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def create_error_preventer() -> ErrorPreventer:
    """Factory function to create error preventer"""
    return ErrorPreventer()
