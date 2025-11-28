"""
ADVANCED ERROR HANDLING SYSTEM

Main entry point for error handling module.

Components:
- Error Detector: Detects all error types
- Error Analyzer: Analyzes and categorizes errors
- Error Rectifier: Automatically fixes errors
- Error Preventer: Prevents future errors
- Error Learner: Learns from errors

Usage:
    from modules.error_handling import ErrorHandlingSystem
    
    error_system = ErrorHandlingSystem()
    
    # Detect error
    error_info = error_system.detect_error(exception)
    
    # Analyze error
    analysis = error_system.analyze_error(error_info)
    
    # Rectify error
    result = error_system.rectify_error(error_info)
"""

from .core.error_detector import ErrorDetector, create_error_detector
from .core.error_analyzer import ErrorAnalyzer, create_error_analyzer
from .core.error_rectifier import ErrorRectifier, create_error_rectifier
from .core.error_preventer import ErrorPreventer, create_error_preventer
from .core.error_learner import ErrorLearner, create_error_learner

__all__ = [
    'ErrorDetector',
    'ErrorAnalyzer',
    'ErrorRectifier',
    'ErrorPreventer',
    'ErrorLearner',
    'create_error_detector',
    'create_error_analyzer',
    'create_error_rectifier',
    'create_error_preventer',
    'create_error_learner',
    'ErrorHandlingSystem',
]

# ============================================================================
# INTEGRATED ERROR HANDLING SYSTEM
# ============================================================================

class ErrorHandlingSystem:
    """
    Integrated error handling system combining detection, analysis, and rectification
    """
    
    def __init__(self):
        """Initialize error handling system"""
        self.detector = create_error_detector()
        self.analyzer = create_error_analyzer()
        self.rectifier = create_error_rectifier()
        self.preventer = create_error_preventer()
        self.learner = create_error_learner()
        self.error_log = []
    
    # ========================================================================
    # MAIN ERROR HANDLING FLOW
    # ========================================================================
    
    def handle_error(self, error: Exception = None, code: str = None, 
                    context: dict = None, operation_result: dict = None) -> dict:
        """
        Complete error handling flow: detect -> analyze -> rectify
        
        Args:
            error: Exception object
            code: Python code to check for syntax errors
            context: Operation context
            operation_result: Result of operation
            
        Returns:
            Complete error handling result
        """
        # Step 1: Detect error
        error_info = self.detector.detect_code_errors(code, error)
        
        if not error_info and context:
            error_info = self.detector.detect_logic_errors(context)
        
        if not error_info and operation_result:
            error_info = self.detector.detect_silent_errors(operation_result)
        
        if not error_info:
            return {'success': True, 'message': 'No errors detected'}
        
        # Step 2: Analyze error
        analysis = self.analyzer.analyze_error(error_info)
        
        # Step 3: Rectify error
        rectification = self.rectifier.rectify_error(error_info, context)
        
        # Log complete error handling
        complete_result = {
            'error_detected': True,
            'error_info': error_info,
            'analysis': analysis,
            'rectification': rectification,
            'timestamp': __import__('datetime').datetime.now()
        }
        
        self.error_log.append(complete_result)
        return complete_result
    
    # ========================================================================
    # INDIVIDUAL OPERATIONS
    # ========================================================================
    
    def detect_error(self, error: Exception = None, code: str = None, 
                    context: dict = None, operation_result: dict = None):
        """Detect error"""
        error_info = self.detector.detect_code_errors(code, error)
        if not error_info and context:
            error_info = self.detector.detect_logic_errors(context)
        if not error_info and operation_result:
            error_info = self.detector.detect_silent_errors(operation_result)
        return error_info
    
    def analyze_error(self, error_info: dict):
        """Analyze error"""
        return self.analyzer.analyze_error(error_info)
    
    def rectify_error(self, error_info: dict, context: dict = None):
        """Rectify error"""
        return self.rectifier.rectify_error(error_info, context)
    
    # ========================================================================
    # STATISTICS & REPORTING
    # ========================================================================
    
    def get_error_statistics(self):
        """Get error statistics"""
        return {
            'detector_stats': self.detector.get_error_history(),
            'analyzer_stats': self.analyzer.get_error_statistics(),
            'rectifier_stats': self.rectifier.get_fix_statistics(),
            'total_errors_handled': len(self.error_log)
        }
    
    def get_error_history(self, limit: int = 100):
        """Get error history"""
        return self.error_log[-limit:]
    
    def get_error_trends(self, hours: int = 24):
        """Get error trends"""
        return self.analyzer.get_error_trends(hours)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def clear_history(self):
        """Clear error history"""
        self.error_log = []
        self.detector.clear_history()
    
    def get_system_health(self):
        """Get system health based on errors"""
        stats = self.get_error_statistics()
        total = stats['total_errors_handled']
        
        if total == 0:
            return 'healthy'
        elif total < 10:
            return 'good'
        elif total < 50:
            return 'warning'
        else:
            return 'critical'
    
    # ========================================================================
    # PREVENTION OPERATIONS
    # ========================================================================
    
    def validate_input(self, input_data, validation_rules):
        """Validate input"""
        return self.preventer.validate_input(input_data, validation_rules)
    
    def monitor_resources(self):
        """Monitor system resources"""
        return self.preventer.monitor_resource_usage()
    
    def detect_resource_exhaustion(self):
        """Detect resource exhaustion"""
        return self.preventer.detect_resource_exhaustion()
    
    def detect_anomalies(self, metrics):
        """Detect anomalies"""
        return self.preventer.detect_anomalies(metrics)
    
    # ========================================================================
    # LEARNING OPERATIONS
    # ========================================================================
    
    def learn_from_error(self, error_info, fix_applied, result):
        """Learn from error"""
        return self.learner.learn_from_error(error_info, fix_applied, result)
    
    def analyze_patterns(self):
        """Analyze error patterns"""
        return self.learner.analyze_error_patterns()
    
    def get_best_solution(self, error_type):
        """Get best solution for error"""
        return self.learner.get_best_solution(error_type)
    
    def predict_future_errors(self, context):
        """Predict future errors"""
        return self.learner.predict_future_errors(context)
    
    def get_improvement_recommendations(self):
        """Get improvement recommendations"""
        return self.learner.improve_error_detection()
    
    def get_learning_summary(self):
        """Get learning summary"""
        return self.learner.get_learning_summary()

# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_error_handling_system() -> ErrorHandlingSystem:
    """Factory function to create error handling system"""
    return ErrorHandlingSystem()
