"""
ERROR ANALYZER - Analyzes and categorizes errors

Provides:
- Error categorization
- Severity assessment
- Root cause analysis
- Impact analysis
- Dependency analysis
- Pattern detection
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================================
# ERROR ANALYSIS ENUMS
# ============================================================================

class ErrorImpact(Enum):
    """Error impact levels"""
    SYSTEM_DOWN = "system_down"
    FEATURE_BROKEN = "feature_broken"
    PARTIAL_FAILURE = "partial_failure"
    DEGRADED_PERFORMANCE = "degraded_performance"
    NO_IMPACT = "no_impact"

# ============================================================================
# ERROR ANALYZER CLASS
# ============================================================================

class ErrorAnalyzer:
    """Analyzes errors comprehensively"""
    
    def __init__(self):
        self.error_patterns = {}
        self.error_history = []
        self.root_causes = {}
        self.dependencies = {}
    
    # ========================================================================
    # ERROR ANALYSIS
    # ========================================================================
    
    def analyze_error(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive error analysis
        
        Args:
            error_info: Error information dict
            
        Returns:
            Detailed analysis dict
        """
        analysis = {
            'error_type': error_info.get('type'),
            'category': error_info.get('category'),
            'severity': error_info.get('severity'),
            'timestamp': error_info.get('timestamp'),
            'message': error_info.get('message'),
            'root_cause': self.find_root_cause(error_info),
            'impact': self.analyze_impact(error_info),
            'dependencies': self.find_dependencies(error_info),
            'similar_errors': self.find_similar_errors(error_info),
            'cascading_risks': self.predict_cascading_errors(error_info),
            'recommendations': self.generate_recommendations(error_info),
            'auto_fixable': error_info.get('auto_fixable', False),
            'fix_type': error_info.get('fix_type'),
            'analysis_timestamp': datetime.now()
        }
        
        self.error_history.append(analysis)
        return analysis
    
    # ========================================================================
    # ROOT CAUSE ANALYSIS
    # ========================================================================
    
    def find_root_cause(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find root cause of error
        
        Args:
            error_info: Error information
            
        Returns:
            Root cause analysis
        """
        error_type = error_info.get('type')
        context = error_info.get('context', {})
        
        root_causes = {
            'SyntaxError': self._analyze_syntax_error_cause,
            'IndentationError': self._analyze_indentation_error_cause,
            'NameError': self._analyze_name_error_cause,
            'TypeError': self._analyze_type_error_cause,
            'ValueError': self._analyze_value_error_cause,
            'InvalidExtractionParams': self._analyze_invalid_params_cause,
            'InvalidStateTransition': self._analyze_state_transition_cause,
            'IncompleteExtraction': self._analyze_incomplete_extraction_cause,
            'ConsentNotGiven': self._analyze_consent_not_given_cause,
            'DeviceOffline': self._analyze_device_offline_cause,
            'StorageFull': self._analyze_storage_full_cause,
        }
        
        analyzer = root_causes.get(error_type)
        if analyzer:
            return analyzer(error_info)
        
        return {
            'probable_cause': 'Unknown',
            'contributing_factors': [],
            'confidence': 0.0
        }
    
    def _analyze_syntax_error_cause(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze syntax error root cause"""
        return {
            'probable_cause': 'Code syntax violation',
            'contributing_factors': [
                'Missing colon',
                'Unclosed bracket',
                'Invalid operator',
                'Incorrect indentation'
            ],
            'line': error_info.get('line'),
            'confidence': 0.95
        }
    
    def _analyze_indentation_error_cause(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze indentation error root cause"""
        return {
            'probable_cause': 'Incorrect code indentation',
            'contributing_factors': [
                'Mixed tabs and spaces',
                'Inconsistent indentation level',
                'Missing indentation'
            ],
            'line': error_info.get('line'),
            'confidence': 0.98
        }
    
    def _analyze_name_error_cause(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze name error root cause"""
        return {
            'probable_cause': 'Undefined variable or function',
            'contributing_factors': [
                'Variable not initialized',
                'Typo in variable name',
                'Variable out of scope',
                'Module not imported'
            ],
            'confidence': 0.85
        }
    
    def _analyze_type_error_cause(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze type error root cause"""
        return {
            'probable_cause': 'Type mismatch in operation',
            'contributing_factors': [
                'Wrong data type passed',
                'Type conversion not performed',
                'Incompatible operation',
                'Missing type check'
            ],
            'confidence': 0.80
        }
    
    def _analyze_value_error_cause(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze value error root cause"""
        return {
            'probable_cause': 'Invalid value provided',
            'contributing_factors': [
                'Value out of range',
                'Invalid format',
                'Null/None value',
                'Missing validation'
            ],
            'confidence': 0.75
        }
    
    def _analyze_invalid_params_cause(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze invalid parameters cause"""
        return {
            'probable_cause': 'Invalid extraction parameters',
            'contributing_factors': [
                'Missing required parameter',
                'Invalid parameter value',
                'Parameter type mismatch',
                'Parameter validation failed'
            ],
            'confidence': 0.90
        }
    
    def _analyze_state_transition_cause(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze state transition error cause"""
        return {
            'probable_cause': 'Invalid state transition',
            'contributing_factors': [
                'Transition not allowed',
                'Precondition not met',
                'State not initialized',
                'Concurrent modification'
            ],
            'confidence': 0.88
        }
    
    def _analyze_incomplete_extraction_cause(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze incomplete extraction cause"""
        return {
            'probable_cause': 'Extraction did not complete all modules',
            'contributing_factors': [
                'Module extraction failed',
                'Timeout during extraction',
                'Device disconnected',
                'Insufficient permissions'
            ],
            'confidence': 0.85
        }
    
    def _analyze_consent_not_given_cause(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consent not given cause"""
        return {
            'probable_cause': 'Extraction consent not provided',
            'contributing_factors': [
                'Nominee did not approve',
                'Approval link expired',
                'Nominee not contacted',
                'Invalid approval process'
            ],
            'confidence': 0.92
        }
    
    def _analyze_device_offline_cause(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze device offline cause"""
        return {
            'probable_cause': 'Device is not connected',
            'contributing_factors': [
                'USB cable disconnected',
                'Device powered off',
                'ADB connection lost',
                'Network connection lost'
            ],
            'confidence': 0.95
        }
    
    def _analyze_storage_full_cause(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze storage full cause"""
        return {
            'probable_cause': 'Insufficient storage space',
            'contributing_factors': [
                'Large files not cleaned up',
                'Old cases not archived',
                'Temporary files accumulated',
                'Database grew too large'
            ],
            'confidence': 0.98
        }
    
    # ========================================================================
    # IMPACT ANALYSIS
    # ========================================================================
    
    def analyze_impact(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze error impact
        
        Args:
            error_info: Error information
            
        Returns:
            Impact analysis
        """
        severity = error_info.get('severity')
        error_type = error_info.get('type')
        
        # Map severity to impact
        impact_map = {
            'CRITICAL': ErrorImpact.SYSTEM_DOWN,
            'HIGH': ErrorImpact.FEATURE_BROKEN,
            'MEDIUM': ErrorImpact.PARTIAL_FAILURE,
            'LOW': ErrorImpact.DEGRADED_PERFORMANCE,
            'INFO': ErrorImpact.NO_IMPACT
        }
        
        impact = impact_map.get(str(severity), ErrorImpact.NO_IMPACT)
        
        return {
            'impact_level': impact,
            'affected_modules': self._get_affected_modules(error_type),
            'user_facing': self._is_user_facing(error_type),
            'data_loss_risk': self._assess_data_loss_risk(error_type),
            'recovery_possible': error_info.get('auto_fixable', False),
            'estimated_downtime': self._estimate_downtime(error_type)
        }
    
    def _get_affected_modules(self, error_type: str) -> List[str]:
        """Get modules affected by error"""
        module_map = {
            'ExtractionError': ['extraction', 'analysis'],
            'ConsentError': ['extraction', 'consent'],
            'StorageError': ['storage', 'extraction', 'report'],
            'DatabaseError': ['database', 'analysis', 'report'],
            'CodeError': ['all'],
        }
        
        for key, modules in module_map.items():
            if key in error_type:
                return modules
        return ['unknown']
    
    def _is_user_facing(self, error_type: str) -> bool:
        """Check if error is user-facing"""
        user_facing_errors = [
            'ConsentNotGiven', 'DeviceOffline', 'StorageFull',
            'ExtractionTimeout', 'ApprovalPending'
        ]
        return error_type in user_facing_errors
    
    def _assess_data_loss_risk(self, error_type: str) -> str:
        """Assess data loss risk"""
        high_risk = ['StorageFull', 'DatabaseError', 'CorruptedData']
        medium_risk = ['PartialExtraction', 'IncompleteExtraction']
        
        if error_type in high_risk:
            return 'high'
        elif error_type in medium_risk:
            return 'medium'
        return 'low'
    
    def _estimate_downtime(self, error_type: str) -> str:
        """Estimate downtime"""
        if 'Critical' in error_type:
            return '> 1 hour'
        elif 'High' in error_type:
            return '15-60 minutes'
        elif 'Medium' in error_type:
            return '5-15 minutes'
        return '< 5 minutes'
    
    # ========================================================================
    # DEPENDENCY ANALYSIS
    # ========================================================================
    
    def find_dependencies(self, error_info: Dict[str, Any]) -> List[str]:
        """
        Find error dependencies
        
        Args:
            error_info: Error information
            
        Returns:
            List of dependent errors
        """
        error_type = error_info.get('type')
        
        dependencies = {
            'DeviceOffline': ['ExtractionFailed', 'ConsentVerificationFailed'],
            'StorageFull': ['ExtractionFailed', 'ReportGenerationFailed'],
            'ConsentNotGiven': ['ExtractionBlocked'],
            'PartialExtraction': ['IncompleteAnalysis', 'InvalidReport'],
        }
        
        return dependencies.get(error_type, [])
    
    # ========================================================================
    # PATTERN DETECTION
    # ========================================================================
    
    def find_similar_errors(self, error_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find similar errors in history
        
        Args:
            error_info: Error information
            
        Returns:
            List of similar errors
        """
        error_type = error_info.get('type')
        similar = []
        
        for hist_error in self.error_history[-100:]:
            if hist_error.get('error_type') == error_type:
                similar.append({
                    'timestamp': hist_error.get('analysis_timestamp'),
                    'message': hist_error.get('message'),
                    'fix_applied': hist_error.get('fix_type')
                })
        
        return similar[-5:]  # Return last 5 similar errors
    
    def predict_cascading_errors(self, error_info: Dict[str, Any]) -> List[str]:
        """
        Predict cascading errors
        
        Args:
            error_info: Error information
            
        Returns:
            List of predicted cascading errors
        """
        error_type = error_info.get('type')
        
        cascading_map = {
            'DeviceOffline': ['ExtractionFailed', 'ConsentVerificationFailed', 'ReportGenerationFailed'],
            'StorageFull': ['ExtractionFailed', 'ReportGenerationFailed', 'AnalysisFailed'],
            'ConsentNotGiven': ['ExtractionBlocked', 'CaseCreationFailed'],
            'PartialExtraction': ['IncompleteAnalysis', 'InvalidReport', 'MissingData'],
            'DatabaseError': ['AllOperationsFailed', 'DataLoss'],
        }
        
        return cascading_map.get(error_type, [])
    
    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================
    
    def generate_recommendations(self, error_info: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations for error resolution
        
        Args:
            error_info: Error information
            
        Returns:
            List of recommendations
        """
        error_type = error_info.get('type')
        
        recommendations = {
            'SyntaxError': [
                'Check code syntax',
                'Review line mentioned in error',
                'Use IDE syntax checker'
            ],
            'DeviceOffline': [
                'Check USB connection',
                'Restart device',
                'Reinstall ADB drivers',
                'Try different USB port'
            ],
            'ConsentNotGiven': [
                'Request consent from nominee',
                'Send approval link',
                'Wait for approval',
                'Follow up with nominee'
            ],
            'StorageFull': [
                'Delete old cases',
                'Archive completed cases',
                'Clear temporary files',
                'Expand storage capacity'
            ],
            'PartialExtraction': [
                'Retry extraction',
                'Check device connectivity',
                'Verify permissions',
                'Check available storage'
            ],
        }
        
        return recommendations.get(error_type, ['Review error details', 'Contact support'])
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        if not self.error_history:
            return {}
        
        total_errors = len(self.error_history)
        error_types = {}
        
        for error in self.error_history:
            error_type = error.get('error_type')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': total_errors,
            'error_types': error_types,
            'most_common': max(error_types, key=error_types.get) if error_types else None,
            'analysis_count': len(self.error_history)
        }
    
    def get_error_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get error trends over time"""
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_history 
                        if e.get('analysis_timestamp', datetime.now()) > cutoff_time]
        
        return {
            'period_hours': hours,
            'errors_in_period': len(recent_errors),
            'trend': 'increasing' if len(recent_errors) > len(self.error_history) / 2 else 'decreasing'
        }

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def create_error_analyzer() -> ErrorAnalyzer:
    """Factory function to create error analyzer"""
    return ErrorAnalyzer()
