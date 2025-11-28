"""
ERROR LEARNER - Learns from errors to improve

Provides:
- Pattern learning
- Root cause analysis
- Solution optimization
- Prevention rule generation
- Predictive modeling
- Continuous improvement
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

# ============================================================================
# ERROR LEARNER CLASS
# ============================================================================

class ErrorLearner:
    """Learns from errors to improve error handling"""
    
    def __init__(self):
        self.error_patterns = defaultdict(list)
        self.knowledge_base = {}
        self.error_solutions = {}
        self.learning_history = []
        self.max_history = 1000
        self.confidence_threshold = 0.7
    
    # ========================================================================
    # LEARNING FROM ERRORS
    # ========================================================================
    
    def learn_from_error(self, error_info: Dict[str, Any], fix_applied: str, 
                        result: bool) -> Dict[str, Any]:
        """
        Learn from error and fix outcome
        
        Args:
            error_info: Error information
            fix_applied: Fix that was applied
            result: Whether fix was successful
            
        Returns:
            Learning result
        """
        error_type = error_info.get('type')
        
        # Record learning
        learning_record = {
            'error_type': error_type,
            'error_message': error_info.get('message'),
            'fix_applied': fix_applied,
            'success': result,
            'timestamp': datetime.now(),
            'context': error_info.get('context', {})
        }
        
        self.learning_history.append(learning_record)
        if len(self.learning_history) > self.max_history:
            self.learning_history.pop(0)
        
        # Update knowledge base
        self._update_knowledge_base(error_type, fix_applied, result)
        
        # Update error patterns
        self._update_error_patterns(error_type, error_info)
        
        # Update solutions
        self._update_solutions(error_type, fix_applied, result)
        
        return {
            'learned': True,
            'error_type': error_type,
            'fix_applied': fix_applied,
            'success': result,
            'timestamp': datetime.now()
        }
    
    def _update_knowledge_base(self, error_type: str, fix_applied: str, success: bool) -> None:
        """Update knowledge base with new learning"""
        if error_type not in self.knowledge_base:
            self.knowledge_base[error_type] = {
                'total_occurrences': 0,
                'successful_fixes': 0,
                'failed_fixes': 0,
                'fixes_tried': defaultdict(lambda: {'success': 0, 'fail': 0})
            }
        
        kb = self.knowledge_base[error_type]
        kb['total_occurrences'] += 1
        
        if success:
            kb['successful_fixes'] += 1
            kb['fixes_tried'][fix_applied]['success'] += 1
        else:
            kb['failed_fixes'] += 1
            kb['fixes_tried'][fix_applied]['fail'] += 1
    
    def _update_error_patterns(self, error_type: str, error_info: Dict[str, Any]) -> None:
        """Update error patterns"""
        pattern = {
            'type': error_type,
            'message': error_info.get('message'),
            'severity': error_info.get('severity'),
            'timestamp': datetime.now()
        }
        
        self.error_patterns[error_type].append(pattern)
    
    def _update_solutions(self, error_type: str, fix_applied: str, success: bool) -> None:
        """Update solution effectiveness"""
        if error_type not in self.error_solutions:
            self.error_solutions[error_type] = {}
        
        if fix_applied not in self.error_solutions[error_type]:
            self.error_solutions[error_type][fix_applied] = {
                'attempts': 0,
                'successes': 0,
                'effectiveness': 0.0
            }
        
        solution = self.error_solutions[error_type][fix_applied]
        solution['attempts'] += 1
        
        if success:
            solution['successes'] += 1
        
        # Calculate effectiveness
        solution['effectiveness'] = (solution['successes'] / solution['attempts']) if solution['attempts'] > 0 else 0
    
    # ========================================================================
    # PATTERN ANALYSIS
    # ========================================================================
    
    def analyze_error_patterns(self) -> Dict[str, Any]:
        """
        Analyze error patterns
        
        Returns:
            Pattern analysis
        """
        analysis = {
            'total_errors': len(self.learning_history),
            'unique_error_types': len(self.error_patterns),
            'most_common_errors': self._get_most_common_errors(),
            'error_frequency': self._get_error_frequency(),
            'error_trends': self._get_error_trends(),
            'timestamp': datetime.now()
        }
        
        return analysis
    
    def _get_most_common_errors(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get most common errors"""
        error_counts = Counter(e.get('error_type') for e in self.learning_history)
        return error_counts.most_common(limit)
    
    def _get_error_frequency(self) -> Dict[str, int]:
        """Get error frequency"""
        frequency = defaultdict(int)
        
        for error in self.learning_history:
            error_type = error.get('error_type')
            frequency[error_type] += 1
        
        return dict(frequency)
    
    def _get_error_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get error trends over time"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [e for e in self.learning_history 
                        if e.get('timestamp', datetime.now()) > cutoff_time]
        
        successful = sum(1 for e in recent_errors if e.get('success'))
        total = len(recent_errors)
        
        return {
            'period_hours': hours,
            'total_errors': total,
            'successful_fixes': successful,
            'failed_fixes': total - successful,
            'success_rate': (successful / total * 100) if total > 0 else 0
        }
    
    # ========================================================================
    # ROOT CAUSE ANALYSIS
    # ========================================================================
    
    def find_root_causes(self) -> Dict[str, Any]:
        """
        Find root causes of errors
        
        Returns:
            Root cause analysis
        """
        root_causes = {}
        
        for error_type, patterns in self.error_patterns.items():
            if not patterns:
                continue
            
            # Analyze patterns
            messages = [p.get('message') for p in patterns]
            message_freq = Counter(messages)
            
            # Get most common message
            most_common_msg = message_freq.most_common(1)
            
            root_causes[error_type] = {
                'most_common_message': most_common_msg[0][0] if most_common_msg else None,
                'frequency': len(patterns),
                'patterns': patterns[-5:]  # Last 5 occurrences
            }
        
        return root_causes
    
    # ========================================================================
    # SOLUTION OPTIMIZATION
    # ========================================================================
    
    def get_best_solution(self, error_type: str) -> Optional[str]:
        """
        Get best solution for error type
        
        Args:
            error_type: Type of error
            
        Returns:
            Best solution or None
        """
        if error_type not in self.error_solutions:
            return None
        
        solutions = self.error_solutions[error_type]
        
        # Find solution with highest effectiveness
        best_solution = None
        best_effectiveness = 0
        
        for solution, metrics in solutions.items():
            if metrics['effectiveness'] > best_effectiveness:
                best_effectiveness = metrics['effectiveness']
                best_solution = solution
        
        return best_solution if best_effectiveness > 0 else None
    
    def get_solution_effectiveness(self, error_type: str) -> Dict[str, Any]:
        """
        Get solution effectiveness for error type
        
        Args:
            error_type: Type of error
            
        Returns:
            Solution effectiveness metrics
        """
        if error_type not in self.error_solutions:
            return {}
        
        solutions = self.error_solutions[error_type]
        
        effectiveness = {}
        for solution, metrics in solutions.items():
            effectiveness[solution] = {
                'attempts': metrics['attempts'],
                'successes': metrics['successes'],
                'effectiveness': f"{metrics['effectiveness']*100:.1f}%"
            }
        
        return effectiveness
    
    # ========================================================================
    # PREVENTION RULE GENERATION
    # ========================================================================
    
    def generate_prevention_rules(self) -> List[Dict[str, Any]]:
        """
        Generate prevention rules from patterns
        
        Returns:
            List of prevention rules
        """
        rules = []
        
        for error_type, patterns in self.error_patterns.items():
            if len(patterns) < 3:  # Need at least 3 occurrences
                continue
            
            # Analyze patterns
            frequency = len(patterns)
            
            if frequency > 10:  # Frequently occurring error
                rule = {
                    'type': 'prevention',
                    'error_type': error_type,
                    'action': f'Add validation for {error_type}',
                    'priority': 'high' if frequency > 20 else 'medium',
                    'confidence': min(frequency / 100, 1.0),
                    'timestamp': datetime.now()
                }
                rules.append(rule)
        
        return rules
    
    # ========================================================================
    # PREDICTIVE MODELING
    # ========================================================================
    
    def predict_future_errors(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Predict future errors based on context
        
        Args:
            context: Current context
            
        Returns:
            List of predicted errors
        """
        predictions = []
        
        # Check for patterns that lead to errors
        for error_type, patterns in self.error_patterns.items():
            # Simple prediction: if we see similar context, predict error
            for pattern in patterns[-5:]:  # Check last 5 occurrences
                pattern_context = pattern.get('context', {})
                
                # Compare contexts
                similarity = self._calculate_context_similarity(context, pattern_context)
                
                if similarity > self.confidence_threshold:
                    predictions.append({
                        'predicted_error': error_type,
                        'confidence': similarity,
                        'reason': 'Similar context detected',
                        'timestamp': datetime.now()
                    })
        
        return predictions
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], 
                                     context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts"""
        if not context1 or not context2:
            return 0.0
        
        matching_keys = 0
        total_keys = max(len(context1), len(context2))
        
        for key in context1:
            if key in context2 and context1[key] == context2[key]:
                matching_keys += 1
        
        return matching_keys / total_keys if total_keys > 0 else 0.0
    
    # ========================================================================
    # CONTINUOUS IMPROVEMENT
    # ========================================================================
    
    def improve_error_detection(self) -> Dict[str, Any]:
        """
        Improve error detection based on learning
        
        Returns:
            Improvement recommendations
        """
        improvements = {
            'detection_improvements': [],
            'prevention_improvements': [],
            'fix_improvements': []
        }
        
        # Analyze patterns
        patterns = self.analyze_error_patterns()
        
        # Recommend detection improvements
        for error_type, count in patterns['most_common_errors']:
            if count > 5:
                improvements['detection_improvements'].append({
                    'error_type': error_type,
                    'recommendation': f'Add specific detector for {error_type}',
                    'occurrences': count
                })
        
        # Recommend prevention improvements
        prevention_rules = self.generate_prevention_rules()
        improvements['prevention_improvements'] = prevention_rules
        
        # Recommend fix improvements
        for error_type, solutions in self.error_solutions.items():
            best_solution = self.get_best_solution(error_type)
            if best_solution:
                effectiveness = solutions[best_solution]['effectiveness']
                if effectiveness < 0.8:
                    improvements['fix_improvements'].append({
                        'error_type': error_type,
                        'current_solution': best_solution,
                        'effectiveness': f"{effectiveness*100:.1f}%",
                        'recommendation': 'Consider alternative fix strategies'
                    })
        
        return improvements
    
    def improve_error_fixes(self) -> Dict[str, Any]:
        """
        Improve error fixes based on learning
        
        Returns:
            Fix improvement recommendations
        """
        improvements = {}
        
        for error_type, solutions in self.error_solutions.items():
            best_solution = self.get_best_solution(error_type)
            
            if best_solution:
                effectiveness = solutions[best_solution]['effectiveness']
                
                if effectiveness < 1.0:
                    improvements[error_type] = {
                        'current_best': best_solution,
                        'effectiveness': f"{effectiveness*100:.1f}%",
                        'alternative_solutions': [
                            s for s in solutions.keys() if s != best_solution
                        ]
                    }
        
        return improvements
    
    def optimize_prevention(self) -> Dict[str, Any]:
        """
        Optimize prevention strategies
        
        Returns:
            Prevention optimization recommendations
        """
        optimization = {
            'high_priority_errors': [],
            'prevention_rules': self.generate_prevention_rules(),
            'resource_allocation': {}
        }
        
        # Identify high priority errors
        patterns = self.analyze_error_patterns()
        for error_type, count in patterns['most_common_errors'][:3]:
            optimization['high_priority_errors'].append({
                'error_type': error_type,
                'occurrences': count,
                'priority': 'high'
            })
        
        # Allocate resources
        total_errors = patterns['total_errors']
        for error_type, count in patterns['most_common_errors']:
            percentage = (count / total_errors * 100) if total_errors > 0 else 0
            optimization['resource_allocation'][error_type] = f"{percentage:.1f}%"
        
        return optimization
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get learning summary"""
        return {
            'total_learning_records': len(self.learning_history),
            'unique_error_types': len(self.knowledge_base),
            'knowledge_base_size': len(self.knowledge_base),
            'error_patterns_tracked': len(self.error_patterns),
            'solutions_learned': len(self.error_solutions),
            'timestamp': datetime.now()
        }
    
    def export_knowledge_base(self) -> Dict[str, Any]:
        """Export knowledge base"""
        return {
            'knowledge_base': dict(self.knowledge_base),
            'error_solutions': dict(self.error_solutions),
            'timestamp': datetime.now()
        }
    
    def clear_learning_history(self) -> None:
        """Clear learning history"""
        self.learning_history = []
        logger.info("Cleared learning history")

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def create_error_learner() -> ErrorLearner:
    """Factory function to create error learner"""
    return ErrorLearner()
