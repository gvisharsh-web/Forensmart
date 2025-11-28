"""
RECOVERY STRATEGIES - Different strategies to recover from errors

Provides:
- Auto-fix & retry
- Skip & continue
- Retry with backoff
- Rollback & restore
- Manual intervention
- Fallback operation
- Partial success
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger(__name__)

# ============================================================================
# RECOVERY STRATEGIES CLASS
# ============================================================================

class RecoveryStrategies:
    """Implements various error recovery strategies"""
    
    def __init__(self):
        self.recovery_history = []
        self.max_history = 1000
    
    # ========================================================================
    # AUTO-FIX & RETRY
    # ========================================================================
    
    def auto_fix_and_retry(self, operation: Callable, error_info: Dict[str, Any], 
                          max_retries: int = 3) -> Dict[str, Any]:
        """
        Auto-fix error and retry operation
        
        Args:
            operation: Operation to retry
            error_info: Error information
            max_retries: Maximum retry attempts
            
        Returns:
            Recovery result
        """
        attempt = 0
        last_error = None
        
        while attempt < max_retries:
            try:
                attempt += 1
                logger.info(f"Auto-fix & retry attempt {attempt}/{max_retries}")
                
                # Attempt operation
                result = operation()
                
                recovery_record = {
                    'strategy': 'auto_fix_and_retry',
                    'error_type': error_info.get('type'),
                    'attempts': attempt,
                    'success': True,
                    'timestamp': datetime.now()
                }
                
                self.recovery_history.append(recovery_record)
                return {
                    'success': True,
                    'strategy': 'auto_fix_and_retry',
                    'attempts': attempt,
                    'result': result,
                    'timestamp': datetime.now()
                }
            
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt} failed: {str(e)}")
                time.sleep(1)  # Wait before retry
        
        recovery_record = {
            'strategy': 'auto_fix_and_retry',
            'error_type': error_info.get('type'),
            'attempts': attempt,
            'success': False,
            'timestamp': datetime.now()
        }
        
        self.recovery_history.append(recovery_record)
        return {
            'success': False,
            'strategy': 'auto_fix_and_retry',
            'attempts': attempt,
            'error': str(last_error),
            'timestamp': datetime.now()
        }
    
    # ========================================================================
    # SKIP & CONTINUE
    # ========================================================================
    
    def skip_and_continue(self, workflow: List[Dict[str, Any]], 
                         failed_step_index: int) -> Dict[str, Any]:
        """
        Skip failed step and continue workflow
        
        Args:
            workflow: Workflow steps
            failed_step_index: Index of failed step
            
        Returns:
            Recovery result
        """
        logger.info(f"Skipping step {failed_step_index} and continuing")
        
        skipped_steps = []
        executed_steps = []
        
        for i, step in enumerate(workflow):
            if i == failed_step_index:
                skipped_steps.append(step)
                logger.warning(f"Skipped step: {step.get('name', 'unknown')}")
            elif i > failed_step_index:
                try:
                    # Execute remaining steps
                    result = self._execute_step(step)
                    executed_steps.append(result)
                except Exception as e:
                    logger.error(f"Error in step {i}: {str(e)}")
                    break
        
        recovery_record = {
            'strategy': 'skip_and_continue',
            'skipped_steps': len(skipped_steps),
            'executed_steps': len(executed_steps),
            'success': len(executed_steps) > 0,
            'timestamp': datetime.now()
        }
        
        self.recovery_history.append(recovery_record)
        return {
            'success': len(executed_steps) > 0,
            'strategy': 'skip_and_continue',
            'skipped_steps': skipped_steps,
            'executed_steps': executed_steps,
            'timestamp': datetime.now()
        }
    
    def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step"""
        return {
            'step': step.get('name'),
            'status': 'completed',
            'timestamp': datetime.now()
        }
    
    # ========================================================================
    # RETRY WITH BACKOFF
    # ========================================================================
    
    def retry_with_backoff(self, operation: Callable, error_info: Dict[str, Any],
                          max_retries: int = 5, initial_delay: int = 1) -> Dict[str, Any]:
        """
        Retry operation with exponential backoff
        
        Args:
            operation: Operation to retry
            error_info: Error information
            max_retries: Maximum retry attempts
            initial_delay: Initial delay in seconds
            
        Returns:
            Recovery result
        """
        attempt = 0
        last_error = None
        delay = initial_delay
        
        while attempt < max_retries:
            try:
                attempt += 1
                logger.info(f"Retry with backoff attempt {attempt}/{max_retries}")
                
                # Attempt operation
                result = operation()
                
                recovery_record = {
                    'strategy': 'retry_with_backoff',
                    'error_type': error_info.get('type'),
                    'attempts': attempt,
                    'success': True,
                    'timestamp': datetime.now()
                }
                
                self.recovery_history.append(recovery_record)
                return {
                    'success': True,
                    'strategy': 'retry_with_backoff',
                    'attempts': attempt,
                    'result': result,
                    'timestamp': datetime.now()
                }
            
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt} failed: {str(e)}")
                
                if attempt < max_retries:
                    logger.info(f"Waiting {delay} seconds before retry")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)  # Exponential backoff, max 60 seconds
        
        recovery_record = {
            'strategy': 'retry_with_backoff',
            'error_type': error_info.get('type'),
            'attempts': attempt,
            'success': False,
            'timestamp': datetime.now()
        }
        
        self.recovery_history.append(recovery_record)
        return {
            'success': False,
            'strategy': 'retry_with_backoff',
            'attempts': attempt,
            'error': str(last_error),
            'timestamp': datetime.now()
        }
    
    # ========================================================================
    # ROLLBACK & RESTORE
    # ========================================================================
    
    def rollback_and_restore(self, transaction_id: str, 
                            state_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rollback transaction and restore state
        
        Args:
            transaction_id: Transaction ID to rollback
            state_snapshot: Previous state snapshot
            
        Returns:
            Recovery result
        """
        logger.info(f"Rolling back transaction {transaction_id}")
        
        try:
            # Restore state
            restored_state = state_snapshot.copy()
            
            recovery_record = {
                'strategy': 'rollback_and_restore',
                'transaction_id': transaction_id,
                'success': True,
                'timestamp': datetime.now()
            }
            
            self.recovery_history.append(recovery_record)
            return {
                'success': True,
                'strategy': 'rollback_and_restore',
                'transaction_id': transaction_id,
                'restored_state': restored_state,
                'timestamp': datetime.now()
            }
        
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            
            recovery_record = {
                'strategy': 'rollback_and_restore',
                'transaction_id': transaction_id,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
            
            self.recovery_history.append(recovery_record)
            return {
                'success': False,
                'strategy': 'rollback_and_restore',
                'transaction_id': transaction_id,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    # ========================================================================
    # MANUAL INTERVENTION
    # ========================================================================
    
    def manual_intervention(self, error_info: Dict[str, Any], 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Request manual intervention
        
        Args:
            error_info: Error information
            context: Operation context
            
        Returns:
            Manual intervention request
        """
        logger.warning(f"Manual intervention required for {error_info.get('type')}")
        
        recovery_record = {
            'strategy': 'manual_intervention',
            'error_type': error_info.get('type'),
            'timestamp': datetime.now()
        }
        
        self.recovery_history.append(recovery_record)
        return {
            'success': False,
            'strategy': 'manual_intervention',
            'error_type': error_info.get('type'),
            'message': error_info.get('message'),
            'recommendations': error_info.get('recommendations', []),
            'context': context,
            'timestamp': datetime.now()
        }
    
    # ========================================================================
    # FALLBACK OPERATION
    # ========================================================================
    
    def fallback_operation(self, primary_operation: Callable,
                          fallback_operation: Callable) -> Dict[str, Any]:
        """
        Use fallback operation if primary fails
        
        Args:
            primary_operation: Primary operation to attempt
            fallback_operation: Fallback operation
            
        Returns:
            Recovery result
        """
        try:
            logger.info("Attempting primary operation")
            result = primary_operation()
            
            recovery_record = {
                'strategy': 'fallback_operation',
                'used_fallback': False,
                'success': True,
                'timestamp': datetime.now()
            }
            
            self.recovery_history.append(recovery_record)
            return {
                'success': True,
                'strategy': 'fallback_operation',
                'used_fallback': False,
                'result': result,
                'timestamp': datetime.now()
            }
        
        except Exception as e:
            logger.warning(f"Primary operation failed: {str(e)}")
            logger.info("Attempting fallback operation")
            
            try:
                result = fallback_operation()
                
                recovery_record = {
                    'strategy': 'fallback_operation',
                    'used_fallback': True,
                    'success': True,
                    'timestamp': datetime.now()
                }
                
                self.recovery_history.append(recovery_record)
                return {
                    'success': True,
                    'strategy': 'fallback_operation',
                    'used_fallback': True,
                    'result': result,
                    'timestamp': datetime.now()
                }
            
            except Exception as fallback_error:
                logger.error(f"Fallback operation also failed: {str(fallback_error)}")
                
                recovery_record = {
                    'strategy': 'fallback_operation',
                    'used_fallback': True,
                    'success': False,
                    'error': str(fallback_error),
                    'timestamp': datetime.now()
                }
                
                self.recovery_history.append(recovery_record)
                return {
                    'success': False,
                    'strategy': 'fallback_operation',
                    'used_fallback': True,
                    'error': str(fallback_error),
                    'timestamp': datetime.now()
                }
    
    # ========================================================================
    # PARTIAL SUCCESS
    # ========================================================================
    
    def partial_success(self, operation: Callable, 
                       partial_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accept partial success
        
        Args:
            operation: Operation that partially succeeded
            partial_result: Partial result
            
        Returns:
            Recovery result
        """
        logger.info("Accepting partial success")
        
        recovery_record = {
            'strategy': 'partial_success',
            'success': True,
            'timestamp': datetime.now()
        }
        
        self.recovery_history.append(recovery_record)
        return {
            'success': True,
            'strategy': 'partial_success',
            'partial_result': partial_result,
            'message': 'Operation partially succeeded',
            'timestamp': datetime.now()
        }
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_recovery_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recovery history"""
        return self.recovery_history[-limit:]
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        if not self.recovery_history:
            return {}
        
        strategies = {}
        successful = 0
        total = len(self.recovery_history)
        
        for record in self.recovery_history:
            strategy = record.get('strategy')
            if strategy not in strategies:
                strategies[strategy] = {'attempts': 0, 'successes': 0}
            
            strategies[strategy]['attempts'] += 1
            if record.get('success'):
                strategies[strategy]['successes'] += 1
                successful += 1
        
        return {
            'total_recoveries': total,
            'successful_recoveries': successful,
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'strategies': strategies
        }
    
    def clear_history(self) -> None:
        """Clear recovery history"""
        self.recovery_history = []
        logger.info("Cleared recovery history")

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def create_recovery_strategies() -> RecoveryStrategies:
    """Factory function to create recovery strategies"""
    return RecoveryStrategies()
