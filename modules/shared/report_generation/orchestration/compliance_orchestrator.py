"""
COMPLIANCE ORCHESTRATOR - Compliance Validation Orchestration

Orchestrates compliance validation:
- IT Act 2000 Validator
- Evidence Act 1872 Validator
- Chain of Custody Validator
- Signature Validator
- Admissibility Checker

Features:
- Comprehensive docstrings
- Error handling with custom exceptions
- Structured logging
- Performance optimization
"""

import logging
import json
from typing import Dict, Any, Tuple, List
from datetime import datetime
from functools import lru_cache

logger = logging.getLogger(__name__)

class OrchestratorException(Exception):
    """Base exception for orchestrator errors"""
    pass

class ComplianceError(OrchestratorException):
    """Raised when compliance check fails"""
    pass

class StructuredLogger:
    """Structured logging with JSON context"""
    
    @staticmethod
    def log_with_context(level: str, message: str, **context) -> None:
        """Log with context information"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message,
                'context': context
            }
            log_level = getattr(logging, level.upper(), logging.INFO)
            logger.log(log_level, json.dumps(log_entry))
        except Exception as e:
            logger.error(f"Error in structured logging: {str(e)}")

class ComplianceOrchestrator:
    """
    Orchestrate compliance validation.
    
    Coordinates comprehensive compliance validation of forensic reports
    against Indian legal standards and evidence requirements.
    """
    
    def __init__(self):
        """Initialize Compliance Orchestrator"""
        logger.debug("ComplianceOrchestrator initialized")
    
    def validate(self, report_data: Dict[str, Any], case_id: str = "") -> Tuple[bool, Dict[str, Any]]:
        """
        Validate report compliance.
        
        Orchestrates comprehensive compliance validation including IT Act,
        Evidence Act, chain of custody, signatures, and admissibility checks.
        
        Args:
            report_data (Dict[str, Any]): Report data to validate
            case_id (str): Case ID for logging (optional)
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (is_compliant, validation_results)
            
        Raises:
            ComplianceError: If validation fails
            
        Example:
            >>> orchestrator = ComplianceOrchestrator()
            >>> is_compliant, results = orchestrator.validate(report_data, "CASE-001")
            >>> if is_compliant:
            ...     print("Report is court-ready")
            ... else:
            ...     print(f"Issues: {results['issues']}")
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "INFO",
                "Starting compliance validation",
                case_id=case_id
            )
            
            validation_results = {
                'case_id': case_id,
                'validation_date': datetime.now().isoformat(),
                'validators': {},
                'issues': [],
                'compliant': True
            }
            
            # Run IT Act validation
            try:
                from ..compliance import ITActValidator
                validator = ITActValidator()
                is_valid, errors = validator.validate(report_data, case_id)
                validation_results['validators']['it_act'] = {
                    'compliant': is_valid,
                    'errors': errors
                }
                if not is_valid:
                    validation_results['issues'].extend(errors)
                    validation_results['compliant'] = False
                structured_logger.log_with_context(
                    "DEBUG",
                    "IT Act validation completed",
                    case_id=case_id,
                    compliant=is_valid
                )
            except Exception as e:
                logger.error(f"Error in IT Act validation: {e}")
                validation_results['validators']['it_act'] = {'error': str(e)}
            
            # Run Evidence Act validation
            try:
                from ..compliance import EvidenceActValidator
                validator = EvidenceActValidator()
                is_valid, errors = validator.validate(report_data, case_id)
                validation_results['validators']['evidence_act'] = {
                    'compliant': is_valid,
                    'errors': errors
                }
                if not is_valid:
                    validation_results['issues'].extend(errors)
                    validation_results['compliant'] = False
                structured_logger.log_with_context(
                    "DEBUG",
                    "Evidence Act validation completed",
                    case_id=case_id,
                    compliant=is_valid
                )
            except Exception as e:
                logger.error(f"Error in Evidence Act validation: {e}")
                validation_results['validators']['evidence_act'] = {'error': str(e)}
            
            # Run Chain of Custody validation
            try:
                from ..compliance import ChainOfCustodyValidator
                validator = ChainOfCustodyValidator()
                coc_data = report_data.get('chain_of_custody', {})
                is_valid, errors = validator.validate(coc_data, case_id)
                validation_results['validators']['chain_of_custody'] = {
                    'compliant': is_valid,
                    'errors': errors
                }
                if not is_valid:
                    validation_results['issues'].extend(errors)
                    validation_results['compliant'] = False
                structured_logger.log_with_context(
                    "DEBUG",
                    "Chain of custody validation completed",
                    case_id=case_id,
                    compliant=is_valid
                )
            except Exception as e:
                logger.error(f"Error in Chain of Custody validation: {e}")
                validation_results['validators']['chain_of_custody'] = {'error': str(e)}
            
            # Run Signature validation
            try:
                from ..compliance import SignatureValidator
                validator = SignatureValidator()
                sig_data = report_data.get('signatures', {})
                is_valid, errors = validator.validate(sig_data, case_id)
                validation_results['validators']['signatures'] = {
                    'compliant': is_valid,
                    'errors': errors
                }
                if not is_valid:
                    validation_results['issues'].extend(errors)
                    validation_results['compliant'] = False
                structured_logger.log_with_context(
                    "DEBUG",
                    "Signature validation completed",
                    case_id=case_id,
                    compliant=is_valid
                )
            except Exception as e:
                logger.error(f"Error in Signature validation: {e}")
                validation_results['validators']['signatures'] = {'error': str(e)}
            
            # Run Admissibility check
            try:
                from ..compliance import AdmissibilityChecker
                checker = AdmissibilityChecker()
                is_admissible, details = checker.check(report_data, case_id)
                validation_results['validators']['admissibility'] = {
                    'compliant': is_admissible,
                    'details': details
                }
                if not is_admissible:
                    validation_results['issues'].extend(details.get('issues', []))
                    validation_results['compliant'] = False
                structured_logger.log_with_context(
                    "DEBUG",
                    "Admissibility check completed",
                    case_id=case_id,
                    admissible=is_admissible
                )
            except Exception as e:
                logger.error(f"Error in Admissibility check: {e}")
                validation_results['validators']['admissibility'] = {'error': str(e)}
            
            # Determine overall status
            validation_results['status'] = 'COURT-READY' if validation_results['compliant'] else 'REQUIRES REVISION'
            validation_results['issue_count'] = len(validation_results['issues'])
            
            structured_logger.log_with_context(
                "INFO",
                "Compliance validation completed",
                case_id=case_id,
                compliant=validation_results['compliant'],
                status=validation_results['status'],
                issue_count=validation_results['issue_count']
            )
            
            return validation_results['compliant'], validation_results
        
        except Exception as e:
            error_msg = f"Error validating compliance: {str(e)}"
            logger.error(error_msg)
            raise ComplianceError(error_msg) from e
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _get_status_message(is_compliant: bool) -> str:
        """Get status message (cached)"""
        return 'COURT-READY' if is_compliant else 'REQUIRES REVISION'
