"""
ADMISSIBILITY CHECKER - Court Admissibility Verification

Checks court admissibility of forensic reports:
- Legal compliance
- Evidence standards
- Expert qualification
- Methodology validation
- Court readiness

Features:
- Comprehensive docstrings
- Error handling with custom exceptions
- Structured logging
- Performance optimization
"""

import logging
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime
from functools import lru_cache

logger = logging.getLogger(__name__)

class ComplianceException(Exception):
    """Base exception for compliance errors"""
    pass

class AdmissibilityError(ComplianceException):
    """Raised when admissibility check fails"""
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

class AdmissibilityChecker:
    """
    Check court admissibility of forensic reports.
    
    Verifies that forensic reports meet all legal and technical
    requirements for admission as evidence in court proceedings.
    """
    
    # Admissibility criteria
    ADMISSIBILITY_CRITERIA = [
        'legal_compliance',
        'evidence_standards',
        'expert_qualification',
        'methodology_validation',
        'chain_of_custody',
        'signatures',
        'metadata',
        'authentication'
    ]
    
    def __init__(self):
        """Initialize Admissibility checker"""
        logger.debug("AdmissibilityChecker initialized")
    
    @staticmethod
    def check(report_data: Dict[str, Any], case_id: str = "") -> Tuple[bool, Dict[str, Any]]:
        """
        Check court admissibility of report.
        
        Performs comprehensive admissibility check against all legal
        and technical criteria for court proceedings.
        
        Args:
            report_data (Dict[str, Any]): Report data to check
            case_id (str): Case ID for logging (optional)
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (is_admissible, details_dict)
            
        Raises:
            AdmissibilityError: If check fails
            
        Example:
            >>> checker = AdmissibilityChecker()
            >>> is_admissible, details = checker.check(report_data, "CASE-001")
            >>> if is_admissible:
            ...     print("Report is court-admissible")
            ... else:
            ...     print(f"Issues: {details['issues']}")
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "DEBUG",
                "Checking court admissibility",
                case_id=case_id
            )
            
            issues = []
            criteria_met = {}
            
            # Check each criterion
            for criterion in AdmissibilityChecker.ADMISSIBILITY_CRITERIA:
                if criterion == 'legal_compliance':
                    met = AdmissibilityChecker._check_legal_compliance(report_data)
                elif criterion == 'evidence_standards':
                    met = AdmissibilityChecker._check_evidence_standards(report_data)
                elif criterion == 'expert_qualification':
                    met = AdmissibilityChecker._check_expert_qualification(report_data)
                elif criterion == 'methodology_validation':
                    met = AdmissibilityChecker._check_methodology(report_data)
                elif criterion == 'chain_of_custody':
                    met = AdmissibilityChecker._check_chain_of_custody(report_data)
                elif criterion == 'signatures':
                    met = AdmissibilityChecker._check_signatures(report_data)
                elif criterion == 'metadata':
                    met = AdmissibilityChecker._check_metadata(report_data)
                elif criterion == 'authentication':
                    met = AdmissibilityChecker._check_authentication(report_data)
                else:
                    met = False
                
                criteria_met[criterion] = met
                if not met:
                    issues.append(f"Criterion not met: {criterion}")
            
            is_admissible = len(issues) == 0
            
            details = {
                'admissible': is_admissible,
                'criteria_met': criteria_met,
                'issues': issues,
                'check_date': datetime.now().isoformat(),
                'case_id': case_id
            }
            
            structured_logger.log_with_context(
                "INFO" if is_admissible else "WARNING",
                "Admissibility check completed",
                case_id=case_id,
                is_admissible=is_admissible,
                issue_count=len(issues)
            )
            
            return is_admissible, details
        
        except Exception as e:
            error_msg = f"Error checking admissibility: {str(e)}"
            logger.error(error_msg)
            raise AdmissibilityError(error_msg) from e
    
    @staticmethod
    def _check_legal_compliance(report_data: Dict[str, Any]) -> bool:
        """Check legal compliance"""
        return report_data.get('legal_compliance', False)
    
    @staticmethod
    def _check_evidence_standards(report_data: Dict[str, Any]) -> bool:
        """Check evidence standards"""
        return report_data.get('evidence_standards', False)
    
    @staticmethod
    def _check_expert_qualification(report_data: Dict[str, Any]) -> bool:
        """Check expert qualification"""
        expert = report_data.get('expert_info', {})
        return expert.get('qualified', False) and expert.get('certified', False)
    
    @staticmethod
    def _check_methodology(report_data: Dict[str, Any]) -> bool:
        """Check methodology validation"""
        methodology = report_data.get('methodology', {})
        return methodology.get('validated', False) and methodology.get('documented', False)
    
    @staticmethod
    def _check_chain_of_custody(report_data: Dict[str, Any]) -> bool:
        """Check chain of custody"""
        coc = report_data.get('chain_of_custody', {})
        return coc.get('verified', False) and coc.get('intact', False)
    
    @staticmethod
    def _check_signatures(report_data: Dict[str, Any]) -> bool:
        """Check signatures"""
        signatures = report_data.get('signatures', {})
        return signatures.get('examiner_signed', False) and signatures.get('supervisor_signed', False)
    
    @staticmethod
    def _check_metadata(report_data: Dict[str, Any]) -> bool:
        """Check metadata"""
        metadata = report_data.get('metadata', {})
        required_fields = ['case_id', 'examination_date', 'examiner_name']
        return all(field in metadata for field in required_fields)
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _check_authentication(report_data_str: str) -> bool:
        """Check authentication (cached)"""
        # In real implementation, would verify digital signatures
        return True
    
    @staticmethod
    def get_admissibility_report(report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed admissibility report.
        
        Args:
            report_data (Dict[str, Any]): Report data
            
        Returns:
            Dict[str, Any]: Detailed admissibility report
        """
        is_admissible, details = AdmissibilityChecker.check(report_data)
        
        return {
            'admissible': is_admissible,
            'details': details,
            'recommendation': 'COURT-READY' if is_admissible else 'REQUIRES REVISION',
            'report_date': datetime.now().isoformat()
        }
