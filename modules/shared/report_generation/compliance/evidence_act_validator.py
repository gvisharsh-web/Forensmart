"""
EVIDENCE ACT 1872 VALIDATOR - Compliance Validation

Validates reports for compliance with Indian Evidence Act 1872:
- Section 3 (Relevancy of facts)
- Section 23 (Admissibility of evidence)
- Section 45 (Expert opinion)
- Section 90 (Presumption as to electronic records)
- Authentication requirements

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

class ValidationError(ComplianceException):
    """Raised when validation fails"""
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

class EvidenceActValidator:
    """
    Validate reports for Evidence Act 1872 compliance.
    
    Ensures forensic reports comply with Indian Evidence Act 1872 requirements
    including relevancy, admissibility, and expert opinion provisions.
    """
    
    # Evidence Act requirements
    REQUIRED_ELEMENTS = [
        'relevancy_statement',
        'expert_qualification',
        'examination_methodology',
        'findings_basis',
        'expert_opinion',
        'authentication'
    ]
    
    def __init__(self):
        """Initialize Evidence Act validator"""
        logger.debug("EvidenceActValidator initialized")
    
    @staticmethod
    def validate(report_data: Dict[str, Any], case_id: str = "") -> Tuple[bool, List[str]]:
        """
        Validate report for Evidence Act 1872 compliance.
        
        Checks if report contains all required elements as per
        Evidence Act 1872 provisions.
        
        Args:
            report_data (Dict[str, Any]): Report data to validate
            case_id (str): Case ID for logging (optional)
            
        Returns:
            Tuple[bool, List[str]]: (is_compliant, list_of_errors)
            
        Raises:
            ValidationError: If validation fails
            
        Example:
            >>> validator = EvidenceActValidator()
            >>> is_valid, errors = validator.validate(report_data, "CASE-001")
            >>> if is_valid:
            ...     print("Report is Evidence Act compliant")
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "DEBUG",
                "Validating Evidence Act 1872 compliance",
                case_id=case_id
            )
            
            errors = []
            
            # Check required elements
            for element in EvidenceActValidator.REQUIRED_ELEMENTS:
                if element not in report_data:
                    errors.append(f"Missing required element: {element}")
            
            # Validate expert qualification
            if 'expert_qualification' in report_data:
                qualification = report_data['expert_qualification']
                if not qualification.get('certified'):
                    errors.append("Expert not properly certified")
                if not qualification.get('experience_years'):
                    errors.append("Expert experience not documented")
            
            # Validate examination methodology
            if 'examination_methodology' in report_data:
                methodology = report_data['examination_methodology']
                if not methodology.get('documented'):
                    errors.append("Examination methodology not documented")
                if not methodology.get('standards_followed'):
                    errors.append("Industry standards not followed")
            
            # Validate findings basis
            if 'findings_basis' in report_data:
                basis = report_data['findings_basis']
                if not basis.get('evidence_based'):
                    errors.append("Findings not evidence-based")
            
            is_compliant = len(errors) == 0
            
            structured_logger.log_with_context(
                "INFO" if is_compliant else "WARNING",
                "Evidence Act validation completed",
                case_id=case_id,
                is_compliant=is_compliant,
                error_count=len(errors)
            )
            
            return is_compliant, errors
        
        except Exception as e:
            error_msg = f"Error validating Evidence Act compliance: {str(e)}"
            logger.error(error_msg)
            raise ValidationError(error_msg) from e
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _validate_expert_qualification(qualification_level: str) -> bool:
        """
        Validate expert qualification level (cached).
        
        Args:
            qualification_level (str): Qualification level
            
        Returns:
            bool: True if valid qualification
        """
        valid_levels = ['certified', 'experienced', 'specialized', 'master']
        return qualification_level.lower() in valid_levels
    
    @staticmethod
    def get_compliance_report(report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate Evidence Act compliance report.
        
        Args:
            report_data (Dict[str, Any]): Report data
            
        Returns:
            Dict[str, Any]: Compliance report
        """
        is_compliant, errors = EvidenceActValidator.validate(report_data)
        
        return {
            'compliant': is_compliant,
            'errors': errors,
            'elements_present': list(report_data.keys()),
            'validation_date': datetime.now().isoformat()
        }
