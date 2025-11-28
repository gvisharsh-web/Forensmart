"""
IT ACT 2000 VALIDATOR - Compliance Validation

Validates reports for compliance with Indian IT Act 2000:
- Section 65 (Computer-generated evidence)
- Section 65A (Secure digital signature)
- Section 65B (Admissibility of computer output)
- Metadata requirements
- Chain of custody

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

class ITActValidator:
    """
    Validate reports for IT Act 2000 compliance.
    
    Ensures forensic reports comply with Indian IT Act 2000 requirements
    including Section 65, 65A, and 65B provisions.
    """
    
    # IT Act 2000 requirements
    REQUIRED_SECTIONS = [
        'cover_page',
        'investigator_declaration',
        'chain_of_custody',
        'technical_details',
        'findings_analysis',
        'conclusions',
        'certification'
    ]
    
    REQUIRED_METADATA = [
        'case_id',
        'investigator_name',
        'examination_date',
        'device_details',
        'extraction_method',
        'hash_value'
    ]
    
    def __init__(self):
        """Initialize IT Act validator"""
        logger.debug("ITActValidator initialized")
    
    @staticmethod
    def validate(report_data: Dict[str, Any], case_id: str = "") -> Tuple[bool, List[str]]:
        """
        Validate report for IT Act 2000 compliance.
        
        Checks if report contains all required sections and metadata
        as per IT Act 2000 provisions.
        
        Args:
            report_data (Dict[str, Any]): Report data to validate
            case_id (str): Case ID for logging (optional)
            
        Returns:
            Tuple[bool, List[str]]: (is_compliant, list_of_errors)
            
        Raises:
            ValidationError: If validation fails
            
        Example:
            >>> validator = ITActValidator()
            >>> is_valid, errors = validator.validate(report_data, "CASE-001")
            >>> if is_valid:
            ...     print("Report is IT Act compliant")
            ... else:
            ...     print(f"Errors: {errors}")
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "DEBUG",
                "Validating IT Act 2000 compliance",
                case_id=case_id
            )
            
            errors = []
            
            # Check required sections
            for section in ITActValidator.REQUIRED_SECTIONS:
                if section not in report_data:
                    errors.append(f"Missing required section: {section}")
            
            # Check required metadata
            metadata = report_data.get('metadata', {})
            for field in ITActValidator.REQUIRED_METADATA:
                if field not in metadata:
                    errors.append(f"Missing required metadata: {field}")
            
            # Validate investigator declaration
            if 'investigator_declaration' in report_data:
                if not report_data['investigator_declaration'].get('signed'):
                    errors.append("Investigator declaration not signed")
            
            # Validate chain of custody
            if 'chain_of_custody' in report_data:
                if not report_data['chain_of_custody'].get('verified'):
                    errors.append("Chain of custody not verified")
            
            is_compliant = len(errors) == 0
            
            structured_logger.log_with_context(
                "INFO" if is_compliant else "WARNING",
                "IT Act validation completed",
                case_id=case_id,
                is_compliant=is_compliant,
                error_count=len(errors)
            )
            
            return is_compliant, errors
        
        except Exception as e:
            error_msg = f"Error validating IT Act compliance: {str(e)}"
            logger.error(error_msg)
            raise ValidationError(error_msg) from e
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _validate_hash_format(hash_value: str) -> bool:
        """
        Validate hash format (cached).
        
        Args:
            hash_value (str): Hash value to validate
            
        Returns:
            bool: True if valid hash format
        """
        # SHA256 hash is 64 hex characters
        if len(hash_value) == 64 and all(c in '0123456789abcdefABCDEF' for c in hash_value):
            return True
        # MD5 hash is 32 hex characters
        if len(hash_value) == 32 and all(c in '0123456789abcdefABCDEF' for c in hash_value):
            return True
        return False
    
    @staticmethod
    def get_compliance_report(report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate compliance report.
        
        Args:
            report_data (Dict[str, Any]): Report data
            
        Returns:
            Dict[str, Any]: Compliance report
        """
        is_compliant, errors = ITActValidator.validate(report_data)
        
        return {
            'compliant': is_compliant,
            'errors': errors,
            'sections_present': list(report_data.keys()),
            'validation_date': datetime.now().isoformat()
        }
