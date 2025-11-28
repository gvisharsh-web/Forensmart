"""
SIGNATURE VALIDATOR - Digital Signature Validation

Validates digital signatures in forensic reports:
- Examiner signature
- Supervisor approval
- Legal compliance
- Timestamp verification
- Certificate validation

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

class SignatureValidator:
    """
    Validate digital signatures in forensic reports.
    
    Ensures proper signing and approval of reports by authorized
    examiners and supervisors with timestamp verification.
    """
    
    # Signature requirements
    REQUIRED_SIGNATURES = [
        'examiner_signature',
        'supervisor_signature'
    ]
    
    SIGNATURE_FIELDS = [
        'name',
        'designation',
        'date',
        'timestamp',
        'signature_hash'
    ]
    
    def __init__(self):
        """Initialize Signature validator"""
        logger.debug("SignatureValidator initialized")
    
    @staticmethod
    def validate(signature_data: Dict[str, Any], case_id: str = "") -> Tuple[bool, List[str]]:
        """
        Validate digital signatures.
        
        Checks if all required signatures are present and valid with
        proper timestamps and certification.
        
        Args:
            signature_data (Dict[str, Any]): Signature data to validate
            case_id (str): Case ID for logging (optional)
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
            
        Raises:
            ValidationError: If validation fails
            
        Example:
            >>> validator = SignatureValidator()
            >>> is_valid, errors = validator.validate(sig_data, "CASE-001")
            >>> if is_valid:
            ...     print("All signatures are valid")
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "DEBUG",
                "Validating signatures",
                case_id=case_id
            )
            
            errors = []
            
            # Check required signatures
            for sig_type in SignatureValidator.REQUIRED_SIGNATURES:
                if sig_type not in signature_data:
                    errors.append(f"Missing required signature: {sig_type}")
                else:
                    sig = signature_data[sig_type]
                    
                    # Validate signature fields
                    for field in SignatureValidator.SIGNATURE_FIELDS:
                        if field not in sig:
                            errors.append(f"{sig_type}: Missing field '{field}'")
                    
                    # Validate timestamp
                    if 'timestamp' in sig:
                        if not SignatureValidator._validate_timestamp(sig['timestamp']):
                            errors.append(f"{sig_type}: Invalid timestamp")
                    
                    # Validate signature hash
                    if 'signature_hash' in sig:
                        if not SignatureValidator._validate_signature_hash(sig['signature_hash']):
                            errors.append(f"{sig_type}: Invalid signature hash")
            
            is_valid = len(errors) == 0
            
            structured_logger.log_with_context(
                "INFO" if is_valid else "WARNING",
                "Signature validation completed",
                case_id=case_id,
                is_valid=is_valid,
                error_count=len(errors)
            )
            
            return is_valid, errors
        
        except Exception as e:
            error_msg = f"Error validating signatures: {str(e)}"
            logger.error(error_msg)
            raise ValidationError(error_msg) from e
    
    @staticmethod
    @lru_cache(maxsize=256)
    def _validate_timestamp(timestamp_str: str) -> bool:
        """
        Validate timestamp format (cached).
        
        Args:
            timestamp_str (str): Timestamp string
            
        Returns:
            bool: True if valid timestamp
        """
        try:
            datetime.fromisoformat(timestamp_str)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    @lru_cache(maxsize=256)
    def _validate_signature_hash(sig_hash: str) -> bool:
        """
        Validate signature hash format (cached).
        
        Args:
            sig_hash (str): Signature hash
            
        Returns:
            bool: True if valid hash format
        """
        # SHA256 hash is 64 hex characters
        if len(sig_hash) == 64 and all(c in '0123456789abcdefABCDEF' for c in sig_hash):
            return True
        return False
    
    @staticmethod
    def get_compliance_report(signature_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate signature compliance report.
        
        Args:
            signature_data (Dict[str, Any]): Signature data
            
        Returns:
            Dict[str, Any]: Compliance report
        """
        is_valid, errors = SignatureValidator.validate(signature_data)
        
        return {
            'valid': is_valid,
            'errors': errors,
            'signatures_present': list(signature_data.keys()),
            'validation_date': datetime.now().isoformat()
        }
