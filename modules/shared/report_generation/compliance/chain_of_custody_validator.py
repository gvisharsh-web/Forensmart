"""
CHAIN OF CUSTODY VALIDATOR - Compliance Validation

Validates chain of custody for forensic reports:
- Initial seizure documentation
- Custody transfers
- Storage conditions
- Final examination
- Integrity verification

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

class ChainOfCustodyValidator:
    """
    Validate chain of custody for forensic reports.
    
    Ensures proper documentation and integrity of evidence chain
    from seizure through examination and storage.
    """
    
    # Chain of custody requirements
    REQUIRED_FIELDS = [
        'seizure_date',
        'seized_by',
        'seizure_location',
        'device_condition',
        'initial_hash',
        'custody_transfers',
        'storage_location',
        'final_hash',
        'verification_status'
    ]
    
    def __init__(self):
        """Initialize Chain of Custody validator"""
        logger.debug("ChainOfCustodyValidator initialized")
    
    @staticmethod
    def validate(coc_data: Dict[str, Any], case_id: str = "") -> Tuple[bool, List[str]]:
        """
        Validate chain of custody documentation.
        
        Checks if chain of custody contains all required fields and
        maintains integrity throughout the evidence lifecycle.
        
        Args:
            coc_data (Dict[str, Any]): Chain of custody data
            case_id (str): Case ID for logging (optional)
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
            
        Raises:
            ValidationError: If validation fails
            
        Example:
            >>> validator = ChainOfCustodyValidator()
            >>> is_valid, errors = validator.validate(coc_data, "CASE-001")
            >>> if is_valid:
            ...     print("Chain of custody is valid")
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "DEBUG",
                "Validating chain of custody",
                case_id=case_id
            )
            
            errors = []
            
            # Check required fields
            for field in ChainOfCustodyValidator.REQUIRED_FIELDS:
                if field not in coc_data:
                    errors.append(f"Missing required field: {field}")
            
            # Validate seizure information
            if 'seizure_date' in coc_data and not coc_data['seizure_date']:
                errors.append("Seizure date not documented")
            
            if 'seized_by' in coc_data and not coc_data['seized_by']:
                errors.append("Seized by information missing")
            
            # Validate hash integrity
            if 'initial_hash' in coc_data and 'final_hash' in coc_data:
                if not ChainOfCustodyValidator._validate_hash_integrity(
                    coc_data['initial_hash'],
                    coc_data['final_hash']
                ):
                    errors.append("Hash mismatch detected - integrity compromised")
            
            # Validate custody transfers
            if 'custody_transfers' in coc_data:
                transfers = coc_data['custody_transfers']
                if isinstance(transfers, list):
                    for i, transfer in enumerate(transfers):
                        if not transfer.get('received_by'):
                            errors.append(f"Transfer {i+1}: Received by not documented")
                        if not transfer.get('received_date'):
                            errors.append(f"Transfer {i+1}: Received date not documented")
            
            is_valid = len(errors) == 0
            
            structured_logger.log_with_context(
                "INFO" if is_valid else "WARNING",
                "Chain of custody validation completed",
                case_id=case_id,
                is_valid=is_valid,
                error_count=len(errors)
            )
            
            return is_valid, errors
        
        except Exception as e:
            error_msg = f"Error validating chain of custody: {str(e)}"
            logger.error(error_msg)
            raise ValidationError(error_msg) from e
    
    @staticmethod
    @lru_cache(maxsize=256)
    def _validate_hash_integrity(initial_hash: str, final_hash: str) -> bool:
        """
        Validate hash integrity (cached).
        
        Args:
            initial_hash (str): Initial hash value
            final_hash (str): Final hash value
            
        Returns:
            bool: True if hashes match
        """
        return initial_hash.lower() == final_hash.lower()
    
    @staticmethod
    def get_compliance_report(coc_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate chain of custody compliance report.
        
        Args:
            coc_data (Dict[str, Any]): Chain of custody data
            
        Returns:
            Dict[str, Any]: Compliance report
        """
        is_valid, errors = ChainOfCustodyValidator.validate(coc_data)
        
        return {
            'valid': is_valid,
            'errors': errors,
            'fields_present': list(coc_data.keys()),
            'validation_date': datetime.now().isoformat()
        }
