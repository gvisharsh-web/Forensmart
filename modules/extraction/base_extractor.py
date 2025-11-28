"""
BASE EXTRACTOR CLASS
All extraction modules inherit from this class to ensure consent verification

Features:
- Consent checking before extraction
- Module filtering based on consent level
- Extraction logging and audit trail
- Error handling
- Results aggregation
"""

from typing import Dict, List, Tuple
from datetime import datetime
from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    """Base class for all extractors with consent verification"""
    
    def __init__(self, consent_data: Dict):
        """
        Initialize extractor with consent data
        
        Args:
            consent_data: Consent token data containing:
                - case_id: Case identifier
                - consent_level: STANDARD/LEGAL/FULL
                - modules_allowed: List of allowed modules
                - modules_blocked: List of blocked modules
        """
        self.consent_data = consent_data
        self.case_id = consent_data.get('case_id', 'UNKNOWN')
        self.consent_level = consent_data.get('consent_level', 'UNKNOWN')
        self.modules_allowed = consent_data.get('modules_allowed', [])
        self.modules_blocked = consent_data.get('modules_blocked', [])
        self.extraction_log = []
        self.module_name = self.__class__.__name__
    
    def check_consent(self, module_name: str) -> Tuple[bool, str]:
        """
        Check if module extraction is allowed by consent
        
        Args:
            module_name: Name of module to extract
            
        Returns:
            (is_allowed, reason)
        """
        
        # Check if module is in allowed list
        if module_name not in self.modules_allowed:
            reason = f"Module '{module_name}' not allowed by consent level '{self.consent_level}'"
            self.log_extraction(module_name, 'blocked', reason)
            return False, reason
        
        # Check if module is not blocked
        if module_name in self.modules_blocked:
            reason = f"Module '{module_name}' is blocked"
            self.log_extraction(module_name, 'blocked', reason)
            return False, reason
        
        return True, "Consent verified"
    
    def log_extraction(self, module_name: str, status: str, details: str = ""):
        """
        Log extraction attempt
        
        Args:
            module_name: Name of module
            status: Status (started/completed/blocked/failed)
            details: Additional details
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'case_id': self.case_id,
            'module': module_name,
            'status': status,
            'details': details,
            'consent_level': self.consent_level
        }
        self.extraction_log.append(log_entry)
    
    def get_extraction_log(self) -> List[Dict]:
        """Get extraction log"""
        return self.extraction_log
    
    def get_module_name(self) -> str:
        """Get module name"""
        return self.module_name
    
    def get_consent_level(self) -> str:
        """Get consent level"""
        return self.consent_level
    
    def get_case_id(self) -> str:
        """Get case ID"""
        return self.case_id
    
    @abstractmethod
    def extract(self, device_id: str) -> Dict:
        """
        Extract data from device
        
        Must be implemented by subclasses
        
        Args:
            device_id: Device identifier
            
        Returns:
            {
                'module': module_name,
                'status': 'completed'/'blocked'/'failed',
                'data': extracted_data or None,
                'files': file_count,
                'size_mb': size_in_mb,
                'reason': reason_if_blocked,
                'error': error_if_failed
            }
        """
        raise NotImplementedError("Subclasses must implement extract()")
    
    def format_results(self, module_name: str, status: str, data: Dict = None, 
                      files: int = 0, size_mb: float = 0, reason: str = "", 
                      error: str = "") -> Dict:
        """
        Format extraction results
        
        Args:
            module_name: Name of module
            status: Status (completed/blocked/failed)
            data: Extracted data
            files: Number of files
            size_mb: Size in MB
            reason: Reason if blocked
            error: Error message if failed
            
        Returns:
            Formatted results dictionary
        """
        return {
            'module': module_name,
            'status': status,
            'data': data,
            'files': files,
            'size_mb': size_mb,
            'reason': reason,
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'case_id': self.case_id,
            'consent_level': self.consent_level
        }
