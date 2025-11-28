"""
BASE ADAPTER CLASS - Abstract Base for All Adapters
Defines the interface and common functionality for all device adapters

This module provides:
- AdapterBase (abstract base class)
- Common adapter methods
- Consent integration
- Artifact storage
- Error handling
- Logging
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

from modules.consent.models import ConsentLevel, get_consent_manager
from modules.shared.utils import ArtifactPathBuilder, ResultsRepository, ErrorHandlingLoopholes
from .exceptions import AdapterException, ConnectionFailed, ExtractionFailed

logger = logging.getLogger(__name__)


# ============================================================================
# BASE ADAPTER CLASS
# ============================================================================

class AdapterBase(ABC):
    """Abstract base class for all device adapters"""
    
    def __init__(self, device_id: str, case_id: str, consent_manager=None):
        """Initialize adapter with device and case information"""
        self.device_id = device_id
        self.case_id = case_id
        self.consent_manager = consent_manager or get_consent_manager()
        self.is_connected = False
        self.extraction_status = "idle"
        self.extracted_data = {}
        self.error_handler = ErrorHandlingLoopholes()
        self.artifact_builder = ArtifactPathBuilder()
        self.results_repo = ResultsRepository()
        
        logger.info(f"✅ Adapter initialized for device: {device_id}, case: {case_id}")
    
    # ========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ========================================================================
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to device"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Close connection to device"""
        pass
    
    @abstractmethod
    def extract_data(self) -> Dict[str, Any]:
        """Extract data from device"""
        pass
    
    # ========================================================================
    # COMMON ADAPTER METHODS
    # ========================================================================
    
    def validate_connection(self) -> bool:
        """Validate that connection is active"""
        if not self.is_connected:
            logger.warning(f"⚠️ Device {self.device_id} is not connected")
            return False
        logger.info(f"✅ Connection validated for device: {self.device_id}")
        return True
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        try:
            info = {
                'device_id': self.device_id,
                'case_id': self.case_id,
                'is_connected': self.is_connected,
                'extraction_status': self.extraction_status,
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"✅ Device info retrieved: {self.device_id}")
            return info
        except Exception as e:
            logger.error(f"❌ Error getting device info: {e}")
            return {}
    
    def get_extraction_status(self) -> Dict[str, Any]:
        """Get current extraction status"""
        return {
            'device_id': self.device_id,
            'case_id': self.case_id,
            'status': self.extraction_status,
            'data_extracted': len(self.extracted_data),
            'timestamp': datetime.now().isoformat()
        }
    
    def check_consent(self, module_name: str, required_level: ConsentLevel = ConsentLevel.STANDARD) -> bool:
        """Check if consent level allows extraction for module"""
        try:
            session = self.consent_manager.get_session(self.case_id)
            
            if not session or not session.level:
                logger.warning(f"⚠️ No consent found for case: {self.case_id}")
                return False
            
            if session.level.value >= required_level.value:
                logger.info(f"✅ Consent check PASSED for {module_name}: {session.level.name}")
                return True
            else:
                logger.warning(f"❌ Insufficient consent for {module_name}: {session.level.name} < {required_level.name}")
                return False
        except Exception as e:
            logger.error(f"❌ Error checking consent: {e}")
            return False
    
    def save_results(self, data: Dict[str, Any], module_name: str) -> bool:
        """Save extraction results to artifacts"""
        try:
            artifact_path = self.artifact_builder.build_path(
                case_id=self.case_id,
                device_id=self.device_id,
                module=module_name
            )
            
            result = self.results_repo.save_results(
                case_id=self.case_id,
                device_id=self.device_id,
                module=module_name,
                data=data,
                artifact_path=artifact_path
            )
            
            if result:
                logger.info(f"✅ Results saved for {module_name}: {artifact_path}")
                self.extracted_data[module_name] = data
                return True
            else:
                logger.error(f"❌ Failed to save results for {module_name}")
                return False
        except Exception as e:
            logger.error(f"❌ Error saving results: {e}")
            return False
    
    def handle_error(self, error: Exception, operation: str = None) -> Dict[str, Any]:
        """Handle and log errors"""
        error_info = {
            'device_id': self.device_id,
            'case_id': self.case_id,
            'operation': operation or 'unknown',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.error(f"❌ Error in {operation or 'operation'}: {error}")
        return error_info
    
    def log_operation(self, operation: str, status: str = "success", details: str = None):
        """Log adapter operation"""
        message = f"{operation} - Status: {status}"
        if details:
            message += f" - {details}"
        
        if status == "success":
            logger.info(f"✅ {message}")
        elif status == "warning":
            logger.warning(f"⚠️ {message}")
        else:
            logger.error(f"❌ {message}")
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """Get summary of extracted data"""
        return {
            'device_id': self.device_id,
            'case_id': self.case_id,
            'modules_extracted': list(self.extracted_data.keys()),
            'total_items': sum(len(v) if isinstance(v, (list, dict)) else 1 for v in self.extracted_data.values()),
            'timestamp': datetime.now().isoformat()
        }
