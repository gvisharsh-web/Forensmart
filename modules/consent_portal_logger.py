"""Persistent logging for consent portal with file and JSON handlers."""
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import json


class ConsentPortalLogger:
    """Persistent logging for consent portal."""
    
    _instance = None
    _logger = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize logger with file handlers."""
        self._logger = logging.getLogger('consent_portal')
        self._logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self._logger.handlers = []
        
        # Create audit directory
        audit_dir = Path('audit/consent_portal')
        audit_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler (text log)
        log_file = audit_dir / f'portal_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self._logger.addHandler(file_handler)
        
        # Rotating file handler
        rotating_handler = logging.handlers.RotatingFileHandler(
            audit_dir / 'portal_current.log',
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        rotating_handler.setLevel(logging.INFO)
        rotating_handler.setFormatter(file_formatter)
        self._logger.addHandler(rotating_handler)
    
    def get_logger(self):
        """Get the configured logger."""
        return self._logger
    
    @staticmethod
    def log_approval(case_id: str, decision: str, nominee_name: str, 
                     device_id: str = "UNKNOWN", purpose: str = "Not specified"):
        """Log approval decision."""
        logger = ConsentPortalLogger().get_logger()
        logger.info(f"Approval: {decision.upper()} | Case: {case_id} | Nominee: {nominee_name} | Device: {device_id}")
    
    @staticmethod
    def log_device_detection(case_id: str, detected_device: str, method: str = "auto"):
        """Log device detection."""
        logger = ConsentPortalLogger().get_logger()
        logger.info(f"Device detected for {case_id}: {detected_device} (method: {method})")
    
    @staticmethod
    def log_error(error: Exception, context: str, case_id: str = None):
        """Log error with context."""
        logger = ConsentPortalLogger().get_logger()
        logger.error(f"Error in {context}: {str(error)}", exc_info=True, extra={
            'case_id': case_id,
            'error_type': type(error).__name__
        })


__all__ = ["ConsentPortalLogger"]
