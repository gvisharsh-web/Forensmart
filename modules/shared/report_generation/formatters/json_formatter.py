"""
JSON FORMATTER - JSON Format Export

Exports reports to JSON (.json) format with:
- Structured data
- Machine-readable format
- Nested organization
- Metadata preservation

Features:
- Comprehensive docstrings
- Error handling with custom exceptions
- Structured logging
- Performance optimization
"""

import logging
import json
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class FormatterException(Exception):
    """Base exception for formatter errors"""
    pass

class FormattingError(FormatterException):
    """Raised when formatting fails"""
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

class JSONFormatter:
    """
    Format reports to JSON (.json) format.
    
    Provides structured JSON formatting for machine-readable output
    with proper nesting and metadata preservation.
    """
    
    def __init__(self):
        """Initialize JSON formatter"""
        logger.debug("JSONFormatter initialized")
    
    @staticmethod
    def format(report_data: Dict[str, Any], case_id: str = "") -> str:
        """
        Format report data to JSON.
        
        Converts report data to properly formatted JSON with indentation
        and metadata preservation for machine-readable output.
        
        Args:
            report_data (Dict[str, Any]): Report data dictionary
            case_id (str): Case ID for logging (optional)
            
        Returns:
            str: Formatted JSON string ready for export
            
        Raises:
            FormattingError: If formatting fails
            
        Example:
            >>> formatter = JSONFormatter()
            >>> data = {'case_id': 'CASE-001', 'findings': [...]}
            >>> json_str = formatter.format(data, "CASE-001")
            >>> json.loads(json_str)  # Should be valid JSON
            {'case_id': 'CASE-001', ...}
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "DEBUG",
                "Formatting report to JSON",
                case_id=case_id,
                data_keys=list(report_data.keys())
            )
            
            # Add metadata
            report_with_metadata = {
                'metadata': {
                    'case_id': case_id,
                    'formatted_at': datetime.now().isoformat(),
                    'format': 'json',
                    'version': '1.0'
                },
                'data': report_data
            }
            
            # Format as JSON with indentation
            formatted = json.dumps(report_with_metadata, indent=2, default=str)
            
            structured_logger.log_with_context(
                "DEBUG",
                "JSON formatting completed",
                case_id=case_id,
                json_length=len(formatted)
            )
            
            return formatted
        
        except Exception as e:
            error_msg = f"Error formatting to JSON: {str(e)}"
            logger.error(error_msg)
            raise FormattingError(error_msg) from e
    
    @staticmethod
    def parse(json_str: str) -> Dict[str, Any]:
        """
        Parse JSON string to dictionary.
        
        Args:
            json_str (str): JSON string to parse
            
        Returns:
            Dict[str, Any]: Parsed data dictionary
            
        Raises:
            FormattingError: If parsing fails
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing JSON: {str(e)}"
            logger.error(error_msg)
            raise FormattingError(error_msg) from e
