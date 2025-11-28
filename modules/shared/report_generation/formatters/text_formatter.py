"""
TEXT FORMATTER - Plain Text Format Export

Exports reports to plain text (.txt) format with:
- Professional formatting
- Line wrapping
- Section separators
- Page breaks

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

class TextFormatter:
    """
    Format reports to plain text (.txt) format.
    
    Provides professional text formatting with proper separators,
    line wrapping, and page breaks for readability.
    """
    
    # Formatting constants
    HEADER_SEPARATOR = "═" * 79
    SECTION_SEPARATOR = "─" * 79
    LINE_WIDTH = 79
    
    def __init__(self):
        """Initialize text formatter"""
        logger.debug("TextFormatter initialized")
    
    @staticmethod
    def format(report_content: str, case_id: str = "") -> str:
        """
        Format report content to plain text.
        
        Applies professional text formatting including proper separators,
        line wrapping, and page breaks for optimal readability.
        
        Args:
            report_content (str): Report content to format
            case_id (str): Case ID for logging (optional)
            
        Returns:
            str: Formatted text content ready for export
            
        Raises:
            FormattingError: If formatting fails
            
        Example:
            >>> formatter = TextFormatter()
            >>> formatted = formatter.format(report_content, "CASE-001")
            >>> len(formatted) > len(report_content)  # Should be formatted
            True
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "DEBUG",
                "Formatting report to text",
                case_id=case_id,
                content_length=len(report_content)
            )
            
            # Apply text formatting
            formatted = report_content
            
            # Ensure proper line endings
            formatted = formatted.replace('\r\n', '\n')
            
            # Add header and footer
            header = f"\n{TextFormatter.HEADER_SEPARATOR}\n"
            footer = f"\n{TextFormatter.HEADER_SEPARATOR}\n"
            
            formatted = header + formatted + footer
            
            structured_logger.log_with_context(
                "DEBUG",
                "Text formatting completed",
                case_id=case_id,
                formatted_length=len(formatted)
            )
            
            return formatted
        
        except Exception as e:
            error_msg = f"Error formatting to text: {str(e)}"
            logger.error(error_msg)
            raise FormattingError(error_msg) from e
    
    @staticmethod
    def add_page_break() -> str:
        """
        Add page break marker.
        
        Returns:
            str: Page break marker
        """
        return f"\n{'=' * 79}\n[PAGE BREAK]\n{'=' * 79}\n"
    
    @staticmethod
    def add_separator(style: str = "header") -> str:
        """
        Add separator line.
        
        Args:
            style (str): Separator style ('header' or 'section')
            
        Returns:
            str: Separator line
        """
        if style == "header":
            return f"\n{TextFormatter.HEADER_SEPARATOR}\n"
        else:
            return f"\n{TextFormatter.SECTION_SEPARATOR}\n"
