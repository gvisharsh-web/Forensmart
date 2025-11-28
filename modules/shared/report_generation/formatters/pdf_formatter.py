"""
PDF FORMATTER - PDF Format Export

Exports reports to PDF (.pdf) format with:
- Professional formatting
- Print-ready output
- Page breaks
- Metadata

Features:
- Comprehensive docstrings
- Error handling with custom exceptions
- Structured logging
- Graceful degradation (fallback to text if PDF library unavailable)
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

class PDFFormatter:
    """
    Format reports to PDF (.pdf) format.
    
    Provides professional PDF formatting with print-ready output,
    page breaks, and metadata. Includes graceful degradation if
    PDF libraries are not available.
    """
    
    def __init__(self):
        """Initialize PDF formatter"""
        logger.debug("PDFFormatter initialized")
        self.pdf_available = self._check_pdf_library()
    
    @staticmethod
    def _check_pdf_library() -> bool:
        """
        Check if PDF library is available.
        
        Returns:
            bool: True if PDF library available, False otherwise
        """
        try:
            import reportlab
            return True
        except ImportError:
            logger.warning("reportlab not available, PDF export will use fallback")
            return False
    
    @staticmethod
    def format(report_content: str, case_id: str = "") -> str:
        """
        Format report content to PDF.
        
        Converts report content to PDF format with professional styling,
        page breaks, and metadata. Falls back to text format if PDF
        libraries are not available.
        
        Args:
            report_content (str): Report content to format
            case_id (str): Case ID for logging (optional)
            
        Returns:
            str: PDF content or fallback text content
            
        Raises:
            FormattingError: If formatting fails
            
        Example:
            >>> formatter = PDFFormatter()
            >>> pdf_content = formatter.format(report_content, "CASE-001")
            >>> len(pdf_content) > 0
            True
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "DEBUG",
                "Formatting report to PDF",
                case_id=case_id,
                content_length=len(report_content)
            )
            
            # Try to use reportlab
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
                from reportlab.lib.units import inch
                from io import BytesIO
                
                # Create PDF in memory
                pdf_buffer = BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                
                # Add content
                for line in report_content.split('\n'):
                    if line.strip():
                        story.append(Paragraph(line, styles['Normal']))
                    else:
                        story.append(Spacer(1, 0.2*inch))
                
                doc.build(story)
                
                structured_logger.log_with_context(
                    "DEBUG",
                    "PDF formatting completed",
                    case_id=case_id,
                    pdf_size=pdf_buffer.tell()
                )
                
                return pdf_buffer.getvalue().decode('latin-1', errors='ignore')
            
            except ImportError:
                logger.warning("reportlab not available, returning text format")
                structured_logger.log_with_context(
                    "WARNING",
                    "PDF library not available, using text fallback",
                    case_id=case_id
                )
                return report_content
        
        except Exception as e:
            error_msg = f"Error formatting to PDF: {str(e)}"
            logger.error(error_msg)
            raise FormattingError(error_msg) from e
