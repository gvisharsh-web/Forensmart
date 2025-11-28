"""
HTML FORMATTER - HTML Format Export

Exports reports to HTML (.html) format with:
- Web-ready output
- Professional styling
- Interactive features
- Metadata

Features:
- Comprehensive docstrings
- Error handling with custom exceptions
- Structured logging
- Professional CSS styling
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

class HTMLFormatter:
    """
    Format reports to HTML (.html) format.
    
    Provides web-ready HTML formatting with professional styling,
    interactive features, and metadata preservation for browser viewing.
    """
    
    # Professional CSS styling
    CSS_STYLE = """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2, h3 {
            color: #004E89;
            border-bottom: 2px solid #004E89;
            padding-bottom: 10px;
        }
        h1 {
            font-size: 2.5em;
            margin-top: 40px;
        }
        h2 {
            font-size: 2em;
            margin-top: 30px;
        }
        h3 {
            font-size: 1.5em;
            margin-top: 20px;
        }
        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #004E89;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #004E89;
            color: white;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .metadata {
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .page-break {
            page-break-after: always;
            margin: 40px 0;
            border-top: 2px dashed #ccc;
            padding-top: 20px;
        }
    </style>
    """
    
    def __init__(self):
        """Initialize HTML formatter"""
        logger.debug("HTMLFormatter initialized")
    
    @staticmethod
    def format(report_content: str, case_id: str = "") -> str:
        """
        Format report content to HTML.
        
        Converts report content to professional HTML format with
        CSS styling, metadata, and web-ready output.
        
        Args:
            report_content (str): Report content to format
            case_id (str): Case ID for logging (optional)
            
        Returns:
            str: Formatted HTML string ready for export
            
        Raises:
            FormattingError: If formatting fails
            
        Example:
            >>> formatter = HTMLFormatter()
            >>> html_content = formatter.format(report_content, "CASE-001")
            >>> '<html>' in html_content
            True
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "DEBUG",
                "Formatting report to HTML",
                case_id=case_id,
                content_length=len(report_content)
            )
            
            # Escape HTML special characters
            escaped_content = report_content.replace('&', '&amp;')\
                                           .replace('<', '&lt;')\
                                           .replace('>', '&gt;')\
                                           .replace('"', '&quot;')\
                                           .replace("'", '&#39;')
            
            # Create HTML document
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forensic Report - {case_id}</title>
    {HTMLFormatter.CSS_STYLE}
</head>
<body>
    <div class="metadata">
        <h1>🔍 Forensic Examination Report</h1>
        <p><strong>Case ID:</strong> {case_id}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Format:</strong> HTML</p>
    </div>
    
    <pre>{escaped_content}</pre>
    
    <div class="metadata">
        <p><em>This report was automatically generated by Forensmart.</em></p>
        <p><em>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    </div>
</body>
</html>
"""
            
            structured_logger.log_with_context(
                "DEBUG",
                "HTML formatting completed",
                case_id=case_id,
                html_length=len(html)
            )
            
            return html
        
        except Exception as e:
            error_msg = f"Error formatting to HTML: {str(e)}"
            logger.error(error_msg)
            raise FormattingError(error_msg) from e
