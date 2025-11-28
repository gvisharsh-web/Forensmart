"""
EXPORT ORCHESTRATOR - Report Export Orchestration

Orchestrates report export to multiple formats:
- Text Format (.txt)
- JSON Format (.json)
- PDF Format (.pdf)
- DOCX Format (.docx)
- HTML Format (.html)

Features:
- Comprehensive docstrings
- Error handling with custom exceptions
- Structured logging
- Performance optimization
"""

import logging
import json
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class OrchestratorException(Exception):
    """Base exception for orchestrator errors"""
    pass

class ExportError(OrchestratorException):
    """Raised when export fails"""
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

class ExportOrchestrator:
    """
    Orchestrate report export to multiple formats.
    
    Coordinates the export of forensic reports to various formats
    including text, JSON, PDF, DOCX, and HTML.
    """
    
    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize Export Orchestrator.
        
        Args:
            output_dir (str): Output directory for exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"ExportOrchestrator initialized with output dir: {output_dir}")
    
    def export(self, report_content: str, case_id: str = "",
               formats: List[str] = None) -> Dict[str, str]:
        """
        Export report to multiple formats.
        
        Orchestrates the export of a report to specified formats,
        saving files to the output directory.
        
        Args:
            report_content (str): Report content to export
            case_id (str): Case ID for file naming (optional)
            formats (List[str]): List of formats to export (default: all)
            
        Returns:
            Dict[str, str]: Dictionary of format -> file_path
            
        Raises:
            ExportError: If export fails
            
        Example:
            >>> orchestrator = ExportOrchestrator("./reports")
            >>> exported = orchestrator.export(
            ...     report_content,
            ...     "CASE-001",
            ...     ["txt", "pdf", "html"]
            ... )
            >>> "txt" in exported
            True
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "INFO",
                "Starting report export",
                case_id=case_id,
                formats=formats or ["txt", "json", "pdf", "docx", "html"]
            )
            
            if formats is None:
                formats = ["txt", "json", "pdf", "docx", "html"]
            
            exported_files = {}
            
            for fmt in formats:
                try:
                    file_path = self._export_format(report_content, case_id, fmt)
                    exported_files[fmt] = str(file_path)
                    structured_logger.log_with_context(
                        "DEBUG",
                        f"Report exported to {fmt}",
                        case_id=case_id,
                        format=fmt,
                        file_path=str(file_path)
                    )
                except Exception as e:
                    logger.error(f"Error exporting to {fmt}: {e}")
                    exported_files[fmt] = f"Error: {str(e)}"
            
            structured_logger.log_with_context(
                "INFO",
                "Report export completed",
                case_id=case_id,
                exported_count=len([f for f in exported_files.values() if not f.startswith("Error")])
            )
            
            return exported_files
        
        except Exception as e:
            error_msg = f"Error exporting report: {str(e)}"
            logger.error(error_msg)
            raise ExportError(error_msg) from e
    
    def _export_format(self, report_content: str, case_id: str, fmt: str) -> Path:
        """
        Export report to specific format.
        
        Args:
            report_content (str): Report content
            case_id (str): Case ID
            fmt (str): Format type
            
        Returns:
            Path: File path
        """
        from ..formatters import (
            TextFormatter,
            JSONFormatter,
            PDFFormatter,
            DOCXFormatter,
            HTMLFormatter
        )
        
        # Create case directory
        case_dir = self.output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        
        if fmt == "txt":
            formatter = TextFormatter()
            content = formatter.format(report_content, case_id)
            file_path = case_dir / f"report.txt"
        elif fmt == "json":
            formatter = JSONFormatter()
            content = formatter.format({"content": report_content}, case_id)
            file_path = case_dir / f"report.json"
        elif fmt == "pdf":
            formatter = PDFFormatter()
            content = formatter.format(report_content, case_id)
            file_path = case_dir / f"report.pdf"
        elif fmt == "docx":
            formatter = DOCXFormatter()
            content = formatter.format(report_content, case_id)
            file_path = case_dir / f"report.docx"
        elif fmt == "html":
            formatter = HTMLFormatter()
            content = formatter.format(report_content, case_id)
            file_path = case_dir / f"report.html"
        else:
            raise ExportError(f"Unknown format: {fmt}")
        
        # Write file
        if isinstance(content, bytes):
            file_path.write_bytes(content)
        else:
            file_path.write_text(content, encoding='utf-8')
        
        return file_path
