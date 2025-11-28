"""
REPORT EXPORTER - Export Reports to Different Formats

Provides functionality to export reports in multiple formats:
- Text (.txt)
- JSON (.json)
- PDF (.pdf)
- DOCX (.docx)
- HTML (.html)
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# REPORT EXPORTER CLASS
# ============================================================================

class ReportExporter:
    """
    Export reports to different formats.
    
    This class provides methods to export report content
    in various formats for different use cases.
    """
    
    def __init__(self, output_dir: str = "reports/generated"):
        """
        Initialize report exporter
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
        logger.info(f"ReportExporter initialized with output_dir: {output_dir}")
    
    def _ensure_output_dir(self) -> None:
        """Ensure output directory exists"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.debug(f"Output directory ensured: {self.output_dir}")
        except Exception as e:
            logger.error(f"Error creating output directory: {str(e)}")
            raise
    
    # ========================================================================
    # EXPORT METHODS
    # ========================================================================
    
    def export_to_text(self, report_content: str, case_id: str, 
                      report_type: str = "report") -> Optional[str]:
        """
        Export report to text file
        
        Args:
            report_content: Report content
            case_id: Case ID
            report_type: Type of report
            
        Returns:
            str: File path if successful, None otherwise
        """
        try:
            filename = f"{case_id}_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = os.path.join(self.output_dir, case_id, filename)
            
            # Create case directory
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Write file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Report exported to text: {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"Error exporting to text: {str(e)}")
            return None
    
    def export_to_json(self, report_data: Dict[str, Any], case_id: str,
                      report_type: str = "report") -> Optional[str]:
        """
        Export report data to JSON file
        
        Args:
            report_data: Report data dictionary
            case_id: Case ID
            report_type: Type of report
            
        Returns:
            str: File path if successful, None otherwise
        """
        try:
            filename = f"{case_id}_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.output_dir, case_id, filename)
            
            # Create case directory
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Write file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"Report exported to JSON: {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            return None
    
    def export_to_pdf(self, report_content: str, case_id: str,
                     report_type: str = "report") -> Optional[str]:
        """
        Export report to PDF file
        
        Args:
            report_content: Report content
            case_id: Case ID
            report_type: Type of report
            
        Returns:
            str: File path if successful, None otherwise
        """
        try:
            filename = f"{case_id}_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            filepath = os.path.join(self.output_dir, case_id, filename)
            
            # Create case directory
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Try to use reportlab
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
                
                doc = SimpleDocTemplate(filepath, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                
                # Add content
                for line in report_content.split('\n'):
                    if line.strip():
                        story.append(Paragraph(line, styles['Normal']))
                    else:
                        story.append(Spacer(1, 0.2*inch))
                
                doc.build(story)
                logger.info(f"Report exported to PDF: {filepath}")
                return filepath
            
            except ImportError:
                logger.warning("reportlab not installed. Saving as text instead.")
                return self.export_to_text(report_content, case_id, report_type)
        
        except Exception as e:
            logger.error(f"Error exporting to PDF: {str(e)}")
            return None
    
    def export_to_docx(self, report_content: str, case_id: str,
                      report_type: str = "report") -> Optional[str]:
        """
        Export report to DOCX file
        
        Args:
            report_content: Report content
            case_id: Case ID
            report_type: Type of report
            
        Returns:
            str: File path if successful, None otherwise
        """
        try:
            filename = f"{case_id}_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
            filepath = os.path.join(self.output_dir, case_id, filename)
            
            # Create case directory
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Try to use python-docx
            try:
                from docx import Document
                
                doc = Document()
                
                # Add content
                for line in report_content.split('\n'):
                    if line.strip():
                        doc.add_paragraph(line)
                
                doc.save(filepath)
                logger.info(f"Report exported to DOCX: {filepath}")
                return filepath
            
            except ImportError:
                logger.warning("python-docx not installed. Saving as text instead.")
                return self.export_to_text(report_content, case_id, report_type)
        
        except Exception as e:
            logger.error(f"Error exporting to DOCX: {str(e)}")
            return None
    
    def export_to_html(self, report_content: str, case_id: str,
                      report_type: str = "report") -> Optional[str]:
        """
        Export report to HTML file
        
        Args:
            report_content: Report content
            case_id: Case ID
            report_type: Type of report
            
        Returns:
            str: File path if successful, None otherwise
        """
        try:
            filename = f"{case_id}_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.output_dir, case_id, filename)
            
            # Create case directory
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Create HTML
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Forensic Report - {case_id}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #004E89;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #004E89;
            color: white;
        }}
    </style>
</head>
<body>
    <pre>{report_content}</pre>
</body>
</html>
"""
            
            # Write file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Report exported to HTML: {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"Error exporting to HTML: {str(e)}")
            return None
    
    def export_to_all_formats(self, report_content: str, report_data: Dict[str, Any],
                             case_id: str, report_type: str = "report") -> Dict[str, Optional[str]]:
        """
        Export report to all available formats
        
        Args:
            report_content: Report content
            report_data: Report data dictionary
            case_id: Case ID
            report_type: Type of report
            
        Returns:
            Dict: Dictionary with format as key and file path as value
        """
        try:
            results = {
                'text': self.export_to_text(report_content, case_id, report_type),
                'json': self.export_to_json(report_data, case_id, report_type),
                'pdf': self.export_to_pdf(report_content, case_id, report_type),
                'docx': self.export_to_docx(report_content, case_id, report_type),
                'html': self.export_to_html(report_content, case_id, report_type)
            }
            
            logger.info(f"Report exported to all formats for case: {case_id}")
            return results
        
        except Exception as e:
            logger.error(f"Error exporting to all formats: {str(e)}")
            return {}
    
    def get_output_dir(self) -> str:
        """
        Get output directory
        
        Returns:
            str: Output directory path
        """
        return self.output_dir
    
    def set_output_dir(self, output_dir: str) -> None:
        """
        Set output directory
        
        Args:
            output_dir: New output directory
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
        logger.info(f"Output directory changed to: {output_dir}")
