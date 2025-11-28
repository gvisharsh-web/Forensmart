"""
EXECUTIVE SUMMARY TEMPLATE - 1-2 Page Summary Report

Provides a concise executive summary template for quick overview.

Features:
- Comprehensive docstrings
- Error handling with custom exceptions
- Structured logging
- Data validation
"""

import logging
import json
from typing import Dict, Any
from datetime import datetime
from ..base_template import BaseTemplate, TemplateValidationError, TemplateDataError

logger = logging.getLogger(__name__)

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

class ExecutiveSummaryTemplate(BaseTemplate):
    """Executive summary report template (1-2 pages)"""
    
    def get_template_name(self) -> str:
        """Get template name"""
        return "Executive Summary Report"
    
    def get_template_type(self) -> str:
        """Get template type"""
        return "executive_summary"
    
    def get_sections(self) -> list:
        """Get template sections"""
        return [
            "cover_page",
            "executive_summary",
            "risk_assessment",
            "next_steps"
        ]
    
    def generate(self) -> str:
        """
        Generate executive summary report with comprehensive formatting.
        
        Creates a concise 1-2 page report with:
        - Cover page with case information
        - Executive summary with key findings
        - Risk assessment with priority items
        - Next steps for investigation
        
        The report is suitable for quick overview by decision makers and
        includes all critical information needed for immediate action.
        
        Returns:
            str: Complete formatted report ready for export
            
        Raises:
            TemplateValidationError: If data validation fails
            TemplateDataError: If required data is missing
            Exception: If report generation fails
            
        Example:
            >>> template = ExecutiveSummaryTemplate(
            ...     case_id="CASE-001",
            ...     case_details={...},
            ...     extraction_results={...}
            ... )
            >>> report = template.generate()
            >>> len(report) > 1000  # Should be substantial
            True
        """
        try:
            structured_logger.log_with_context(
                "INFO",
                "Generating executive summary report",
                case_id=self.case_id
            )
            
            # Validate data
            if not self.validate_data():
                error_msg = "Data validation failed for executive summary"
                structured_logger.log_with_context("ERROR", error_msg, case_id=self.case_id)
                raise TemplateValidationError(error_msg)
            
            # Import section generators
            from ..sections import (
                CoverPageSection,
                ExecutiveSummarySection
            )
            
            # Generate sections
            cover = CoverPageSection.generate(self.case_details)
            summary = ExecutiveSummarySection.generate(self.extraction_results)
            
            # Combine sections
            report = cover + summary
            
            # Add to template
            self.add_section("Cover Page", cover)
            self.add_section("Executive Summary", summary)
            
            # Mark as final
            self.mark_as_final()
            
            logger.info(f"Executive summary report generated successfully")
            return report
        
        except Exception as e:
            logger.error(f"Error generating executive summary report: {str(e)}")
            raise
