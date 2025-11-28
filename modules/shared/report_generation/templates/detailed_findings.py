"""
DETAILED FINDINGS TEMPLATE - 5-10 Page Detailed Report

Provides comprehensive findings template with in-depth analysis.

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

class DetailedFindingsTemplate(BaseTemplate):
    """Detailed findings report template (5-10 pages) with comprehensive analysis"""
    
    def get_template_name(self) -> str:
        """Get template name"""
        return "Detailed Findings Report"
    
    def get_template_type(self) -> str:
        """Get template type"""
        return "detailed_findings"
    
    def get_sections(self) -> list:
        """Get template sections"""
        return [
            "cover_page",
            "executive_summary",
            "findings_analysis",
            "conclusions",
            "recommendations",
            "appendices"
        ]
    
    def generate(self) -> str:
        """
        Generate detailed findings report with comprehensive analysis.
        
        Creates a comprehensive 5-10 page report with:
        - Cover page with case information
        - Executive summary with key findings
        - Findings & analysis with detailed insights
        - Conclusions with evidence linking
        - Recommendations with actionable guidance
        - Appendices with supporting documentation
        
        The report is suitable for detailed investigation review and includes
        all critical information needed for comprehensive case analysis.
        
        Returns:
            str: Complete formatted report ready for export
            
        Raises:
            TemplateValidationError: If data validation fails
            TemplateDataError: If required data is missing
            Exception: If report generation fails
            
        Example:
            >>> template = DetailedFindingsTemplate(
            ...     case_id="CASE-001",
            ...     case_details={...},
            ...     extraction_results={...}
            ... )
            >>> report = template.generate()
            >>> len(report) > 5000  # Should be substantial
            True
        """
        try:
            structured_logger.log_with_context(
                "INFO",
                "Generating detailed findings report",
                case_id=self.case_id
            )
            
            # Validate data
            if not self.validate_data():
                error_msg = "Data validation failed for detailed findings"
                structured_logger.log_with_context("ERROR", error_msg, case_id=self.case_id)
                raise TemplateValidationError(error_msg)
            
            # Import section generators
            from ..sections import (
                CoverPageSection,
                ExecutiveSummarySection,
                FindingsAnalysisSection,
                ConclusionsSection,
                RecommendationsSection,
                AppendicesSection
            )
            
            # Generate sections
            cover = CoverPageSection.generate(self.case_details)
            summary = ExecutiveSummarySection.generate(self.extraction_results)
            findings = FindingsAnalysisSection.generate(self.extraction_results)
            conclusions = ConclusionsSection.generate(self.extraction_results)
            recommendations = RecommendationsSection.generate(self.extraction_results)
            appendices = AppendicesSection.generate(self.extraction_results)
            
            # Combine sections
            report = cover + summary + findings + conclusions + recommendations + appendices
            
            # Add to template
            self.add_section("Cover Page", cover)
            self.add_section("Executive Summary", summary)
            self.add_section("Findings & Analysis", findings)
            self.add_section("Conclusions", conclusions)
            self.add_section("Recommendations", recommendations)
            self.add_section("Appendices", appendices)
            
            # Mark as final
            self.mark_as_final()
            
            logger.info(f"Detailed findings report generated successfully")
            return report
        
        except Exception as e:
            logger.error(f"Error generating detailed findings report: {str(e)}")
            raise
