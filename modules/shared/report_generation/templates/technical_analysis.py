"""
TECHNICAL ANALYSIS TEMPLATE - 3-5 Page Technical Report

Provides technical analysis template with methodology and specifications.

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

class TechnicalAnalysisTemplate(BaseTemplate):
    """Technical analysis report template (3-5 pages) with detailed methodology"""
    
    def get_template_name(self) -> str:
        """Get template name"""
        return "Technical Analysis Report"
    
    def get_template_type(self) -> str:
        """Get template type"""
        return "technical_analysis"
    
    def get_sections(self) -> list:
        """Get template sections"""
        return [
            "cover_page",
            "technical_details",
            "findings_analysis",
            "appendices"
        ]
    
    def generate(self) -> str:
        """
        Generate technical analysis report.
        
        Creates a technical 3-5 page report with:
        - Cover page
        - Technical details
        - Findings & analysis
        - Appendices
        
        Returns:
            str: Complete formatted report
        """
        try:
            logger.info(f"Generating technical analysis report for case: {self.case_id}")
            
            # Validate data
            if not self.validate_data():
                raise ValueError("Data validation failed")
            
            # Import section generators
            from ..sections import (
                CoverPageSection,
                TechnicalDetailsSection,
                FindingsAnalysisSection,
                AppendicesSection
            )
            
            # Generate sections
            cover = CoverPageSection.generate(self.case_details)
            technical = TechnicalDetailsSection.generate(self.extraction_results)
            findings = FindingsAnalysisSection.generate(self.extraction_results)
            appendices = AppendicesSection.generate(self.extraction_results)
            
            # Combine sections
            report = cover + technical + findings + appendices
            
            # Add to template
            self.add_section("Cover Page", cover)
            self.add_section("Technical Details", technical)
            self.add_section("Findings & Analysis", findings)
            self.add_section("Appendices", appendices)
            
            # Mark as final
            self.mark_as_final()
            
            logger.info(f"Technical analysis report generated successfully")
            return report
        
        except Exception as e:
            logger.error(f"Error generating technical analysis report: {str(e)}")
            raise
