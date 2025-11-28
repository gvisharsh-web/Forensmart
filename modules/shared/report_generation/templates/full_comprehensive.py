"""
FULL COMPREHENSIVE TEMPLATE - 20-30 Page Complete Report

Provides the most comprehensive template with all sections and details.

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

class FullComprehensiveTemplate(BaseTemplate):
    """Full comprehensive report template (20-30 pages)"""
    
    def get_template_name(self) -> str:
        """Get template name"""
        return "Full Comprehensive Report"
    
    def get_template_type(self) -> str:
        """Get template type"""
        return "full_comprehensive"
    
    def get_sections(self) -> list:
        """Get template sections"""
        return [
            "cover_page",
            "investigator_declaration",
            "executive_summary",
            "chain_of_custody",
            "technical_details",
            "findings_analysis",
            "conclusions",
            "recommendations",
            "appendices",
            "certification"
        ]
    
    def generate(self) -> str:
        """
        Generate full comprehensive report.
        
        Creates the most comprehensive 20-30 page report with:
        - Cover page
        - Investigator declaration
        - Executive summary
        - Chain of custody
        - Technical details
        - Findings & analysis
        - Conclusions
        - Recommendations
        - Appendices
        - Certification
        
        This template includes all available sections and is suitable
        for complete documentation and legal proceedings.
        
        Returns:
            str: Complete formatted report
        """
        try:
            logger.info(f"Generating full comprehensive report for case: {self.case_id}")
            
            # Validate data
            if not self.validate_data():
                raise ValueError("Data validation failed")
            
            # Import section generators
            from ..sections import (
                CoverPageSection,
                InvestigatorDeclarationSection,
                ExecutiveSummarySection,
                ChainOfCustodySection,
                TechnicalDetailsSection,
                FindingsAnalysisSection,
                ConclusionsSection,
                RecommendationsSection,
                AppendicesSection,
                CertificationSection
            )
            
            # Generate sections
            cover = CoverPageSection.generate(self.case_details)
            declaration = InvestigatorDeclarationSection.generate(self.case_details)
            summary = ExecutiveSummarySection.generate(self.extraction_results)
            coc = ChainOfCustodySection.generate(self.case_details)
            technical = TechnicalDetailsSection.generate(self.extraction_results)
            findings = FindingsAnalysisSection.generate(self.extraction_results)
            conclusions = ConclusionsSection.generate(self.extraction_results)
            recommendations = RecommendationsSection.generate(self.extraction_results)
            appendices = AppendicesSection.generate(self.extraction_results)
            certification = CertificationSection.generate(self.case_details)
            
            # Combine sections
            report = (cover + declaration + summary + coc + technical + 
                     findings + conclusions + recommendations + appendices + certification)
            
            # Add to template
            self.add_section("Cover Page", cover)
            self.add_section("Investigator Declaration", declaration)
            self.add_section("Executive Summary", summary)
            self.add_section("Chain of Custody", coc)
            self.add_section("Technical Details", technical)
            self.add_section("Findings & Analysis", findings)
            self.add_section("Conclusions", conclusions)
            self.add_section("Recommendations", recommendations)
            self.add_section("Appendices", appendices)
            self.add_section("Certification", certification)
            
            # Mark as final
            self.mark_as_final()
            
            logger.info(f"Full comprehensive report generated successfully")
            return report
        
        except Exception as e:
            logger.error(f"Error generating full comprehensive report: {str(e)}")
            raise
