"""
REPORT ORCHESTRATOR - Main Report Generation Orchestration

Orchestrates the complete main report generation workflow:
- Template selection
- Section generation
- Module report integration
- Report combination
- Validation

Features:
- Comprehensive docstrings
- Error handling with custom exceptions
- Structured logging
- Performance optimization
"""

import logging
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime
from functools import lru_cache

logger = logging.getLogger(__name__)

class OrchestratorException(Exception):
    """Base exception for orchestrator errors"""
    pass

class ReportGenerationError(OrchestratorException):
    """Raised when report generation fails"""
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

class ReportOrchestrator:
    """
    Orchestrate main forensic report generation.
    
    Coordinates the complete workflow of generating a comprehensive
    forensic report including template selection, section generation,
    module report integration, and final report assembly.
    """
    
    def __init__(self, case_id: str, case_details: Dict[str, Any]):
        """
        Initialize Report Orchestrator.
        
        Args:
            case_id (str): Unique case identifier
            case_details (Dict[str, Any]): Case information
        """
        self.case_id = case_id
        self.case_details = case_details
        logger.debug(f"ReportOrchestrator initialized for case: {case_id}")
    
    def generate(self, extraction_results: Dict[str, Any], 
                 module_reports: Dict[str, str] = None,
                 template_type: str = "full_comprehensive") -> str:
        """
        Generate complete forensic report.
        
        Orchestrates the entire report generation process including
        template selection, section generation, module report integration,
        and final report assembly.
        
        Args:
            extraction_results (Dict[str, Any]): Extracted device data
            module_reports (Dict[str, str]): Module-specific reports (optional)
            template_type (str): Type of template to use (default: full_comprehensive)
            
        Returns:
            str: Complete formatted forensic report
            
        Raises:
            ReportGenerationError: If report generation fails
            
        Example:
            >>> orchestrator = ReportOrchestrator("CASE-001", case_details)
            >>> report = orchestrator.generate(
            ...     extraction_results,
            ...     module_reports,
            ...     "detailed_findings"
            ... )
            >>> len(report) > 5000
            True
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "INFO",
                "Starting main report generation",
                case_id=self.case_id,
                template_type=template_type
            )
            
            # Step 1: Import template
            template = self._select_template(template_type)
            structured_logger.log_with_context(
                "DEBUG",
                "Template selected",
                case_id=self.case_id,
                template_type=template_type
            )
            
            # Step 2: Generate sections
            sections = self._generate_sections(extraction_results, module_reports)
            structured_logger.log_with_context(
                "DEBUG",
                "Sections generated",
                case_id=self.case_id,
                section_count=len(sections)
            )
            
            # Step 3: Combine sections using template
            report = self._combine_sections(template, sections)
            structured_logger.log_with_context(
                "DEBUG",
                "Sections combined",
                case_id=self.case_id,
                report_length=len(report)
            )
            
            # Step 4: Validate report structure
            self._validate_report_structure(report)
            structured_logger.log_with_context(
                "DEBUG",
                "Report structure validated",
                case_id=self.case_id
            )
            
            structured_logger.log_with_context(
                "INFO",
                "Main report generation completed successfully",
                case_id=self.case_id,
                report_length=len(report)
            )
            
            return report
        
        except Exception as e:
            error_msg = f"Error generating main report: {str(e)}"
            logger.error(error_msg)
            raise ReportGenerationError(error_msg) from e
    
    def _select_template(self, template_type: str):
        """
        Select appropriate template.
        
        Args:
            template_type (str): Type of template
            
        Returns:
            Template class
        """
        from ..templates import (
            ExecutiveSummaryTemplate,
            DetailedFindingsTemplate,
            TechnicalAnalysisTemplate,
            RiskAssessmentTemplate,
            TimelineReportTemplate,
            ITActIndiaTemplate,
            FullComprehensiveTemplate
        )
        
        templates = {
            'executive_summary': ExecutiveSummaryTemplate,
            'detailed_findings': DetailedFindingsTemplate,
            'technical_analysis': TechnicalAnalysisTemplate,
            'risk_assessment': RiskAssessmentTemplate,
            'timeline_report': TimelineReportTemplate,
            'it_act_india': ITActIndiaTemplate,
            'full_comprehensive': FullComprehensiveTemplate
        }
        
        template_class = templates.get(template_type, FullComprehensiveTemplate)
        return template_class(self.case_id, self.case_details, {})
    
    def _generate_sections(self, extraction_results: Dict[str, Any],
                          module_reports: Dict[str, str] = None) -> Dict[str, str]:
        """
        Generate all report sections.
        
        Args:
            extraction_results (Dict[str, Any]): Extracted data
            module_reports (Dict[str, str]): Module reports
            
        Returns:
            Dict[str, str]: Generated sections
        """
        from ..sections import (
            CoverPageSection,
            ExecutiveSummarySection,
            InvestigatorDeclarationSection,
            ChainOfCustodySection,
            TechnicalDetailsSection,
            FindingsAnalysisSection,
            ConclusionsSection,
            RecommendationsSection,
            AppendicesSection,
            CertificationSection
        )
        
        sections = {}
        
        try:
            sections['cover_page'] = CoverPageSection.generate(self.case_details)
            sections['executive_summary'] = ExecutiveSummarySection.generate(extraction_results)
            sections['investigator_declaration'] = InvestigatorDeclarationSection.generate(self.case_details)
            sections['chain_of_custody'] = ChainOfCustodySection.generate(self.case_details)
            sections['technical_details'] = TechnicalDetailsSection.generate(extraction_results)
            
            # Integrate module reports into findings
            findings_data = extraction_results.copy()
            if module_reports:
                findings_data['module_reports'] = module_reports
            sections['findings_analysis'] = FindingsAnalysisSection.generate(findings_data)
            
            sections['conclusions'] = ConclusionsSection.generate(extraction_results)
            sections['recommendations'] = RecommendationsSection.generate(extraction_results)
            sections['appendices'] = AppendicesSection.generate(extraction_results)
            sections['certification'] = CertificationSection.generate(self.case_details)
            
            return sections
        
        except Exception as e:
            raise ReportGenerationError(f"Error generating sections: {str(e)}")
    
    def _combine_sections(self, template, sections: Dict[str, str]) -> str:
        """
        Combine sections using template.
        
        Args:
            template: Template instance
            sections (Dict[str, str]): Generated sections
            
        Returns:
            str: Combined report
        """
        try:
            for section_name, section_content in sections.items():
                template.add_section(section_name, section_content)
            
            template.mark_as_final()
            return template.get_final_report()
        
        except Exception as e:
            raise ReportGenerationError(f"Error combining sections: {str(e)}")
    
    def _validate_report_structure(self, report: str) -> bool:
        """
        Validate report structure.
        
        Args:
            report (str): Report content
            
        Returns:
            bool: True if valid
        """
        required_sections = [
            'COVER PAGE',
            'EXECUTIVE SUMMARY',
            'FINDINGS',
            'CONCLUSIONS',
            'CERTIFICATION'
        ]
        
        for section in required_sections:
            if section not in report.upper():
                raise ReportGenerationError(f"Missing required section: {section}")
        
        return True
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _get_template_name(template_type: str) -> str:
        """Get template display name (cached)"""
        names = {
            'executive_summary': 'Executive Summary Report',
            'detailed_findings': 'Detailed Findings Report',
            'technical_analysis': 'Technical Analysis Report',
            'risk_assessment': 'Risk Assessment Report',
            'timeline_report': 'Timeline Report',
            'it_act_india': 'IT Act India Compliant Report',
            'full_comprehensive': 'Full Comprehensive Report'
        }
        return names.get(template_type, 'Unknown Template')
