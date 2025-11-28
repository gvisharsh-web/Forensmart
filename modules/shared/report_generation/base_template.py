"""
BASE TEMPLATE CLASS - Foundation for All Report Templates

Provides the base class and structure for all report templates.
All specific report templates inherit from this class.

Features:
- Template structure definition
- Section management
- Data handling
- Common methods
- Validation framework
- Custom exception handling
- Structured logging
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class TemplateException(Exception):
    """Base exception for template errors"""
    pass

class TemplateValidationError(TemplateException):
    """Raised when template validation fails"""
    pass

class TemplateGenerationError(TemplateException):
    """Raised when template generation fails"""
    pass

class TemplateDataError(TemplateException):
    """Raised when template data is invalid"""
    pass

# ============================================================================
# STRUCTURED LOGGER
# ============================================================================

class StructuredLogger:
    """Structured logging with JSON context"""
    
    def __init__(self, name: str):
        """Initialize structured logger"""
        self.logger = logging.getLogger(name)
    
    def log_with_context(self, level: str, message: str, **context) -> None:
        """Log with context information"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message,
                'context': context
            }
            log_level = getattr(logging, level.upper(), logging.INFO)
            self.logger.log(log_level, json.dumps(log_entry))
        except Exception as e:
            self.logger.error(f"Error in structured logging: {str(e)}")

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)
structured_logger = StructuredLogger(__name__)

# ============================================================================
# BASE TEMPLATE CLASS
# ============================================================================

class BaseTemplate(ABC):
    """
    Base class for all report templates.
    
    All specific report templates (ExecutiveSummary, DetailedFindings, etc.)
    should inherit from this class and implement required methods.
    """
    
    def __init__(self, case_id: str, case_details: Dict[str, Any], 
                 extraction_results: Dict[str, Any]):
        """
        Initialize base template
        
        Args:
            case_id: Case ID
            case_details: Case details dictionary
            extraction_results: Extraction results dictionary
        """
        self.case_id = case_id
        self.case_details = case_details
        self.extraction_results = extraction_results
        self.generated_at = datetime.now()
        self.sections: List[str] = []
        self.report_content = ""
        self.metadata = {
            'template_version': '1.0',
            'generated_at': self.generated_at.isoformat(),
            'case_id': case_id,
            'status': 'DRAFT'
        }
        
        logger.info(f"Template initialized for case: {case_id}")
    
    # ========================================================================
    # ABSTRACT METHODS - MUST BE IMPLEMENTED BY SUBCLASSES
    # ========================================================================
    
    @abstractmethod
    def get_template_name(self) -> str:
        """
        Get template name
        
        Returns:
            str: Template name
        """
        pass
    
    @abstractmethod
    def get_template_type(self) -> str:
        """
        Get template type
        
        Returns:
            str: Template type (e.g., 'executive_summary', 'detailed_findings')
        """
        pass
    
    @abstractmethod
    def get_sections(self) -> List[str]:
        """
        Get list of sections for this template
        
        Returns:
            List[str]: List of section names
        """
        pass
    
    @abstractmethod
    def generate(self) -> str:
        """
        Generate the complete report
        
        Returns:
            str: Generated report content
        """
        pass
    
    # ========================================================================
    # COMMON METHODS
    # ========================================================================
    
    def add_section(self, section_name: str, section_content: str) -> None:
        """
        Add a section to the report
        
        Args:
            section_name: Name of the section
            section_content: Content of the section
        """
        try:
            if section_name not in self.sections:
                self.sections.append(section_name)
            
            self.report_content += f"\n{section_content}\n"
            logger.debug(f"Section added: {section_name}")
        
        except Exception as e:
            logger.error(f"Error adding section {section_name}: {str(e)}")
            raise
    
    def validate_data(self) -> bool:
        """
        Validate required data is present.
        
        Checks that all required fields are present and valid.
        Raises TemplateValidationError if validation fails.
        
        Returns:
            bool: True if valid
            
        Raises:
            TemplateValidationError: If validation fails
            TemplateDataError: If data is invalid
        """
        try:
            # Check case ID
            if not self.case_id:
                error_msg = "Case ID is missing"
                structured_logger.log_with_context("ERROR", error_msg, field="case_id")
                raise TemplateDataError(error_msg)
            
            if not isinstance(self.case_id, str):
                error_msg = f"Case ID must be string, got {type(self.case_id).__name__}"
                structured_logger.log_with_context("ERROR", error_msg, field="case_id", type=type(self.case_id).__name__)
                raise TemplateDataError(error_msg)
            
            # Check case details
            if not self.case_details:
                error_msg = "Case details are missing"
                structured_logger.log_with_context("ERROR", error_msg, field="case_details")
                raise TemplateDataError(error_msg)
            
            if not isinstance(self.case_details, dict):
                error_msg = f"Case details must be dict, got {type(self.case_details).__name__}"
                structured_logger.log_with_context("ERROR", error_msg, field="case_details", type=type(self.case_details).__name__)
                raise TemplateDataError(error_msg)
            
            # Check extraction results
            if not self.extraction_results:
                error_msg = "Extraction results are missing"
                structured_logger.log_with_context("ERROR", error_msg, field="extraction_results")
                raise TemplateDataError(error_msg)
            
            if not isinstance(self.extraction_results, dict):
                error_msg = f"Extraction results must be dict, got {type(self.extraction_results).__name__}"
                structured_logger.log_with_context("ERROR", error_msg, field="extraction_results", type=type(self.extraction_results).__name__)
                raise TemplateDataError(error_msg)
            
            structured_logger.log_with_context("DEBUG", "Data validation passed", case_id=self.case_id)
            return True
        
        except TemplateDataError:
            raise
        except Exception as e:
            error_msg = f"Unexpected error validating data: {str(e)}"
            structured_logger.log_with_context("ERROR", error_msg, exception=str(e))
            raise TemplateValidationError(error_msg) from e
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get report metadata
        
        Returns:
            Dict: Report metadata
        """
        return self.metadata
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata value
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
        logger.debug(f"Metadata set: {key} = {value}")
    
    def get_report_content(self) -> str:
        """
        Get generated report content
        
        Returns:
            str: Report content
        """
        return self.report_content
    
    def get_sections_list(self) -> List[str]:
        """
        Get list of sections in report
        
        Returns:
            List[str]: List of section names
        """
        return self.sections
    
    def get_case_id(self) -> str:
        """
        Get case ID
        
        Returns:
            str: Case ID
        """
        return self.case_id
    
    def get_case_details(self) -> Dict[str, Any]:
        """
        Get case details
        
        Returns:
            Dict: Case details
        """
        return self.case_details
    
    def get_extraction_results(self) -> Dict[str, Any]:
        """
        Get extraction results
        
        Returns:
            Dict: Extraction results
        """
        return self.extraction_results
    
    def get_generated_at(self) -> datetime:
        """
        Get generation timestamp
        
        Returns:
            datetime: Generation timestamp
        """
        return self.generated_at
    
    def mark_as_final(self) -> None:
        """
        Mark report as final
        """
        self.metadata['status'] = 'FINAL'
        logger.info("Report marked as FINAL")
    
    def mark_as_draft(self) -> None:
        """
        Mark report as draft
        """
        self.metadata['status'] = 'DRAFT'
        logger.info("Report marked as DRAFT")
    
    def get_status(self) -> str:
        """
        Get report status
        
        Returns:
            str: Report status (DRAFT or FINAL)
        """
        return self.metadata.get('status', 'DRAFT')
    
    def clear_content(self) -> None:
        """
        Clear report content
        """
        self.report_content = ""
        self.sections = []
        logger.debug("Report content cleared")
    
    def get_page_count_estimate(self) -> int:
        """
        Estimate page count based on content length
        
        Returns:
            int: Estimated page count
        """
        # Rough estimate: ~3000 characters per page
        char_count = len(self.report_content)
        page_count = max(1, char_count // 3000)
        return page_count
    
    def get_word_count(self) -> int:
        """
        Get word count of report
        
        Returns:
            int: Word count
        """
        words = self.report_content.split()
        return len(words)
    
    def get_section_count(self) -> int:
        """
        Get number of sections in report
        
        Returns:
            int: Section count
        """
        return len(self.sections)
    
    def __str__(self) -> str:
        """
        String representation of template
        
        Returns:
            str: String representation
        """
        return f"Template: {self.get_template_name()} (Case: {self.case_id})"
    
    def __repr__(self) -> str:
        """
        Detailed string representation
        
        Returns:
            str: Detailed representation
        """
        return (f"BaseTemplate(case_id='{self.case_id}', "
                f"template_type='{self.get_template_type()}', "
                f"sections={len(self.sections)}, "
                f"status='{self.get_status()}')")
