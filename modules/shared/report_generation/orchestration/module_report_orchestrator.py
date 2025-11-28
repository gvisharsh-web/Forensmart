"""
MODULE REPORT ORCHESTRATOR - Module-Specific Report Orchestration

Orchestrates module-specific report generation:
- Communications Analyzer Report
- Location Intelligence Report
- Media Viewer Report
- Device Information Report
- Cloud Analysis Report
- AI Analysis Report

Features:
- Comprehensive docstrings
- Error handling with custom exceptions
- Structured logging
- Performance optimization
"""

import logging
import json
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class OrchestratorException(Exception):
    """Base exception for orchestrator errors"""
    pass

class ModuleReportError(OrchestratorException):
    """Raised when module report generation fails"""
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

class ModuleReportOrchestrator:
    """
    Orchestrate module-specific report generation.
    
    Coordinates the generation of all module-specific reports from
    analysis results, creating comprehensive module documentation.
    """
    
    def __init__(self):
        """Initialize Module Report Orchestrator"""
        logger.debug("ModuleReportOrchestrator initialized")
    
    def generate(self, analysis_results: Dict[str, Any], case_id: str = "") -> Dict[str, str]:
        """
        Generate all module-specific reports.
        
        Orchestrates the generation of reports from all analysis modules
        including communications, location, media, device, cloud, and AI.
        
        Args:
            analysis_results (Dict[str, Any]): Analysis results from all modules
            case_id (str): Case ID for logging (optional)
            
        Returns:
            Dict[str, str]: Dictionary of module reports
            
        Raises:
            ModuleReportError: If report generation fails
            
        Example:
            >>> orchestrator = ModuleReportOrchestrator()
            >>> analysis_results = {
            ...     'comms': {...},
            ...     'location': {...},
            ...     'media': {...},
            ...     'device': {...},
            ...     'cloud': {...},
            ...     'ai': {...}
            ... }
            >>> reports = orchestrator.generate(analysis_results, "CASE-001")
            >>> len(reports) == 6
            True
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "INFO",
                "Starting module report generation",
                case_id=case_id
            )
            
            module_reports = {}
            
            # Generate each module report
            try:
                from ..module_reports import CommsAnalyzerReport
                comms_report = CommsAnalyzerReport(case_id)
                module_reports['comms'] = comms_report.generate(
                    analysis_results.get('comms', {})
                )
                structured_logger.log_with_context(
                    "DEBUG",
                    "Communications report generated",
                    case_id=case_id
                )
            except Exception as e:
                logger.error(f"Error generating communications report: {e}")
                module_reports['comms'] = f"Error: {str(e)}"
            
            try:
                from ..module_reports import LocationIntelligenceReport
                location_report = LocationIntelligenceReport(case_id)
                module_reports['location'] = location_report.generate(
                    analysis_results.get('location', {})
                )
                structured_logger.log_with_context(
                    "DEBUG",
                    "Location intelligence report generated",
                    case_id=case_id
                )
            except Exception as e:
                logger.error(f"Error generating location report: {e}")
                module_reports['location'] = f"Error: {str(e)}"
            
            try:
                from ..module_reports import MediaViewerReport
                media_report = MediaViewerReport(case_id)
                module_reports['media'] = media_report.generate(
                    analysis_results.get('media', {})
                )
                structured_logger.log_with_context(
                    "DEBUG",
                    "Media viewer report generated",
                    case_id=case_id
                )
            except Exception as e:
                logger.error(f"Error generating media report: {e}")
                module_reports['media'] = f"Error: {str(e)}"
            
            try:
                from ..module_reports import DeviceInformationReport
                device_report = DeviceInformationReport(case_id)
                module_reports['device'] = device_report.generate(
                    analysis_results.get('device', {})
                )
                structured_logger.log_with_context(
                    "DEBUG",
                    "Device information report generated",
                    case_id=case_id
                )
            except Exception as e:
                logger.error(f"Error generating device report: {e}")
                module_reports['device'] = f"Error: {str(e)}"
            
            try:
                from ..module_reports import CloudAnalysisReport
                cloud_report = CloudAnalysisReport(case_id)
                module_reports['cloud'] = cloud_report.generate(
                    analysis_results.get('cloud', {})
                )
                structured_logger.log_with_context(
                    "DEBUG",
                    "Cloud analysis report generated",
                    case_id=case_id
                )
            except Exception as e:
                logger.error(f"Error generating cloud report: {e}")
                module_reports['cloud'] = f"Error: {str(e)}"
            
            try:
                from ..module_reports import AIAnalysisReport
                ai_report = AIAnalysisReport(case_id)
                module_reports['ai'] = ai_report.generate(
                    analysis_results.get('ai', {})
                )
                structured_logger.log_with_context(
                    "DEBUG",
                    "AI analysis report generated",
                    case_id=case_id
                )
            except Exception as e:
                logger.error(f"Error generating AI report: {e}")
                module_reports['ai'] = f"Error: {str(e)}"
            
            structured_logger.log_with_context(
                "INFO",
                "Module report generation completed",
                case_id=case_id,
                report_count=len(module_reports)
            )
            
            return module_reports
        
        except Exception as e:
            error_msg = f"Error generating module reports: {str(e)}"
            logger.error(error_msg)
            raise ModuleReportError(error_msg) from e
