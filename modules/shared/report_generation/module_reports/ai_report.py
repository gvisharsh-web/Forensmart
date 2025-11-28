"""
AI ANALYSIS REPORT - Module-Specific Report

Generates detailed reports for AI analysis:
- Pattern detection
- Anomaly detection
- Predictive analysis
- Risk scoring
- Recommendations

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
from functools import lru_cache

logger = logging.getLogger(__name__)

class ModuleReportException(Exception):
    """Base exception for module report errors"""
    pass

class ReportGenerationError(ModuleReportException):
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

class AIAnalysisReport:
    """
    Generate detailed AI analysis reports.
    
    Creates comprehensive reports from AI analysis module including
    pattern detection, anomalies, predictions, and recommendations.
    """
    
    def __init__(self, case_id: str = ""):
        """Initialize AI Analysis Report"""
        self.case_id = case_id
        logger.debug(f"AIAnalysisReport initialized for case: {case_id}")
    
    def generate(self, ai_data: Dict[str, Any]) -> str:
        """
        Generate AI analysis report.
        
        Creates detailed report including pattern detection, anomaly
        detection, predictive analysis, risk scoring, and recommendations.
        
        Args:
            ai_data (Dict[str, Any]): AI analysis data
            
        Returns:
            str: Formatted AI analysis report
            
        Raises:
            ReportGenerationError: If report generation fails
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "DEBUG",
                "Generating AI analysis report",
                case_id=self.case_id
            )
            
            patterns = ai_data.get('patterns', [])
            anomalies = ai_data.get('anomalies', [])
            predictions = ai_data.get('predictions', [])
            
            report = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                       AI ANALYSIS REPORT                                      ║
║                          Case ID: {self.case_id:<50} ║
╚═══════════════════════════════════════════════════════════════════════════════╝

EXECUTIVE SUMMARY
─────────────────────────────────────────────────────────────────────────────────
Patterns Detected:        {len(patterns)}
Anomalies Found:          {len(anomalies)}
Predictions Made:         {len(predictions)}
Overall Risk Score:       {ai_data.get('overall_risk_score', 0)}/100
Report Generated:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PATTERN DETECTION
─────────────────────────────────────────────────────────────────────────────────
"""
            
            if patterns:
                for i, pattern in enumerate(patterns, 1):
                    confidence = pattern.get('confidence', 0)
                    report += f"""
{i}. {pattern.get('name', 'Unknown Pattern')}
   Type: {pattern.get('type', 'N/A')}
   Confidence: {confidence:.1%}
   Occurrences: {pattern.get('occurrences', 0)}
   Description: {pattern.get('description', 'N/A')}
"""
            else:
                report += "No significant patterns detected.\n"
            
            report += f"""
ANOMALY DETECTION
─────────────────────────────────────────────────────────────────────────────────
"""
            
            if anomalies:
                critical = len([a for a in anomalies if a.get('severity') == 'CRITICAL'])
                high = len([a for a in anomalies if a.get('severity') == 'HIGH'])
                medium = len([a for a in anomalies if a.get('severity') == 'MEDIUM'])
                low = len([a for a in anomalies if a.get('severity') == 'LOW'])
                
                report += f"""
Critical Anomalies:       {critical}
High Severity:            {high}
Medium Severity:          {medium}
Low Severity:             {low}

Detected Anomalies:
"""
                for i, anomaly in enumerate(anomalies[:10], 1):
                    report += f"""
{i}. {anomaly.get('description', 'Unknown Anomaly')}
   Severity: {anomaly.get('severity', 'N/A')}
   Confidence: {anomaly.get('confidence', 0):.1%}
   Timestamp: {anomaly.get('timestamp', 'N/A')}
"""
            else:
                report += "No anomalies detected.\n"
            
            report += f"""
PREDICTIVE ANALYSIS
─────────────────────────────────────────────────────────────────────────────────
"""
            
            if predictions:
                for i, prediction in enumerate(predictions, 1):
                    report += f"""
{i}. {prediction.get('prediction', 'Unknown Prediction')}
   Probability: {prediction.get('probability', 0):.1%}
   Time Horizon: {prediction.get('time_horizon', 'N/A')}
   Confidence: {prediction.get('confidence', 0):.1%}
"""
            else:
                report += "No predictions available.\n"
            
            report += f"""
RISK ASSESSMENT
─────────────────────────────────────────────────────────────────────────────────
Overall Risk Level:       {self._get_risk_level(ai_data.get('overall_risk_score', 0))}
Risk Score:               {ai_data.get('overall_risk_score', 0)}/100
Threat Level:             {ai_data.get('threat_level', 'N/A')}
Confidence:               {ai_data.get('confidence', 0):.1%}

Risk Factors:
"""
            
            risk_factors = ai_data.get('risk_factors', [])
            for factor in risk_factors[:5]:
                report += f"  • {factor.get('name', 'Unknown')}: {factor.get('impact', 'N/A')}\n"
            
            report += f"""
RECOMMENDATIONS
─────────────────────────────────────────────────────────────────────────────────
"""
            
            recommendations = ai_data.get('recommendations', [])
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    report += f"""
{i}. {rec.get('action', 'Unknown Action')}
   Priority: {rec.get('priority', 'N/A')}
   Impact: {rec.get('impact', 'N/A')}
"""
            else:
                report += "No specific recommendations at this time.\n"
            
            report += f"""
═══════════════════════════════════════════════════════════════════════════════
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            structured_logger.log_with_context(
                "DEBUG",
                "AI analysis report generated successfully",
                case_id=self.case_id,
                report_length=len(report)
            )
            
            return report
        
        except Exception as e:
            error_msg = f"Error generating AI analysis report: {str(e)}"
            logger.error(error_msg)
            raise ReportGenerationError(error_msg) from e
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _get_risk_level(risk_score: int) -> str:
        """Get risk level from score (cached)"""
        if risk_score >= 80:
            return "CRITICAL"
        elif risk_score >= 60:
            return "HIGH"
        elif risk_score >= 40:
            return "MEDIUM"
        elif risk_score >= 20:
            return "LOW"
        else:
            return "MINIMAL"
