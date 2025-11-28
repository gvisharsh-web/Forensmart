"""
EXECUTIVE SUMMARY SECTION - High-level Report Summary

Generates executive summary with:
- Investigation overview
- Extraction summary
- Key findings
- Risk assessment
- Next steps

Features:
- Comprehensive docstrings
- Error handling
- Structured logging
- Caching for performance
"""

import logging
import json
from typing import Dict, Any
from functools import lru_cache
from datetime import datetime

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

class ExecutiveSummarySection:
    """Generate executive summary section"""
    
    @staticmethod
    def generate(extraction_results: Dict[str, Any]) -> str:
        """
        Generate executive summary section.
        
        Args:
            extraction_results: Extraction results dictionary
            
        Returns:
            str: Formatted executive summary content
        """
        try:
            logger.debug("Generating executive summary section")
            
            summary = f"""
EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════════════════════

EXTRACTION OVERVIEW
────────────────────────────────────────────────────────────────────────────────
Total Data Extracted:       {ExecutiveSummarySection._format_size(extraction_results.get('total_size', 0))}
Files Extracted:            {extraction_results.get('file_count', 0):,}
Communications Found:       {extraction_results.get('message_count', 0):,}
Media Items:                {extraction_results.get('media_count', 0):,}
Locations Tracked:          {extraction_results.get('location_count', 0):,}

KEY FINDINGS
────────────────────────────────────────────────────────────────────────────────
{ExecutiveSummarySection._generate_key_findings(extraction_results)}

RISK ASSESSMENT
────────────────────────────────────────────────────────────────────────────────
Overall Risk Level:         {ExecutiveSummarySection._assess_risk_level(extraction_results)}
Critical Findings:          {extraction_results.get('critical_count', 0)}
High Priority Items:        {extraction_results.get('high_count', 0)}
Medium Priority Items:      {extraction_results.get('medium_count', 0)}

═══════════════════════════════════════════════════════════════════════════════
"""
            logger.debug("Executive summary section generated successfully")
            return summary
        
        except Exception as e:
            logger.error(f"Error generating executive summary: {str(e)}")
            raise
    
    @staticmethod
    @lru_cache(maxsize=256)
    def _format_size(bytes_size: int) -> str:
        """
        Format bytes to human-readable size with caching.
        
        Uses LRU cache to avoid repeated formatting of same sizes.
        
        Args:
            bytes_size: Size in bytes
            
        Returns:
            str: Formatted size (e.g., "1.50 GB")
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} PB"
    
    @staticmethod
    def _generate_key_findings(results: Dict[str, Any]) -> str:
        """Generate key findings"""
        findings = []
        
        if results.get('suspicious_messages', 0) > 0:
            findings.append(f"• {results['suspicious_messages']} suspicious communications detected")
        
        if results.get('location_count', 0) > 10:
            findings.append(f"• Device tracked in {results['location_count']} different locations")
        
        if results.get('malware_count', 0) > 0:
            findings.append(f"• {results['malware_count']} potential malware/suspicious apps found")
        
        if results.get('critical_evidence', 0) > 0:
            findings.append(f"• {results['critical_evidence']} critical evidence items identified")
        
        if not findings:
            findings.append("• No critical findings at this time")
        
        return "\n".join(findings)
    
    @staticmethod
    def _assess_risk_level(results: Dict[str, Any]) -> str:
        """Assess risk level"""
        risk_score = results.get('risk_score', 0)
        
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
