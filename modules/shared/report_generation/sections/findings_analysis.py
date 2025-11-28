"""
FINDINGS & ANALYSIS SECTION - Detailed Analysis

Generates findings and analysis section with:
- Communications analysis
- Location intelligence
- Media analysis
- Device information
- Security findings
- Evidence summary

Features:
- Comprehensive docstrings
- Error handling
- Structured logging
"""

import logging
import json
from typing import Dict, Any
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

class FindingsAnalysisSection:
    """Generate findings and analysis section with detailed insights"""
    
    @staticmethod
    def generate(extraction_results: Dict[str, Any]) -> str:
        """
        Generate findings and analysis section.
        
        Creates detailed analysis of extracted data including communications,
        locations, media, device information, security findings, and evidence.
        
        Args:
            extraction_results: Extraction results dictionary
            
        Returns:
            str: Formatted findings and analysis content
            
        Raises:
            Exception: If generation fails
        """
        try:
            logger.debug("Generating findings and analysis section")
            
            findings = f"""
FINDINGS & ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

COMMUNICATIONS ANALYSIS
────────────────────────────────────────────────────────────────────────────────
Total Messages:             {extraction_results.get('message_count', 0):,}
SMS Messages:               {extraction_results.get('sms_count', 0):,}
Email Messages:             {extraction_results.get('email_count', 0):,}
Chat Applications:          {extraction_results.get('chat_app_count', 0)}
Suspicious Communications:  {extraction_results.get('suspicious_messages', 0)}

Key Contacts:
{FindingsAnalysisSection._format_contacts(extraction_results.get('top_contacts', []))}

LOCATION INTELLIGENCE
────────────────────────────────────────────────────────────────────────────────
Unique Locations:           {extraction_results.get('location_count', 0):,}
GPS Coordinates Found:      {extraction_results.get('gps_count', 0):,}

Frequent Locations:
{FindingsAnalysisSection._format_locations(extraction_results.get('frequent_locations', []))}

MEDIA ANALYSIS
────────────────────────────────────────────────────────────────────────────────
Total Media Files:          {extraction_results.get('media_count', 0):,}
Photos:                     {extraction_results.get('photo_count', 0):,}
Videos:                     {extraction_results.get('video_count', 0):,}
Audio Files:                {extraction_results.get('audio_count', 0):,}

DEVICE INFORMATION
────────────────────────────────────────────────────────────────────────────────
Device Model:               {extraction_results.get('device_model', 'N/A')}
Operating System:           {extraction_results.get('os_version', 'N/A')}
Last Boot Time:             {extraction_results.get('last_boot', 'N/A')}
Storage Used:               {FindingsAnalysisSection._format_size(extraction_results.get('storage_used', 0))}
Storage Available:          {FindingsAnalysisSection._format_size(extraction_results.get('storage_available', 0))}

SECURITY FINDINGS
────────────────────────────────────────────────────────────────────────────────
Installed Applications:     {extraction_results.get('app_count', 0):,}
Suspicious Apps:            {extraction_results.get('suspicious_apps', 0)}
Malware Detected:           {extraction_results.get('malware_count', 0)}
Security Issues:            {extraction_results.get('security_issues', 0)}

EVIDENCE SUMMARY
────────────────────────────────────────────────────────────────────────────────
Total Evidence Items:       {extraction_results.get('evidence_count', 0):,}
Critical Evidence:          {extraction_results.get('critical_evidence', 0)}
Supporting Evidence:        {extraction_results.get('supporting_evidence', 0)}

═══════════════════════════════════════════════════════════════════════════════
"""
            logger.debug("Findings and analysis section generated successfully")
            return findings
        
        except Exception as e:
            logger.error(f"Error generating findings and analysis: {str(e)}")
            raise
    
    @staticmethod
    def _format_size(bytes_size: int) -> str:
        """Format bytes to human-readable size"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} PB"
    
    @staticmethod
    def _format_contacts(contacts: list) -> str:
        """Format contacts list"""
        if not contacts:
            return "  No significant contacts identified"
        
        formatted = []
        for contact in contacts[:5]:
            formatted.append(f"  • {contact.get('name', 'Unknown')}: {contact.get('message_count', 0)} messages")
        
        return "\n".join(formatted)
    
    @staticmethod
    def _format_locations(locations: list) -> str:
        """Format locations list"""
        if not locations:
            return "  No location data available"
        
        formatted = []
        for loc in locations[:5]:
            formatted.append(f"  • {loc.get('name', 'Unknown')}: {loc.get('visits', 0)} visits")
        
        return "\n".join(formatted)
