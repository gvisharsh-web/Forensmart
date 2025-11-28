"""
APPENDICES SECTION - Supporting Documentation

Generates appendices section with:
- Detailed data tables
- Screenshots & evidence
- Technical specifications
- Glossary
- References

Features:
- Comprehensive docstrings
- Error handling
- Structured logging
"""

import logging
import json
from typing import Dict, Any, List
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

class AppendicesSection:
    """Generate appendices section with supporting documentation"""
    
    @staticmethod
    def generate(extraction_results: Dict[str, Any]) -> str:
        """
        Generate appendices section.
        
        Creates appendices including data tables, evidence references,
        technical specifications, glossary, and references.
        
        Args:
            extraction_results: Extraction results dictionary
            
        Returns:
            str: Formatted appendices content
            
        Raises:
            Exception: If generation fails
        """
        try:
            logger.debug("Generating appendices section")
            
            appendices = f"""
APPENDICES
═══════════════════════════════════════════════════════════════════════════════

APPENDIX A: DETAILED DATA TABLES
────────────────────────────────────────────────────────────────────────────────

Communications Table:
  Total Messages:           {extraction_results.get('message_count', 0):,}
  SMS Messages:             {extraction_results.get('sms_count', 0):,}
  Email Messages:           {extraction_results.get('email_count', 0):,}
  Chat Messages:            {extraction_results.get('chat_count', 0):,}

Location Table:
  Unique Locations:         {extraction_results.get('location_count', 0):,}
  GPS Coordinates:          {extraction_results.get('gps_count', 0):,}
  Frequent Locations:       {extraction_results.get('frequent_location_count', 0)}

Media Table:
  Total Media:              {extraction_results.get('media_count', 0):,}
  Photos:                   {extraction_results.get('photo_count', 0):,}
  Videos:                   {extraction_results.get('video_count', 0):,}
  Audio:                    {extraction_results.get('audio_count', 0):,}

Device Information Table:
  Device Model:             {extraction_results.get('device_model', 'N/A')}
  OS Version:               {extraction_results.get('os_version', 'N/A')}
  Storage Capacity:         {extraction_results.get('storage_capacity', 'N/A')}

APPENDIX B: SCREENSHOTS & EVIDENCE
────────────────────────────────────────────────────────────────────────────────

Evidence items are documented and stored separately with the following references:

• Screenshot 1: Device home screen
• Screenshot 2: Communications application
• Screenshot 3: Location history
• Screenshot 4: Installed applications
• Screenshot 5: System settings

All screenshots are stored in the evidence folder with proper naming convention.

APPENDIX C: TECHNICAL SPECIFICATIONS
────────────────────────────────────────────────────────────────────────────────

Device Specifications:
  Device Type:              {extraction_results.get('device_type', 'N/A')}
  Device Model:             {extraction_results.get('device_model', 'N/A')}
  Operating System:         {extraction_results.get('os_version', 'N/A')}
  Processor:                {extraction_results.get('processor', 'N/A')}
  RAM:                      {extraction_results.get('ram', 'N/A')}

Extraction Tool Specifications:
  Tool Name:                {extraction_results.get('extraction_tool', 'N/A')}
  Tool Version:             {extraction_results.get('tool_version', 'N/A')}
  Hash Algorithm:           {extraction_results.get('hash_algorithm', 'SHA-256')}

APPENDIX D: GLOSSARY
────────────────────────────────────────────────────────────────────────────────

IMEI: International Mobile Equipment Identity
MAC: Media Access Control
GPS: Global Positioning System
SMS: Short Message Service
CoC: Chain of Custody
SHA: Secure Hash Algorithm
IT Act: Information Technology Act, 2000

APPENDIX E: REFERENCES
────────────────────────────────────────────────────────────────────────────────

Legal References:
• Information Technology Act, 2000 - Section 65, 66, 67
• Indian Evidence Act, 1872 - Section 3, 45, 65
• Indian Penal Code - Section 379, 380, 381

Technical References:
• Digital Forensics Standards and Procedures
• NIST Guidelines for Mobile Device Forensics
• ISO/IEC 27037 - Guidelines for identification, collection, acquisition and preservation

═══════════════════════════════════════════════════════════════════════════════
"""
            logger.debug("Appendices section generated successfully")
            return appendices
        
        except Exception as e:
            logger.error(f"Error generating appendices: {str(e)}")
            raise
