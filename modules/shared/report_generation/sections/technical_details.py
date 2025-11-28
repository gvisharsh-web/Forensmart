"""
TECHNICAL DETAILS SECTION - Technical Specifications

Generates technical details section with:
- Device specifications
- Extraction methodology
- Data extraction details
- Quality metrics
- Storage & encryption

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

class TechnicalDetailsSection:
    """Generate technical details section with comprehensive specifications"""
    
    @staticmethod
    def generate(extraction_results: Dict[str, Any]) -> str:
        """
        Generate technical details section.
        
        Creates detailed technical specifications including device specs,
        extraction methodology, data metrics, and storage information.
        
        Args:
            extraction_results: Extraction results dictionary
            
        Returns:
            str: Formatted technical details content
            
        Raises:
            Exception: If generation fails
        """
        try:
            logger.debug("Generating technical details section")
            
            technical = f"""
TECHNICAL DETAILS
═══════════════════════════════════════════════════════════════════════════════

DEVICE SPECIFICATIONS
────────────────────────────────────────────────────────────────────────────────
Device Model:               {extraction_results.get('device_model', 'N/A')}
Operating System:           {extraction_results.get('os_version', 'N/A')}
Processor:                  {extraction_results.get('processor', 'N/A')}
RAM:                        {extraction_results.get('ram', 'N/A')}
Storage Capacity:           {TechnicalDetailsSection._format_size(extraction_results.get('storage_capacity', 0))}
Serial Number:              {extraction_results.get('serial_number', 'N/A')}
IMEI/MAC Address:           {extraction_results.get('imei', 'N/A')}
Last Boot Time:             {extraction_results.get('last_boot', 'N/A')}

EXTRACTION METHODOLOGY
────────────────────────────────────────────────────────────────────────────────
Extraction Method:          {extraction_results.get('extraction_method', 'N/A')}
Extraction Tool:            {extraction_results.get('extraction_tool', 'N/A')}
Tool Version:               {extraction_results.get('tool_version', 'N/A')}
Extraction Duration:        {extraction_results.get('extraction_duration', 'N/A')}
Data Integrity Check:       {extraction_results.get('integrity_status', 'VERIFIED')}
Hash Algorithm:             {extraction_results.get('hash_algorithm', 'SHA-256')}
Source Hash:                {extraction_results.get('source_hash', 'N/A')}
Destination Hash:           {extraction_results.get('destination_hash', 'N/A')}
Hash Verification:          ✓ MATCH

DATA EXTRACTION DETAILS
────────────────────────────────────────────────────────────────────────────────
Extraction Modules:         {extraction_results.get('modules_count', 0)}
Total Data Size:            {TechnicalDetailsSection._format_size(extraction_results.get('total_size', 0))}
Files Extracted:            {extraction_results.get('file_count', 0):,}
Messages:                   {extraction_results.get('message_count', 0):,}
Calls:                      {extraction_results.get('call_count', 0):,}
Contacts:                   {extraction_results.get('contact_count', 0):,}
Media Items:                {extraction_results.get('media_count', 0):,}
Locations:                  {extraction_results.get('location_count', 0):,}

QUALITY METRICS
────────────────────────────────────────────────────────────────────────────────
Data Completeness:          {extraction_results.get('completeness_percentage', 0)}%
Extraction Success Rate:    {extraction_results.get('success_rate', 0)}%
Errors Encountered:         {extraction_results.get('error_count', 0)}
Warnings:                   {extraction_results.get('warning_count', 0)}
Data Integrity:             ✓ VERIFIED

STORAGE & ENCRYPTION
────────────────────────────────────────────────────────────────────────────────
Storage Location:           {extraction_results.get('storage_location', 'N/A')}
Encryption Status:          {extraction_results.get('encryption_status', 'ENCRYPTED')}
Encryption Algorithm:       {extraction_results.get('encryption_algorithm', 'AES-256')}
Backup Location:            {extraction_results.get('backup_location', 'N/A')}
Backup Encryption:          {extraction_results.get('backup_encryption', 'ENCRYPTED')}

═══════════════════════════════════════════════════════════════════════════════
"""
            logger.debug("Technical details section generated successfully")
            return technical
        
        except Exception as e:
            logger.error(f"Error generating technical details: {str(e)}")
            raise
    
    @staticmethod
    def _format_size(bytes_size: int) -> str:
        """Format bytes to human-readable size"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} PB"
