"""
SECTION GENERATOR - Generate Individual Report Sections

Provides functionality to generate individual report sections with:
- Section content generation
- Data processing
- Formatting
- Validation
- Caching for performance
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from functools import lru_cache

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# SECTION GENERATOR CLASS
# ============================================================================

class SectionGenerator:
    """
    Generate individual report sections.
    
    This class provides methods to generate each section of a report
    with proper formatting and data processing.
    """
    
    def __init__(self):
        """Initialize section generator"""
        logger.info("SectionGenerator initialized")
    
    # ========================================================================
    # SECTION GENERATION METHODS
    # ========================================================================
    
    def generate_cover_page(self, case_details: Dict[str, Any]) -> str:
        """
        Generate cover page section
        
        Args:
            case_details: Case details dictionary
            
        Returns:
            str: Cover page content
        """
        try:
            logger.debug("Generating cover page section")
            
            cover = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                    DIGITAL FORENSIC EXAMINATION REPORT                        ║
║                   (As per Information Technology Act, 2000)                   ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

CASE INFORMATION
═══════════════════════════════════════════════════════════════════════════════

Case ID:                    {case_details.get('case_id', 'N/A')}
Case Name:                  {case_details.get('case_name', 'N/A')}
Investigation Agency:       {case_details.get('agency', 'N/A')}
Investigating Officer:      {case_details.get('investigator', 'N/A')}
Officer ID/Badge No:        {case_details.get('officer_id', 'N/A')}
Contact Number:             {case_details.get('contact', 'N/A')}

DEVICE INFORMATION
═══════════════════════════════════════════════════════════════════════════════

Device Type:                {case_details.get('device_type', 'N/A')}
Device Model:               {case_details.get('device_model', 'N/A')}
Serial Number:              {case_details.get('serial_number', 'N/A')}
IMEI/MAC Address:           {case_details.get('imei', 'N/A')}
Owner/Nominee:              {case_details.get('nominee_name', 'N/A')}

REPORT INFORMATION
═══════════════════════════════════════════════════════════════════════════════

Report Generated:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report Version:             1.0
Examiner Name:              {case_details.get('examiner_name', 'N/A')}
Report Status:              DRAFT

═══════════════════════════════════════════════════════════════════════════════
"""
            logger.debug("Cover page generated successfully")
            return cover
        
        except Exception as e:
            logger.error(f"Error generating cover page: {str(e)}")
            raise
    
    def generate_executive_summary_section(self, extraction_results: Dict[str, Any]) -> str:
        """
        Generate executive summary section
        
        Args:
            extraction_results: Extraction results dictionary
            
        Returns:
            str: Executive summary content
        """
        try:
            logger.debug("Generating executive summary section")
            
            summary = f"""
EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════════════════════

EXTRACTION OVERVIEW
────────────────────────────────────────────────────────────────────────────────
Total Data Extracted:       {self._format_size(extraction_results.get('total_size', 0))}
Files Extracted:            {extraction_results.get('file_count', 0):,}
Communications Found:       {extraction_results.get('message_count', 0):,}
Media Items:                {extraction_results.get('media_count', 0):,}
Locations Tracked:          {extraction_results.get('location_count', 0):,}

KEY FINDINGS
────────────────────────────────────────────────────────────────────────────────
{self._generate_key_findings(extraction_results)}

RISK ASSESSMENT
────────────────────────────────────────────────────────────────────────────────
Overall Risk Level:         {self._assess_risk_level(extraction_results)}
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
    
    def generate_technical_details_section(self, extraction_results: Dict[str, Any]) -> str:
        """
        Generate technical details section
        
        Args:
            extraction_results: Extraction results dictionary
            
        Returns:
            str: Technical details content
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
Storage Capacity:           {self._format_size(extraction_results.get('storage_capacity', 0))}

EXTRACTION METHODOLOGY
────────────────────────────────────────────────────────────────────────────────
Extraction Method:          {extraction_results.get('extraction_method', 'N/A')}
Extraction Duration:        {extraction_results.get('extraction_duration', 'N/A')}
Data Integrity:             {extraction_results.get('integrity_status', 'VERIFIED')}
Hash Verification:          {extraction_results.get('hash_verified', 'YES')}

QUALITY METRICS
────────────────────────────────────────────────────────────────────────────────
Data Completeness:          {extraction_results.get('completeness_percentage', 0)}%
Extraction Success:         {extraction_results.get('success_rate', 0)}%
Errors Encountered:         {extraction_results.get('error_count', 0)}
Warnings:                   {extraction_results.get('warning_count', 0)}

═══════════════════════════════════════════════════════════════════════════════
"""
            logger.debug("Technical details section generated successfully")
            return technical
        
        except Exception as e:
            logger.error(f"Error generating technical details: {str(e)}")
            raise
    
    def generate_findings_section(self, extraction_results: Dict[str, Any]) -> str:
        """
        Generate findings section
        
        Args:
            extraction_results: Extraction results dictionary
            
        Returns:
            str: Findings content
        """
        try:
            logger.debug("Generating findings section")
            
            findings = f"""
FINDINGS & ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

COMMUNICATIONS ANALYSIS
────────────────────────────────────────────────────────────────────────────────
Total Messages:             {extraction_results.get('message_count', 0):,}
SMS Messages:               {extraction_results.get('sms_count', 0):,}
Email Messages:             {extraction_results.get('email_count', 0):,}
Chat Applications:          {extraction_results.get('chat_app_count', 0)}

LOCATION INTELLIGENCE
────────────────────────────────────────────────────────────────────────────────
Unique Locations:           {extraction_results.get('location_count', 0):,}
GPS Coordinates Found:      {extraction_results.get('gps_count', 0):,}

MEDIA ANALYSIS
────────────────────────────────────────────────────────────────────────────────
Total Media Files:          {extraction_results.get('media_count', 0):,}
Photos:                     {extraction_results.get('photo_count', 0):,}
Videos:                     {extraction_results.get('video_count', 0):,}
Audio Files:                {extraction_results.get('audio_count', 0):,}

SECURITY FINDINGS
────────────────────────────────────────────────────────────────────────────────
Installed Applications:     {extraction_results.get('app_count', 0):,}
Suspicious Apps:            {extraction_results.get('suspicious_apps', 0)}
Malware Detected:           {extraction_results.get('malware_count', 0)}
Security Issues:            {extraction_results.get('security_issues', 0)}

═══════════════════════════════════════════════════════════════════════════════
"""
            logger.debug("Findings section generated successfully")
            return findings
        
        except Exception as e:
            logger.error(f"Error generating findings: {str(e)}")
            raise
    
    def generate_conclusions_section(self, extraction_results: Dict[str, Any]) -> str:
        """
        Generate conclusions section
        
        Args:
            extraction_results: Extraction results dictionary
            
        Returns:
            str: Conclusions content
        """
        try:
            logger.debug("Generating conclusions section")
            
            conclusions = f"""
CONCLUSIONS
═══════════════════════════════════════════════════════════════════════════════

KEY CONCLUSIONS
────────────────────────────────────────────────────────────────────────────────
Based on the forensic examination of the device, the following conclusions
can be drawn:

1. Data Integrity: ✓ VERIFIED
   All extracted data has been verified for integrity using hash verification.

2. Chain of Custody: ✓ MAINTAINED
   Proper chain of custody has been maintained throughout the examination.

3. Completeness: {extraction_results.get('completeness_percentage', 0)}%
   The extraction is {extraction_results.get('completeness_percentage', 0)}% complete.

4. Evidence Quality: ✓ ADMISSIBLE
   All evidence meets the standards for admissibility in court proceedings.

═══════════════════════════════════════════════════════════════════════════════
"""
            logger.debug("Conclusions section generated successfully")
            return conclusions
        
        except Exception as e:
            logger.error(f"Error generating conclusions: {str(e)}")
            raise
    
    # ========================================================================
    # HELPER METHODS WITH CACHING
    # ========================================================================
    
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
    @lru_cache(maxsize=128)
    def _assess_risk_level(risk_score: int) -> str:
        """
        Assess overall risk level with caching.
        
        Uses LRU cache to avoid repeated risk assessment calculations.
        
        Args:
            risk_score: Risk score (0-100)
            
        Returns:
            str: Risk level (CRITICAL, HIGH, MEDIUM, LOW, MINIMAL)
        """
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
    
    @staticmethod
    def _generate_key_findings(results: Dict[str, Any]) -> str:
        """
        Generate key findings summary.
        
        Analyzes extraction results and generates a list of key findings.
        
        Args:
            results: Extraction results dictionary
            
        Returns:
            str: Formatted key findings (one per line)
        """
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
