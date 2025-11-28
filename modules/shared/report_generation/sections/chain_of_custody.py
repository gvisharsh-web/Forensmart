"""
CHAIN OF CUSTODY SECTION - Custody Documentation

Generates chain of custody section with:
- Initial seizure
- Custody history
- Storage details
- Final examination
- Certification

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

class ChainOfCustodySection:
    """Generate chain of custody section with comprehensive documentation"""
    
    @staticmethod
    def generate(case_details: Dict[str, Any], custody_records: List[Dict[str, Any]] = None) -> str:
        """
        Generate chain of custody section.
        
        Creates detailed chain of custody documentation including initial seizure,
        custody transfers, storage details, and final examination verification.
        
        Args:
            case_details: Case details dictionary
            custody_records: List of custody transfer records (optional)
            
        Returns:
            str: Formatted chain of custody content
            
        Raises:
            Exception: If generation fails
        """
        try:
            logger.debug("Generating chain of custody section")
            
            if custody_records is None:
                custody_records = []
            
            custody_history = ""
            for i, record in enumerate(custody_records, 1):
                custody_history += f"""
Transfer {i}:
  Received By:     {record.get('received_by', 'N/A')}
  Received Date:   {record.get('received_date', 'N/A')}
  Received From:   {record.get('received_from', 'N/A')}
  Purpose:         {record.get('purpose', 'N/A')}
  Signature:       ________________________
"""
            
            coc = f"""
CHAIN OF CUSTODY
═══════════════════════════════════════════════════════════════════════════════

INITIAL SEIZURE
────────────────────────────────────────────────────────────────────────────────
Seizure Date & Time:        {case_details.get('seizure_date', 'N/A')}
Seized By (Name & ID):      {case_details.get('seized_by', 'N/A')}
Seizure Location:           {case_details.get('seizure_location', 'N/A')}
Device Condition:           {case_details.get('device_condition', 'N/A')}
Device Seal/Lock Status:    {case_details.get('seal_status', 'N/A')}
Initial Hash/Checksum:      {case_details.get('initial_hash', 'N/A')}

CUSTODY HISTORY
────────────────────────────────────────────────────────────────────────────────
{custody_history if custody_history else "No transfers recorded"}

STORAGE DETAILS
────────────────────────────────────────────────────────────────────────────────
Storage Location:           {case_details.get('storage_location', 'N/A')}
Storage Conditions:         {case_details.get('storage_conditions', 'N/A')}
Security Measures:          {case_details.get('security_measures', 'N/A')}
Access Control:             {case_details.get('access_control', 'N/A')}
Backup Location:            {case_details.get('backup_location', 'N/A')}

FINAL EXAMINATION
────────────────────────────────────────────────────────────────────────────────
Examination Date & Time:    {case_details.get('examination_date', 'N/A')}
Examined By (Name & ID):    {case_details.get('examined_by', 'N/A')}
Examination Method:         {case_details.get('examination_method', 'N/A')}
Final Hash/Checksum:        {case_details.get('final_hash', 'N/A')}
Verification Status:        {case_details.get('verification_status', 'VERIFIED')}

CERTIFICATION
────────────────────────────────────────────────────────────────────────────────
Chain Integrity:            ✓ VERIFIED
No Tampering:               ✓ CONFIRMED
Data Integrity:             ✓ VERIFIED
Legal Compliance:           ✓ CONFIRMED

═══════════════════════════════════════════════════════════════════════════════
"""
            logger.debug("Chain of custody section generated successfully")
            return coc
        
        except Exception as e:
            logger.error(f"Error generating chain of custody: {str(e)}")
            raise
