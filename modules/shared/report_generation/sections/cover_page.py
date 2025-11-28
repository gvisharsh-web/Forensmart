"""
COVER PAGE SECTION - Report Cover Page

Generates the cover page for forensic reports with:
- Report title
- Case information
- Device information
- Report metadata
- IT Act compliance notice
"""

import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CoverPageSection:
    """Generate cover page section for reports with professional formatting"""
    
    @staticmethod
    def generate(case_details: Dict[str, Any]) -> str:
        """
        Generate cover page section with professional formatting.
        
        Creates a formatted cover page containing case information, device details,
        and report metadata suitable for forensic reports. Includes IT Act compliance
        notice and confidentiality warnings.
        
        Args:
            case_details (Dict[str, Any]): Dictionary containing case information
                with keys: 'case_id', 'case_name', 'agency', 'investigator',
                'officer_id', 'contact', 'device_type', 'device_model',
                'serial_number', 'imei', 'nominee_name', 'examiner_name'
            
        Returns:
            str: Formatted cover page content ready for inclusion in report
            
        Raises:
            Exception: If cover page generation fails
            
        Example:
            >>> case_details = {
            ...     'case_id': 'CASE-001',
            ...     'investigator': 'John Smith',
            ...     'device_type': 'Android'
            ... }
            >>> cover = CoverPageSection.generate(case_details)
            >>> print(cover[:50])
            '╔═══════════════════════════════════════════════'
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

CONFIDENTIALITY NOTICE
═══════════════════════════════════════════════════════════════════════════════

This report contains confidential forensic examination results and is intended
only for authorized personnel. Unauthorized access, use, or distribution is
prohibited by law.

═══════════════════════════════════════════════════════════════════════════════
"""
            logger.debug("Cover page section generated successfully")
            return cover
        
        except Exception as e:
            logger.error(f"Error generating cover page: {str(e)}")
            raise
