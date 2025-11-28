"""
RECOMMENDATIONS SECTION - Investigation Recommendations

Generates recommendations section with:
- Immediate actions
- Follow-up investigation
- Evidence handling
- Legal proceedings

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

class RecommendationsSection:
    """Generate recommendations section with actionable guidance"""
    
    @staticmethod
    def generate(extraction_results: Dict[str, Any]) -> str:
        """
        Generate recommendations section.
        
        Creates recommendations for immediate actions, follow-up investigation,
        evidence handling, and legal proceedings.
        
        Args:
            extraction_results: Extraction results dictionary
            
        Returns:
            str: Formatted recommendations content
            
        Raises:
            Exception: If generation fails
        """
        try:
            logger.debug("Generating recommendations section")
            
            recommendations = f"""
RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════════════════

IMMEDIATE ACTIONS
────────────────────────────────────────────────────────────────────────────────

{RecommendationsSection._generate_immediate_actions(extraction_results)}

FOLLOW-UP INVESTIGATION
────────────────────────────────────────────────────────────────────────────────

1. Conduct detailed analysis of suspicious communications
   - Review message content and timestamps
   - Identify communication patterns
   - Cross-reference with other evidence

2. Cross-reference locations with known associates
   - Map device locations to known addresses
   - Identify location patterns
   - Correlate with timeline of events

3. Investigate potential malware or spyware
   - Analyze suspicious applications
   - Review system logs for unauthorized access
   - Check for remote access tools

4. Review security settings and access logs
   - Examine device security configuration
   - Review access control settings
   - Identify potential vulnerabilities

EVIDENCE HANDLING
────────────────────────────────────────────────────────────────────────────────

1. Preservation: Maintain secure storage of extracted data
   - Keep data in encrypted storage
   - Maintain backup copies
   - Document all access

2. Storage: Store in secure facility with access control
   - Restricted access
   - Environmental controls
   - Backup power supply

3. Access Control: Limit access to authorized personnel only
   - Maintain access logs
   - Require authentication
   - Document all access

LEGAL PROCEEDINGS
────────────────────────────────────────────────────────────────────────────────

1. Court Submission: Submit report and evidence to court
   - Prepare certified copies
   - Organize evidence exhibits
   - Prepare for presentation

2. Expert Testimony: Be prepared to provide expert testimony
   - Review findings thoroughly
   - Prepare testimony notes
   - Anticipate cross-examination

3. Evidence Presentation: Present evidence clearly and professionally
   - Use visual aids
   - Explain technical details
   - Maintain professional demeanor

═══════════════════════════════════════════════════════════════════════════════
"""
            logger.debug("Recommendations section generated successfully")
            return recommendations
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise
    
    @staticmethod
    def _generate_immediate_actions(results: Dict[str, Any]) -> str:
        """Generate immediate actions"""
        actions = []
        
        if results.get('suspicious_messages', 0) > 5:
            actions.append("1. Conduct detailed analysis of suspicious communications")
        
        if results.get('location_count', 0) > 20:
            actions.append("2. Cross-reference locations with known associates")
        
        if results.get('malware_count', 0) > 0:
            actions.append("3. Investigate potential malware or spyware installation")
        
        if results.get('security_risk', 'LOW') == 'HIGH':
            actions.append("4. Review security settings and access logs")
        
        if not actions:
            actions.append("1. Continue standard investigation procedures")
        
        return "\n".join(actions)
