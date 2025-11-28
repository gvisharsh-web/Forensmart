"""
COMMUNICATIONS ANALYZER REPORT - Module-Specific Report

Generates detailed reports for communications analysis:
- Message analysis
- Call records
- Contact analysis
- Communication patterns
- Timeline analysis

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

class CommsAnalyzerReport:
    """
    Generate detailed communications analysis reports.
    
    Creates comprehensive reports from communications analyzer module
    including message analysis, call records, contacts, and patterns.
    """
    
    def __init__(self, case_id: str = ""):
        """
        Initialize Communications Analyzer Report.
        
        Args:
            case_id (str): Case ID for logging
        """
        self.case_id = case_id
        logger.debug(f"CommsAnalyzerReport initialized for case: {case_id}")
    
    def generate(self, comms_data: Dict[str, Any]) -> str:
        """
        Generate communications analysis report.
        
        Creates detailed report including message analysis, call records,
        contact information, communication patterns, and timeline analysis.
        
        Args:
            comms_data (Dict[str, Any]): Communications analysis data
            
        Returns:
            str: Formatted communications report
            
        Raises:
            ReportGenerationError: If report generation fails
            
        Example:
            >>> report_gen = CommsAnalyzerReport("CASE-001")
            >>> comms_data = {
            ...     'messages': [...],
            ...     'calls': [...],
            ...     'contacts': [...]
            ... }
            >>> report = report_gen.generate(comms_data)
            >>> len(report) > 1000
            True
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "DEBUG",
                "Generating communications analysis report",
                case_id=self.case_id
            )
            
            # Extract data
            messages = comms_data.get('messages', [])
            calls = comms_data.get('calls', [])
            contacts = comms_data.get('contacts', [])
            
            # Generate sections
            report = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    COMMUNICATIONS ANALYSIS REPORT                             ║
║                          Case ID: {self.case_id:<50} ║
╚═══════════════════════════════════════════════════════════════════════════════╝

EXECUTIVE SUMMARY
─────────────────────────────────────────────────────────────────────────────────
Total Messages:           {len(messages):,}
Total Calls:              {len(calls):,}
Total Contacts:           {len(contacts):,}
Report Generated:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MESSAGE ANALYSIS
─────────────────────────────────────────────────────────────────────────────────
"""
            
            # Message statistics
            if messages:
                sms_count = len([m for m in messages if m.get('type') == 'SMS'])
                mms_count = len([m for m in messages if m.get('type') == 'MMS'])
                app_count = len([m for m in messages if m.get('type') == 'APP'])
                
                report += f"""
SMS Messages:             {sms_count:,}
MMS Messages:             {mms_count:,}
App Messages:             {app_count:,}

Top Contacts (by message count):
"""
                # Get top contacts
                contact_msgs = {}
                for msg in messages:
                    contact = msg.get('contact', 'Unknown')
                    contact_msgs[contact] = contact_msgs.get(contact, 0) + 1
                
                for contact, count in sorted(contact_msgs.items(), key=lambda x: x[1], reverse=True)[:5]:
                    report += f"  • {contact}: {count:,} messages\n"
            
            report += f"""
CALL ANALYSIS
─────────────────────────────────────────────────────────────────────────────────
"""
            
            # Call statistics
            if calls:
                incoming = len([c for c in calls if c.get('direction') == 'INCOMING'])
                outgoing = len([c for c in calls if c.get('direction') == 'OUTGOING'])
                missed = len([c for c in calls if c.get('direction') == 'MISSED'])
                
                total_duration = sum(c.get('duration', 0) for c in calls)
                avg_duration = total_duration / len(calls) if calls else 0
                
                report += f"""
Incoming Calls:           {incoming:,}
Outgoing Calls:           {outgoing:,}
Missed Calls:             {missed:,}
Total Call Duration:      {self._format_duration(total_duration)}
Average Call Duration:    {self._format_duration(avg_duration)}

Top Callers:
"""
                # Get top callers
                caller_calls = {}
                for call in calls:
                    contact = call.get('contact', 'Unknown')
                    caller_calls[contact] = caller_calls.get(contact, 0) + 1
                
                for contact, count in sorted(caller_calls.items(), key=lambda x: x[1], reverse=True)[:5]:
                    report += f"  • {contact}: {count:,} calls\n"
            
            report += f"""
CONTACT ANALYSIS
─────────────────────────────────────────────────────────────────────────────────
Total Contacts:           {len(contacts):,}

Contact Details:
"""
            
            for contact in contacts[:10]:
                report += f"""
  Contact: {contact.get('name', 'Unknown')}
    Phone: {contact.get('phone', 'N/A')}
    Email: {contact.get('email', 'N/A')}
    Messages: {contact.get('message_count', 0):,}
    Calls: {contact.get('call_count', 0):,}
"""
            
            report += f"""
COMMUNICATION PATTERNS
─────────────────────────────────────────────────────────────────────────────────
Peak Communication Hours: {self._analyze_peak_hours(messages + calls)}
Most Active Day: {self._analyze_peak_day(messages + calls)}
Communication Frequency: {self._analyze_frequency(messages + calls)}

═══════════════════════════════════════════════════════════════════════════════
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            structured_logger.log_with_context(
                "DEBUG",
                "Communications report generated successfully",
                case_id=self.case_id,
                report_length=len(report)
            )
            
            return report
        
        except Exception as e:
            error_msg = f"Error generating communications report: {str(e)}"
            logger.error(error_msg)
            raise ReportGenerationError(error_msg) from e
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _format_duration(seconds: int) -> str:
        """Format duration in seconds to readable format (cached)"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs}s"
    
    @staticmethod
    def _analyze_peak_hours(communications: List[Dict[str, Any]]) -> str:
        """Analyze peak communication hours"""
        if not communications:
            return "N/A"
        hours = {}
        for comm in communications:
            if 'timestamp' in comm:
                hour = datetime.fromisoformat(comm['timestamp']).hour
                hours[hour] = hours.get(hour, 0) + 1
        if hours:
            peak_hour = max(hours, key=hours.get)
            return f"{peak_hour:02d}:00 - {peak_hour+1:02d}:00"
        return "N/A"
    
    @staticmethod
    def _analyze_peak_day(communications: List[Dict[str, Any]]) -> str:
        """Analyze peak communication day"""
        if not communications:
            return "N/A"
        days = {}
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for comm in communications:
            if 'timestamp' in comm:
                day = datetime.fromisoformat(comm['timestamp']).weekday()
                days[day] = days.get(day, 0) + 1
        if days:
            peak_day = max(days, key=days.get)
            return day_names[peak_day]
        return "N/A"
    
    @staticmethod
    def _analyze_frequency(communications: List[Dict[str, Any]]) -> str:
        """Analyze communication frequency"""
        if not communications:
            return "N/A"
        return f"{len(communications):,} communications"
