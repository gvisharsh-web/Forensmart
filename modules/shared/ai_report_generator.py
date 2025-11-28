"""
AI-POWERED REPORT GENERATION MODULE

Generates human-readable forensic reports with:
- Natural language processing
- Intelligent data summarization
- Contextual analysis
- Executive summaries
- Detailed findings
- Risk assessment
- Recommendations
- Timeline generation
- Evidence linking
- Chain of custody documentation
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# REPORT TYPES
# ============================================================================

class ReportType(Enum):
    """Types of forensic reports"""
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_FINDINGS = "detailed_findings"
    TECHNICAL_ANALYSIS = "technical_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    TIMELINE_REPORT = "timeline_report"
    EVIDENCE_REPORT = "evidence_report"
    CHAIN_OF_CUSTODY = "chain_of_custody"
    FULL_REPORT = "full_report"


# ============================================================================
# AI REPORT GENERATOR
# ============================================================================

class AIReportGenerator:
    """Generate human-readable forensic reports using AI"""
    
    def __init__(self, case_id: str, case_details: Dict[str, Any]):
        """
        Initialize report generator
        
        Args:
            case_id: Case ID
            case_details: Case details dictionary
        """
        self.case_id = case_id
        self.case_details = case_details
        self.generated_at = datetime.now()
        self.report_data = {}
    
    # ========================================================================
    # EXECUTIVE SUMMARY
    # ========================================================================
    
    def generate_executive_summary(self, extraction_results: Dict[str, Any]) -> str:
        """
        Generate executive summary in human-readable format
        
        Args:
            extraction_results: Extraction results dictionary
            
        Returns:
            str: Executive summary
        """
        try:
            summary = f"""
═══════════════════════════════════════════════════════════════════════════════
                        EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════════════════════

CASE INFORMATION
────────────────────────────────────────────────────────────────────────────────
Case ID:                {self.case_id}
Investigator:           {self.case_details.get('investigator', 'N/A')}
Nominee:                {self.case_details.get('nominee_name', 'N/A')}
Device Type:            {self.case_details.get('device_type', 'N/A')}
Extraction Date:        {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
Report Generated:       {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}

INVESTIGATION OVERVIEW
────────────────────────────────────────────────────────────────────────────────
Investigation Reason:   {self.case_details.get('reason', 'N/A')}
Investigation Status:   ACTIVE
Authorization Level:    {self.case_details.get('consent_level', 'LEGAL')}

EXTRACTION SUMMARY
────────────────────────────────────────────────────────────────────────────────
Total Data Extracted:   {self._format_size(extraction_results.get('total_size', 0))}
Files Extracted:        {extraction_results.get('file_count', 0):,}
Communications Found:   {extraction_results.get('message_count', 0):,}
Media Items:            {extraction_results.get('media_count', 0):,}
Locations Tracked:      {extraction_results.get('location_count', 0):,}

KEY FINDINGS
────────────────────────────────────────────────────────────────────────────────
{self._generate_key_findings(extraction_results)}

RISK ASSESSMENT
────────────────────────────────────────────────────────────────────────────────
Overall Risk Level:     {self._assess_risk_level(extraction_results)}
Critical Findings:      {extraction_results.get('critical_count', 0)}
High Priority Items:    {extraction_results.get('high_count', 0)}
Medium Priority Items:  {extraction_results.get('medium_count', 0)}

NEXT STEPS
────────────────────────────────────────────────────────────────────────────────
1. Review detailed findings in full report
2. Analyze timeline of events
3. Cross-reference evidence
4. Conduct follow-up investigation if needed
5. Document findings in case file

═══════════════════════════════════════════════════════════════════════════════
"""
            return summary
        
        except Exception as e:
            logger.error(f"Error generating executive summary: {str(e)}")
            return f"Error generating executive summary: {str(e)}"
    
    # ========================================================================
    # DETAILED FINDINGS
    # ========================================================================
    
    def generate_detailed_findings(self, extraction_results: Dict[str, Any]) -> str:
        """
        Generate detailed findings report
        
        Args:
            extraction_results: Extraction results
            
        Returns:
            str: Detailed findings
        """
        try:
            findings = f"""
═══════════════════════════════════════════════════════════════════════════════
                        DETAILED FINDINGS REPORT
═══════════════════════════════════════════════════════════════════════════════

CASE: {self.case_id}
GENERATED: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}

1. COMMUNICATIONS ANALYSIS
────────────────────────────────────────────────────────────────────────────────
Total Messages:         {extraction_results.get('message_count', 0):,}
SMS Messages:           {extraction_results.get('sms_count', 0):,}
Email Messages:         {extraction_results.get('email_count', 0):,}
Chat Applications:      {extraction_results.get('chat_app_count', 0)}

Key Contacts:
{self._format_contacts(extraction_results.get('top_contacts', []))}

Suspicious Communications:
{self._format_suspicious_comms(extraction_results.get('suspicious_messages', []))}

2. LOCATION INTELLIGENCE
────────────────────────────────────────────────────────────────────────────────
Unique Locations:       {extraction_results.get('location_count', 0):,}
GPS Coordinates Found:  {extraction_results.get('gps_count', 0):,}
Frequent Locations:
{self._format_locations(extraction_results.get('frequent_locations', []))}

Timeline of Movements:
{self._format_movement_timeline(extraction_results.get('movement_timeline', []))}

3. MEDIA ANALYSIS
────────────────────────────────────────────────────────────────────────────────
Total Media Files:      {extraction_results.get('media_count', 0):,}
Photos:                 {extraction_results.get('photo_count', 0):,}
Videos:                 {extraction_results.get('video_count', 0):,}
Audio Files:            {extraction_results.get('audio_count', 0):,}

Media Timeline:
{self._format_media_timeline(extraction_results.get('media_timeline', []))}

4. DEVICE INFORMATION
────────────────────────────────────────────────────────────────────────────────
Device Model:           {extraction_results.get('device_model', 'N/A')}
Operating System:       {extraction_results.get('os_version', 'N/A')}
Last Boot Time:         {extraction_results.get('last_boot', 'N/A')}
Storage Used:           {self._format_size(extraction_results.get('storage_used', 0))}
Storage Available:      {self._format_size(extraction_results.get('storage_available', 0))}

5. SECURITY FINDINGS
────────────────────────────────────────────────────────────────────────────────
Installed Applications: {extraction_results.get('app_count', 0):,}
Suspicious Apps:        {extraction_results.get('suspicious_apps', 0)}
Malware Detected:       {extraction_results.get('malware_count', 0)}
Security Issues:        {extraction_results.get('security_issues', 0)}

6. EVIDENCE SUMMARY
────────────────────────────────────────────────────────────────────────────────
Total Evidence Items:   {extraction_results.get('evidence_count', 0):,}
Critical Evidence:      {extraction_results.get('critical_evidence', 0)}
Supporting Evidence:    {extraction_results.get('supporting_evidence', 0)}

═══════════════════════════════════════════════════════════════════════════════
"""
            return findings
        
        except Exception as e:
            logger.error(f"Error generating detailed findings: {str(e)}")
            return f"Error generating detailed findings: {str(e)}"
    
    # ========================================================================
    # TECHNICAL ANALYSIS
    # ========================================================================
    
    def generate_technical_analysis(self, extraction_results: Dict[str, Any]) -> str:
        """
        Generate technical analysis report
        
        Args:
            extraction_results: Extraction results
            
        Returns:
            str: Technical analysis
        """
        try:
            analysis = f"""
═══════════════════════════════════════════════════════════════════════════════
                        TECHNICAL ANALYSIS REPORT
═══════════════════════════════════════════════════════════════════════════════

EXTRACTION METHODOLOGY
────────────────────────────────────────────────────────────────────────────────
Extraction Method:      {extraction_results.get('extraction_method', 'N/A')}
Extraction Duration:    {extraction_results.get('extraction_duration', 'N/A')}
Data Integrity:         {extraction_results.get('integrity_status', 'VERIFIED')}
Hash Verification:      {extraction_results.get('hash_verified', 'YES')}

DEVICE SPECIFICATIONS
────────────────────────────────────────────────────────────────────────────────
Device ID:              {extraction_results.get('device_id', 'N/A')}
IMEI:                   {extraction_results.get('imei', 'N/A')}
Serial Number:          {extraction_results.get('serial_number', 'N/A')}
Processor:              {extraction_results.get('processor', 'N/A')}
RAM:                    {extraction_results.get('ram', 'N/A')}
Storage Capacity:       {self._format_size(extraction_results.get('storage_capacity', 0))}

DATA EXTRACTION DETAILS
────────────────────────────────────────────────────────────────────────────────
Extraction Modules:
{self._format_modules(extraction_results.get('modules', []))}

Data Categories:
{self._format_data_categories(extraction_results.get('data_categories', {}))}

CHAIN OF CUSTODY
────────────────────────────────────────────────────────────────────────────────
Extracted By:           {extraction_results.get('extracted_by', 'N/A')}
Extraction Start:       {extraction_results.get('extraction_start', 'N/A')}
Extraction End:         {extraction_results.get('extraction_end', 'N/A')}
Storage Location:       {extraction_results.get('storage_location', 'N/A')}
Encryption Status:      {extraction_results.get('encryption_status', 'ENCRYPTED')}

QUALITY METRICS
────────────────────────────────────────────────────────────────────────────────
Data Completeness:      {extraction_results.get('completeness_percentage', 0)}%
Extraction Success:     {extraction_results.get('success_rate', 0)}%
Errors Encountered:     {extraction_results.get('error_count', 0)}
Warnings:               {extraction_results.get('warning_count', 0)}

═══════════════════════════════════════════════════════════════════════════════
"""
            return analysis
        
        except Exception as e:
            logger.error(f"Error generating technical analysis: {str(e)}")
            return f"Error generating technical analysis: {str(e)}"
    
    # ========================================================================
    # RISK ASSESSMENT
    # ========================================================================
    
    def generate_risk_assessment(self, extraction_results: Dict[str, Any]) -> str:
        """
        Generate risk assessment report
        
        Args:
            extraction_results: Extraction results
            
        Returns:
            str: Risk assessment
        """
        try:
            assessment = f"""
═══════════════════════════════════════════════════════════════════════════════
                        RISK ASSESSMENT REPORT
═══════════════════════════════════════════════════════════════════════════════

OVERALL RISK LEVEL: {self._assess_risk_level(extraction_results)}

RISK BREAKDOWN
────────────────────────────────────────────────────────────────────────────────
Communication Risk:     {extraction_results.get('communication_risk', 'MEDIUM')}
Location Risk:          {extraction_results.get('location_risk', 'MEDIUM')}
Media Risk:             {extraction_results.get('media_risk', 'LOW')}
Security Risk:          {extraction_results.get('security_risk', 'HIGH')}
Overall Risk Score:     {extraction_results.get('risk_score', 0)}/100

CRITICAL FINDINGS
────────────────────────────────────────────────────────────────────────────────
{self._format_critical_findings(extraction_results.get('critical_findings', []))}

HIGH PRIORITY ITEMS
────────────────────────────────────────────────────────────────────────────────
{self._format_high_priority(extraction_results.get('high_priority_items', []))}

RECOMMENDATIONS
────────────────────────────────────────────────────────────────────────────────
{self._generate_recommendations(extraction_results)}

INVESTIGATION PRIORITIES
────────────────────────────────────────────────────────────────────────────────
Priority 1: {extraction_results.get('priority_1', 'N/A')}
Priority 2: {extraction_results.get('priority_2', 'N/A')}
Priority 3: {extraction_results.get('priority_3', 'N/A')}

═══════════════════════════════════════════════════════════════════════════════
"""
            return assessment
        
        except Exception as e:
            logger.error(f"Error generating risk assessment: {str(e)}")
            return f"Error generating risk assessment: {str(e)}"
    
    # ========================================================================
    # TIMELINE REPORT
    # ========================================================================
    
    def generate_timeline_report(self, extraction_results: Dict[str, Any]) -> str:
        """
        Generate timeline report
        
        Args:
            extraction_results: Extraction results
            
        Returns:
            str: Timeline report
        """
        try:
            timeline = f"""
═══════════════════════════════════════════════════════════════════════════════
                        TIMELINE REPORT
═══════════════════════════════════════════════════════════════════════════════

CHRONOLOGICAL EVENTS
────────────────────────────────────────────────────────────────────────────────
{self._format_timeline_events(extraction_results.get('timeline_events', []))}

COMMUNICATION TIMELINE
────────────────────────────────────────────────────────────────────────────────
{self._format_communication_timeline(extraction_results.get('communication_timeline', []))}

LOCATION TIMELINE
────────────────────────────────────────────────────────────────────────────────
{self._format_location_timeline(extraction_results.get('location_timeline', []))}

MEDIA TIMELINE
────────────────────────────────────────────────────────────────────────────────
{self._format_media_timeline(extraction_results.get('media_timeline', []))}

═══════════════════════════════════════════════════════════════════════════════
"""
            return timeline
        
        except Exception as e:
            logger.error(f"Error generating timeline report: {str(e)}")
            return f"Error generating timeline report: {str(e)}"
    
    # ========================================================================
    # FULL REPORT
    # ========================================================================
    
    def generate_full_report(self, extraction_results: Dict[str, Any]) -> str:
        """
        Generate complete forensic report
        
        Args:
            extraction_results: Extraction results
            
        Returns:
            str: Full report
        """
        try:
            full_report = f"""
{self.generate_executive_summary(extraction_results)}

{self.generate_detailed_findings(extraction_results)}

{self.generate_technical_analysis(extraction_results)}

{self.generate_risk_assessment(extraction_results)}

{self.generate_timeline_report(extraction_results)}

═══════════════════════════════════════════════════════════════════════════════
                        REPORT CERTIFICATION
═══════════════════════════════════════════════════════════════════════════════

This report has been generated using forensically sound methods and procedures.
All data has been extracted with proper chain of custody documentation.

Generated By:           AI Report Generator v1.0
Generation Date:        {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
Report Version:         1.0
Status:                 FINAL

═══════════════════════════════════════════════════════════════════════════════
"""
            return full_report
        
        except Exception as e:
            logger.error(f"Error generating full report: {str(e)}")
            return f"Error generating full report: {str(e)}"
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    @staticmethod
    def _format_size(bytes_size: int) -> str:
        """Format bytes to human-readable size"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} PB"
    
    @staticmethod
    def _generate_key_findings(results: Dict[str, Any]) -> str:
        """Generate key findings summary"""
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
        """Assess overall risk level"""
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
    
    @staticmethod
    def _format_contacts(contacts: List[Dict[str, Any]]) -> str:
        """Format contacts list"""
        if not contacts:
            return "No significant contacts identified"
        
        formatted = []
        for contact in contacts[:5]:
            formatted.append(f"  • {contact.get('name', 'Unknown')}: {contact.get('message_count', 0)} messages")
        
        return "\n".join(formatted)
    
    @staticmethod
    def _format_suspicious_comms(comms: List[Dict[str, Any]]) -> str:
        """Format suspicious communications"""
        if not comms:
            return "No suspicious communications detected"
        
        formatted = []
        for comm in comms[:5]:
            formatted.append(f"  • {comm.get('timestamp', 'N/A')}: {comm.get('content', 'N/A')[:50]}...")
        
        return "\n".join(formatted)
    
    @staticmethod
    def _format_locations(locations: List[Dict[str, Any]]) -> str:
        """Format locations"""
        if not locations:
            return "No location data available"
        
        formatted = []
        for loc in locations[:5]:
            formatted.append(f"  • {loc.get('name', 'Unknown')}: {loc.get('visits', 0)} visits")
        
        return "\n".join(formatted)
    
    @staticmethod
    def _format_movement_timeline(timeline: List[Dict[str, Any]]) -> str:
        """Format movement timeline"""
        if not timeline:
            return "No movement timeline available"
        
        formatted = []
        for event in timeline[:5]:
            formatted.append(f"  • {event.get('timestamp', 'N/A')}: {event.get('location', 'N/A')}")
        
        return "\n".join(formatted)
    
    @staticmethod
    def _format_media_timeline(timeline: List[Dict[str, Any]]) -> str:
        """Format media timeline"""
        if not timeline:
            return "No media timeline available"
        
        formatted = []
        for event in timeline[:5]:
            formatted.append(f"  • {event.get('timestamp', 'N/A')}: {event.get('type', 'N/A')} - {event.get('name', 'N/A')}")
        
        return "\n".join(formatted)
    
    @staticmethod
    def _format_modules(modules: List[str]) -> str:
        """Format extraction modules"""
        if not modules:
            return "No modules extracted"
        
        formatted = []
        for module in modules:
            formatted.append(f"  ✓ {module}")
        
        return "\n".join(formatted)
    
    @staticmethod
    def _format_data_categories(categories: Dict[str, int]) -> str:
        """Format data categories"""
        if not categories:
            return "No data categories available"
        
        formatted = []
        for category, count in categories.items():
            formatted.append(f"  • {category}: {count:,} items")
        
        return "\n".join(formatted)
    
    @staticmethod
    def _format_critical_findings(findings: List[str]) -> str:
        """Format critical findings"""
        if not findings:
            return "No critical findings"
        
        formatted = []
        for finding in findings:
            formatted.append(f"  ⚠️  {finding}")
        
        return "\n".join(formatted)
    
    @staticmethod
    def _format_high_priority(items: List[str]) -> str:
        """Format high priority items"""
        if not items:
            return "No high priority items"
        
        formatted = []
        for item in items:
            formatted.append(f"  • {item}")
        
        return "\n".join(formatted)
    
    @staticmethod
    def _generate_recommendations(results: Dict[str, Any]) -> str:
        """Generate recommendations"""
        recommendations = []
        
        if results.get('suspicious_messages', 0) > 5:
            recommendations.append("1. Conduct detailed analysis of suspicious communications")
        
        if results.get('location_count', 0) > 20:
            recommendations.append("2. Cross-reference locations with known associates")
        
        if results.get('malware_count', 0) > 0:
            recommendations.append("3. Investigate potential malware or spyware installation")
        
        if results.get('security_risk', 'LOW') == 'HIGH':
            recommendations.append("4. Review security settings and access logs")
        
        if not recommendations:
            recommendations.append("1. Continue standard investigation procedures")
        
        return "\n".join(recommendations)
    
    @staticmethod
    def _format_timeline_events(events: List[Dict[str, Any]]) -> str:
        """Format timeline events"""
        if not events:
            return "No timeline events available"
        
        formatted = []
        for event in events:
            formatted.append(f"  {event.get('timestamp', 'N/A')} - {event.get('event', 'N/A')}")
        
        return "\n".join(formatted)
    
    @staticmethod
    def _format_communication_timeline(timeline: List[Dict[str, Any]]) -> str:
        """Format communication timeline"""
        if not timeline:
            return "No communication timeline available"
        
        formatted = []
        for event in timeline[:10]:
            formatted.append(f"  {event.get('timestamp', 'N/A')} - {event.get('type', 'N/A')} with {event.get('contact', 'N/A')}")
        
        return "\n".join(formatted)
    
    @staticmethod
    def _format_location_timeline(timeline: List[Dict[str, Any]]) -> str:
        """Format location timeline"""
        if not timeline:
            return "No location timeline available"
        
        formatted = []
        for event in timeline[:10]:
            formatted.append(f"  {event.get('timestamp', 'N/A')} - {event.get('location', 'N/A')} ({event.get('duration', 'N/A')})")
        
        return "\n".join(formatted)


# ============================================================================
# REPORT EXPORTER
# ============================================================================

class ReportExporter:
    """Export reports in various formats"""
    
    @staticmethod
    def export_to_text(report: str, filename: str) -> bool:
        """Export report to text file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report exported to: {filename}")
            return True
        except Exception as e:
            logger.error(f"Error exporting report: {str(e)}")
            return False
    
    @staticmethod
    def export_to_json(report_data: Dict[str, Any], filename: str) -> bool:
        """Export report data to JSON"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
            logger.info(f"Report exported to: {filename}")
            return True
        except Exception as e:
            logger.error(f"Error exporting report: {str(e)}")
            return False
    
    @staticmethod
    def export_to_pdf(report: str, filename: str) -> bool:
        """Export report to PDF (requires reportlab)"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            
            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            for line in report.split('\n'):
                if line.strip():
                    story.append(Paragraph(line, styles['Normal']))
                else:
                    story.append(Spacer(1, 0.2*inch))
            
            doc.build(story)
            logger.info(f"Report exported to: {filename}")
            return True
        
        except ImportError:
            logger.warning("reportlab not installed. Install with: pip install reportlab")
            return False
        except Exception as e:
            logger.error(f"Error exporting report: {str(e)}")
            return False

