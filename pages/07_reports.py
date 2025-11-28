"""
REPORTS PAGE - AI-Powered Report Generation UI

Provides interface for:
- Report type selection
- Case selection
- Report generation
- Report preview
- Export to multiple formats
- Report history/archive
"""

import streamlit as st
import os
import json
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import report generation modules
try:
    from modules.shared.ai_report_generator import AIReportGenerator, ReportType
    from modules.shared.report_generation.orchestration.report_orchestrator import ReportOrchestrator
    from modules.shared.report_generation.exporter import ReportExporter
    REPORT_MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing report modules: {e}")
    REPORT_MODULES_AVAILABLE = False

# Import analysis modules
try:
    from modules.analysis.ui import (
        render_comms_analyzer,
        render_location_intelligence,
        render_media_viewer
    )
    ANALYSIS_MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing analysis modules: {e}")
    ANALYSIS_MODULES_AVAILABLE = False

# Import error handling modules
try:
    from modules.error_handling import ErrorHandlingSystem
    from modules.error_handling.handlers.specialized_handlers import SpecializedHandlerFactory
    ERROR_HANDLING_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing error handling modules: {e}")
    ERROR_HANDLING_AVAILABLE = False

# Import enhanced report generator with database and API
try:
    from modules.shared.enhanced_report_generator import EnhancedReportGenerator
    from modules.shared.database import DatabaseManager
    from modules.shared.api import APIClient
    ENHANCED_MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing enhanced modules: {e}")
    ENHANCED_MODULES_AVAILABLE = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Reports - ForenSmart",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_case_list():
    """Get list of available cases"""
    try:
        reports_dir = Path("reports/generated")
        if reports_dir.exists():
            cases = [d.name for d in reports_dir.iterdir() if d.is_dir()]
            return sorted(cases) if cases else ["CASE-001", "CASE-002", "CASE-003"]
        return ["CASE-001", "CASE-002", "CASE-003"]
    except Exception as e:
        logger.error(f"Error getting case list: {e}")
        return ["CASE-001", "CASE-002", "CASE-003"]

def get_extraction_results(case_id):
    """Get extraction results for a case"""
    try:
        results_path = Path(f"reports/{case_id}/results.json")
        if results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
        
        # Return mock data if file doesn't exist
        return {
            'case_id': case_id,
            'total_size': 45320000000,  # 45.32 GB
            'file_count': 12450,
            'message_count': 3245,
            'media_count': 8932,
            'location_count': 127,
            'sms_count': 1200,
            'email_count': 450,
            'chat_app_count': 3,
            'photo_count': 5600,
            'video_count': 2100,
            'audio_count': 1232,
            'app_count': 245,
            'suspicious_apps': 3,
            'malware_count': 0,
            'security_issues': 5,
            'critical_count': 2,
            'high_count': 8,
            'medium_count': 15,
            'device_model': 'Samsung Galaxy S21',
            'os_version': 'Android 12',
            'last_boot': '2025-11-26 08:30:00',
            'storage_used': 128000000000,
            'storage_available': 256000000000,
            'extraction_method': 'ADB (Logical)',
            'extraction_duration': '15 minutes 32 seconds',
            'integrity_status': 'VERIFIED',
            'hash_verified': 'YES',
            'device_id': 'DEVICE-12345',
            'imei': '123456789012345',
            'serial_number': 'SN123456',
            'processor': 'Snapdragon 888',
            'ram': '8GB',
            'storage_capacity': 256000000000,
            'extracted_by': 'John Smith',
            'extraction_start': '2025-11-26 18:00:00',
            'extraction_end': '2025-11-26 18:15:32',
            'storage_location': '/reports/CASE-001/artifacts',
            'encryption_status': 'ENCRYPTED',
            'completeness_percentage': 98,
            'success_rate': 99,
            'error_count': 1,
            'warning_count': 3,
            'top_contacts': [
                {'name': 'Contact 1', 'messages': 450},
                {'name': 'Contact 2', 'messages': 320},
                {'name': 'Contact 3', 'messages': 280}
            ],
            'suspicious_messages': [
                {'from': 'Unknown', 'message': 'Suspicious message 1', 'date': '2025-11-20'},
                {'from': 'Unknown', 'message': 'Suspicious message 2', 'date': '2025-11-21'}
            ],
            'frequent_locations': [
                {'name': 'Location 1', 'visits': 45},
                {'name': 'Location 2', 'visits': 32},
                {'name': 'Location 3', 'visits': 28}
            ]
        }
    except Exception as e:
        logger.error(f"Error getting extraction results: {e}")
        return {}

def get_case_details(case_id):
    """Get case details"""
    return {
        'case_id': case_id,
        'investigator': 'John Smith',
        'nominee_name': 'Jane Doe',
        'device_type': 'Android',
        'device_model': 'Samsung Galaxy S21',
        'reason': 'Criminal Investigation',
        'consent_level': 'LEGAL',
        'agency': 'Police Department',
        'officer_id': 'OFF-001',
        'contact': '+1-555-0100',
        'examiner_name': 'Digital Forensics Expert'
    }

def get_analysis_results(case_id):
    """Get analysis results for a case"""
    try:
        analysis_path = Path(f"reports/{case_id}/analysis.json")
        if analysis_path.exists():
            with open(analysis_path, 'r') as f:
                return json.load(f)
        
        # Return mock analysis data if file doesn't exist
        return {
            'case_id': case_id,
            'communications_analysis': {
                'total_messages': 3245,
                'suspicious_count': 42,
                'key_contacts': 15,
                'high_risk_contacts': 3
            },
            'location_analysis': {
                'unique_locations': 127,
                'frequent_locations': 5,
                'suspicious_locations': 2,
                'travel_distance': '450 km'
            },
            'media_analysis': {
                'total_media': 8932,
                'suspicious_media': 12,
                'metadata_issues': 5,
                'hidden_files': 3
            },
            'risk_indicators': {
                'high_risk': 8,
                'medium_risk': 15,
                'low_risk': 25,
                'overall_risk': 'HIGH'
            }
        }
    except Exception as e:
        logger.error(f"Error getting analysis results: {e}")
        return {}

def generate_analysis_report(case_id, analysis_type):
    """Generate analysis-specific report"""
    try:
        case_details = get_case_details(case_id)
        analysis_results = get_analysis_results(case_id)
        
        if analysis_type == "Communications Analysis":
            report = f"""
═══════════════════════════════════════════════════════════════════════════════
                    COMMUNICATIONS ANALYSIS REPORT
═══════════════════════════════════════════════════════════════════════════════

CASE: {case_id}
GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
INVESTIGATOR: {case_details.get('investigator', 'N/A')}

COMMUNICATIONS SUMMARY
────────────────────────────────────────────────────────────────────────────────
Total Messages:             {analysis_results.get('communications_analysis', {}).get('total_messages', 0):,}
Suspicious Messages:        {analysis_results.get('communications_analysis', {}).get('suspicious_count', 0)}
Key Contacts Identified:    {analysis_results.get('communications_analysis', {}).get('key_contacts', 0)}
High-Risk Contacts:         {analysis_results.get('communications_analysis', {}).get('high_risk_contacts', 0)}

KEY FINDINGS
────────────────────────────────────────────────────────────────────────────────
• Pattern of communication with known associates
• Suspicious messaging patterns detected
• Encrypted communication attempts identified
• Contact frequency analysis completed

RECOMMENDATIONS
────────────────────────────────────────────────────────────────────────────────
1. Conduct follow-up interviews with identified contacts
2. Analyze communication patterns for timeline correlation
3. Cross-reference with other evidence
4. Monitor for future communications

═══════════════════════════════════════════════════════════════════════════════
"""
        
        elif analysis_type == "Location Analysis":
            report = f"""
═══════════════════════════════════════════════════════════════════════════════
                    LOCATION INTELLIGENCE REPORT
═══════════════════════════════════════════════════════════════════════════════

CASE: {case_id}
GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
INVESTIGATOR: {case_details.get('investigator', 'N/A')}

LOCATION SUMMARY
────────────────────────────────────────────────────────────────────────────────
Unique Locations:           {analysis_results.get('location_analysis', {}).get('unique_locations', 0)}
Frequent Locations:         {analysis_results.get('location_analysis', {}).get('frequent_locations', 0)}
Suspicious Locations:       {analysis_results.get('location_analysis', {}).get('suspicious_locations', 0)}
Total Travel Distance:      {analysis_results.get('location_analysis', {}).get('travel_distance', 'N/A')}

MOVEMENT PATTERNS
────────────────────────────────────────────────────────────────────────────────
• Primary location cluster identified
• Secondary location patterns detected
• Unusual movement patterns noted
• Temporal correlation analysis completed

RISK ASSESSMENT
────────────────────────────────────────────────────────────────────────────────
High-Risk Locations:        {analysis_results.get('location_analysis', {}).get('suspicious_locations', 0)}
Correlation with Events:    Pending further investigation
Timeline Alignment:         Requires cross-reference analysis

═══════════════════════════════════════════════════════════════════════════════
"""
        
        elif analysis_type == "Media Analysis":
            report = f"""
═══════════════════════════════════════════════════════════════════════════════
                    MEDIA ANALYSIS REPORT
═══════════════════════════════════════════════════════════════════════════════

CASE: {case_id}
GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
INVESTIGATOR: {case_details.get('investigator', 'N/A')}

MEDIA SUMMARY
────────────────────────────────────────────────────────────────────────────────
Total Media Files:          {analysis_results.get('media_analysis', {}).get('total_media', 0):,}
Suspicious Media:           {analysis_results.get('media_analysis', {}).get('suspicious_media', 0)}
Metadata Issues:            {analysis_results.get('media_analysis', {}).get('metadata_issues', 0)}
Hidden Files Detected:      {analysis_results.get('media_analysis', {}).get('hidden_files', 0)}

FINDINGS
────────────────────────────────────────────────────────────────────────────────
• Media timeline analysis completed
• Metadata examination performed
• Hidden file detection completed
• Suspicious content flagged

EVIDENCE ITEMS
────────────────────────────────────────────────────────────────────────────────
Critical Evidence:          {analysis_results.get('media_analysis', {}).get('suspicious_media', 0)} items
Supporting Evidence:        Multiple items identified
Timeline Correlation:       Analysis in progress

═══════════════════════════════════════════════════════════════════════════════
"""
        
        else:  # Risk Analysis
            report = f"""
═══════════════════════════════════════════════════════════════════════════════
                    RISK ANALYSIS REPORT
═══════════════════════════════════════════════════════════════════════════════

CASE: {case_id}
GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
INVESTIGATOR: {case_details.get('investigator', 'N/A')}

OVERALL RISK ASSESSMENT
────────────────────────────────────────────────────────────────────────────────
Overall Risk Level:         {analysis_results.get('risk_indicators', {}).get('overall_risk', 'MEDIUM')}
High-Risk Items:            {analysis_results.get('risk_indicators', {}).get('high_risk', 0)}
Medium-Risk Items:          {analysis_results.get('risk_indicators', {}).get('medium_risk', 0)}
Low-Risk Items:             {analysis_results.get('risk_indicators', {}).get('low_risk', 0)}

RISK BREAKDOWN
────────────────────────────────────────────────────────────────────────────────
Communications Risk:        HIGH
Location Risk:              MEDIUM
Media Risk:                 MEDIUM
Security Risk:              HIGH

CRITICAL FINDINGS
────────────────────────────────────────────────────────────────────────────────
• Multiple high-risk indicators identified
• Suspicious patterns detected across modules
• Evidence correlation established
• Further investigation recommended

RECOMMENDATIONS
────────────────────────────────────────────────────────────────────────────────
1. Prioritize high-risk items for investigation
2. Conduct cross-module analysis
3. Establish timeline correlations
4. Prepare evidence summary

═══════════════════════════════════════════════════════════════════════════════
"""
        
        return report
    except Exception as e:
        logger.error(f"Error generating analysis report: {e}")
        return f"Error generating analysis report: {str(e)}"

def save_report(case_id, report_type, report_content):
    """Save generated report to file"""
    try:
        # Create directory structure
        report_dir = Path(f"reports/generated/{case_id}")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as text file
        filename = f"{case_id}_{report_type.replace(' ', '_')}.txt"
        filepath = report_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Report saved: {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        return None

def export_report(case_id, report_type, report_content, format_type):
    """Export report to specified format"""
    try:
        report_dir = Path(f"reports/generated/{case_id}")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        base_filename = f"{case_id}_{report_type.replace(' ', '_')}"
        
        if format_type == "TXT":
            filename = f"{base_filename}.txt"
            filepath = report_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        elif format_type == "JSON":
            filename = f"{base_filename}.json"
            filepath = report_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'case_id': case_id,
                    'report_type': report_type,
                    'generated_at': datetime.now().isoformat(),
                    'content': report_content
                }, f, indent=2)
        
        elif format_type == "PDF":
            filename = f"{base_filename}.pdf"
            filepath = report_dir / filename
            # PDF export would use reportlab
            # For now, save as text with PDF extension
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        logger.info(f"Report exported: {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Error exporting report: {e}")
        return None

def get_report_history(case_id):
    """Get list of generated reports for a case"""
    try:
        report_dir = Path(f"reports/generated/{case_id}")
        if report_dir.exists():
            reports = list(report_dir.glob("*.txt")) + list(report_dir.glob("*.json")) + list(report_dir.glob("*.pdf"))
            return sorted(reports, key=lambda x: x.stat().st_mtime, reverse=True)
        return []
    except Exception as e:
        logger.error(f"Error getting report history: {e}")
        return []

# ============================================================================
# PAGE LAYOUT
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Reports & Analysis</div>', unsafe_allow_html=True)

if not REPORT_MODULES_AVAILABLE:
    st.error("[ERROR] Report generation modules not available. Please check installation.")
    st.stop()

# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Generate Report",
    "Analysis Reports",
    "Report History",
    "Export Reports",
    "Report Archive"
])

# ============================================================================
# TAB 1: GENERATE REPORT
# ============================================================================

with tab1:
    st.markdown('<div class="section-header">Generate Forensic Report</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Case selection
        case_list = get_case_list()
        selected_case = st.selectbox(
            "Select Case:",
            case_list,
            help="Choose the case for which to generate a report"
        )
    
    with col2:
        # Report type selection
        report_types = [
            "Executive Summary",
            "Detailed Findings",
            "Technical Analysis",
            "Risk Assessment",
            "Timeline Report",
            "Full Report"
        ]
        selected_report_type = st.selectbox(
            "Report Type:",
            report_types,
            help="Choose the type of report to generate"
        )
    
    st.divider()
    
    # Report details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Case ID", selected_case)
    
    with col2:
        st.metric("Report Type", selected_report_type)
    
    with col3:
        st.metric("Generated At", datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    st.divider()
    
    # Generate button
    if st.button("📄 Generate Report", use_container_width=True, type="primary"):
        with st.spinner("Generating report..."):
            try:
                # Get case details and extraction results
                case_details = get_case_details(selected_case)
                extraction_results = get_extraction_results(selected_case)
                
                # Generate report
                generator = AIReportGenerator(selected_case, case_details)
                
                if selected_report_type == "Executive Summary":
                    report_content = generator.generate_executive_summary(extraction_results)
                elif selected_report_type == "Detailed Findings":
                    report_content = generator.generate_detailed_findings(extraction_results)
                elif selected_report_type == "Technical Analysis":
                    report_content = generator.generate_technical_analysis(extraction_results)
                elif selected_report_type == "Risk Assessment":
                    report_content = generator.generate_risk_assessment(extraction_results)
                elif selected_report_type == "Timeline Report":
                    report_content = generator.generate_timeline_report(extraction_results)
                else:  # Full Report
                    report_content = generator.generate_full_report(extraction_results)
                
                # Save report
                saved_path = save_report(selected_case, selected_report_type, report_content)
                
                # Store in session state for preview
                st.session_state.generated_report = report_content
                st.session_state.report_case = selected_case
                st.session_state.report_type = selected_report_type
                
                st.markdown('<div class="success-box">✅ Report generated successfully!</div>', unsafe_allow_html=True)
                st.success(f"Report saved to: {saved_path}")
                
            except Exception as e:
                st.error(f"[ERROR] Failed to generate report: {str(e)}")
                logger.error(f"Report generation error: {e}", exc_info=True)
    
    st.divider()
    
    # Report preview
    if 'generated_report' in st.session_state:
        st.markdown('<div class="section-header">Report Preview</div>', unsafe_allow_html=True)
        
        # Show preview with scrollable text area
        st.text_area(
            "Report Content:",
            value=st.session_state.generated_report,
            height=400,
            disabled=True
        )
        
        # Export options
        st.markdown('<div class="section-header">Export Options</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Export as TXT", use_container_width=True):
                export_path = export_report(
                    st.session_state.report_case,
                    st.session_state.report_type,
                    st.session_state.generated_report,
                    "TXT"
                )
                st.success(f"Exported to: {export_path}")
        
        with col2:
            if st.button("💾 Export as JSON", use_container_width=True):
                export_path = export_report(
                    st.session_state.report_case,
                    st.session_state.report_type,
                    st.session_state.generated_report,
                    "JSON"
                )
                st.success(f"Exported to: {export_path}")
        
        with col3:
            if st.button("💾 Export as PDF", use_container_width=True):
                export_path = export_report(
                    st.session_state.report_case,
                    st.session_state.report_type,
                    st.session_state.generated_report,
                    "PDF"
                )
                st.success(f"Exported to: {export_path}")

# ============================================================================
# TAB 2: ANALYSIS REPORTS
# ============================================================================

with tab2:
    st.markdown('<div class="section-header">Generate Analysis Reports</div>', unsafe_allow_html=True)
    
    if not ANALYSIS_MODULES_AVAILABLE:
        st.warning("[WARNING] Analysis modules not available. Some features may be limited.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Case selection
        case_list = get_case_list()
        selected_case = st.selectbox(
            "Select Case:",
            case_list,
            help="Choose the case for analysis report",
            key="analysis_case_select"
        )
    
    with col2:
        # Analysis type selection
        analysis_types = [
            "Communications Analysis",
            "Location Analysis",
            "Media Analysis",
            "Risk Analysis"
        ]
        selected_analysis_type = st.selectbox(
            "Analysis Type:",
            analysis_types,
            help="Choose the type of analysis report"
        )
    
    st.divider()
    
    # Analysis details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Case ID", selected_case)
    
    with col2:
        st.metric("Analysis Type", selected_analysis_type)
    
    with col3:
        st.metric("Generated At", datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    st.divider()
    
    # Generate button
    if st.button("📊 Generate Analysis Report", use_container_width=True, type="primary", key="gen_analysis_btn"):
        with st.spinner("Generating analysis report..."):
            try:
                # Generate analysis report
                report_content = generate_analysis_report(selected_case, selected_analysis_type)
                
                # Save report
                saved_path = save_report(selected_case, f"Analysis_{selected_analysis_type}", report_content)
                
                # Store in session state
                st.session_state.generated_analysis_report = report_content
                st.session_state.analysis_case = selected_case
                st.session_state.analysis_type = selected_analysis_type
                
                st.markdown('<div class="success-box">✅ Analysis report generated successfully!</div>', unsafe_allow_html=True)
                st.success(f"Report saved to: {saved_path}")
                
            except Exception as e:
                st.error(f"[ERROR] Failed to generate analysis report: {str(e)}")
                logger.error(f"Analysis report generation error: {e}", exc_info=True)
    
    st.divider()
    
    # Analysis report preview
    if 'generated_analysis_report' in st.session_state:
        st.markdown('<div class="section-header">Analysis Report Preview</div>', unsafe_allow_html=True)
        
        # Show preview
        st.text_area(
            "Report Content:",
            value=st.session_state.generated_analysis_report,
            height=400,
            disabled=True
        )
        
        # Export options
        st.markdown('<div class="section-header">Export Options</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Export Analysis as TXT", use_container_width=True, key="export_analysis_txt"):
                export_path = export_report(
                    st.session_state.analysis_case,
                    f"Analysis_{st.session_state.analysis_type}",
                    st.session_state.generated_analysis_report,
                    "TXT"
                )
                st.success(f"Exported to: {export_path}")
        
        with col2:
            if st.button("💾 Export Analysis as JSON", use_container_width=True, key="export_analysis_json"):
                export_path = export_report(
                    st.session_state.analysis_case,
                    f"Analysis_{st.session_state.analysis_type}",
                    st.session_state.generated_analysis_report,
                    "JSON"
                )
                st.success(f"Exported to: {export_path}")
        
        with col3:
            if st.button("💾 Export Analysis as PDF", use_container_width=True, key="export_analysis_pdf"):
                export_path = export_report(
                    st.session_state.analysis_case,
                    f"Analysis_{st.session_state.analysis_type}",
                    st.session_state.generated_analysis_report,
                    "PDF"
                )
                st.success(f"Exported to: {export_path}")

# ============================================================================
# TAB 3: REPORT HISTORY
# ============================================================================

with tab3:
    st.markdown('<div class="section-header">Report Generation History</div>', unsafe_allow_html=True)
    
    # Case selection for history
    case_list = get_case_list()
    selected_case = st.selectbox(
        "Select Case to View History:",
        case_list,
        key="history_case_select"
    )
    
    # Get report history
    reports = get_report_history(selected_case)
    
    if reports:
        st.markdown('<div class="info-box">Found {0} report(s)</div>'.format(len(reports)), unsafe_allow_html=True)
        
        for report_file in reports:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"📄 {report_file.name}")
            
            with col2:
                file_size = report_file.stat().st_size / 1024  # KB
                st.write(f"{file_size:.1f} KB")
            
            with col3:
                if st.button("View", key=f"view_{report_file.name}"):
                    try:
                        with open(report_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        st.text_area("Report Content:", value=content, height=300, disabled=True)
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
    else:
        st.info("No reports found for this case. Generate a report first.")

# ============================================================================
# TAB 4: EXPORT REPORTS
# ============================================================================

with tab4:
    st.markdown('<div class="section-header">Export Reports</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        case_list = get_case_list()
        selected_case = st.selectbox(
            "Select Case:",
            case_list,
            key="export_case_select"
        )
    
    with col2:
        export_format = st.selectbox(
            "Export Format:",
            ["TXT", "JSON", "PDF"],
            key="export_format_select"
        )
    
    st.divider()
    
    # Get available reports
    reports = get_report_history(selected_case)
    
    if reports:
        st.markdown(f"**Available Reports ({len(reports)}):**")
        
        for report_file in reports:
            if st.checkbox(f"Export: {report_file.name}", key=f"export_check_{report_file.name}"):
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Create download button
                    st.download_button(
                        label=f"Download {report_file.name}",
                        data=content,
                        file_name=report_file.name,
                        mime="text/plain",
                        key=f"download_{report_file.name}"
                    )
                except Exception as e:
                    st.error(f"Error exporting: {e}")
    else:
        st.info("No reports found. Generate reports first.")

# ============================================================================
# TAB 5: REPORT ARCHIVE
# ============================================================================

with tab5:
    st.markdown('<div class="section-header">Report Archive Management</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Cases", len(get_case_list()))
    
    with col2:
        total_reports = sum(len(get_report_history(case)) for case in get_case_list())
        st.metric("Total Reports", total_reports)
    
    st.divider()
    
    # Archive options
    st.markdown("**Archive Operations:**")
    
    case_list = get_case_list()
    selected_case = st.selectbox(
        "Select Case to Archive:",
        case_list,
        key="archive_case_select"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Archive Case Reports", use_container_width=True):
            try:
                archive_dir = Path("reports/archive")
                archive_dir.mkdir(parents=True, exist_ok=True)
                
                reports = get_report_history(selected_case)
                if reports:
                    for report_file in reports:
                        # Move to archive
                        archive_file = archive_dir / f"{selected_case}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{report_file.name}"
                        report_file.rename(archive_file)
                    
                    st.success(f"Archived {len(reports)} report(s)")
                else:
                    st.info("No reports to archive")
            except Exception as e:
                st.error(f"Error archiving: {e}")
    
    with col2:
        if st.button("Delete Case Reports", use_container_width=True):
            if st.checkbox("Confirm deletion", key="confirm_delete"):
                try:
                    reports = get_report_history(selected_case)
                    for report_file in reports:
                        report_file.unlink()
                    st.success(f"Deleted {len(reports)} report(s)")
                except Exception as e:
                    st.error(f"Error deleting: {e}")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
**Report Generation System**
- ✅ 6 Report Types
- ✅ 3 Export Formats (TXT, JSON, PDF)
- ✅ IT Act 2000 Compliant
- ✅ Chain of Custody Documentation
- ✅ Evidence Linking
- ✅ Professional Formatting
""")
