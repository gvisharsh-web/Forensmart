"""
EXTRACTION UI MODULE - Streamlit UI Components
Handles extraction interface and progress display

This module provides:
- Extraction form rendering
- Progress tracking UI
- Results display
- Error handling UI
- Pause/Resume extraction
- Extraction history
- Export results (PDF, CSV, JSON)
- Comparison with previous extractions
- Detailed error messages
"""

import streamlit as st
import json
import csv
import io
from datetime import datetime
from typing import Optional, Dict, Any, List
from modules.extraction.orchestrator import get_orchestrator
from modules.consent.models import get_consent_manager

# Try to import reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ============================================================================
# EXTRACTION FORM RENDERING
# ============================================================================

def render_extraction_form() -> tuple[str, str]:
    """Render extraction form"""
    
    st.markdown("## 📊 Data Extraction")
    
    # Dev mode toggle
    from modules.consent.models import get_consent_manager
    consent_manager = get_consent_manager()
    
    col_dev1, col_dev2 = st.columns([4, 1])
    with col_dev2:
        if st.checkbox("🧪 Dev Mode", value=consent_manager.connectivity_manager.is_dev_mode(), key="extraction_dev_mode"):
            consent_manager.connectivity_manager.set_dev_mode(True)
            st.success("Dev mode enabled")
        else:
            consent_manager.connectivity_manager.set_dev_mode(False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        case_id = st.text_input(
            "Case ID:",
            placeholder="CASE-001",
            help="Unique case identifier"
        )
    
    with col2:
        device_id = st.text_input(
            "Device ID:",
            placeholder="DEVICE-001",
            help="Target device identifier"
        )
    
    return case_id, device_id


# ============================================================================
# EXTRACTION PROGRESS DISPLAY
# ============================================================================

def render_extraction_progress(
    case_id: str,
    device_id: str
) -> None:
    """Render extraction progress with live updates"""
    
    if not case_id or not device_id:
        st.warning("⚠️ Please enter Case ID and Device ID")
        return
    
    orchestrator = get_orchestrator()
    consent_manager = get_consent_manager()
    
    # Check consent
    session = consent_manager.get_session(case_id)
    if not session:
        if not consent_manager.connectivity_manager.is_dev_mode():
            st.error(f"❌ No consent found for case {case_id}")
            return
        else:
            st.warning(f"⚠️ No consent found for case {case_id} (Dev Mode: Bypassing)")
            st.info("🧪 Dev Mode: Proceeding with extraction without consent")
    else:
        st.info(f"📋 Extracting with {session.level.name} consent level")
    
    # Display consent level and module requirements
    st.markdown("---")
    st.markdown("### 🔐 Consent & Module Requirements")
    
    from modules.extraction.orchestrator import MODULE_MIN_LEVELS, check_module_consent
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if session and session.level:
            st.metric("Current Consent", session.level.name)
        else:
            st.metric("Current Consent", "NONE")
    
    with col2:
        if session and hasattr(session, 'locked') and session.locked:
            st.metric("Status", "🔒 Locked")
        else:
            st.metric("Status", "🔓 Unlocked")
    
    with col3:
        if session and session.level:
            st.metric("Consent Level", f"{session.level.value}/4")
        else:
            st.metric("Consent Level", "0/4")
    
    with col4:
        if consent_manager.connectivity_manager.is_dev_mode():
            st.metric("Dev Mode", "🧪 ON")
        else:
            st.metric("Dev Mode", "OFF")
    
    # Show module requirements
    st.markdown("**Module Requirements:**")
    
    module_cols = st.columns(3)
    col_idx = 0
    
    for module_name, min_level in MODULE_MIN_LEVELS.items():
        with module_cols[col_idx % 3]:
            if session and session.level:
                allowed, message = check_module_consent(session.level, module_name)
                if allowed:
                    st.success(f"✅ {module_name.title()}: {min_level.name}")
                else:
                    st.error(f"❌ {module_name.title()}: Requires {min_level.name}")
            else:
                st.warning(f"⚠️ {module_name.title()}: Requires {min_level.name}")
            col_idx += 1
    
    # Dev mode info
    if consent_manager.connectivity_manager.is_dev_mode():
        st.info("🧪 **Dev Mode Active** - Consent checks bypassed for testing")
        st.success("✅ Testing all extraction modules")
    
    # Start extraction
    if st.button("🚀 Start Extraction", use_container_width=True, type="primary"):
        
        # Create progress containers
        st.markdown("---")
        st.markdown("## ⏳ Extraction Progress")
        
        # Main progress bar
        progress_bar = st.progress(0)
        
        # Status containers
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_module = st.empty()
            current_module.write("🔄 **Current Module**: Initializing...")
        
        with col2:
            progress_percent = st.empty()
            progress_percent.write("📊 **Progress**: 0%")
        
        with col3:
            elapsed_time = st.empty()
            elapsed_time.write("⏱️ **Elapsed**: 0s")
        
        # Detailed status
        status_container = st.container()
        status_log = st.empty()
        
        # Module status list
        module_status = st.empty()
        
        import time
        start_time = time.time()
        module_list = []
        module_times = {}
        
        def progress_callback(message: str, current: int):
            """Update progress with live updates"""
            
            # Calculate progress
            total_modules = 6
            progress = current / total_modules
            
            # Update progress bar
            progress_bar.progress(min(progress, 1.0))
            
            # Update current module
            current_module.write(f"🔄 **Current Module**: {message}")
            
            # Update progress percentage
            progress_percent.write(f"📊 **Progress**: {int(progress * 100)}%")
            
            # Update elapsed time
            elapsed = int(time.time() - start_time)
            elapsed_time.write(f"⏱️ **Elapsed**: {elapsed}s")
            
            # Track module start time
            if message not in module_times:
                module_times[message] = time.time()
            
            # Add to module list
            module_list.append({
                'module': message,
                'status': '⏳ Processing...',
                'time': elapsed
            })
            
            # Update module status display with detailed info
            status_text = "### 📋 Module Status\n\n"
            
            # Show completed modules
            for i, item in enumerate(module_list[:-1]):
                module_time = time.time() - module_times.get(item['module'], time.time())
                status_text += f"✅ {item['module']} ({module_time:.2f}s)\n"
            
            # Show current module
            if module_list:
                current_item = module_list[-1]
                module_time = time.time() - module_times.get(current_item['module'], time.time())
                status_text += f"🔄 {current_item['module']} ({module_time:.2f}s)\n"
            
            status_log.markdown(status_text)
        
        # Run extraction
        results = orchestrator.extract_all_data(
            case_id=case_id,
            device_id=device_id,
            consent_manager=consent_manager,
            progress_callback=progress_callback
        )
        
        # Final progress update
        progress_bar.progress(1.0)
        current_module.write("✅ **Current Module**: Extraction Complete!")
        progress_percent.write("📊 **Progress**: 100%")
        elapsed = int(time.time() - start_time)
        elapsed_time.write(f"⏱️ **Total Time**: {elapsed}s")
        
        # Success message
        st.success(f"✅ Extraction completed in {elapsed}s!")
        st.balloons()
        
        # Display results
        render_extraction_results(results)


# ============================================================================
# EXTRACTION RESULTS DISPLAY
# ============================================================================

def render_extraction_results(results: Dict[str, Any]) -> None:
    """Render extraction results"""
    
    st.markdown("---")
    st.markdown("## 📊 Extraction Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Artifacts", results.get('total_artifacts', 0))
    
    with col2:
        st.metric("Extraction Time", f"{results.get('total_time', 0):.2f}s")
    
    with col3:
        successful = len([m for m in results.get('modules', {}).values() if m.get('status') == 'success'])
        st.metric("Successful Modules", successful)
    
    with col4:
        blocked = len(results.get('blocked_modules', []))
        st.metric("Blocked Modules", blocked)
    
    st.markdown("---")
    
    # Module results
    st.markdown("### 📦 Module Results")
    
    tab1, tab2, tab3 = st.tabs(["Successful", "Blocked", "Errors"])
    
    with tab1:
        st.markdown("#### ✅ Successful Extractions")
        
        successful_modules = {
            name: data for name, data in results.get('modules', {}).items()
            if data.get('status') == 'success'
        }
        
        if not successful_modules:
            st.info("No successful extractions")
        else:
            for module_name, module_data in successful_modules.items():
                with st.expander(f"📦 {module_name.replace('_', ' ').title()}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Artifacts**: {module_data.get('artifact_count', 0)}")
                    
                    with col2:
                        st.write(f"**Time**: {module_data.get('extraction_time', 0):.2f}s")
                    
                    # Show sample data
                    if module_data.get('data'):
                        st.json(module_data.get('data'), expanded=False)
    
    with tab2:
        st.markdown("#### ❌ Blocked Extractions")
        
        blocked_modules = results.get('blocked_modules', [])
        
        if not blocked_modules:
            st.info("No blocked extractions")
        else:
            for blocked in blocked_modules:
                with st.expander(f"🚫 {blocked.get('module', 'Unknown').replace('_', ' ').title()}"):
                    st.warning(f"**Reason**: {blocked.get('reason')}")
                    st.write(f"**Required Level**: {blocked.get('required_level')}")
                    st.write(f"**Current Level**: {blocked.get('current_level')}")
    
    with tab3:
        st.markdown("#### ⚠️ Extraction Errors")
        
        error_modules = {
            name: data for name, data in results.get('modules', {}).items()
            if data.get('status') == 'error'
        }
        
        if not error_modules:
            st.info("No extraction errors")
        else:
            for module_name, module_data in error_modules.items():
                with st.expander(f"⚠️ {module_name.replace('_', ' ').title()}"):
                    st.error(f"**Error**: {module_data.get('error')}")


# ============================================================================
# MODULE INFORMATION DISPLAY
# ============================================================================

def render_module_info() -> None:
    """Render module information"""
    
    st.markdown("## 📚 Extraction Modules")
    
    orchestrator = get_orchestrator()
    module_info = orchestrator.get_module_info()
    
    for module_name, info in module_info.items():
        with st.expander(f"📦 {info.get('name')}"):
            st.write(f"**Description**: {info.get('description')}")
            st.write(f"**Module ID**: `{module_name}`")


# ============================================================================
# EXTRACTION TESTING LOOPHOLES
# ============================================================================

def render_extraction_testing_loopholes() -> None:
    """Render extraction testing loopholes"""
    
    import os
    
    if not os.getenv('TESTING', 'false').lower() == 'true':
        return
    
    st.markdown("---")
    st.markdown("## 🧪 Extraction Testing Loopholes")
    st.warning("⚠️ Testing mode enabled")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ✅ Quick Extract")
        if st.button("Extract All (Auto-Approve)", use_container_width=True):
            from modules.consent.models import ConsentTestingLoopholes, ConsentLevel, get_consent_manager
            
            case_id = "TEST-CASE-001"
            device_id = "TEST-DEVICE-001"
            
            # Auto-approve
            consent_manager = get_consent_manager()
            ConsentTestingLoopholes.auto_approve_consent(consent_manager, case_id, 'FULL')
            
            # Extract
            orchestrator = get_orchestrator()
            results = orchestrator.extract_all_data(
                case_id=case_id,
                device_id=device_id,
                consent_manager=consent_manager
            )
            
            st.success("✅ Quick extraction completed!")
            render_extraction_results(results)
    
    with col2:
        st.markdown("### 🔗 Extract with Link")
        if st.button("Extract via Approval Link", use_container_width=True):
            from modules.consent.models import ApprovalLinkGenerator, get_consent_manager
            
            case_id = "TEST-CASE-002"
            device_id = "TEST-DEVICE-002"
            
            # Generate link
            link_gen = ApprovalLinkGenerator()
            link = link_gen.generate_link(case_id, 1)
            
            st.success("✅ Approval link generated!")
            st.code(link)
    
    with col3:
        st.markdown("### 🔄 Reset & Extract")
        if st.button("Reset & Extract", use_container_width=True):
            from modules.consent.models import ConsentTestingLoopholes, get_consent_manager
            
            case_id = "TEST-CASE-003"
            device_id = "TEST-DEVICE-003"
            
            # Reset
            consent_manager = get_consent_manager()
            ConsentTestingLoopholes.reset_case_consent(consent_manager, case_id)
            
            # Auto-approve
            ConsentTestingLoopholes.auto_approve_consent(consent_manager, case_id, 'LEGAL')
            
            # Extract
            orchestrator = get_orchestrator()
            results = orchestrator.extract_all_data(
                case_id=case_id,
                device_id=device_id,
                consent_manager=consent_manager
            )
            
            st.success("✅ Reset and extraction completed!")
            render_extraction_results(results)


# ============================================================================
# PAUSE/RESUME EXTRACTION
# ============================================================================

def render_extraction_controls(extraction_id: str) -> None:
    """Render pause/resume and cancel controls with real functionality"""
    
    st.markdown("## ⏸️ Extraction Controls")
    
    orchestrator = get_orchestrator()
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get current status
    is_paused = orchestrator.is_extraction_paused(extraction_id)
    is_cancelled = orchestrator.is_extraction_cancelled(extraction_id)
    pause_duration = orchestrator.get_extraction_pause_duration(extraction_id)
    
    with col1:
        if not is_paused and not is_cancelled:
            if st.button("⏸️ Pause Extraction", use_container_width=True):
                success = orchestrator.pause_extraction(extraction_id)
                if success:
                    st.info("⏸️ Extraction paused")
                    st.rerun()
                else:
                    st.error("Failed to pause extraction")
        else:
            st.button("⏸️ Pause Extraction", use_container_width=True, disabled=True)
    
    with col2:
        if is_paused and not is_cancelled:
            if st.button("▶️ Resume Extraction", use_container_width=True):
                success = orchestrator.resume_extraction(extraction_id)
                if success:
                    st.info("▶️ Extraction resumed")
                    st.rerun()
                else:
                    st.error("Failed to resume extraction")
        else:
            st.button("▶️ Resume Extraction", use_container_width=True, disabled=True)
    
    with col3:
        if not is_cancelled:
            if st.button("🛑 Cancel Extraction", use_container_width=True, type="secondary"):
                success = orchestrator.cancel_active_extraction(extraction_id)
                if success:
                    st.error("🛑 Extraction cancelled")
                    st.rerun()
                else:
                    st.error("Failed to cancel extraction")
        else:
            st.button("🛑 Cancel Extraction", use_container_width=True, disabled=True)
    
    with col4:
        st.metric("Pause Duration", f"{pause_duration:.1f}s")
    
    # Show status
    st.markdown("### Status")
    status_cols = st.columns(3)
    
    with status_cols[0]:
        if is_paused:
            st.warning("⏸️ PAUSED")
        else:
            st.success("▶️ RUNNING")
    
    with status_cols[1]:
        if is_cancelled:
            st.error("🛑 CANCELLED")
        else:
            st.info("✅ ACTIVE")
    
    with status_cols[2]:
        st.metric("Total Pause Time", f"{pause_duration:.2f}s")


# ============================================================================
# EXTRACTION HISTORY VIEW
# ============================================================================

def render_extraction_history(case_id: str) -> None:
    """Render extraction history"""
    
    st.markdown("## 📋 Extraction History")
    
    orchestrator = get_orchestrator()
    
    # Simulated history (in production, load from database)
    history = [
        {
            'extraction_id': f'{case_id}_001',
            'timestamp': '2025-11-25 10:30:00',
            'modules': 6,
            'artifacts': 1245,
            'status': 'completed',
            'time': '45.2s'
        },
        {
            'extraction_id': f'{case_id}_002',
            'timestamp': '2025-11-25 11:15:00',
            'modules': 4,
            'artifacts': 856,
            'status': 'completed',
            'time': '32.1s'
        },
        {
            'extraction_id': f'{case_id}_003',
            'timestamp': '2025-11-25 12:00:00',
            'modules': 6,
            'artifacts': 1512,
            'status': 'completed',
            'time': '52.8s'
        }
    ]
    
    for item in history:
        with st.expander(f"📦 {item['extraction_id']} - {item['timestamp']}"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Modules", item['modules'])
            with col2:
                st.metric("Artifacts", item['artifacts'])
            with col3:
                st.metric("Status", item['status'].upper())
            with col4:
                st.metric("Time", item['time'])


# ============================================================================
# MODULE-LEVEL FILTERING
# ============================================================================

def render_module_filter() -> List[str]:
    """Render module-level filtering"""
    
    st.markdown("## 🔍 Module Filter")
    
    modules = [
        'device_info',
        'communications',
        'location',
        'security',
        'media',
        'system'
    ]
    
    selected_modules = st.multiselect(
        "Select modules to extract:",
        modules,
        default=modules,
        help="Choose which modules to include in extraction"
    )
    
    return selected_modules


# ============================================================================
# EXPORT RESULTS
# ============================================================================

def generate_pdf_report(results: Dict[str, Any]) -> bytes:
    """Generate PDF report from extraction results"""
    
    if not PDF_AVAILABLE:
        return None
    
    # Create PDF in memory
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=1  # Center
    )
    story.append(Paragraph("FORENSMART EXTRACTION REPORT", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Case Information
    story.append(Paragraph("Case Information", styles['Heading2']))
    case_data = [
        ['Case ID', results.get('case_id', 'N/A')],
        ['Device ID', results.get('device_id', 'N/A')],
        ['Start Time', results.get('start_time', 'N/A')],
        ['End Time', results.get('end_time', 'N/A')],
        ['Total Time', f"{results.get('total_time', 0):.2f}s"],
        ['Total Artifacts', str(results.get('total_artifacts', 0))]
    ]
    
    case_table = Table(case_data, colWidths=[2*inch, 4*inch])
    case_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(case_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Module Results
    story.append(Paragraph("Module Results", styles['Heading2']))
    
    module_data = [['Module', 'Status', 'Artifacts', 'Time']]
    for module_name, module_info in results.get('modules', {}).items():
        module_data.append([
            module_name.replace('_', ' ').title(),
            module_info.get('status', 'unknown').upper(),
            str(module_info.get('artifact_count', 0)),
            f"{module_info.get('extraction_time', 0):.2f}s"
        ])
    
    module_table = Table(module_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    module_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(module_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Blocked Modules
    if results.get('blocked_modules'):
        story.append(Paragraph("Blocked Modules", styles['Heading2']))
        
        blocked_data = [['Module', 'Reason', 'Required Level']]
        for blocked in results.get('blocked_modules', []):
            blocked_data.append([
                blocked.get('module', 'N/A'),
                blocked.get('reason', 'N/A'),
                blocked.get('required_level', 'N/A')
            ])
        
        blocked_table = Table(blocked_data, colWidths=[2*inch, 2.5*inch, 1.5*inch])
        blocked_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.red),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(blocked_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Footer
    story.append(Spacer(1, 0.3*inch))
    footer_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    story.append(Paragraph(footer_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def render_export_results(results: Dict[str, Any]) -> None:
    """Render export results functionality"""
    
    st.markdown("## 📤 Export Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 Export as JSON", use_container_width=True):
            json_str = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"extraction_{results.get('case_id', 'unknown')}.json",
                mime="application/json"
            )
            st.success("✅ JSON export ready")
    
    with col2:
        if st.button("📕 Export as PDF", use_container_width=True):
            if PDF_AVAILABLE:
                pdf_data = generate_pdf_report(results)
                if pdf_data:
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name=f"extraction_{results.get('case_id', 'unknown')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("✅ PDF export ready")
                else:
                    st.error("❌ Failed to generate PDF")
            else:
                st.warning("⚠️ PDF export requires reportlab library")
                st.info("Install with: pip install reportlab")
    
    with col3:
        if st.button("📊 Export as CSV", use_container_width=True):
            # Convert results to CSV format
            csv_data = []
            for module_name, module_data in results.get('modules', {}).items():
                csv_data.append({
                    'Module': module_name,
                    'Status': module_data.get('status', 'unknown'),
                    'Artifacts': module_data.get('artifact_count', 0),
                    'Time': f"{module_data.get('extraction_time', 0):.2f}s"
                })
            
            # Create CSV string
            import io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=['Module', 'Status', 'Artifacts', 'Time'])
            writer.writeheader()
            writer.writerows(csv_data)
            csv_str = output.getvalue()
            
            st.download_button(
                label="Download CSV",
                data=csv_str,
                file_name=f"extraction_{results.get('case_id', 'unknown')}.csv",
                mime="text/csv"
            )
            st.success("✅ CSV export ready")
    
    with col3:
        if st.button("📋 Export Summary", use_container_width=True):
            summary = f"""
EXTRACTION SUMMARY
==================

Case ID: {results.get('case_id')}
Device ID: {results.get('device_id')}
Start Time: {results.get('start_time')}
End Time: {results.get('end_time')}
Total Time: {results.get('total_time', 0):.2f}s
Total Artifacts: {results.get('total_artifacts', 0)}

MODULES:
--------
"""
            for module_name, module_data in results.get('modules', {}).items():
                summary += f"\n{module_name}: {module_data.get('status')} ({module_data.get('artifact_count', 0)} artifacts)"
            
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name=f"extraction_{results.get('case_id', 'unknown')}_summary.txt",
                mime="text/plain"
            )
            st.success("✅ Summary export ready")


# ============================================================================
# COMPARISON WITH PREVIOUS EXTRACTIONS
# ============================================================================

def render_extraction_comparison(case_id: str, current_results: Dict[str, Any]) -> None:
    """Render comparison with previous extractions"""
    
    st.markdown("## 📊 Comparison with Previous Extraction")
    
    # Simulated previous extraction (in production, load from database)
    previous_results = {
        'case_id': case_id,
        'total_artifacts': 1245,
        'total_time': 45.2,
        'modules': {
            'device_info': {'status': 'success', 'artifact_count': 1},
            'communications': {'status': 'success', 'artifact_count': 245},
            'location': {'status': 'success', 'artifact_count': 156},
            'security': {'status': 'success', 'artifact_count': 1},
            'media': {'status': 'success', 'artifact_count': 342},
            'system': {'status': 'success', 'artifact_count': 500}
        }
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_artifacts = current_results.get('total_artifacts', 0)
        previous_artifacts = previous_results.get('total_artifacts', 0)
        diff = current_artifacts - previous_artifacts
        st.metric(
            "Total Artifacts",
            current_artifacts,
            delta=f"{diff:+d}" if diff != 0 else "No change"
        )
    
    with col2:
        current_time = current_results.get('total_time', 0)
        previous_time = previous_results.get('total_time', 0)
        diff = current_time - previous_time
        st.metric(
            "Extraction Time",
            f"{current_time:.2f}s",
            delta=f"{diff:+.2f}s" if diff != 0 else "No change"
        )
    
    with col3:
        current_modules = len([m for m in current_results.get('modules', {}).values() if m.get('status') == 'success'])
        previous_modules = len([m for m in previous_results.get('modules', {}).values() if m.get('status') == 'success'])
        diff = current_modules - previous_modules
        st.metric(
            "Successful Modules",
            current_modules,
            delta=f"{diff:+d}" if diff != 0 else "No change"
        )
    
    # Detailed comparison
    st.markdown("### Module Comparison")
    
    comparison_data = []
    for module_name in current_results.get('modules', {}).keys():
        current_module = current_results['modules'].get(module_name, {})
        previous_module = previous_results['modules'].get(module_name, {})
        
        comparison_data.append({
            'Module': module_name,
            'Current': current_module.get('artifact_count', 0),
            'Previous': previous_module.get('artifact_count', 0),
            'Change': current_module.get('artifact_count', 0) - previous_module.get('artifact_count', 0)
        })
    
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)


# ============================================================================
# DETAILED ERROR MESSAGES PER MODULE
# ============================================================================

def render_detailed_error_messages(results: Dict[str, Any]) -> None:
    """Render detailed error messages per module"""
    
    st.markdown("## ⚠️ Detailed Error Messages")
    
    error_modules = {
        name: data for name, data in results.get('modules', {}).items()
        if data.get('status') == 'error'
    }
    
    if not error_modules:
        st.success("✅ No errors - all modules completed successfully")
        return
    
    st.error(f"❌ {len(error_modules)} module(s) failed")
    
    for module_name, module_data in error_modules.items():
        with st.expander(f"❌ {module_name.replace('_', ' ').title()} - Error Details"):
            st.markdown("### Error Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Module**: {module_name}")
                st.write(f"**Status**: {module_data.get('status')}")
            
            with col2:
                st.write(f"**Error Type**: {type(module_data.get('error')).__name__}")
                st.write(f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            st.markdown("### Error Message")
            st.error(module_data.get('error', 'Unknown error'))
            
            st.markdown("### Troubleshooting")
            st.info("""
            **Possible Solutions:**
            1. Check internet connectivity
            2. Verify device is accessible
            3. Check consent level requirements
            4. Review logs for more details
            5. Try extraction again
            """)
            
            st.markdown("### Retry Options")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"🔄 Retry {module_name}", use_container_width=True):
                    st.info(f"Retrying {module_name}...")
            
            with col2:
                if st.button(f"⏭️ Skip {module_name}", use_container_width=True):
                    st.warning(f"Skipped {module_name}")
