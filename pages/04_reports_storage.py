"""
Reports & Storage Page
Handles report generation and storage management
"""

import streamlit as st
from pathlib import Path
import sys
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.consent.manager import get_consent_manager
from modules.storage.manager import StorageManager, StorageAnalytics
from modules.shared.error_checker import ErrorChecker

# Setup logging
logger = logging.getLogger(__name__)

def render_reports_storage_page():
    """Render reports and storage page"""
    
    st.markdown("# 📊 Reports & Storage")
    st.markdown("Generate reports and manage storage")
    
    # Initialize session state
    if 'case_id' not in st.session_state:
        st.session_state['case_id'] = None
    
    # Sidebar: Case Selection
    st.sidebar.markdown("### 📋 Case Selection")
    
    cm = get_consent_manager()
    cases = list(cm.sessions.keys())
    
    if cases:
        selected_case = st.sidebar.selectbox(
            "Select Case",
            cases,
            key="reports_case"
        )
        if st.button("Load Case", key="load_reports_case"):
            st.session_state['case_id'] = selected_case
            st.rerun()
    else:
        st.sidebar.info("No cases found")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs([
        "📄 Reports",
        "💾 Storage",
        "🧹 Cleanup"
    ])
    
    with tab1:
        render_reports_tab()
    
    with tab2:
        render_storage_tab()
    
    with tab3:
        render_cleanup_tab()

def render_reports_tab():
    """Render reports tab"""
    
    st.markdown("### 📄 Report Generation")
    
    case_id = st.session_state.get('case_id')
    
    if not case_id:
        st.info("Select a case to generate reports")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Generate Extraction Report", key="gen_extraction_report"):
            with st.spinner("Generating extraction report..."):
                try:
                    # Create report
                    report_dir = Path(f"reports/{case_id}")
                    report_dir.mkdir(parents=True, exist_ok=True)
                    
                    report_file = report_dir / "extraction_report.txt"
                    report_file.write_text(f"Extraction Report for Case {case_id}\n")
                    
                    st.success(f"✅ Report generated: {report_file}")
                except Exception as e:
                    st.error(f"Error generating report: {e}")
                    logger.error(f"Report generation error: {e}")
    
    with col2:
        if st.button("📊 Generate Intelligence Report", key="gen_intelligence_report"):
            with st.spinner("Generating intelligence report..."):
                try:
                    # Create report
                    report_dir = Path(f"reports/{case_id}")
                    report_dir.mkdir(parents=True, exist_ok=True)
                    
                    report_file = report_dir / "intelligence_report.txt"
                    report_file.write_text(f"Intelligence Report for Case {case_id}\n")
                    
                    st.success(f"✅ Report generated: {report_file}")
                except Exception as e:
                    st.error(f"Error generating report: {e}")
                    logger.error(f"Report generation error: {e}")
    
    with col3:
        if st.button("📦 Generate Full Report", key="gen_full_report"):
            with st.spinner("Generating full report..."):
                try:
                    # Create report
                    report_dir = Path(f"reports/{case_id}")
                    report_dir.mkdir(parents=True, exist_ok=True)
                    
                    report_file = report_dir / "full_report.txt"
                    report_file.write_text(f"Full Report for Case {case_id}\n")
                    
                    st.success(f"✅ Report generated: {report_file}")
                except Exception as e:
                    st.error(f"Error generating report: {e}")
                    logger.error(f"Report generation error: {e}")

def render_storage_tab():
    """Render storage tab"""
    
    st.markdown("### 💾 Storage Management")
    
    try:
        # Storage analytics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_info = StorageAnalytics.get_total_storage_info()
            st.metric("Total Storage", total_info['total_size_formatted'])
        
        with col2:
            artifacts_size = StorageAnalytics.get_directory_size(Path("artifacts"))
            st.metric("Artifacts", StorageAnalytics.format_size(artifacts_size))
        
        with col3:
            reports_size = StorageAnalytics.get_directory_size(Path("reports"))
            st.metric("Reports", StorageAnalytics.format_size(reports_size))
        
        with col4:
            audit_size = StorageAnalytics.get_directory_size(Path("audit"))
            st.metric("Audit", StorageAnalytics.format_size(audit_size))
        
        st.divider()
        
        # Storage integrity check
        st.markdown("#### 🔍 Storage Integrity")
        
        if st.button("Check Storage Integrity", key="check_integrity"):
            with st.spinner("Checking storage integrity..."):
                try:
                    integrity = ErrorChecker.check_storage_integrity()
                    
                    if integrity.get('status') == 'healthy':
                        st.success("✅ Storage is healthy")
                    else:
                        st.warning("⚠️ Storage issues detected")
                    
                    st.json(integrity)
                except Exception as e:
                    st.error(f"Error checking storage: {e}")
                    logger.error(f"Storage check error: {e}")
    
    except Exception as e:
        st.error(f"Error in storage management: {e}")
        logger.error(f"Storage management error: {e}")

def render_cleanup_tab():
    """Render cleanup tab"""
    
    st.markdown("### 🧹 Storage Cleanup")
    
    case_id = st.session_state.get('case_id')
    
    if not case_id:
        st.info("Select a case to perform cleanup operations")
        return
    
    st.warning("⚠️ Cleanup operations are permanent and cannot be undone")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Delete Artifacts", key="delete_artifacts"):
            confirm = st.checkbox("I confirm to delete all artifacts for this case", key="confirm_artifacts")
            if confirm and st.button("Confirm Delete", key="confirm_delete_artifacts"):
                with st.spinner("Deleting artifacts..."):
                    try:
                        success, msg, info = StorageManager.delete_artifact_directory(case_id)
                        if success:
                            st.success(f"✅ {msg}")
                        else:
                            st.error(f"❌ {msg}")
                    except Exception as e:
                        st.error(f"Error deleting artifacts: {e}")
                        logger.error(f"Deletion error: {e}")
    
    with col2:
        if st.button("🗑️ Delete Entire Case", key="delete_case"):
            confirm = st.checkbox("I confirm to delete entire case", key="confirm_case")
            if confirm and st.button("Confirm Delete Case", key="confirm_delete_case"):
                with st.spinner("Deleting case..."):
                    try:
                        success, msg, info = StorageManager.delete_entire_case(case_id)
                        if success:
                            st.success(f"✅ {msg}")
                            st.session_state['case_id'] = None
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
                    except Exception as e:
                        st.error(f"Error deleting case: {e}")
                        logger.error(f"Case deletion error: {e}")

def main():
    """Main function"""
    render_reports_storage_page()

if __name__ == "__main__":
    main()
