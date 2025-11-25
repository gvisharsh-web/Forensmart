"""
Intelligence Page
Handles location intelligence and communications analysis
"""

import streamlit as st
from pathlib import Path
import sys
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.approval.sync import ApprovalSync
from modules.consent.manager import get_consent_manager
from modules.analysis.location_intelligence import render_ui as render_location_ui
from modules.analysis.comms_analyzer import render_ui as render_comms_ui

# Setup logging
logger = logging.getLogger(__name__)

def render_intelligence_page():
    """Render intelligence page"""
    
    st.markdown("# 🧠 Intelligence Analysis")
    st.markdown("Analyze location and communications data")
    
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
            key="intelligence_case"
        )
        if st.button("Load Case", key="load_intelligence_case"):
            st.session_state['case_id'] = selected_case
            st.rerun()
    else:
        st.sidebar.info("No cases found")
    
    # Current case info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Current Case")
    
    case_id = st.session_state.get('case_id')
    
    if case_id:
        st.sidebar.metric("Case ID", case_id)
        
        # Check approval status
        approval = ApprovalSync.get_approval_status(case_id)
        if approval:
            status = approval.get('decision', 'pending').upper()
            if status == 'APPROVED':
                st.sidebar.success(f"✅ {status}")
            elif status == 'DENIED':
                st.sidebar.error(f"❌ {status}")
            else:
                st.sidebar.warning(f"⏳ {status}")
        else:
            st.sidebar.warning("⏳ Pending")
    else:
        st.sidebar.info("No case selected")
    
    # Main content
    if not case_id:
        st.info("👈 Select a case from the sidebar to begin intelligence analysis")
        return
    
    # Check approval status
    approval = ApprovalSync.get_approval_status(case_id)
    
    if not approval or approval.get('decision') != 'approved':
        st.warning("⏳ Awaiting approval from nominee")
        st.info("The nominee must approve this extraction request before you can proceed with intelligence analysis.")
        return
    
    # Approval granted - show intelligence options
    st.success("✅ **APPROVED** - Ready for intelligence analysis")
    
    st.divider()
    
    # Intelligence tabs
    tab1, tab2 = st.tabs([
        "📍 Location Intelligence",
        "💬 Communications Analysis"
    ])
    
    with tab1:
        render_location_intelligence(case_id)
    
    with tab2:
        render_communications_analysis(case_id)

def render_location_intelligence(case_id: str):
    """Render location intelligence UI"""
    
    try:
        # Check if extraction data exists
        artifacts_dir = Path(f"artifacts/{case_id}")
        if not artifacts_dir.exists():
            st.warning("No extraction data found for this case")
            return
        
        # Render location intelligence UI
        render_location_ui(case_id)
    
    except Exception as e:
        st.error(f"Error in location intelligence: {e}")
        logger.error(f"Location intelligence error: {e}")

def render_communications_analysis(case_id: str):
    """Render communications analysis UI"""
    
    st.markdown("### 💬 Communications Analysis")
    
    try:
        # Check if extraction data exists
        artifacts_dir = Path(f"artifacts/{case_id}")
        if not artifacts_dir.exists():
            st.warning("No extraction data found for this case")
            return
        
        # Run communications analysis
        comms_analyzer = CommsAnalyzer()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Analyze Communications", key="analyze_comms"):
                with st.spinner("Analyzing communications data..."):
                    try:
                        results = comms_analyzer.analyze(case_id)
                        
                        if results:
                            st.success("✅ Analysis complete")
                            
                            # Display results
                            st.markdown("#### 📊 Results")
                            st.json(results)
                            
                            # Save results
                            results_file = Path(f"reports/{case_id}/communications_analysis.json")
                            results_file.parent.mkdir(parents=True, exist_ok=True)
                            results_file.write_text(str(results))
                            
                            st.success(f"✅ Results saved to {results_file}")
                        else:
                            st.info("No communications data found")
                    except Exception as e:
                        st.error(f"Error analyzing communications: {e}")
                        logger.error(f"Communications analysis error: {e}")
        
        with col2:
            st.info("💬 Analyzes messages, calls, contacts, and communication patterns")
    
    except Exception as e:
        st.error(f"Error in communications analysis: {e}")
        logger.error(f"Communications analysis error: {e}")

def main():
    """Main function"""
    render_intelligence_page()

if __name__ == "__main__":
    main()
