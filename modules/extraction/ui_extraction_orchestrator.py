"""
Extraction Orchestrator UI Component

Main extraction page that controls the entire workflow:
1. Device selection
2. Module selection
3. Consent check
4. Extraction progress
5. Results display

This is the main page that investigator sees.
"""

import streamlit as st
from typing import Dict, List
from datetime import datetime

# Import UI components
from ui_device_selector import render_device_selector
from ui_extraction_progress import render_extraction_progress
from ui_extraction_results import render_extraction_results
from ui_consent_check import render_consent_check, show_consent_summary
from ui_module_selector import render_module_selector


def render_extraction_page() -> None:
    """
    Render main extraction page with complete workflow.
    
    This is the orchestrator that controls the entire extraction process:
    1. Device selection
    2. Module selection
    3. Consent check
    4. Extraction control
    5. Progress display
    6. Results display
    """
    st.set_page_config(
        page_title="Forensmart - Extraction",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.header("📱 Device Extraction Workflow")
    
    # Initialize session state
    initialize_session_state()
    
    # Show consent summary in sidebar
    show_consent_summary()
    
    # Create tabs for different stages
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📱 Device",
        "📦 Modules",
        "🔐 Consent",
        "⚙️ Extract",
        "📊 Results"
    ])
    
    # ===== TAB 1: DEVICE SELECTION =====
    with tab1:
        st.subheader("Step 1: Select Device")
        
        device_info = render_device_selector()
        
        if device_info:
            st.session_state['selected_device'] = device_info
            st.success(f"✅ Device selected: {device_info['device_name']}")
        else:
            st.info("👈 Select a device to continue")
    
    # ===== TAB 2: MODULE SELECTION =====
    with tab2:
        st.subheader("Step 2: Select Modules to Extract")
        
        if not st.session_state.get('selected_device'):
            st.warning("⚠️ Please select a device first (Step 1)")
        else:
            modules = render_module_selector()
            
            if modules:
                st.session_state['selected_modules'] = modules
                
                # Show selected modules
                selected_count = sum(1 for m in modules.values() if m)
                st.success(f"✅ {selected_count} module(s) selected")
                
                # Show details
                with st.expander("📋 View Selected Modules"):
                    for module_name, is_selected in modules.items():
                        status = "✅" if is_selected else "❌"
                        st.write(f"{status} {module_name}")
            else:
                st.info("👈 Select modules to continue")
    
    # ===== TAB 3: CONSENT CHECK =====
    with tab3:
        st.subheader("Step 3: Verify Consent")
        
        if not st.session_state.get('selected_device'):
            st.warning("⚠️ Please select a device first (Step 1)")
        elif not st.session_state.get('selected_modules'):
            st.warning("⚠️ Please select modules first (Step 2)")
        else:
            # Render consent check
            is_approved, consent_details = render_consent_check()
            
            if is_approved:
                st.session_state['consent_approved'] = True
                st.success("✅ Consent verified - Ready to extract")
            else:
                st.session_state['consent_approved'] = False
                st.warning("⏳ Waiting for nominee approval")
    
    # ===== TAB 4: EXTRACTION CONTROL =====
    with tab4:
        st.subheader("Step 4: Start Extraction")
        
        if not st.session_state.get('selected_device'):
            st.warning("⚠️ Please select a device first (Step 1)")
        elif not st.session_state.get('selected_modules'):
            st.warning("⚠️ Please select modules first (Step 2)")
        elif not st.session_state.get('consent_approved'):
            st.warning("⚠️ Consent not approved yet (Step 3)")
        else:
            # Show extraction summary
            st.subheader("📋 Extraction Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Device", st.session_state['selected_device']['device_name'])
            
            with col2:
                module_count = sum(1 for m in st.session_state['selected_modules'].values() if m)
                st.metric("Modules", module_count)
            
            with col3:
                st.metric("Consent", "✅ Approved")
            
            # Extraction controls
            st.subheader("⚙️ Extraction Controls")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("▶️ Start Extraction", key="start_extraction"):
                    st.session_state['extraction_started'] = True
                    st.session_state['extraction_status'] = 'RUNNING'
                    st.success("✅ Extraction started")
                    st.rerun()
            
            with col2:
                if st.button("⏸️ Pause", key="pause_extraction", disabled=not st.session_state.get('extraction_started')):
                    st.session_state['extraction_status'] = 'PAUSED'
                    st.info("⏸️ Extraction paused")
            
            with col3:
                if st.button("▶️ Resume", key="resume_extraction", disabled=st.session_state.get('extraction_status') != 'PAUSED'):
                    st.session_state['extraction_status'] = 'RUNNING'
                    st.info("▶️ Extraction resumed")
            
            with col4:
                if st.button("⏹️ Stop", key="stop_extraction", disabled=not st.session_state.get('extraction_started')):
                    st.session_state['extraction_started'] = False
                    st.session_state['extraction_status'] = 'STOPPED'
                    st.warning("⏹️ Extraction stopped")
    
    # ===== TAB 5: RESULTS =====
    with tab5:
        st.subheader("Step 5: View Results")
        
        if not st.session_state.get('extraction_started'):
            st.info("👈 Start extraction first (Step 4)")
        else:
            # Show progress
            if st.session_state.get('extraction_status') == 'RUNNING':
                st.info("⏳ Extraction in progress...")
                render_extraction_progress()
            
            elif st.session_state.get('extraction_status') == 'PAUSED':
                st.warning("⏸️ Extraction paused")
                render_extraction_progress()
            
            else:
                st.success("✅ Extraction complete")
                render_extraction_results()
    
    # Show workflow diagram
    show_workflow_diagram()


def initialize_session_state() -> None:
    """Initialize session state variables."""
    
    if 'selected_device' not in st.session_state:
        st.session_state['selected_device'] = None
    
    if 'selected_modules' not in st.session_state:
        st.session_state['selected_modules'] = {}
    
    if 'consent_approved' not in st.session_state:
        st.session_state['consent_approved'] = False
    
    if 'extraction_started' not in st.session_state:
        st.session_state['extraction_started'] = False
    
    if 'extraction_status' not in st.session_state:
        st.session_state['extraction_status'] = 'IDLE'
    
    if 'case_id' not in st.session_state:
        st.session_state['case_id'] = 'case_001'


def show_workflow_diagram() -> None:
    """Show extraction workflow diagram."""
    
    with st.expander("📊 View Workflow Diagram"):
        st.write("""
        ```
        ┌─────────────────────────────────────────────────────────────┐
        │                    EXTRACTION WORKFLOW                       │
        ├─────────────────────────────────────────────────────────────┤
        │                                                              │
        │  Step 1: Device Selection                                   │
        │  ├─ Select device type (Android, iOS, HDD)                 │
        │  ├─ Select specific device                                 │
        │  └─ Confirm device details                                 │
        │         ↓                                                   │
        │  Step 2: Module Selection                                   │
        │  ├─ Device Info                                            │
        │  ├─ Communications                                         │
        │  ├─ Location                                               │
        │  ├─ Media                                                  │
        │  ├─ Security                                               │
        │  └─ Social Media                                           │
        │         ↓                                                   │
        │  Step 3: Consent Verification                              │
        │  ├─ Check consent status                                   │
        │  ├─ Send approval link to nominee                          │
        │  ├─ Nominee enters PIN/Pattern                             │
        │  └─ Consent UNLOCKED                                       │
        │         ↓                                                   │
        │  Step 4: Extraction Control                                │
        │  ├─ Start extraction                                       │
        │  ├─ Monitor progress                                       │
        │  ├─ Pause/Resume/Stop                                      │
        │  └─ Handle errors                                          │
        │         ↓                                                   │
        │  Step 5: Results Display                                    │
        │  ├─ Show extraction summary                                │
        │  ├─ Display extracted data                                 │
        │  ├─ Filter and search                                      │
        │  └─ Export results                                         │
        │                                                              │
        └─────────────────────────────────────────────────────────────┘
        ```
        """)


def show_quick_stats() -> None:
    """Show quick statistics in sidebar."""
    
    with st.sidebar:
        st.subheader("📊 Quick Stats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Cases", "5")
        
        with col2:
            st.metric("Completed", "3")
        
        st.divider()
        
        st.subheader("🔧 Tools")
        
        if st.button("🔄 Refresh"):
            st.rerun()
        
        if st.button("⚙️ Settings"):
            st.info("Settings page coming soon")
        
        if st.button("📋 History"):
            st.info("History page coming soon")


def main() -> None:
    """Main entry point."""
    render_extraction_page()
    show_quick_stats()


if __name__ == "__main__":
    main()
