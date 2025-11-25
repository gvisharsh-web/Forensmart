"""
Unified ForenSmart Application
==============================

Merged Consent Portal + Dashboard into a single Streamlit application.
This solves file synchronization and access issues by running both
functionalities in the same process.

Run with: streamlit run modules/unified_app.py
"""

from __future__ import annotations

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from urllib.parse import unquote

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st

# Import all necessary modules
from modules.consent.models import ConsentManager, ConsentLevel
from modules.approval.utils import get_approvals_file, save_approval_decision, get_approval_decision
from modules.approval.sync import ApprovalSync
from modules.approval.redirect import ApprovalRedirect, ApprovalNotifier
from modules.consent.portal import ConsentPortalEnhancer, ConsentAuditTrail, ConsentPortalLogger
from modules.extraction.orchestrator import DataExtractionOrchestrator
from modules.extraction.ui import render_extraction_tab, render_intelligence_tab
from modules.shared.utils import ResultsRepository

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="ForenSmart - Unified Console",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def get_consent_manager() -> ConsentManager:
    """Get or create consent manager."""
    if 'consent_manager' not in st.session_state:
        st.session_state['consent_manager'] = ConsentManager()
    return st.session_state['consent_manager']


def initialize_session():
    """Initialize session state variables."""
    if 'current_tab' not in st.session_state:
        st.session_state['current_tab'] = 'Dashboard'
    if 'case_id' not in st.session_state:
        st.session_state['case_id'] = None
    if 'approval_link' not in st.session_state:
        st.session_state['approval_link'] = None


# ============================================================================
# UNIFIED SIDEBAR
# ============================================================================

def render_sidebar():
    """Render unified sidebar with navigation."""
    with st.sidebar:
        st.markdown("# 🔍 ForenSmart Console")
        st.divider()
        
        # Navigation tabs
        st.markdown("## 📋 Navigation")
        tab = st.radio(
            "Select Module",
            ["Dashboard", "Consent Portal", "System Status"],
            key="nav_tabs"
        )
        st.session_state['current_tab'] = tab
        
        st.divider()
        
        # Case selection
        st.markdown("## 📁 Case Management")
        cm = get_consent_manager()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            case_id = st.text_input(
                "Case ID",
                value=st.session_state.get('case_id', ''),
                placeholder="e.g., CASE-001"
            )
            if case_id:
                st.session_state['case_id'] = case_id
        
        with col2:
            if st.button("✓", help="Confirm case"):
                st.session_state['case_id'] = case_id
                st.rerun()
        
        st.divider()
        
        # System status
        st.markdown("## ⚙️ System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Approval File", "✅ Ready" if get_approvals_file().exists() else "⏳ Pending")
        with col2:
            st.metric("Cache TTL", "30s")
        
        st.divider()
        
        # Quick actions
        st.markdown("## ⚡ Quick Actions")
        if st.button("🔄 Refresh All", use_container_width=True):
            ApprovalSync.clear_cache()
            st.rerun()
        
        if st.button("📊 View Approvals", use_container_width=True):
            st.session_state['current_tab'] = 'System Status'
            st.rerun()


# ============================================================================
# DASHBOARD TAB
# ============================================================================

def render_dashboard_tab():
    """Render dashboard tab."""
    st.markdown("# 📊 ForenSmart Dashboard")
    
    case_id = st.session_state.get('case_id')
    if not case_id:
        st.warning("⚠️ Please select a case from the sidebar")
        return
    
    cm = get_consent_manager()
    session = cm.get_session(case_id)
    
    if not session:
        st.info("📝 Creating new consent session...")
        session = cm.create_session(case_id)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Consent", "Extraction", "Intelligence", "Reports"])
    
    with tab1:
        render_consent_management(cm, case_id)
    
    with tab2:
        render_extraction_tab(case_id)
    
    with tab3:
        render_intelligence_tab(case_id)
    
    with tab4:
        st.markdown("### 📄 Reports")
        st.info("Report generation coming soon...")


def render_consent_management(cm: ConsentManager, case_id: str):
    """Render consent management section."""
    st.markdown("## 🔐 Consent Management")
    
    session = cm.get_session(case_id)
    if not session:
        st.error("No consent session found")
        return
    
    # Current consent level
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Consent Level", session.level.name if session.level else "NONE")
    with col2:
        approval_status = ApprovalSync.is_approved(case_id)
        st.metric("Approval Status", "✅ APPROVED" if approval_status else "⏳ PENDING")
    with col3:
        st.metric("Device ID", session.device_id or "Not set")
    
    st.divider()
    
    # Consent level selector
    st.markdown("### 📋 Set Consent Level")
    new_level = st.selectbox(
        "Consent Level",
        [level.name for level in ConsentLevel],
        index=[level.name for level in ConsentLevel].index(session.level.name) if session.level else 0
    )
    
    if st.button("✅ Update Consent Level"):
        cm.set_consent_level(case_id, ConsentLevel[new_level])
        st.success(f"✅ Consent level updated to {new_level}")
        st.rerun()
    
    st.divider()
    
    # Approval status
    st.markdown("### 📤 Approval Status")
    
    approval_decision = get_approval_decision(case_id)
    if approval_decision:
        if approval_decision == 'approved':
            st.success(f"✅ **Approved** - Extraction is unlocked!")
        elif approval_decision == 'denied':
            st.error(f"❌ **Denied** - Extraction request was rejected")
        else:
            st.info(f"⏳ **{approval_decision.upper()}** - Waiting for response")
    else:
        st.info("⏳ No approval decision yet")
    
    if st.button("🔄 Refresh Approval Status"):
        ApprovalSync.clear_cache(case_id)
        st.rerun()
    
    st.divider()
    
    # Approval diagnostics
    with st.expander("🔍 Approval System Diagnostics"):
        try:
            approvals_file = get_approvals_file()
            st.write(f"**File Location**: `{approvals_file}`")
            st.write(f"**File Exists**: {'✅ Yes' if approvals_file.exists() else '❌ No'}")
            
            if approvals_file.exists():
                try:
                    content = json.loads(approvals_file.read_text(encoding="utf-8"))
                    st.write(f"**Total Cases**: {len(content)}")
                    
                    if case_id in content:
                        st.write(f"**Case {case_id} Status**:")
                        st.json(content[case_id])
                    else:
                        st.warning(f"Case {case_id} not found in approval file")
                except Exception as e:
                    st.error(f"Error reading approval file: {e}")
            else:
                st.info("Approval file not yet created. It will be created when nominee approves.")
        except Exception as e:
            st.error(f"Diagnostics error: {e}")


# ============================================================================
# CONSENT PORTAL TAB
# ============================================================================

def render_consent_portal_tab():
    """Render consent portal tab."""
    st.markdown("# 🔐 Consent Portal")
    
    # Check for approval link in URL
    query_params = st.query_params
    approval_link_param = query_params.get('approval_link')
    
    if approval_link_param:
        render_approval_link_view(approval_link_param)
    else:
        render_approval_link_generator()


def render_approval_link_generator():
    """Render approval link generator."""
    st.markdown("## 📝 Generate Approval Link")
    
    col1, col2 = st.columns(2)
    
    with col1:
        case_id = st.text_input("Case ID", placeholder="e.g., CASE-001")
    
    with col2:
        nominee_name = st.text_input("Nominee Name", placeholder="e.g., John Doe")
    
    device_id = st.text_input("Device ID", placeholder="Optional")
    
    if st.button("🔗 Generate Approval Link"):
        if not case_id:
            st.error("❌ Case ID is required")
            return
        
        if not nominee_name:
            st.error("❌ Nominee name is required")
            return
        
        # Create approval link
        approval_link = ConsentPortalEnhancer.create_approval_link(
            case_id=case_id,
            nominee_name=nominee_name,
            device_id=device_id or "UNKNOWN"
        )
        
        # Save link
        from modules.approval.utils import get_approvals_file
        approvals_file = get_approvals_file()
        approvals = {}
        
        if approvals_file.exists():
            try:
                approvals = json.loads(approvals_file.read_text())
            except Exception:
                approvals = {}
        
        if case_id not in approvals:
            approvals[case_id] = {}
        
        approvals[case_id].update({
            'approval_link': approval_link,
            'link_created_at': datetime.now().isoformat(),
            'nominee_name': nominee_name,
            'device_id': device_id or 'UNKNOWN',
            'status': 'pending'
        })
        
        approvals_file.write_text(json.dumps(approvals, indent=2))
        
        st.success("✅ Approval link generated!")
        st.markdown("### 📋 Approval Link")
        st.code(approval_link, language='text')
        
        if st.button("📋 Copy Link"):
            st.success("Link copied to clipboard!")


def render_approval_link_view(approval_link_param: str):
    """Render approval link view for nominee."""
    st.markdown("# 📋 Approval Request")
    
    try:
        # Decode approval link
        import base64
        from urllib.parse import unquote
        
        decoded = base64.b64decode(unquote(approval_link_param)).decode('utf-8')
        approval_data = json.loads(decoded)
        
        case_id = approval_data.get('case_id')
        nominee_name = approval_data.get('nominee_name')
        
        st.markdown(f"### Case: **{case_id}**")
        st.markdown(f"### Nominee: **{nominee_name}**")
        
        st.divider()
        
        # Approval buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Approve", use_container_width=True):
                # Save approval
                success = save_approval_decision(case_id, 'approved', nominee_name)
                
                if success:
                    # Clear cache
                    try:
                        ApprovalSync.clear_cache(case_id)
                    except Exception:
                        pass
                    
                    st.success("✅ Approval saved successfully!")
                    st.balloons()
                    
                    # Show redirect message
                    st.info("📊 Redirecting to dashboard...")
                    st.markdown(f"[Open Dashboard](/?case_id={case_id})")
                else:
                    st.error("❌ Failed to save approval")
        
        with col2:
            if st.button("❌ Deny", use_container_width=True):
                # Save denial
                success = save_approval_decision(case_id, 'denied', nominee_name)
                
                if success:
                    # Clear cache
                    try:
                        ApprovalSync.clear_cache(case_id)
                    except Exception:
                        pass
                    
                    st.warning("⚠️ Approval denied")
                else:
                    st.error("❌ Failed to save denial")
    
    except Exception as e:
        st.error(f"❌ Invalid approval link: {e}")


# ============================================================================
# SYSTEM STATUS TAB
# ============================================================================

def render_system_status_tab():
    """Render system status tab."""
    st.markdown("# ⚙️ System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        approvals_file = get_approvals_file()
        st.metric("Approval File", "✅ Exists" if approvals_file.exists() else "⏳ Pending")
    
    with col2:
        st.metric("Cache TTL", "30 seconds")
    
    with col3:
        st.metric("Status", "✅ Running")
    
    st.divider()
    
    st.markdown("## 📊 Approval Records")
    
    approvals_file = get_approvals_file()
    if approvals_file.exists():
        try:
            content = json.loads(approvals_file.read_text(encoding="utf-8"))
            st.json(content)
        except Exception as e:
            st.error(f"Error reading approvals: {e}")
    else:
        st.info("No approval records yet")
    
    st.divider()
    
    st.markdown("## 📁 File Locations")
    st.write(f"**Approvals File**: `{get_approvals_file()}`")
    st.write(f"**Project Root**: `{PROJECT_ROOT}`")
    st.write(f"**Audit Directory**: `{PROJECT_ROOT / 'audit'}`")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    initialize_session()
    render_sidebar()
    
    current_tab = st.session_state.get('current_tab', 'Dashboard')
    
    if current_tab == 'Dashboard':
        render_dashboard_tab()
    elif current_tab == 'Consent Portal':
        render_consent_portal_tab()
    elif current_tab == 'System Status':
        render_system_status_tab()


if __name__ == '__main__':
    main()
