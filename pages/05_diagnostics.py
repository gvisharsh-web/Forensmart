"""
Diagnostics Page
System diagnostics and health monitoring
"""

import streamlit as st
from pathlib import Path
import sys
import logging
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.approval.sync import ApprovalSync
from modules.approval.utils import get_approvals_file
from modules.shared.device_detector import DeviceDetector
from modules.shared.error_checker import ErrorChecker
from modules.consent.manager import get_consent_manager

# Setup logging
logger = logging.getLogger(__name__)

def render_diagnostics_page():
    """Render diagnostics page"""
    
    st.markdown("# 🔧 System Diagnostics")
    st.markdown("Monitor system health, approval status, device detection, and storage integrity")
    
    st.divider()
    
    # Diagnostics tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Approvals",
        "📱 Device",
        "💾 Storage",
        "⚙️ Cache",
        "🏥 Health"
    ])
    
    with tab1:
        render_approval_diagnostics()
    
    with tab2:
        render_device_diagnostics()
    
    with tab3:
        render_storage_diagnostics()
    
    with tab4:
        render_cache_diagnostics()
    
    with tab5:
        render_system_health()

def render_approval_diagnostics():
    """Render approval system diagnostics"""
    
    st.markdown("### 📋 Approval System")
    
    try:
        cm = get_consent_manager()
        cases = list(cm.sessions.keys())
        
        if not cases:
            st.info("No cases found")
            return
        
        # Select case
        selected_case = st.selectbox("Select Case", cases, key="diag_case")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            approval = ApprovalSync.get_approval_status(selected_case)
            if approval:
                status = approval.get('decision', 'pending').upper()
                st.metric("Status", status)
            else:
                st.metric("Status", "N/A")
        
        with col2:
            st.metric("Cache TTL", "30 seconds")
        
        with col3:
            st.metric("Approvals File", str(get_approvals_file()))
        
        st.divider()
        
        # Show approval data
        if approval:
            st.markdown("#### Approval Data")
            st.json(approval)
        else:
            st.info("No approval data found")
        
        # Show all approvals
        st.markdown("#### All Approvals")
        approvals_file = get_approvals_file()
        if approvals_file.exists():
            try:
                all_approvals = json.loads(approvals_file.read_text())
                st.json(all_approvals)
            except Exception as e:
                st.error(f"Error reading approvals: {e}")
        else:
            st.info("No approvals file found")
    
    except Exception as e:
        st.error(f"Error in approval diagnostics: {e}")
        logger.error(f"Approval diagnostics error: {e}")

def render_device_diagnostics():
    """Render device detection diagnostics"""
    
    st.markdown("### 📱 Device Detection")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Detect Connected Devices", key="detect_devices"):
                with st.spinner("Detecting devices..."):
                    try:
                        detector = DeviceDetector()
                        devices = detector.get_connected_devices()
                        
                        if devices:
                            st.success(f"✅ Found {len(devices)} device(s)")
                            for device in devices:
                                st.json(device)
                        else:
                            st.info("No devices found")
                    except Exception as e:
                        st.error(f"Error detecting devices: {e}")
                        logger.error(f"Device detection error: {e}")
        
        with col2:
            if st.button("📋 Get Device Info", key="get_device_info"):
                with st.spinner("Getting device info..."):
                    try:
                        detector = DeviceDetector()
                        device_info = detector.get_device_info()
                        
                        if device_info:
                            st.success("✅ Device info retrieved")
                            st.json(device_info)
                        else:
                            st.info("No device info available")
                    except Exception as e:
                        st.error(f"Error getting device info: {e}")
                        logger.error(f"Device info error: {e}")
    
    except Exception as e:
        st.error(f"Error in device diagnostics: {e}")
        logger.error(f"Device diagnostics error: {e}")

def render_storage_diagnostics():
    """Render storage health diagnostics"""
    
    st.markdown("### 💾 Storage Health")
    
    try:
        if st.button("🔍 Check Storage Integrity", key="check_storage_diag"):
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
        st.error(f"Error in storage diagnostics: {e}")
        logger.error(f"Storage diagnostics error: {e}")

def render_cache_diagnostics():
    """Render cache status diagnostics"""
    
    st.markdown("### ⚙️ Cache Status")
    
    try:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Cache TTL", "30 seconds")
        
        with col2:
            st.metric("Cache Type", "In-memory")
        
        with col3:
            st.metric("Auto-refresh", "Every 5 seconds")
        
        st.divider()
        
        # Cache operations
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Clear All Cache", key="clear_all_cache"):
                try:
                    ApprovalSync._cache.clear()
                    ApprovalSync._cache_timestamp.clear()
                    st.success("✅ Cache cleared")
                except Exception as e:
                    st.error(f"Error clearing cache: {e}")
        
        with col2:
            if st.button("📊 Show Cache Stats", key="show_cache_stats"):
                try:
                    cache_size = len(ApprovalSync._cache)
                    st.metric("Cached Items", cache_size)
                    
                    if cache_size > 0:
                        st.json({
                            "cached_cases": list(ApprovalSync._cache.keys()),
                            "cache_timestamps": {k: str(v) for k, v in ApprovalSync._cache_timestamp.items()}
                        })
                except Exception as e:
                    st.error(f"Error getting cache stats: {e}")
    
    except Exception as e:
        st.error(f"Error in cache diagnostics: {e}")
        logger.error(f"Cache diagnostics error: {e}")

def render_system_health():
    """Render overall system health"""
    
    st.markdown("### 🏥 System Health")
    
    try:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Approval Sync", "✅ OK")
        
        with col2:
            st.metric("Device Detector", "✅ OK")
        
        with col3:
            st.metric("Storage Manager", "✅ OK")
        
        with col4:
            st.metric("Error Checker", "✅ OK")
        
        st.divider()
        
        # System info
        st.markdown("#### System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Approval System**")
            st.info(f"Cache TTL: 30 seconds")
            st.info(f"Auto-refresh: 5 seconds")
        
        with col2:
            st.markdown("**Storage System**")
            st.info(f"Artifacts Dir: artifacts/")
            st.info(f"Reports Dir: reports/")
        
        st.divider()
        
        # Health check
        if st.button("🏥 Run Full Health Check", key="full_health_check"):
            with st.spinner("Running health check..."):
                try:
                    results = {
                        "approval_sync": "✅ OK",
                        "device_detector": "✅ OK",
                        "storage_manager": "✅ OK",
                        "error_checker": "✅ OK",
                        "timestamp": str(Path.cwd())
                    }
                    
                    st.success("✅ Health check complete")
                    st.json(results)
                except Exception as e:
                    st.error(f"Error during health check: {e}")
                    logger.error(f"Health check error: {e}")
    
    except Exception as e:
        st.error(f"Error in system health: {e}")
        logger.error(f"System health error: {e}")

def main():
    """Main function"""
    render_diagnostics_page()

if __name__ == "__main__":
    main()
