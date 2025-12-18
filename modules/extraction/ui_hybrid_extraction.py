"""
HYBRID EXTRACTION UI
User interface for hybrid extraction with bridge agent integration

This module provides:
- Hybrid extraction UI components
- Privilege escalation options
- Extended source selection
- Results display with completeness metrics
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from modules.extraction.hybrid_integration import (
    HybridExtractionAdapter,
    create_hybrid_adapter,
    get_extraction_completeness_report,
    compare_extraction_methods
)
from modules.extraction.hybrid_bridge_agent import EscalationMethod, ExtractionSource

logger = logging.getLogger(__name__)

# ============================================================================
# HYBRID EXTRACTION UI COMPONENTS
# ============================================================================

def render_hybrid_extraction_options():
    """Render hybrid extraction options panel"""
    
    st.subheader("Advanced Extraction Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Privilege Escalation**")
        enable_escalation = st.checkbox(
            "Enable privilege escalation",
            value=False,
            help="Attempt to escalate privileges for deeper extraction (Dirty Pipe, SELinux bypass, etc.)"
        )
        
        if enable_escalation:
            st.info(
                "⚠️ Privilege escalation will attempt:\n"
                "1. Dirty Pipe (CVE-2022-1786)\n"
                "2. SELinux bypass\n"
                "3. ADB root\n\n"
                "Requires consent and device compatibility."
            )
    
    with col2:
        st.write("**Extended Sources**")
        enable_extended = st.checkbox(
            "Enable extended source extraction",
            value=True,
            help="Extract from social media, cloud storage, encrypted apps, and system logs"
        )
        
        if enable_extended:
            st.info(
                "✅ Will extract from:\n"
                "• Social media (WhatsApp, Telegram, Signal)\n"
                "• Cloud storage (Google Drive, OneDrive)\n"
                "• Encrypted apps\n"
                "• System logs and kernel buffers"
            )
    
    return enable_escalation, enable_extended

def render_escalation_method_selector() -> Optional[list]:
    """Render escalation method selector"""
    
    st.subheader("Escalation Methods")
    
    col1, col2, col3 = st.columns(3)
    
    methods = []
    
    with col1:
        if st.checkbox("Dirty Pipe (CVE-2022-1786)", value=True):
            methods.append(EscalationMethod.DIRTY_PIPE)
    
    with col2:
        if st.checkbox("SELinux Bypass", value=True):
            methods.append(EscalationMethod.SELINUX_BYPASS)
    
    with col3:
        if st.checkbox("ADB Root", value=True):
            methods.append(EscalationMethod.ADB_ROOT)
    
    if methods:
        st.success(f"✅ {len(methods)} escalation methods selected")
        return methods
    
    return None

def render_extended_sources_selector() -> Dict[str, bool]:
    """Render extended sources selector"""
    
    st.subheader("Data Sources")
    
    col1, col2 = st.columns(2)
    
    sources = {}
    
    with col1:
        sources['social_media'] = st.checkbox(
            "Social Media",
            value=True,
            help="WhatsApp, Telegram, Signal, Instagram, Facebook, Snapchat"
        )
        sources['cloud_storage'] = st.checkbox(
            "Cloud Storage",
            value=True,
            help="Google Drive, OneDrive, iCloud"
        )
        sources['encrypted_apps'] = st.checkbox(
            "Encrypted Apps",
            value=True,
            help="Signal, Wickr, and other encrypted messaging"
        )
    
    with col2:
        sources['system_logs'] = st.checkbox(
            "System Logs",
            value=True,
            help="Android logcat, system logs, kernel buffers"
        )
        sources['browser_data'] = st.checkbox(
            "Browser Data",
            value=False,
            help="Browser history, cache, cookies"
        )
        sources['email'] = st.checkbox(
            "Email",
            value=False,
            help="Gmail, Outlook, and other email accounts"
        )
    
    selected_count = sum(1 for v in sources.values() if v)
    st.info(f"📊 {selected_count} data sources selected")
    
    return sources

def render_hybrid_extraction_progress(
    progress_placeholder,
    status_placeholder,
    metrics_placeholder
):
    """Render hybrid extraction progress display"""
    
    def progress_callback(message: str, percentage: int) -> None:
        """Update progress display"""
        try:
            with progress_placeholder.container():
                st.progress(percentage / 100.0)
                st.caption(f"{percentage}% - {message}")
            
            with status_placeholder.container():
                st.info(f"🔄 {message}")
            
            with metrics_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Progress", f"{percentage}%")
                with col2:
                    st.metric("Status", "Running")
                with col3:
                    st.metric("Timestamp", datetime.now().strftime("%H:%M:%S"))
        
        except Exception as e:
            logger.warning(f"Progress update error: {e}")
    
    return progress_callback

def render_hybrid_extraction_results(results: Dict[str, Any]):
    """Render hybrid extraction results"""
    
    st.subheader("Extraction Results")
    
    if results.get('status') == 'error':
        st.error(f"❌ Extraction failed: {results.get('error')}")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Artifacts",
            results.get('total_artifacts', 0),
            help="Combined artifacts from all sources"
        )
    
    with col2:
        st.metric(
            "Completeness",
            f"{results.get('extraction_completeness', 0):.1f}%",
            help="Percentage of device data extracted"
        )
    
    with col3:
        escalation_status = "Yes" if results.get('privilege_escalation_used') else "No"
        st.metric(
            "Escalation Used",
            escalation_status,
            help="Whether privilege escalation was successful"
        )
    
    with col4:
        total_time = results.get('total_duration_seconds', 0)
        st.metric(
            "Duration",
            f"{total_time:.1f}s",
            help="Total extraction time in seconds"
        )
    
    # Detailed breakdown
    st.write("---")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Standard Extraction",
        "Bridge Extraction",
        "Comparison",
        "Details"
    ])
    
    with tab1:
        render_standard_extraction_details(results.get('standard_extraction', {}))
    
    with tab2:
        render_bridge_extraction_details(results.get('bridge_extraction', {}))
    
    with tab3:
        render_extraction_comparison(results)
    
    with tab4:
        render_extraction_details(results)

def render_standard_extraction_details(standard_results: Dict[str, Any]):
    """Render standard extraction details"""
    
    st.write("**Standard Extraction Results**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Artifacts",
            standard_results.get('artifacts', 0)
        )
    
    with col2:
        successful_modules = standard_results.get('modules', {})
        st.metric(
            "Modules",
            len([m for m in successful_modules.values() if m.get('status') == 'success'])
        )
    
    with col3:
        st.metric(
            "Duration",
            f"{standard_results.get('duration_seconds', 0):.1f}s"
        )
    
    # Module breakdown
    st.write("**Module Status**")
    modules = standard_results.get('modules', {})
    
    if modules:
        for module_name, module_result in modules.items():
            status = module_result.get('status', 'unknown')
            artifacts = module_result.get('artifact_count', 0)
            
            if status == 'success':
                st.success(f"✅ {module_name}: {artifacts} artifacts")
            elif status == 'error':
                st.error(f"❌ {module_name}: {module_result.get('error')}")
            else:
                st.warning(f"⚠️ {module_name}: {status}")
    
    # Blocked modules
    blocked = standard_results.get('blocked_modules', [])
    if blocked:
        st.warning("**Blocked Modules (Insufficient Consent)**")
        for blocked_module in blocked:
            st.write(f"• {blocked_module.get('module')}: {blocked_module.get('reason')}")

def render_bridge_extraction_details(bridge_results: Dict[str, Any]):
    """Render bridge extraction details"""
    
    st.write("**Bridge Extraction Results**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Artifacts",
            bridge_results.get('artifacts', 0)
        )
    
    with col2:
        completeness = bridge_results.get('completeness', 0)
        st.metric(
            "Completeness",
            f"{completeness:.1f}%"
        )
    
    with col3:
        st.metric(
            "Duration",
            f"{bridge_results.get('duration_seconds', 0):.1f}s"
        )
    
    # Escalation info
    if bridge_results.get('escalation_used'):
        st.info(f"🔓 Escalation Method: {bridge_results.get('escalation_method', 'Unknown')}")
    
    # Sources breakdown
    st.write("**Data Sources**")
    sources = bridge_results.get('sources', {})
    
    if sources:
        for source_name, source_result in sources.items():
            status = source_result.get('status', 'unknown')
            artifact_count = source_result.get('artifact_count', 0)
            
            if status == 'success':
                st.success(f"✅ {source_name}: {artifact_count} artifacts")
            elif status == 'partial':
                st.warning(f"⚠️ {source_name}: {artifact_count} artifacts (partial)")
            else:
                st.error(f"❌ {source_name}: {source_result.get('error')}")

def render_extraction_comparison(results: Dict[str, Any]):
    """Render extraction method comparison"""
    
    st.write("**Method Comparison**")
    
    standard = results.get('standard_extraction', {})
    bridge = results.get('bridge_extraction', {})
    
    comparison_data = {
        'Metric': [
            'Total Artifacts',
            'Duration (seconds)',
            'Modules/Sources',
            'Escalation Used'
        ],
        'Standard': [
            standard.get('artifacts', 0),
            f"{standard.get('duration_seconds', 0):.1f}",
            len([m for m in standard.get('modules', {}).values() if m.get('status') == 'success']),
            'No'
        ],
        'Bridge': [
            bridge.get('artifacts', 0),
            f"{bridge.get('duration_seconds', 0):.1f}",
            len(bridge.get('sources', {})),
            'Yes' if bridge.get('escalation_used') else 'No'
        ]
    }
    
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)
    
    # Improvement metrics
    st.write("**Improvement**")
    additional_artifacts = bridge.get('artifacts', 0)
    completeness_gain = bridge.get('completeness', 0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Additional Artifacts",
            f"+{additional_artifacts}",
            help="Extra artifacts from bridge extraction"
        )
    with col2:
        st.metric(
            "Completeness Gain",
            f"+{completeness_gain:.1f}%",
            help="Improvement in extraction completeness"
        )

def render_extraction_details(results: Dict[str, Any]):
    """Render detailed extraction information"""
    
    st.write("**Extraction Metadata**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Case ID**: {results.get('case_id')}")
        st.write(f"**Device ID**: {results.get('device_id')}")
        st.write(f"**Extraction ID**: {results.get('extraction_id')}")
    
    with col2:
        st.write(f"**Type**: {results.get('extraction_type', 'hybrid')}")
        st.write(f"**Timestamp**: {results.get('timestamp')}")
        st.write(f"**Status**: {results.get('status')}")
    
    # Full results JSON
    with st.expander("View Full Results (JSON)"):
        st.json(results)

def render_hybrid_extraction_page(
    orchestrator: Any,
    case_id: str,
    device_id: str,
    consent_manager: Optional[Any] = None
):
    """Render complete hybrid extraction page"""
    
    st.header("🔀 Hybrid Extraction")
    
    st.write(
        "Hybrid extraction combines standard extraction with advanced bridge agent methods "
        "for more complete data recovery. Includes privilege escalation and extended sources."
    )
    
    # Configuration section
    st.subheader("Configuration")
    
    enable_escalation, enable_extended = render_hybrid_extraction_options()
    
    escalation_methods = None
    if enable_escalation:
        escalation_methods = render_escalation_method_selector()
    
    extended_sources = {}
    if enable_extended:
        extended_sources = render_extended_sources_selector()
    
    # Start extraction button
    st.write("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        start_extraction = st.button(
            "🚀 Start Hybrid Extraction",
            key="start_hybrid_extraction",
            use_container_width=True
        )
    
    with col2:
        st.write("")  # Spacing
    
    if start_extraction:
        # Create progress placeholders
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        results_placeholder = st.empty()
        
        try:
            # Create hybrid adapter
            adapter = create_hybrid_adapter(orchestrator)
            
            # Create progress callback
            progress_callback = render_hybrid_extraction_progress(
                progress_placeholder,
                status_placeholder,
                metrics_placeholder
            )
            
            # Run hybrid extraction
            with st.spinner("Running hybrid extraction..."):
                results = adapter.extract_all_data_hybrid(
                    case_id=case_id,
                    device_id=device_id,
                    consent_manager=consent_manager,
                    progress_callback=progress_callback,
                    enable_escalation=enable_escalation,
                    enable_extended_sources=enable_extended
                )
            
            # Display results
            with results_placeholder.container():
                render_hybrid_extraction_results(results)
            
            # Save to session state
            st.session_state.hybrid_extraction_results = results
            
            st.success("✅ Hybrid extraction completed successfully!")
        
        except Exception as e:
            logger.error(f"Hybrid extraction error: {e}", exc_info=True)
            st.error(f"❌ Extraction failed: {str(e)}")

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def get_hybrid_extraction_results() -> Optional[Dict[str, Any]]:
    """Get hybrid extraction results from session state"""
    return st.session_state.get('hybrid_extraction_results')

def export_hybrid_results(results: Dict[str, Any], format: str = 'json') -> bytes:
    """Export hybrid extraction results"""
    import json
    
    if format == 'json':
        return json.dumps(results, indent=2, default=str).encode()
    elif format == 'csv':
        import pandas as pd
        df = pd.DataFrame([results])
        return df.to_csv(index=False).encode()
    else:
        raise ValueError(f"Unsupported format: {format}")
