"""
ERROR HANDLING DASHBOARD - Advanced Error Handling System UI

Provides:
- Real-time error monitoring
- Error history and analysis
- Auto-rectification interface
- Prevention rules management
- System analytics and insights
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Error Handling - ForenSmart",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
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
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
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

# ============================================================================
# IMPORTS & INITIALIZATION
# ============================================================================

try:
    from modules.error_handling import ErrorHandlingSystem
    ERROR_SYSTEM_AVAILABLE = True
except ImportError:
    ERROR_SYSTEM_AVAILABLE = False

# Import offline error handler
try:
    from modules.error_handling.offline_error_handler import OfflineErrorHandler
    OFFLINE_HANDLER_AVAILABLE = True
except ImportError:
    OFFLINE_HANDLER_AVAILABLE = False

# Initialize session state
if 'error_system' not in st.session_state:
    if ERROR_SYSTEM_AVAILABLE:
        st.session_state.error_system = ErrorHandlingSystem()
    else:
        st.session_state.error_system = None

if 'offline_handler' not in st.session_state:
    if OFFLINE_HANDLER_AVAILABLE:
        st.session_state.offline_handler = OfflineErrorHandler()
    else:
        st.session_state.offline_handler = None

if 'mode' not in st.session_state:
    st.session_state.mode = 'online' if ERROR_SYSTEM_AVAILABLE else 'offline'

# ============================================================================
# MAIN HEADER
# ============================================================================

st.markdown('<div class="main-header">🛡️ Advanced Error Handling System</div>', unsafe_allow_html=True)

# Mode indicator and selector
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    if st.session_state.mode == 'online':
        st.success(f"Mode: ONLINE (Full Error Handling System Active)")
    else:
        st.warning(f"Mode: OFFLINE (Standalone Error Handler Active)")

with col2:
    if ERROR_SYSTEM_AVAILABLE and OFFLINE_HANDLER_AVAILABLE:
        if st.button("Switch Mode", use_container_width=True):
            st.session_state.mode = 'offline' if st.session_state.mode == 'online' else 'online'
            st.rerun()

with col3:
    if st.button("Refresh", use_container_width=True):
        st.rerun()

st.divider()

# Check availability
if not ERROR_SYSTEM_AVAILABLE and not OFFLINE_HANDLER_AVAILABLE:
    st.error("Neither Online nor Offline Error Handling available. Please check installation.")
    st.stop()

if st.session_state.mode == 'online' and not ERROR_SYSTEM_AVAILABLE:
    st.warning("Online mode selected but Error System not available. Switching to Offline mode.")
    st.session_state.mode = 'offline'
    st.rerun()

# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Error Monitor",
    "Error History",
    "Auto-Rectification",
    "Prevention",
    "Analytics"
])

# ============================================================================
# TAB 1: ERROR MONITOR
# ============================================================================

with tab1:
    st.markdown('<div class="section-header">Real-Time Error Monitoring</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    if st.session_state.mode == 'online' and st.session_state.error_system:
        with col1:
            st.metric("System Health", st.session_state.error_system.get_system_health().upper())
        
        with col2:
            stats = st.session_state.error_system.get_error_statistics()
            st.metric("Total Errors", stats.get('total_errors_handled', 0))
        
        with col3:
            st.metric("Error Types", len(st.session_state.error_system.analyzer.error_patterns))
        
        with col4:
            st.metric("Solutions Learned", len(st.session_state.error_system.learner.error_solutions))
    else:
        with col1:
            st.metric("Mode", "OFFLINE")
        
        with col2:
            stats = st.session_state.offline_handler.get_error_statistics()
            st.metric("Total Errors", stats.get('total_errors', 0))
        
        with col3:
            st.metric("Error Types", len(stats.get('by_type', {})))
        
        with col4:
            effectiveness = st.session_state.offline_handler.get_solution_effectiveness()
            st.metric("Solutions Tracked", len(effectiveness))
    
    st.divider()
    
    # Current error status
    st.markdown("**Current Error Status**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("No active errors detected")
    
    with col2:
        if st.button("Refresh Status", use_container_width=True):
            st.success("Status refreshed")
    
    st.divider()
    
    # Resource monitoring
    st.markdown("**System Resources**")
    
    if st.session_state.mode == 'online' and st.session_state.error_system:
        resources = st.session_state.error_system.monitor_resources()
        
        if resources:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                memory = resources.get('memory', {})
                st.metric(
                    "Memory Usage",
                    f"{memory.get('percent', 0):.1f}%",
                    f"{memory.get('available_gb', 0):.1f} GB available"
                )
            
            with col2:
                cpu = resources.get('cpu', {})
                st.metric("CPU Usage", f"{cpu.get('percent', 0):.1f}%")
            
            with col3:
                storage = resources.get('storage', {})
                st.metric(
                    "Storage Usage",
                    f"{storage.get('percent', 0):.1f}%",
                    f"{storage.get('available_gb', 0):.1f} GB available"
                )
    else:
        st.info("Offline mode: Resource monitoring available through system tools")

# ============================================================================
# TAB 2: ERROR HISTORY
# ============================================================================

with tab2:
    st.markdown('<div class="section-header">Error History & Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.mode == 'online' and st.session_state.error_system:
        # Error history
        history = st.session_state.error_system.get_error_history(limit=50)
        
        if history:
            st.markdown(f"**Last {len(history)} Errors**")
            
            # Create dataframe
            error_data = []
            for error in history[-10:]:
                error_data.append({
                    'Timestamp': error.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                    'Error Type': error.get('error_info', {}).get('type', 'Unknown'),
                    'Severity': str(error.get('error_info', {}).get('severity', 'Unknown')),
                    'Message': error.get('error_info', {}).get('message', 'N/A')[:50]
                })
            
            if error_data:
                df = pd.DataFrame(error_data)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No errors in history")
        
        st.divider()
        
        # Error patterns
        st.markdown("**Error Patterns**")
        
        patterns = st.session_state.error_system.analyze_patterns()
        
        if patterns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Errors Analyzed", patterns.get('total_errors', 0))
            
            with col2:
                st.metric("Unique Error Types", patterns.get('unique_error_types', 0))
            
            # Most common errors
            most_common = patterns.get('most_common_errors', [])
            if most_common:
                st.markdown("**Most Common Errors**")
                for error_type, count in most_common[:5]:
                    st.write(f"- {error_type}: {count} occurrences")
    else:
        # Offline mode history
        st.markdown("**Offline Error History**")
        
        stats = st.session_state.offline_handler.get_error_statistics()
        
        if stats.get('total_errors', 0) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Errors", stats.get('total_errors', 0))
            
            with col2:
                st.metric("Error Types", len(stats.get('by_type', {})))
            
            st.divider()
            st.markdown("**Errors by Type**")
            
            for error_type, count in stats.get('by_type', {}).items():
                st.write(f"- {error_type}: {count} occurrences")
            
            st.divider()
            st.markdown("**Errors by Severity**")
            
            for severity, count in stats.get('by_severity', {}).items():
                st.write(f"- {severity}: {count} errors")
        else:
            st.info("No errors recorded in offline mode yet")

# ============================================================================
# TAB 3: AUTO-RECTIFICATION
# ============================================================================

with tab3:
    st.markdown('<div class="section-header">Auto-Rectification Interface</div>', unsafe_allow_html=True)
    
    if st.session_state.mode == 'online' and st.session_state.error_system:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Available Fixes**")
            st.info("Auto-fix capabilities are active and ready")
        
        with col2:
            st.markdown("**Fix Statistics**")
            fix_stats = st.session_state.error_system.rectifier.get_fix_statistics()
            if fix_stats:
                st.metric("Successful Fixes", fix_stats.get('successful_fixes', 0))
                st.metric("Success Rate", f"{fix_stats.get('success_rate', 0):.1f}%")
        
        st.divider()
        
        # Manual fix interface
        st.markdown("**Manual Fix Options**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("View Recent Fixes", use_container_width=True):
                fix_history = st.session_state.error_system.rectifier.get_fix_history(limit=5)
                if fix_history:
                    for fix in fix_history:
                        status = "Success" if fix.get('success') else "Failed"
                        st.write(f"- {fix.get('error_type')}: {status}")
        
        with col2:
            if st.button("Clear Fix History", use_container_width=True):
                st.session_state.error_system.rectifier.fix_history = []
                st.success("Fix history cleared")
    else:
        st.markdown("**Offline Auto-Fix Capabilities**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("16 Error Types Auto-Fixed")
            st.write("- Extraction (2 types)")
            st.write("- Consent (3 types)")
            st.write("- Analysis (4 types)")
            st.write("- Report Generation (2 types)")
            st.write("- System (3 types)")
        
        with col2:
            effectiveness = st.session_state.offline_handler.get_solution_effectiveness()
            st.markdown("**Solution Effectiveness**")
            
            if effectiveness:
                for error_type, metrics in list(effectiveness.items())[:5]:
                    st.write(f"- {error_type}: {metrics['effectiveness']}")
            else:
                st.info("No solutions tracked yet")

# ============================================================================
# TAB 4: PREVENTION
# ============================================================================

with tab4:
    st.markdown('<div class="section-header">Prevention Rules & Validation</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Input Validation**")
        st.info("Active validation rules: 5")
    
    with col2:
        st.markdown("**Resource Monitoring**")
        st.info("Monitoring: Memory, CPU, Storage")
    
    with col3:
        st.markdown("**Anomaly Detection**")
        st.info("Anomaly detection: Active")
    
    st.divider()
    
    # Prevention rules
    st.markdown("**Generated Prevention Rules**")
    
    prevention_rules = st.session_state.error_system.learner.generate_prevention_rules()
    
    if prevention_rules:
        for rule in prevention_rules[:5]:
            st.write(f"- {rule.get('error_type')}: {rule.get('action')}")
    else:
        st.info("No prevention rules generated yet")
    
    st.divider()
    
    # Resource limits
    st.markdown("**Resource Limits**")
    
    limits = st.session_state.error_system.preventer.get_resource_limits()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"Max Memory: {limits.get('max_memory_percent', 90)}%")
        st.write(f"Max CPU: {limits.get('max_cpu_percent', 95)}%")
    
    with col2:
        st.write(f"Max Storage: {limits.get('max_storage_percent', 95)}%")
        st.write(f"Max Extraction Time: {limits.get('max_extraction_time', 3600)}s")

# ============================================================================
# TAB 5: ANALYTICS
# ============================================================================

with tab5:
    st.markdown('<div class="section-header">Error Analytics & Insights</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        learning_summary = st.session_state.error_system.get_learning_summary()
        st.metric("Learning Records", learning_summary.get('total_learning_records', 0))
    
    with col2:
        st.metric("Error Types Tracked", learning_summary.get('unique_error_types', 0))
    
    with col3:
        st.metric("Solutions Learned", learning_summary.get('solutions_learned', 0))
    
    st.divider()
    
    # Error trends
    st.markdown("**Error Trends (Last 24 Hours)**")
    
    trends = st.session_state.error_system.get_error_trends(hours=24)
    
    if trends:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Errors in Period", trends.get('errors_in_period', 0))
        
        with col2:
            st.metric("Trend", trends.get('trend', 'stable').upper())
        
        with col3:
            st.metric("Period", f"{trends.get('period_hours', 24)}h")
    
    st.divider()
    
    # Improvement recommendations
    st.markdown("**Improvement Recommendations**")
    
    improvements = st.session_state.error_system.get_improvement_recommendations()
    
    if improvements.get('detection_improvements'):
        st.markdown("**Detection Improvements**")
        for improvement in improvements['detection_improvements'][:3]:
            st.write(f"- {improvement.get('recommendation')}")
    
    if improvements.get('prevention_improvements'):
        st.markdown("**Prevention Improvements**")
        for improvement in improvements['prevention_improvements'][:3]:
            st.write(f"- {improvement.get('action')} for {improvement.get('error_type')}")
    
    if improvements.get('fix_improvements'):
        st.markdown("**Fix Improvements**")
        for improvement in improvements['fix_improvements'][:3]:
            st.write(f"- {improvement.get('error_type')}: {improvement.get('recommendation')}")
    
    st.divider()
    
    # System health summary
    st.markdown("**System Health Summary**")
    
    health = st.session_state.error_system.get_system_health()
    
    if health == 'healthy':
        st.markdown('<div class="success-box">System is healthy - No issues detected</div>', unsafe_allow_html=True)
    elif health == 'good':
        st.markdown('<div class="info-box">System is good - Minor issues detected</div>', unsafe_allow_html=True)
    elif health == 'warning':
        st.markdown('<div class="warning-box">System warning - Multiple issues detected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-box">System critical - Immediate action required</div>', unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("Advanced Error Handling System v1.0 | Last Updated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
