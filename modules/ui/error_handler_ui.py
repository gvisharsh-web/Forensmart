"""
Advanced Error Handler UI for ForenSmart
========================================

Provides interactive error handling UI with:
- Error detection and categorization
- Auto-fix options
- Troubleshooting suggestions
- Error history and patterns
- Recovery options
"""

import streamlit as st
from typing import Dict, Any, Optional
from datetime import datetime
from modules.shared.advanced_error_handler import (
    get_error_handler,
    handle_error_with_fix,
    ErrorSeverity,
    ErrorCategory
)


def render_error_handler_ui():
    """Render advanced error handler UI"""
    
    st.markdown("## 🔧 Error Handler & Troubleshooting")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚨 Current Error",
        "🔍 Error History",
        "📊 Error Patterns",
        "⚙️ Troubleshooting"
    ])
    
    with tab1:
        render_current_error_tab()
    
    with tab2:
        render_error_history_tab()
    
    with tab3:
        render_error_patterns_tab()
    
    with tab4:
        render_troubleshooting_tab()


def render_current_error_tab():
    """Render current error tab"""
    
    st.markdown("### Current Error Status")
    
    # Check if there's an error in session state
    if 'current_error' not in st.session_state:
        st.info("✅ No errors detected. System is running normally.")
        return
    
    error_info = st.session_state['current_error']
    
    # Error summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        severity = error_info.get('severity', 'unknown')
        severity_emoji = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢',
            'info': '🔵'
        }.get(severity, '❓')
        st.metric("Severity", f"{severity_emoji} {severity.upper()}")
    
    with col2:
        category = error_info.get('category', 'unknown')
        st.metric("Category", category.upper())
    
    with col3:
        error_type = error_info.get('type', 'Unknown')
        st.metric("Type", error_type)
    
    with col4:
        timestamp = error_info.get('timestamp', 'Unknown')
        st.metric("Time", timestamp[-8:])
    
    st.divider()
    
    # Error message
    st.markdown("### Error Message")
    error_msg = error_info.get('message', 'No message')
    st.error(f"```\n{error_msg}\n```")
    
    st.divider()
    
    # Available fixes
    st.markdown("### 🔧 Available Fixes")
    
    fixes = error_info.get('fixes', [])
    
    if not fixes:
        st.info("No specific fixes available for this error.")
    else:
        for i, fix in enumerate(fixes):
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{fix['name']}**")
                    st.caption(fix['description'])
                    
                    if fix['auto_fixable']:
                        st.markdown("✅ **Auto-fixable** - Can be applied automatically")
                    else:
                        st.markdown("⚠️ **Manual** - Requires user action")
                    
                    if fix['suggestions']:
                        st.markdown("**Steps:**")
                        for suggestion in fix['suggestions']:
                            st.caption(f"• {suggestion}")
                
                with col2:
                    if fix['auto_fixable']:
                        if st.button("🔧 Apply Fix", key=f"fix_{i}"):
                            st.success("✅ Fix applied! Please retry the operation.")
                            st.session_state['current_error'] = None
                            st.rerun()
                    else:
                        st.info("Manual fix required")
    
    st.divider()
    
    # Troubleshooting suggestions
    st.markdown("### 💡 Troubleshooting Suggestions")
    
    suggestions = error_info.get('suggestions', [])
    
    if suggestions:
        for suggestion in suggestions:
            st.caption(f"• {suggestion}")
    else:
        st.info("No additional suggestions available.")
    
    st.divider()
    
    # Error details
    if st.checkbox("Show detailed error information"):
        st.markdown("### 📋 Detailed Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Error Type:**")
            st.code(error_info.get('type', 'Unknown'))
            
            st.markdown("**Category:**")
            st.code(error_info.get('category', 'Unknown'))
        
        with col2:
            st.markdown("**Severity:**")
            st.code(error_info.get('severity', 'Unknown'))
            
            st.markdown("**Timestamp:**")
            st.code(error_info.get('timestamp', 'Unknown'))
        
        st.markdown("**Traceback:**")
        st.code(error_info.get('traceback', 'No traceback available'))
        
        st.markdown("**Context:**")
        st.json(error_info.get('context', {}))


def render_error_history_tab():
    """Render error history tab"""
    
    st.markdown("### Error History")
    
    handler = get_error_handler()
    history = handler.get_error_history(limit=50)
    
    if not history:
        st.info("No errors in history.")
        return
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            ["critical", "high", "medium", "low", "info"],
            default=["critical", "high", "medium"]
        )
    
    with col2:
        category_filter = st.multiselect(
            "Filter by Category",
            ["device", "extraction", "consent", "approval", "storage", "network", "permission", "validation", "config", "unknown"],
            default=None
        )
    
    with col3:
        limit = st.slider("Show last N errors", 1, 50, 10)
    
    # Filter history
    filtered_history = [
        e for e in history[-limit:]
        if e.get('severity') in severity_filter
        and (not category_filter or e.get('category') in category_filter)
    ]
    
    if not filtered_history:
        st.info("No errors match the selected filters.")
        return
    
    # Display errors
    for error in reversed(filtered_history):
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            
            with col1:
                severity = error.get('severity', 'unknown')
                emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢', 'info': '🔵'}.get(severity, '❓')
                st.markdown(f"**{emoji} {severity.upper()}**")
            
            with col2:
                st.markdown(f"**{error.get('type', 'Unknown')}**")
            
            with col3:
                st.markdown(f"*{error.get('category', 'unknown')}*")
            
            with col4:
                st.caption(error.get('timestamp', 'Unknown')[-8:])
            
            st.caption(error.get('message', 'No message')[:100])
            
            if st.checkbox("View details", key=f"history_{error.get('timestamp')}"):
                st.json(error)


def render_error_patterns_tab():
    """Render error patterns tab"""
    
    st.markdown("### Error Patterns & Analytics")
    
    handler = get_error_handler()
    
    # Get report
    report = handler.get_error_report()
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Errors", report['total_errors'])
    
    with col2:
        patterns = report['error_patterns']
        st.metric("Error Types", len(patterns))
    
    with col3:
        if patterns:
            most_common = max(patterns.items(), key=lambda x: x[1])
            st.metric("Most Common", most_common[0])
    
    st.divider()
    
    # Error patterns chart
    st.markdown("### Error Distribution")
    
    patterns = report['error_patterns']
    
    if patterns:
        # Create chart data
        import pandas as pd
        df = pd.DataFrame(
            list(patterns.items()),
            columns=['Error Type', 'Count']
        ).sort_values('Count', ascending=True)
        
        st.bar_chart(df.set_index('Error Type'))
    else:
        st.info("No error patterns recorded yet.")
    
    st.divider()
    
    # Most common errors
    st.markdown("### Most Common Errors")
    
    most_common = report['most_common']
    
    if most_common:
        for i, (error_type, count) in enumerate(most_common, 1):
            st.markdown(f"{i}. **{error_type}** - {count} occurrences")
    else:
        st.info("No errors recorded yet.")


def render_troubleshooting_tab():
    """Render troubleshooting tab"""
    
    st.markdown("### 🔧 Troubleshooting Guide")
    
    # Common issues
    st.markdown("## Common Issues & Solutions")
    
    issues = {
        "Device Not Found": {
            "symptoms": [
                "ADB not detecting device",
                "Device shows as offline",
                "USB connection issues"
            ],
            "solutions": [
                "1. Disconnect USB cable",
                "2. Wait 5 seconds",
                "3. Reconnect USB cable",
                "4. Accept USB debugging prompt on device",
                "5. Run 'adb devices' to verify",
                "6. Try different USB port",
                "7. Try different USB cable"
            ]
        },
        "No Storage Space": {
            "symptoms": [
                "Extraction fails with storage error",
                "Cannot save artifacts",
                "Disk full error"
            ],
            "solutions": [
                "1. Go to Reports & Storage tab",
                "2. Click Cleanup tab",
                "3. Select old cases to delete",
                "4. Confirm deletion",
                "5. Check available space",
                "6. Retry extraction"
            ]
        },
        "Consent/Approval Issues": {
            "symptoms": [
                "Extraction blocked by consent",
                "Approval not recognized",
                "Permission denied"
            ],
            "solutions": [
                "1. Go to Consent Hub tab",
                "2. Generate approval link",
                "3. Send to nominee",
                "4. Nominee clicks link and approves",
                "5. System detects approval",
                "6. Retry extraction"
            ]
        },
        "Extraction Failures": {
            "symptoms": [
                "Extraction starts but fails",
                "Partial data extracted",
                "Timeout errors"
            ],
            "solutions": [
                "1. Check device is still connected",
                "2. Check storage space",
                "3. Check consent level (STANDARD or LEGAL)",
                "4. Check device battery level",
                "5. Disable screen lock if possible",
                "6. Retry extraction",
                "7. Check logs in Diagnostics"
            ]
        },
        "Network Issues": {
            "symptoms": [
                "Timeout errors",
                "Connection refused",
                "Cannot reach server"
            ],
            "solutions": [
                "1. Check internet connection",
                "2. Check firewall settings",
                "3. Try again in a moment",
                "4. Check if service is down",
                "5. Restart application",
                "6. Contact system administrator"
            ]
        }
    }
    
    # Display issues
    selected_issue = st.selectbox(
        "Select an issue to view solutions",
        list(issues.keys())
    )
    
    if selected_issue:
        issue_data = issues[selected_issue]
        
        st.markdown(f"### {selected_issue}")
        
        st.markdown("**Symptoms:**")
        for symptom in issue_data['symptoms']:
            st.caption(f"• {symptom}")
        
        st.markdown("**Solutions:**")
        for solution in issue_data['solutions']:
            st.caption(solution)
    
    st.divider()
    
    # Quick diagnostics
    st.markdown("## Quick Diagnostics")
    
    if st.button("🔍 Run Diagnostics"):
        with st.spinner("Running diagnostics..."):
            diagnostics = run_diagnostics()
            
            st.markdown("### Diagnostic Results")
            
            for check_name, result in diagnostics.items():
                status = "✅" if result['status'] else "❌"
                st.markdown(f"{status} **{check_name}**: {result['message']}")


def run_diagnostics() -> Dict[str, Dict[str, Any]]:
    """Run system diagnostics"""
    
    diagnostics = {}
    
    # Check ADB
    try:
        import shutil
        adb_path = shutil.which('adb')
        diagnostics['ADB'] = {
            'status': adb_path is not None,
            'message': f"ADB found at {adb_path}" if adb_path else "ADB not found"
        }
    except Exception as e:
        diagnostics['ADB'] = {'status': False, 'message': str(e)}
    
    # Check storage
    try:
        import shutil
        total, used, free = shutil.disk_usage('/')
        free_gb = free / (1024**3)
        diagnostics['Storage'] = {
            'status': free_gb > 1,
            'message': f"{free_gb:.1f} GB free"
        }
    except Exception as e:
        diagnostics['Storage'] = {'status': False, 'message': str(e)}
    
    # Check directories
    try:
        from pathlib import Path
        required_dirs = ['data', 'modules', 'pages']
        all_exist = all(Path(d).exists() for d in required_dirs)
        diagnostics['Directories'] = {
            'status': all_exist,
            'message': "All required directories exist" if all_exist else "Missing directories"
        }
    except Exception as e:
        diagnostics['Directories'] = {'status': False, 'message': str(e)}
    
    # Check permissions
    try:
        from pathlib import Path
        test_file = Path('data/.test')
        test_file.touch()
        test_file.unlink()
        diagnostics['Permissions'] = {
            'status': True,
            'message': "Write permissions OK"
        }
    except Exception as e:
        diagnostics['Permissions'] = {'status': False, 'message': str(e)}
    
    return diagnostics


def show_error_notification(error_info: Dict[str, Any]):
    """Show error notification to user"""
    
    severity = error_info.get('severity', 'unknown')
    message = error_info.get('message', 'An error occurred')
    
    if severity == 'critical':
        st.error(f"🔴 **CRITICAL ERROR**: {message}")
    elif severity == 'high':
        st.error(f"🟠 **ERROR**: {message}")
    elif severity == 'medium':
        st.warning(f"🟡 **WARNING**: {message}")
    else:
        st.info(f"🔵 **INFO**: {message}")
    
    # Store in session for detailed view
    st.session_state['current_error'] = error_info
