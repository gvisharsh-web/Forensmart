"""
ForenSmart Storage Management UI
================================

Modern UI for storage management with:
- Storage analytics and visualization
- Safe case deletion with confirmation
- Selective artifact deletion
- Storage monitoring dashboard
- Cleanup recommendations
- Deletion history

Author: ForenSmart Development Team
"""

import streamlit as st
from typing import Optional, Dict, Any
from datetime import datetime
import json

from modules.storage.manager import (
    StorageManager,
    StorageAnalytics,
    DeletionAudit
)
from modules.consent.models import ConsentManager


def get_consent_manager() -> "ConsentManager":
    """Get consent manager from session state."""
    if 'consent_manager' not in st.session_state:
        st.session_state['consent_manager'] = ConsentManager()
    return st.session_state['consent_manager']



def render_storage_dashboard():
    """Render main storage management dashboard."""
    
    st.markdown("# 💾 Storage Management")
    
    # Get storage info
    total_info = StorageAnalytics.get_total_storage_info()
    
    # Storage overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Storage",
            total_info['total_size_formatted'],
            f"{total_info['case_count']} cases"
        )
    
    with col2:
        st.metric(
            "Artifacts",
            total_info['artifacts_total_formatted']
        )
    
    with col3:
        st.metric(
            "Reports",
            total_info['reports_total_formatted']
        )
    
    with col4:
        st.metric(
            "Consent Data",
            total_info['consent_total_formatted']
        )
    
    st.divider()
    
    # Storage tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Cases by Size",
        "🗑️ Delete Case",
        "🧹 Cleanup",
        "📋 History",
        "⚙️ Tools"
    ])
    
    # ========================================================================
    # Tab 1: Cases by Size
    # ========================================================================
    with tab1:
        st.markdown("### Cases Sorted by Storage Usage")
        
        cases = StorageAnalytics.list_cases_by_size()
        
        if not cases:
            st.info("No cases found")
        else:
            # Display as table
            table_data = []
            for case in cases:
                table_data.append({
                    "Case ID": case['case_id'],
                    "Artifacts": case['artifacts_size_formatted'],
                    "Reports": case['reports_size_formatted'],
                    "Consent": case['consent_size_formatted'],
                    "Total": case['total_size_formatted']
                })
            
            st.dataframe(table_data, use_container_width=True)
            
            # Storage breakdown chart
            st.markdown("### Storage Breakdown")
            
            chart_data = {
                'Case': [c['case_id'] for c in cases[:10]],
                'Artifacts (MB)': [c['artifacts_size'] / (1024*1024) for c in cases[:10]],
                'Reports (MB)': [c['reports_size'] / (1024*1024) for c in cases[:10]],
                'Consent (MB)': [c['consent_size'] / (1024*1024) for c in cases[:10]]
            }
            
            st.bar_chart(
                data={
                    'Artifacts': [c['artifacts_size'] / (1024*1024) for c in cases[:10]],
                    'Reports': [c['reports_size'] / (1024*1024) for c in cases[:10]],
                    'Consent': [c['consent_size'] / (1024*1024) for c in cases[:10]]
                },
                use_container_width=True
            )
    
    # ========================================================================
    # Tab 2: Delete Case
    # ========================================================================
    with tab2:
        st.markdown("### Delete Case & Artifacts")
        st.warning("⚠️ **WARNING**: This action is permanent and cannot be undone!")
        
        # Case selection
        cases = StorageAnalytics.list_cases_by_size()
        case_options = {c['case_id']: c['total_size_formatted'] for c in cases}
        
        if not case_options:
            st.info("No cases available for deletion")
        else:
            selected_case = st.selectbox(
                "Select case to delete",
                options=list(case_options.keys()),
                format_func=lambda x: f"{x} ({case_options[x]})"
            )
            
            if selected_case:
                case_info = StorageAnalytics.get_case_storage_info(selected_case)
                
                # Show what will be deleted
                st.markdown("### What will be deleted:")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if case_info['artifacts_exist']:
                        st.metric("Artifacts", case_info['artifacts_size_formatted'])
                    else:
                        st.metric("Artifacts", "None")
                
                with col2:
                    if case_info['reports_exist']:
                        st.metric("Reports", case_info['reports_size_formatted'])
                    else:
                        st.metric("Reports", "None")
                
                with col3:
                    if case_info['consent_exist']:
                        st.metric("Consent Data", case_info['consent_size_formatted'])
                    else:
                        st.metric("Consent Data", "None")
                
                st.info(f"**Total to be deleted**: {case_info['total_size_formatted']}")
                
                # Deletion options
                st.markdown("### Deletion Options")
                
                delete_option = st.radio(
                    "What to delete?",
                    options=[
                        "Everything (artifacts, reports, consent)",
                        "Artifacts only",
                        "Reports only",
                        "Consent data only"
                    ]
                )
                
                # Confirmation
                st.markdown("### Confirmation")
                
                confirm_text = st.text_input(
                    f"Type '{selected_case}' to confirm deletion:",
                    placeholder="Type case ID here"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🗑️ Delete", key="btn_delete_case"):
                        if confirm_text == selected_case:
                            with st.spinner("Deleting..."):
                                if delete_option == "Everything (artifacts, reports, consent)":
                                    success, message, info = StorageManager.delete_entire_case(selected_case)
                                elif delete_option == "Artifacts only":
                                    success, message, info = StorageManager.delete_artifact_directory(selected_case)
                                elif delete_option == "Reports only":
                                    success, message, info = StorageManager.delete_case_reports(selected_case)
                                else:  # Consent only
                                    success, message, info = StorageManager.delete_case_consent_data(selected_case)
                                
                                if success:
                                    st.success(f"✅ {message}")
                                    st.json(info)
                                else:
                                    st.error(f"❌ {message}")
                        else:
                            st.error("❌ Confirmation text does not match case ID")
                
                with col2:
                    if st.button("Cancel", key="btn_cancel_delete"):
                        st.info("Deletion cancelled")
    
    # ========================================================================
    # Tab 3: Cleanup
    # ========================================================================
    with tab3:
        st.markdown("### Automated Cleanup")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Orphaned Artifacts")
            st.info("Find and delete artifacts without corresponding consent records")
            
            if st.button("🔍 Scan for Orphaned Artifacts"):
                with st.spinner("Scanning..."):
                    files_deleted, size_freed, details = StorageManager.cleanup_orphaned_artifacts(dry_run=True)
                    
                    if details['orphaned_cases']:
                        st.warning(f"Found {len(details['orphaned_cases'])} orphaned cases")
                        
                        for case in details['orphaned_cases']:
                            st.write(f"- {case['case_id']}: {case['size_formatted']}")
                        
                        if st.button("🗑️ Delete Orphaned Artifacts"):
                            with st.spinner("Deleting..."):
                                files_deleted, size_freed, details = StorageManager.cleanup_orphaned_artifacts(dry_run=False)
                                st.success(f"✅ Deleted {files_deleted} orphaned cases, freed {StorageAnalytics.format_size(size_freed)}")
                    else:
                        st.success("✅ No orphaned artifacts found")
        
        with col2:
            st.markdown("#### Old Cases")
            st.info("Find cases that haven't been modified recently")
            
            min_age = st.slider("Minimum age (days)", 1, 365, 30)
            
            if st.button("🔍 Find Old Cases"):
                with st.spinner("Scanning..."):
                    candidates = StorageManager.get_deletion_candidates(min_age_days=min_age)
                    
                    if candidates:
                        st.warning(f"Found {len(candidates)} cases older than {min_age} days")
                        
                        for case in candidates[:10]:
                            with st.expander(f"{case['case_id']} ({case['age_days']} days old)"):
                                st.write(f"**Last Modified**: {case['last_modified']}")
                                st.write(f"**Storage**: {case['storage_info']['total_size_formatted']}")
                                
                                if st.button(f"Delete {case['case_id']}", key=f"delete_old_{case['case_id']}"):
                                    success, message, info = StorageManager.delete_entire_case(case['case_id'])
                                    if success:
                                        st.success(message)
                                    else:
                                        st.error(message)
                    else:
                        st.success(f"✅ No cases older than {min_age} days")
    
    # ========================================================================
    # Tab 4: History
    # ========================================================================
    with tab4:
        st.markdown("### Deletion History")
        
        history = DeletionAudit.get_deletion_history()
        
        if not history:
            st.info("No deletion history")
        else:
            # Filter options
            col1, col2 = st.columns(2)
            
            with col1:
                filter_case = st.text_input("Filter by case ID (optional)")
            
            with col2:
                show_count = st.slider("Show last N deletions", 1, len(history), 10)
            
            # Filter history
            filtered_history = history
            if filter_case:
                filtered_history = [h for h in history if filter_case in h.get('case_id', '')]
            
            # Show history
            for entry in filtered_history[-show_count:]:
                with st.expander(f"{entry['case_id']} - {entry['timestamp'][:10]}"):
                    st.write(f"**Timestamp**: {entry['timestamp']}")
                    st.write(f"**Reason**: {entry.get('reason', 'N/A')}")
                    st.write(f"**Status**: {entry.get('status', 'N/A')}")
                    
                    deleted_items = entry.get('deleted_items', {})
                    if deleted_items:
                        st.write("**Deleted Items**:")
                        st.json(deleted_items)
    
    # ========================================================================
    # Tab 5: Tools
    # ========================================================================
    with tab5:
        st.markdown("### Storage Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Storage Report")
            
            if st.button("📊 Generate Storage Report"):
                report = {
                    'generated_at': datetime.now().isoformat(),
                    'total_storage': StorageAnalytics.get_total_storage_info(),
                    'cases': StorageAnalytics.list_cases_by_size()
                }
                
                st.json(report)
                
                # Download option
                report_json = json.dumps(report, indent=2, default=str)
                st.download_button(
                    label="📥 Download Report",
                    data=report_json,
                    file_name=f"storage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            st.markdown("#### Deletion History Export")
            
            if st.button("📋 Export Deletion History"):
                history = DeletionAudit.get_deletion_history()
                history_json = json.dumps(history, indent=2, default=str)
                
                st.download_button(
                    label="📥 Download History",
                    data=history_json,
                    file_name=f"deletion_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )


def render_storage_quick_view(case_id: str):
    """Render quick storage view for a specific case."""
    
    st.markdown("### Storage Usage")
    
    case_info = StorageAnalytics.get_case_storage_info(case_id)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Artifacts", case_info['artifacts_size_formatted'])
    
    with col2:
        st.metric("Reports", case_info['reports_size_formatted'])
    
    with col3:
        st.metric("Total", case_info['total_size_formatted'])
    
    # Quick delete button
    if st.button(f"🗑️ Delete {case_id}", key=f"quick_delete_{case_id}"):
        st.session_state[f'confirm_delete_{case_id}'] = True
    
    if st.session_state.get(f'confirm_delete_{case_id}'):
        st.warning("⚠️ Are you sure? This cannot be undone!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Yes, Delete", key=f"confirm_yes_{case_id}"):
                success, message, info = StorageManager.delete_entire_case(case_id)
                if success:
                    st.success(message)
                    st.session_state[f'confirm_delete_{case_id}'] = False
                else:
                    st.error(message)
        
        with col2:
            if st.button("❌ Cancel", key=f"confirm_no_{case_id}"):
                st.session_state[f'confirm_delete_{case_id}'] = False
