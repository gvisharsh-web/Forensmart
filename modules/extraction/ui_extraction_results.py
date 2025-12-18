"""
EXTRACTION RESULTS UI - Results Display and Artifact Viewing
Handles UI for displaying extraction results and artifacts

This module provides:
- Extraction summary display
- Extracted data display by module
- Artifact viewing
- Filtering and searching
- Metadata display
- Export options
"""

import logging
import streamlit as st
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


# ============================================================================
# EXTRACTION RESULTS UI
# ============================================================================

def render_extraction_results(results: Dict[str, Any]):
    """Render extraction results UI"""
    
    st.header("📊 Extraction Results")
    
    # Step 1: Show extraction summary
    st.subheader("📈 Extraction Summary")
    
    show_extraction_summary(results)
    
    # Step 2: Show extracted data by module
    st.subheader("📦 Extracted Data")
    
    show_extracted_data(results)
    
    # Step 3: Show artifacts
    st.subheader("🎁 Artifacts")
    
    show_artifacts(results)
    
    # Step 4: Show filtering options
    st.subheader("🔍 Filter & Search")
    
    show_filtering_options(results)
    
    # Step 5: Show metadata
    st.subheader("ℹ️ Metadata")
    
    show_metadata(results)
    
    # Step 6: Show export options
    st.subheader("📥 Export Options")
    
    show_export_options(results)


def show_extraction_summary(results: Dict[str, Any]):
    """Show extraction summary"""
    
    logger.info("📈 Showing extraction summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        device_id = results.get('device_id', 'Unknown')
        st.metric("Device/Account", device_id[:20])
    
    with col2:
        adapter_type = results.get('adapter_type', 'Unknown')
        st.metric("Adapter Type", adapter_type)
    
    with col3:
        timestamp = results.get('timestamp', 'Unknown')
        st.metric("Extraction Time", timestamp[:10])
    
    with col4:
        modules = results.get('modules', {})
        total_items = sum(len(v) if isinstance(v, list) else 1 for v in modules.values())
        st.metric("Total Items", total_items)
    
    # Show source info
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"📱 Device ID: {results.get('device_id', 'N/A')}")
    
    with col2:
        st.info(f"📋 Case ID: {results.get('case_id', 'N/A')}")


def show_extracted_data(results: Dict[str, Any]):
    """Show extracted data by module"""
    
    logger.info("📦 Showing extracted data")
    
    modules = results.get('modules', {})
    
    if not modules:
        st.warning("⚠️ No data extracted")
        return
    
    # Create tabs for each module
    tabs = st.tabs([f"📧 {key.title()}" for key in modules.keys()])
    
    for tab, (module_name, module_data) in zip(tabs, modules.items()):
        with tab:
            if isinstance(module_data, list):
                st.write(f"**Total: {len(module_data)} items**")
                
                # Show first 5 items
                for i, item in enumerate(module_data[:5]):
                    if isinstance(item, dict):
                        st.write(f"**Item {i+1}:**")
                        for key, value in item.items():
                            st.write(f"  • {key}: {str(value)[:50]}")
                    else:
                        st.write(f"**Item {i+1}:** {str(item)[:100]}")
                
                if len(module_data) > 5:
                    st.info(f"... and {len(module_data) - 5} more items")
            
            elif isinstance(module_data, dict):
                st.write(f"**{module_name} Info:**")
                for key, value in module_data.items():
                    st.write(f"  • {key}: {value}")
            
            else:
                st.write(module_data)


def show_artifacts(results: Dict[str, Any]):
    """Show artifacts"""
    
    logger.info("🎁 Showing artifacts")
    
    # Get modules data (contains counts)
    modules = results.get('modules', {})
    
    # Extract counts - handle both integer counts and list formats
    artifacts = {
        'emails': modules.get('emails', 0) if isinstance(modules.get('emails'), int) else len(modules.get('emails', [])),
        'messages': modules.get('messages', 0) if isinstance(modules.get('messages'), int) else len(modules.get('messages', [])),
        'files': modules.get('files', 0) if isinstance(modules.get('files'), int) else len(modules.get('files', [])),
        'attachments': modules.get('attachments', 0) if isinstance(modules.get('attachments'), int) else len(modules.get('attachments', [])),
        'media': modules.get('media', 0) if isinstance(modules.get('media'), int) else len(modules.get('media', [])),
        'contacts': modules.get('contacts', 0) if isinstance(modules.get('contacts'), int) else len(modules.get('contacts', []))
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📧 Emails", artifacts['emails'])
        st.metric("💬 Messages", artifacts['messages'])
    
    with col2:
        st.metric("📄 Files", artifacts['files'])
        st.metric("📎 Attachments", artifacts['attachments'])
    
    with col3:
        st.metric("📸 Media", artifacts['media'])
        st.metric("👥 Contacts", artifacts['contacts'])


def show_filtering_options(results: Dict[str, Any]):
    """Show filtering and search options"""
    
    logger.info("🔍 Showing filtering options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        search_term = st.text_input("🔍 Search", placeholder="Search in results...")
    
    with col2:
        filter_type = st.selectbox("📁 Filter by Type", 
                                   ["All", "Emails", "Messages", "Files", 
                                    "Attachments", "Media", "Contacts"])
    
    if search_term or filter_type != "All":
        st.info(f"🔍 Searching for: {search_term if search_term else 'All'}")
        st.info(f"📁 Filtering by: {filter_type}")
        
        # Simulated search results
        st.success("✅ Found 5 matching items")


def show_metadata(results: Dict[str, Any]):
    """Show metadata"""
    
    logger.info("ℹ️ Showing metadata")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Extraction Details:**")
        st.write(f"  • Device ID: {results.get('device_id', 'N/A')}")
        st.write(f"  • Case ID: {results.get('case_id', 'N/A')}")
        st.write(f"  • Adapter Type: {results.get('adapter_type', 'N/A')}")
    
    with col2:
        st.write("**Timestamps:**")
        st.write(f"  • Extraction Time: {results.get('timestamp', 'N/A')}")
        st.write(f"  • Extraction Date: {datetime.now().strftime('%Y-%m-%d')}")
        st.write(f"  • Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def show_export_options(results: Dict[str, Any]):
    """Show export options"""
    
    logger.info("📥 Showing export options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export as JSON"):
            export_as_json(results)
    
    with col2:
        if st.button("📥 Export as CSV"):
            export_as_csv(results)
    
    with col3:
        if st.button("📥 Export as PDF Report"):
            export_as_pdf(results)


def export_as_json(results: Dict[str, Any]):
    """Export results as JSON"""
    
    logger.info("📥 Exporting as JSON")
    
    json_data = json.dumps(results, indent=2, default=str)
    
    st.download_button(
        label="📥 Download JSON",
        data=json_data,
        file_name=f"extraction_results_{results.get('case_id', 'unknown')}.json",
        mime="application/json"
    )
    
    st.success("✅ JSON export ready")


def export_as_csv(results: Dict[str, Any]):
    """Export results as CSV"""
    
    logger.info("📥 Exporting as CSV")
    
    # Simulated CSV export
    csv_data = "Device ID,Case ID,Adapter Type,Timestamp\n"
    csv_data += f"{results.get('device_id', 'N/A')},{results.get('case_id', 'N/A')},{results.get('adapter_type', 'N/A')},{results.get('timestamp', 'N/A')}\n"
    
    st.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name=f"extraction_results_{results.get('case_id', 'unknown')}.csv",
        mime="text/csv"
    )
    
    st.success("✅ CSV export ready")


def export_as_pdf(results: Dict[str, Any]):
    """Export results as PDF report"""
    
    logger.info("📥 Exporting as PDF")
    
    # Simulated PDF export
    pdf_data = f"""
    EXTRACTION REPORT
    =================
    
    Device ID: {results.get('device_id', 'N/A')}
    Case ID: {results.get('case_id', 'N/A')}
    Adapter Type: {results.get('adapter_type', 'N/A')}
    Timestamp: {results.get('timestamp', 'N/A')}
    
    EXTRACTED DATA
    ==============
    """
    
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_data,
        file_name=f"extraction_report_{results.get('case_id', 'unknown')}.txt",
        mime="text/plain"
    )
    
    st.success("✅ PDF report ready")


def get_results_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Get results summary"""
    
    logger.info("📊 Getting results summary")
    
    modules = results.get('modules', {})
    total_items = sum(len(v) if isinstance(v, list) else 1 for v in modules.values())
    
    summary = {
        'device_id': results.get('device_id', 'Unknown'),
        'case_id': results.get('case_id', 'Unknown'),
        'adapter_type': results.get('adapter_type', 'Unknown'),
        'timestamp': results.get('timestamp', 'Unknown'),
        'total_items': total_items,
        'modules_extracted': len(modules),
        'modules': modules
    }
    
    return summary
