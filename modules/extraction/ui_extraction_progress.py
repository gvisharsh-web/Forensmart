"""
EXTRACTION PROGRESS UI - Real-time Progress Tracking
Handles UI for showing extraction progress and real-time updates

This module provides:
- Progress bar display
- Current operation display
- Extracted items counter
- Extraction speed calculation
- Time estimation
- Error/warning display
"""

import logging
import streamlit as st
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


# ============================================================================
# EXTRACTION PROGRESS UI
# ============================================================================

def render_extraction_progress(adapter_type: str, case_id: str):
    """Render extraction progress UI"""
    
    st.header("⏳ Extraction Progress")
    
    # Initialize session state
    if 'extraction_start_time' not in st.session_state:
        st.session_state.extraction_start_time = datetime.now()
    
    if 'extraction_items' not in st.session_state:
        st.session_state.extraction_items = {
            'emails': 0,
            'messages': 0,
            'files': 0,
            'attachments': 0,
            'contacts': 0,
            'media': 0
        }
    
    # Step 1: Show extraction status
    st.subheader("📊 Extraction Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status", "🟢 Running")
    
    with col2:
        elapsed = datetime.now() - st.session_state.extraction_start_time
        st.metric("Elapsed Time", f"{elapsed.seconds}s")
    
    with col3:
        st.metric("Adapter", adapter_type)
    
    with col4:
        st.metric("Case ID", case_id)
    
    # Step 2: Show progress bar
    st.subheader("📈 Overall Progress")
    
    progress_value = show_progress_bar()
    st.progress(progress_value)
    
    # Step 3: Show current operation
    st.subheader("🔄 Current Operation")
    
    show_current_operation(adapter_type)
    
    # Step 4: Show extracted items
    st.subheader("📦 Extracted Items")
    
    show_extracted_items()
    
    # Step 5: Show extraction speed
    st.subheader("⚡ Extraction Speed")
    
    show_extraction_speed()
    
    # Step 6: Show time estimation
    st.subheader("⏱️ Time Estimation")
    
    show_time_remaining()
    
    # Step 7: Show errors/warnings
    st.subheader("⚠️ Errors & Warnings")
    
    show_errors_warnings()
    
    # Step 8: Show extraction log
    st.subheader("📋 Extraction Log")
    
    show_extraction_log()


def show_progress_bar() -> float:
    """Show and calculate progress bar"""
    
    logger.info("📊 Calculating progress...")
    
    # Simulated progress calculation
    total_items = 100
    extracted_items = sum(st.session_state.extraction_items.values())
    
    progress = min(extracted_items / total_items, 1.0)
    
    st.write(f"Progress: {int(progress * 100)}%")
    
    return progress


def show_current_operation(adapter_type: str):
    """Show current operation being performed"""
    
    logger.info(f"🔄 Showing current operation for {adapter_type}")
    
    operations = {
        'Android': [
            '🔌 Connecting to Android device...',
            '📂 Reading device storage...',
            '📧 Extracting emails...',
            '💬 Extracting messages...',
            '📱 Extracting contacts...',
            '📸 Extracting media...'
        ],
        'iOS': [
            '🔌 Connecting to iOS device...',
            '💾 Creating backup...',
            '📧 Extracting emails...',
            '💬 Extracting iMessages...',
            '📱 Extracting contacts...',
            '📸 Extracting media...'
        ],
        'GoogleDrive': [
            '🔐 Authenticating with Google...',
            '☁️ Fetching file list...',
            '📁 Extracting folders...',
            '📄 Extracting files...',
            '📊 Extracting metadata...'
        ],
        'Email': [
            '🔐 Authenticating with email provider...',
            '📧 Fetching email list...',
            '📨 Extracting emails...',
            '📎 Extracting attachments...',
            '👥 Extracting contacts...'
        ],
        'WhatsApp': [
            '🔌 Connecting to device...',
            '💬 Reading WhatsApp database...',
            '💬 Extracting messages...',
            '👥 Extracting contacts...',
            '📸 Extracting media...',
            '📞 Extracting call logs...'
        ]
    }
    
    current_ops = operations.get(adapter_type, operations['Android'])
    
    # Show current operation with animation
    col1, col2 = st.columns([3, 1])
    
    with col1:
        for i, op in enumerate(current_ops):
            if i == 0:
                st.write(f"**{op}**")
            else:
                st.write(f"⏳ {op}")
    
    with col2:
        st.write("⏳ In Progress")


def show_extracted_items():
    """Show extracted items count"""
    
    logger.info("📦 Showing extracted items")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Emails", st.session_state.extraction_items['emails'])
        st.metric("Messages", st.session_state.extraction_items['messages'])
    
    with col2:
        st.metric("Files", st.session_state.extraction_items['files'])
        st.metric("Attachments", st.session_state.extraction_items['attachments'])
    
    with col3:
        st.metric("Contacts", st.session_state.extraction_items['contacts'])
        st.metric("Media", st.session_state.extraction_items['media'])


def show_extraction_speed():
    """Show extraction speed"""
    
    logger.info("⚡ Calculating extraction speed")
    
    elapsed = datetime.now() - st.session_state.extraction_start_time
    total_items = sum(st.session_state.extraction_items.values())
    
    if elapsed.total_seconds() > 0:
        speed = total_items / elapsed.total_seconds()
    else:
        speed = 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Items/Second", f"{speed:.2f}")
    
    with col2:
        st.metric("Total Items", total_items)


def show_time_remaining():
    """Show estimated time remaining"""
    
    logger.info("⏱️ Calculating time remaining")
    
    elapsed = datetime.now() - st.session_state.extraction_start_time
    total_items = sum(st.session_state.extraction_items.values())
    
    # Estimate: assume 100 total items to extract
    total_to_extract = 100
    remaining_items = total_to_extract - total_items
    
    if elapsed.total_seconds() > 0 and total_items > 0:
        speed = total_items / elapsed.total_seconds()
        if speed > 0:
            time_remaining = remaining_items / speed
        else:
            time_remaining = 0
    else:
        time_remaining = 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Time Remaining", f"{int(time_remaining)}s")
    
    with col2:
        estimated_total = elapsed.total_seconds() + time_remaining
        st.metric("Estimated Total Time", f"{int(estimated_total)}s")


def show_errors_warnings():
    """Show errors and warnings"""
    
    logger.info("⚠️ Checking for errors and warnings")
    
    # Simulated errors and warnings
    errors = []
    warnings = []
    
    if not errors and not warnings:
        st.success("✅ No errors or warnings")
    else:
        if errors:
            st.error("❌ Errors:")
            for error in errors:
                st.write(f"  • {error}")
        
        if warnings:
            st.warning("⚠️ Warnings:")
            for warning in warnings:
                st.write(f"  • {warning}")


def show_extraction_log():
    """Show extraction log"""
    
    logger.info("📋 Showing extraction log")
    
    log_entries = [
        "✅ Connected to device",
        "✅ Started extracting emails",
        "✅ Extracted 10 emails",
        "✅ Started extracting messages",
        "✅ Extracted 25 messages",
        "✅ Started extracting attachments",
        "✅ Extracted 5 attachments",
        "⏳ Processing media files...",
    ]
    
    with st.expander("📋 View Full Log", expanded=False):
        for entry in log_entries:
            st.write(entry)


def update_extraction_progress(item_type: str, count: int):
    """Update extraction progress"""
    
    logger.info(f"📊 Updating progress: {item_type} = {count}")
    
    if item_type in st.session_state.extraction_items:
        st.session_state.extraction_items[item_type] = count


def get_extraction_status() -> Dict[str, Any]:
    """Get current extraction status"""
    
    elapsed = datetime.now() - st.session_state.extraction_start_time
    total_items = sum(st.session_state.extraction_items.values())
    
    status = {
        'elapsed_time': elapsed.total_seconds(),
        'total_items': total_items,
        'items_breakdown': st.session_state.extraction_items,
        'status': 'running'
    }
    
    return status
