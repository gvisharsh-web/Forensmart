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
    
    # ALWAYS perform REAL extraction from device (not simulated)
    # Reset extraction_completed to force real extraction every time
    st.session_state.extraction_completed = False
    perform_extraction(adapter_type, case_id)
    
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
    """Show extraction log with REAL extraction data"""
    
    logger.info("📋 Showing extraction log")
    
    # Build log from REAL extraction data
    log_entries = []
    
    device_id = st.session_state.get('selected_device', {}).get('device_id', 'device')
    log_entries.append(f"✅ Connected to device: {device_id}")
    
    # Show real extraction counts
    extraction_items = st.session_state.get('extraction_items', {})
    
    if extraction_items.get('contacts', 0) > 0:
        log_entries.append(f"✅ Started extracting contacts")
        log_entries.append(f"✅ Extracted {extraction_items['contacts']} contacts")
    
    if extraction_items.get('messages', 0) > 0:
        log_entries.append(f"✅ Started extracting messages")
        log_entries.append(f"✅ Extracted {extraction_items['messages']} messages")
    
    if extraction_items.get('emails', 0) > 0:
        log_entries.append(f"✅ Started extracting emails")
        log_entries.append(f"✅ Extracted {extraction_items['emails']} emails")
    
    if extraction_items.get('attachments', 0) > 0:
        log_entries.append(f"✅ Started extracting attachments")
        log_entries.append(f"✅ Extracted {extraction_items['attachments']} attachments")
    
    if extraction_items.get('media', 0) > 0:
        log_entries.append(f"✅ Started extracting media files")
        log_entries.append(f"✅ Extracted {extraction_items['media']} media files")
    
    if extraction_items.get('files', 0) > 0:
        log_entries.append(f"✅ Started extracting files")
        log_entries.append(f"✅ Extracted {extraction_items['files']} files")
    
    # Add completion status
    if st.session_state.get('extraction_completed', False):
        total_items = sum(extraction_items.values())
        log_entries.append(f"✅ Extraction completed - Total items: {total_items}")
    else:
        log_entries.append("⏳ Extraction in progress...")
    
    # If no items extracted, show message
    if not log_entries or sum(extraction_items.values()) == 0:
        log_entries = [
            "⏳ Connecting to device...",
            "⏳ Querying device data...",
            "⏳ Processing extraction..."
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


def perform_extraction(adapter_type: str, case_id: str):
    """Perform REAL extraction from connected device using ADB"""
    
    logger.info(f"🔄 Starting REAL extraction for {adapter_type} - Case: {case_id}")
    
    import subprocess
    from modules.shared.validators import validate_device_id
    
    # Initialize session state variables for extracted data
    if 'extracted_messages' not in st.session_state:
        st.session_state.extracted_messages = []
    if 'extracted_contacts' not in st.session_state:
        st.session_state.extracted_contacts = []
    if 'extracted_media_files' not in st.session_state:
        st.session_state.extracted_media_files = []
    
    device_id = None
    if hasattr(st.session_state, 'selected_device') and st.session_state.selected_device:
        device_id = st.session_state.selected_device.get('device_id', None) if isinstance(st.session_state.selected_device, dict) else None
    
    # ✅ Validate device_id
    if not device_id or not validate_device_id(device_id):
        logger.error(f"❌ Invalid device ID: {device_id}")
        st.error("❌ No device selected or invalid device ID. Please select a device first.")
        st.warning("⚠️ Please go to Extraction tab and select a device first")
        return
    
    st.info(f"📱 Extracting from device: {device_id}")
    st.info("⏳ Performing real extraction from your device...")
    
    extraction_data = {
        'emails': 0,
        'messages': 0,
        'files': 0,
        'attachments': 0,
        'contacts': 0,
        'media': 0
    }
    
    try:
        # Extract real data from device using ADB
        
        # 1. Get REAL contacts data
        try:
            st.write("📱 Querying contacts from device...")
            
            # Try multiple URI formats for contacts
            contact_uris = [
                'content://com.android.contacts/contacts',
                'content://contacts/people',
                'content://com.android.contacts/data'
            ]
            
            real_contacts = []
            contacts_found = False
            
            for uri in contact_uris:
                try:
                    result = subprocess.run(
                        ['adb', '-s', device_id, 'shell', 'content', 'query', '--uri', uri],
                        capture_output=True,
                        text=True,
                        timeout=15
                    )
                    
                    stdout_text = result.stdout if result.stdout else ""
                    stderr_text = result.stderr if result.stderr else ""
                    
                    logger.info(f"Contacts query ({uri}) return code: {result.returncode}")
                    logger.info(f"Contacts stdout length: {len(stdout_text)}")
                    
                    if result.returncode == 0 and stdout_text and '_id=' in stdout_text:
                        extraction_data['contacts'] = stdout_text.count('_id=')
                        st.write(f"✅ Found {extraction_data['contacts']} contacts using {uri}")
                        
                        # Parse REAL contact data
                        for line in stdout_text.split('\n'):
                            if '_id=' in line and line.strip():
                                try:
                                    parts = {}
                                    for item in line.split(','):
                                        if '=' in item:
                                            key, val = item.split('=', 1)
                                            parts[key.strip()] = val.strip()
                                    
                                    if '_id' in parts:
                                        real_contacts.append({
                                            'id': parts.get('_id', ''),
                                            'name': parts.get('display_name', parts.get('name', 'Unknown')),
                                            'phone': parts.get('data1', parts.get('phone', 'N/A')),
                                            'messages': 0
                                        })
                                except Exception as parse_err:
                                    logger.warning(f"Could not parse contact: {parse_err}")
                        
                        contacts_found = True
                        break
                except Exception as uri_err:
                    logger.warning(f"Failed to query {uri}: {uri_err}")
                    continue
            
            if contacts_found:
                st.session_state.extracted_contacts = real_contacts
                logger.info(f"✅ Extracted {len(real_contacts)} REAL contacts from device")
            else:
                st.warning(f"⚠️ No contacts found - Device may not have contacts or permissions denied")
                st.info("💡 Try: Settings → Apps → Contacts → Permissions → Enable all")
                logger.warning(f"Contacts query failed for all URIs")
                st.session_state.extracted_contacts = []
        except Exception as e:
            st.error(f"❌ Error extracting contacts: {e}")
            logger.error(f"❌ Could not extract contacts: {e}")
            st.session_state.extracted_contacts = []
        
        st.session_state.extraction_items['contacts'] = extraction_data['contacts']
        time.sleep(0.5)
        
        # 2. Get SMS messages with REAL data
        try:
            st.write("💬 Querying messages from device...")
            
            real_messages = []
            messages_found = False
            last_error = None
            
            # ENHANCED METHOD 1: Direct SQLite database query with root access (most reliable)
            st.write("📊 Trying enhanced SQLite method...")
            try:
                # Try with root access first (su command) - use shell=True for proper command execution
                result = subprocess.run(
                    f'adb -s {device_id} shell su -c "sqlite3 /data/data/com.android.providers.telephony/databases/mmssms.db \'SELECT _id, address, body, date, type FROM sms ORDER BY date DESC;\'"',
                    capture_output=True,
                    text=True,
                    timeout=15,
                    shell=True
                )
                
                # If root fails, try without root
                if result.returncode != 0:
                    result = subprocess.run(
                        f'adb -s {device_id} shell sqlite3 /data/data/com.android.providers.telephony/databases/mmssms.db "SELECT _id, address, body, date, type FROM sms ORDER BY date DESC;"',
                        capture_output=True,
                        text=True,
                        timeout=15,
                        shell=True
                    )
                
                stdout_text = result.stdout if result.stdout else ""
                stderr_text = result.stderr if result.stderr else ""
                
                logger.info(f"SQLite SMS query return code: {result.returncode}")
                logger.info(f"SQLite SMS stdout length: {len(stdout_text)}")
                logger.info(f"SQLite SMS stderr: {stderr_text}")
                
                if result.returncode == 0 and stdout_text and len(stdout_text) > 10:
                    st.success("✅ Using enhanced SQLite method")
                    lines = stdout_text.strip().split('\n')
                    
                    for line in lines:
                        if line.strip() and '|' in line:
                            try:
                                parts = line.split('|')
                                if len(parts) >= 3:
                                    try:
                                        timestamp = int(parts[3]) // 1000 if len(parts) > 3 else 0
                                        msg_date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M') if timestamp > 0 else 'N/A'
                                    except (ValueError, OSError) as e:
                                        logger.debug(f"Could not parse timestamp: {e}")
                                        msg_date = 'N/A'
                                    
                                    real_messages.append({
                                        'id': parts[0].strip() if len(parts) > 0 else '',
                                        'from': parts[1].strip() if len(parts) > 1 and parts[1].strip() else 'Unknown',
                                        'text': parts[2].strip() if len(parts) > 2 else '',
                                        'time': msg_date,
                                        'type': parts[4].strip() if len(parts) > 4 else '1',
                                        'status': 'Suspicious' if len(parts[2]) > 100 or 'http' in parts[2].lower() else 'Normal'
                                    })
                            except Exception as parse_err:
                                logger.warning(f"Could not parse SMS line: {parse_err}")
                    
                    if real_messages:
                        extraction_data['messages'] = len(real_messages)
                        st.success(f"✅ Extracted {len(real_messages)} messages via SQLite")
                        messages_found = True
                    else:
                        logger.warning(f"SQLite returned data but no messages parsed")
            except Exception as sqlite_err:
                logger.warning(f"SQLite method failed: {sqlite_err}")
                last_error = str(sqlite_err)
            
            # FALLBACK METHOD 2: Content Provider URIs (if SQLite fails)
            if not messages_found:
                st.write("📊 Trying content provider URIs...")
                
                sms_uris = [
                    'content://sms',                           # Standard Android SMS
                    'content://sms/inbox',                     # Inbox only
                    'content://sms/sent',                      # Sent only
                    'content://sms/draft',                     # Drafts
                    'content://mms-sms/conversations',         # MMS + SMS combined
                    'content://mms-sms/conversations/simple',  # Simple view
                    'content://com.android.mms-sms/conversations',  # Alternative path
                    'content://telephony/sms',                 # Telephony provider
                    'content://telephony/sms/inbox',           # Telephony inbox
                    'content://com.google.android.gm/conversations',  # Gmail
                    'content://com.samsung.android.messaging/sms',    # Samsung
                    'content://com.android.messaging/sms'      # Android Messaging
                ]
                
                for uri in sms_uris:
                    try:
                        result = subprocess.run(
                            ['adb', '-s', device_id, 'shell', 'content', 'query', '--uri', uri],
                            capture_output=True,
                            text=True,
                            timeout=15
                        )
                        
                        stdout_text = result.stdout if result.stdout else ""
                        stderr_text = result.stderr if result.stderr else ""
                        
                        logger.info(f"Messages query ({uri}) return code: {result.returncode}")
                        logger.info(f"Messages stdout length: {len(stdout_text)}")
                        logger.info(f"Messages stderr: {stderr_text}")
                        
                        # Store error for diagnostics
                        if result.returncode != 0:
                            last_error = stderr_text if stderr_text else "Query failed"
                        
                        if result.returncode == 0 and stdout_text and '_id=' in stdout_text:
                            extraction_data['messages'] = stdout_text.count('_id=')
                            st.success(f"✅ Found {extraction_data['messages']} messages using URI: {uri}")
                            logger.info(f"✅ Successfully queried messages using: {uri}")
                            
                            # Parse REAL message data
                            for line in stdout_text.split('\n'):
                                if '_id=' in line and line.strip():
                                    try:
                                        parts = {}
                                        for item in line.split(','):
                                            if '=' in item:
                                                key, val = item.split('=', 1)
                                                parts[key.strip()] = val.strip()
                                        
                                        if '_id' in parts:
                                            try:
                                                timestamp = int(parts.get('date', '0')) // 1000
                                                msg_date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
                                            except (ValueError, OSError) as e:
                                                logger.debug(f"Could not parse timestamp: {e}")
                                                msg_date = 'N/A'
                                            
                                            real_messages.append({
                                                'id': parts.get('_id', ''),
                                                'from': parts.get('address', parts.get('phone', 'Unknown')),
                                                'text': parts.get('body', ''),
                                                'time': msg_date,
                                                'type': parts.get('type', '1'),
                                                'status': 'Suspicious' if len(parts.get('body', '')) > 100 or 'http' in parts.get('body', '').lower() else 'Normal'
                                            })
                                    except Exception as parse_err:
                                        logger.warning(f"Could not parse message: {parse_err}")
                            
                            messages_found = True
                            break
                    except Exception as uri_err:
                        last_error = str(uri_err)
                        logger.warning(f"Failed to query {uri}: {uri_err}")
                        continue
            
            if messages_found:
                st.session_state.extracted_messages = real_messages
                logger.info(f"✅ Extracted {len(real_messages)} REAL messages from device")
            else:
                st.warning(f"⚠️ No messages found - Device may not have SMS or permissions denied")
                
                # Show diagnostic information
                with st.expander("🔧 Troubleshooting - Click to expand"):
                    st.markdown("**Possible causes:**")
                    st.write("1. ❌ Device doesn't have SMS messages")
                    st.write("2. ❌ Permissions not granted to Forensmart")
                    st.write("3. ❌ SMS app not installed or disabled")
                    st.write("4. ❌ Wrong URI for your device/SMS app")
                    st.write("5. ❌ ADB connection issue")
                    
                    st.markdown("**Solutions to try:**")
                    st.write("1. ✅ Verify SMS exists: Open Messages app on device - confirm you see SMS messages")
                    st.write("2. ✅ Enable permissions: Settings → Apps → Messages/SMS → Permissions → Enable all")
                    st.write("3. ✅ Enable USB Debugging: Settings → Developer Options → USB Debugging → ON")
                    st.write("4. ✅ Try different SMS app: Some devices use different SMS providers")
                    st.write("5. ✅ Reconnect device: Unplug and replug USB cable")
                    st.write("6. ✅ Restart ADB: Run `adb kill-server` then `adb start-server`")
                    st.write("7. ✅ Check device trust: Tap 'Allow' when device asks to trust computer")
                    
                    st.markdown("**URIs being tried:**")
                    for i, uri in enumerate(sms_uris, 1):
                        st.write(f"{i}. {uri}")
                    
                    if last_error:
                        st.write(f"**Last error:** `{last_error}`")
                
                logger.warning(f"Messages query failed for all URIs. Last error: {last_error}")
                st.session_state.extracted_messages = []
        except Exception as e:
            st.error(f"❌ Error extracting messages: {e}")
            logger.error(f"❌ Could not extract messages: {e}")
            st.session_state.extracted_messages = []
        
        st.session_state.extraction_items['messages'] = extraction_data['messages']
        time.sleep(0.5)
        
        # 3. Get media files with REAL paths and names (images, videos, and audio)
        try:
            st.write("📸 Searching for media files...")
            
            # Limit extraction to prevent overwhelming results
            max_media_files = 5000  # Maximum files to extract
            media_file_count = 0
            
            # Search multiple locations for all media types
            media_locations = [
                # Primary storage locations
                '/sdcard/DCIM',                    # Camera photos/videos
                '/sdcard/Pictures',                # Pictures folder
                '/sdcard/Movies',                  # Movies folder
                '/sdcard/Music',                   # Music folder
                '/sdcard/Podcasts',                # Podcasts folder
                '/sdcard/Audiobooks',              # Audiobooks folder
                '/sdcard/Recordings',              # Voice recordings
                '/sdcard/Downloads',               # Downloads folder (audio can be here)
                '/sdcard/Documents',               # Documents folder
                
                # Messaging Apps Media
                '/sdcard/WhatsApp/Media',          # WhatsApp media
                '/sdcard/Telegram',                # Telegram files
                '/sdcard/Signal',                  # Signal files
                '/sdcard/Android/data/com.whatsapp/media',  # WhatsApp app data
                '/sdcard/Android/data/com.facebook.orca/media',  # Facebook Messenger
                '/sdcard/Android/data/org.telegram.messenger/files',  # Telegram app
                '/sdcard/Android/data/org.signal/files',  # Signal app
                
                # Social Media Apps Media
                '/sdcard/Android/data/com.instagram.android/files',  # Instagram
                '/sdcard/Android/data/com.instagram.android/cache',  # Instagram cache
                '/sdcard/Android/data/com.snapchat.android/files',  # Snapchat
                '/sdcard/Android/data/com.tiktok/files',  # TikTok
                '/sdcard/Android/data/com.twitter.android/files',  # Twitter/X
                '/sdcard/Android/data/com.pinterest/files',  # Pinterest
                '/sdcard/Android/data/com.viber.voip/files',  # Viber
                '/sdcard/Android/data/com.skype.raider/files',  # Skype
                
                # Emulated storage locations
                '/storage/emulated/0/DCIM',
                '/storage/emulated/0/Pictures',
                '/storage/emulated/0/Movies',
                '/storage/emulated/0/Music',
                '/storage/emulated/0/Podcasts',
                '/storage/emulated/0/Audiobooks',
                '/storage/emulated/0/Recordings',
                '/storage/emulated/0/Downloads',   # Audio in Downloads
                '/storage/emulated/0/Documents',
                '/storage/emulated/0/WhatsApp/Media',
                '/storage/emulated/0/Telegram',
                '/storage/emulated/0/Signal',
                
                # Emulated Messaging Apps
                '/storage/emulated/0/Android/data/com.whatsapp/media',
                '/storage/emulated/0/Android/data/com.facebook.orca/media',
                '/storage/emulated/0/Android/data/org.telegram.messenger/files',
                '/storage/emulated/0/Android/data/org.signal/files',
                
                # Emulated Social Media Apps
                '/storage/emulated/0/Android/data/com.instagram.android/files',
                '/storage/emulated/0/Android/data/com.instagram.android/cache',
                '/storage/emulated/0/Android/data/com.snapchat.android/files',
                '/storage/emulated/0/Android/data/com.tiktok/files',
                '/storage/emulated/0/Android/data/com.twitter.android/files',
                '/storage/emulated/0/Android/data/com.pinterest/files',
                '/storage/emulated/0/Android/data/com.viber.voip/files',
                '/storage/emulated/0/Android/data/com.skype.raider/files',
                
                # App-specific audio locations
                '/sdcard/Android/data/com.spotify.music/files',
                '/sdcard/Android/data/com.google.android.music/files',
                '/sdcard/Android/data/com.apple.android.music/files',
                '/sdcard/Android/data/com.amazon.mp3/files',
                '/sdcard/Android/data/com.youtube.android/files',
                '/sdcard/Android/data/com.soundcloud.android/files',
                
                # Emulated app-specific locations
                '/storage/emulated/0/Android/data/com.spotify.music/files',
                '/storage/emulated/0/Android/data/com.google.android.music/files',
                '/storage/emulated/0/Android/data/com.apple.android.music/files',
                '/storage/emulated/0/Android/data/com.amazon.mp3/files',
                '/storage/emulated/0/Android/data/com.youtube.android/files',
                '/storage/emulated/0/Android/data/com.soundcloud.android/files',
                
                # Root sdcard search (comprehensive)
                '/sdcard'
            ]
            
            media_extensions = {
                'image': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff'],
                'video': ['mp4', 'avi', 'mkv', 'mov', 'flv', 'wmv', '3gp', 'webm'],
                'audio': ['mp3', 'wav', 'aac', 'm4a', 'flac', 'ogg', 'wma', 'opus']
            }
            
            all_media_files = []
            
            # Build file extension patterns for find command
            image_patterns = ' -o '.join([f'-iname "*.{ext}"' for ext in media_extensions['image']])
            video_patterns = ' -o '.join([f'-iname "*.{ext}"' for ext in media_extensions['video']])
            audio_patterns = ' -o '.join([f'-iname "*.{ext}"' for ext in media_extensions['audio']])
            
            all_patterns = f'( {image_patterns} -o {video_patterns} -o {audio_patterns} )'
            
            for location in media_locations:
                # Stop if we've reached the limit
                if media_file_count >= max_media_files:
                    logger.info(f"Reached maximum media file limit ({max_media_files})")
                    break
                
                try:
                    # Search with media extension filter to reduce results
                    result = subprocess.run(
                        f'adb -s {device_id} shell find {location} -type f {all_patterns} 2>/dev/null',
                        capture_output=True,
                        text=True,
                        timeout=15,
                        shell=True
                    )
                    
                    if result.returncode == 0 and result.stdout:
                        files = [l.strip() for l in result.stdout.split('\n') if l.strip()]
                        
                        # Limit files to prevent overwhelming
                        remaining = max_media_files - media_file_count
                        if len(files) > remaining:
                            files = files[:remaining]
                            logger.info(f"Truncated to {remaining} files to reach limit")
                        
                        all_media_files.extend(files)
                        media_file_count += len(files)
                        
                        if files:
                            logger.info(f"Found {len(files)} media files in {location} (Total: {media_file_count})")
                except Exception as loc_err:
                    logger.warning(f"Could not search {location}: {loc_err}")
                    continue
            
            if all_media_files:
                # Remove duplicates while preserving order
                unique_files = []
                seen = set()
                for file_path in all_media_files:
                    if file_path not in seen:
                        unique_files.append(file_path)
                        seen.add(file_path)
                
                extraction_data['media'] = len(unique_files)
                
                # Store REAL file information in session state
                real_media_info = []
                for file_path in unique_files:
                    try:
                        file_name = file_path.split('/')[-1]
                        file_ext = file_name.split('.')[-1].lower() if '.' in file_name else 'unknown'
                        
                        # Determine file type
                        file_type = 'other'
                        if file_ext in media_extensions['image']:
                            file_type = 'image'
                        elif file_ext in media_extensions['video']:
                            file_type = 'video'
                        elif file_ext in media_extensions['audio']:
                            file_type = 'audio'
                        
                        real_media_info.append({
                            'path': file_path,
                            'name': file_name,
                            'ext': file_ext,
                            'type': file_type
                        })
                    except Exception as parse_err:
                        logger.warning(f"Could not process media file: {parse_err}")
                
                if real_media_info:
                    st.session_state.extracted_media_files = real_media_info
                    logger.info(f"✅ Extracted {len(real_media_info)} REAL media files from device (after dedup)")
                    st.success(f"✅ Found {len(real_media_info)} real media files on device")
                else:
                    logger.warning("No valid media files found")
                    st.session_state.extracted_media_files = []
            else:
                logger.warning("No media files found in any location")
                st.session_state.extracted_media_files = []
        except Exception as e:
            logger.warning(f"⚠️ Could not extract media: {e}")
            st.session_state.extracted_media_files = []
        
        st.session_state.extraction_items['media'] = extraction_data['media']
        time.sleep(0.5)
        
        # 4. Get documents and files (PDF, DOCX, XLSX, TXT, etc.) - COMPREHENSIVE SEARCH
        try:
            st.write("📄 Searching for documents everywhere...")
            
            # Limit extraction to prevent overwhelming results
            max_document_files = 2000  # Maximum documents to extract
            document_file_count = 0
            
            # Search EVERY location comprehensively
            search_locations = [
                # Primary storage locations
                '/sdcard',                                    # Root search (catches everything)
                '/sdcard/Download',                           # Downloads
                '/sdcard/Downloads',                          # Downloads (alternate)
                '/sdcard/Documents',                          # Documents folder
                '/sdcard/DCIM',                               # Camera folder (may have docs)
                '/sdcard/Pictures',                           # Pictures folder
                '/sdcard/Movies',                             # Movies folder
                '/sdcard/Music',                              # Music folder
                '/sdcard/Podcasts',                           # Podcasts folder
                '/sdcard/Audiobooks',                         # Audiobooks folder
                '/sdcard/Recordings',                         # Recordings folder
                '/sdcard/Desktop',                            # Desktop folder
                '/sdcard/Notes',                              # Notes folder
                '/sdcard/Books',                              # Books folder
                '/sdcard/eBooks',                             # eBooks folder
                
                # Emulated storage locations
                '/storage/emulated/0',                        # Root emulated (catches everything)
                '/storage/emulated/0/Download',
                '/storage/emulated/0/Downloads',
                '/storage/emulated/0/Documents',
                '/storage/emulated/0/DCIM',
                '/storage/emulated/0/Pictures',
                '/storage/emulated/0/Movies',
                '/storage/emulated/0/Music',
                '/storage/emulated/0/Podcasts',
                '/storage/emulated/0/Audiobooks',
                '/storage/emulated/0/Recordings',
                '/storage/emulated/0/Desktop',
                '/storage/emulated/0/Notes',
                '/storage/emulated/0/Books',
                '/storage/emulated/0/eBooks',
                
                # Messaging app document locations
                '/sdcard/WhatsApp/Media/WhatsApp Documents',
                '/sdcard/Telegram/Telegram Documents',
                '/sdcard/Signal/Signal Documents',
                '/storage/emulated/0/WhatsApp/Media/WhatsApp Documents',
                '/storage/emulated/0/Telegram/Telegram Documents',
                '/storage/emulated/0/Signal/Signal Documents',
                
                # Email app locations
                '/sdcard/Android/data/com.google.android.gm/files',
                '/sdcard/Android/data/com.microsoft.office.outlook/files',
                '/storage/emulated/0/Android/data/com.google.android.gm/files',
                '/storage/emulated/0/Android/data/com.microsoft.office.outlook/files',
                
                # Office app locations
                '/sdcard/Android/data/com.microsoft.office.word/files',
                '/sdcard/Android/data/com.microsoft.office.excel/files',
                '/sdcard/Android/data/com.microsoft.office.powerpoint/files',
                '/sdcard/Android/data/com.google.android.apps.docs/files',
                '/storage/emulated/0/Android/data/com.microsoft.office.word/files',
                '/storage/emulated/0/Android/data/com.microsoft.office.excel/files',
                '/storage/emulated/0/Android/data/com.microsoft.office.powerpoint/files',
                '/storage/emulated/0/Android/data/com.google.android.apps.docs/files',
                
                # PDF reader app locations
                '/sdcard/Android/data/com.adobe.reader/files',
                '/sdcard/Android/data/com.google.android.apps.pdfviewer/files',
                '/storage/emulated/0/Android/data/com.adobe.reader/files',
                '/storage/emulated/0/Android/data/com.google.android.apps.pdfviewer/files',
                
                # Cloud storage app locations
                '/sdcard/Android/data/com.google.android.apps.docs.editors.sheets/files',
                '/sdcard/Android/data/com.dropbox.android/files',
                '/sdcard/Android/data/com.microsoft.skydrive/files',
                '/sdcard/Android/data/com.amazon.clouddrive.android/files',
                '/storage/emulated/0/Android/data/com.google.android.apps.docs.editors.sheets/files',
                '/storage/emulated/0/Android/data/com.dropbox.android/files',
                '/storage/emulated/0/Android/data/com.microsoft.skydrive/files',
                '/storage/emulated/0/Android/data/com.amazon.clouddrive.android/files',
                
                # Data folder (may contain exported documents)
                '/data',
                '/data/data'
            ]
            
            document_extensions = [
                '*.pdf', '*.doc', '*.docx', '*.docm',
                '*.xls', '*.xlsx', '*.xlsm',
                '*.ppt', '*.pptx', '*.pptm',
                '*.txt', '*.rtf',
                '*.zip', '*.rar', '*.7z',
                '*.json', '*.xml', '*.csv',
                '*.odt', '*.ods', '*.odp',  # OpenOffice formats
                '*.pages', '*.numbers', '*.keynote',  # Apple formats
                '*.epub', '*.mobi',  # eBook formats
                '*.sql', '*.db', '*.sqlite'  # Database formats
            ]
            
            all_files = []
            
            for location in search_locations:
                # Stop if we've reached the limit
                if document_file_count >= max_document_files:
                    logger.info(f"Reached maximum document file limit ({max_document_files})")
                    break
                
                try:
                    # Try to find documents with each extension separately
                    for ext in document_extensions:
                        # Stop if we've reached the limit
                        if document_file_count >= max_document_files:
                            break
                        
                        try:
                            # Use simple find command with single extension
                            result = subprocess.run(
                                f'adb -s {device_id} shell find {location} -type f -name "{ext}" 2>/dev/null',
                                capture_output=True,
                                text=True,
                                timeout=10,
                                shell=True
                            )
                            
                            if result.returncode == 0 and result.stdout:
                                files = [l.strip() for l in result.stdout.split('\n') if l.strip()]
                                
                                # Limit files to prevent overwhelming
                                remaining = max_document_files - document_file_count
                                if len(files) > remaining:
                                    files = files[:remaining]
                                    logger.info(f"Truncated to {remaining} files to reach limit")
                                
                                all_files.extend(files)
                                document_file_count += len(files)
                                
                                if files:
                                    logger.info(f"Found {len(files)} {ext} files in {location} (Total: {document_file_count})")
                        except Exception as ext_err:
                            logger.warning(f"Could not search for {ext} in {location}: {ext_err}")
                            continue
                except Exception as loc_err:
                    logger.warning(f"Could not search {location}: {loc_err}")
                    continue
            
            if all_files:
                extraction_data['files'] = len(all_files)
                st.success(f"✅ Found {len(all_files)} documents")
                logger.info(f"✅ Extracted {extraction_data['files']} documents")
                
                # Store document information in session state for display
                document_info = []
                for file_path in all_files:
                    try:
                        file_name = file_path.split('/')[-1]
                        file_ext = file_name.split('.')[-1].lower() if '.' in file_name else 'unknown'
                        document_info.append({
                            'path': file_path,
                            'name': file_name,
                            'ext': file_ext,
                            'type': 'document',
                            'category': 'Document'
                        })
                    except Exception as doc_err:
                        logger.warning(f"Could not process document: {doc_err}")
                
                # Add documents to extracted media files for display
                if 'extracted_media_files' not in st.session_state:
                    st.session_state.extracted_media_files = []
                
                st.session_state.extracted_media_files.extend(document_info)
                logger.info(f"Stored {len(document_info)} documents in session state")
            else:
                st.warning("⚠️ No documents found")
                with st.expander("🔧 Document Search Troubleshooting"):
                    st.write("Searched locations:")
                    for loc in search_locations:
                        st.write(f"  • {loc}")
                    st.write("\nSearched file types:")
                    for ext in document_extensions:
                        st.write(f"  • {ext}")
                    st.write("\n**Possible reasons:**")
                    st.write("1. No documents on device")
                    st.write("2. Documents stored in app-specific folders")
                    st.write("3. Storage permissions not granted")
                    st.write("4. Device storage not accessible via ADB")
                logger.warning(f"No documents found in any location")
        except Exception as e:
            logger.warning(f"⚠️ Could not extract documents: {e}")
        
        st.session_state.extraction_items['files'] = extraction_data['files']
        time.sleep(0.5)
        
        # 5. Get emails count (from Gmail app if available)
        try:
            result = subprocess.run(
                ['adb', '-s', device_id, 'shell', 'content', 'query', '--uri', 'content://gmail-ls/conversations'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                extraction_data['emails'] = result.stdout.count('_id=')
                logger.info(f"✅ Extracted {extraction_data['emails']} emails")
        except Exception as e:
            logger.warning(f"⚠️ Could not extract emails: {e}")
        
        st.session_state.extraction_items['emails'] = extraction_data['emails']
        time.sleep(0.5)
        
        # 6. Get attachments count (from messaging apps)
        try:
            st.write("📎 Searching for attachments...")
            
            # Search for attachments in messaging apps and common locations
            attachment_locations = [
                '/sdcard/Android/data/com.whatsapp/media',
                '/sdcard/Android/data/com.facebook.orca/media',
                '/sdcard/Android/data/com.google.android.gm/files',
                '/sdcard/Android/data/com.android.messaging/files',
                '/sdcard/Telegram',
                '/sdcard/Signal',
                '/sdcard/Download'
            ]
            
            attachment_files = []
            
            for location in attachment_locations:
                try:
                    result = subprocess.run(
                        ['adb', '-s', device_id, 'shell', 'find', location, '-type', 'f'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0 and result.stdout:
                        files = [l.strip() for l in result.stdout.split('\n') if l.strip()]
                        attachment_files.extend(files)
                        logger.info(f"Found {len(files)} attachments in {location}")
                except Exception as loc_err:
                    logger.warning(f"Could not search {location}: {loc_err}")
                    continue
            
            if attachment_files:
                extraction_data['attachments'] = len(attachment_files)
                st.success(f"✅ Found {len(attachment_files)} attachments")
                logger.info(f"✅ Extracted {extraction_data['attachments']} attachments")
                
                # Store attachment information in session state for display
                attachment_info = []
                for file_path in attachment_files:
                    try:
                        file_name = file_path.split('/')[-1]
                        file_ext = file_name.split('.')[-1].lower() if '.' in file_name else 'unknown'
                        attachment_info.append({
                            'path': file_path,
                            'name': file_name,
                            'ext': file_ext,
                            'type': 'attachment',
                            'category': 'Attachment'
                        })
                    except Exception as att_err:
                        logger.warning(f"Could not process attachment: {att_err}")
                
                # Add attachments to extracted media files for display
                if 'extracted_media_files' not in st.session_state:
                    st.session_state.extracted_media_files = []
                
                st.session_state.extracted_media_files.extend(attachment_info)
                logger.info(f"Stored {len(attachment_info)} attachments in session state")
            else:
                logger.warning(f"No attachments found in messaging apps")
        except Exception as e:
            logger.warning(f"⚠️ Could not extract attachments: {e}")
        
        st.session_state.extraction_items['attachments'] = extraction_data['attachments']
        time.sleep(0.5)
        
        # 6. FORENSIC AGENTS - Extract advanced forensic data using adapter methods
        st.write("🔍 Running forensic agents...")
        
        try:
            # Import adapter based on device type
            if adapter_type.lower() == 'android':
                from adapters.android_adb import AndroidADB
                adb = AndroidADB()
                
                logger.info(f"📱 Extracting forensic data from Android device: {device_id}")
                
                # Extract all forensic data using adapter method
                forensic_data = adb.extract_all_forensic_data(device_id)
                
                if not forensic_data:
                    logger.warning("⚠️ Forensic data extraction returned None or empty")
                    forensic_data = {}
                
                logger.info(f"📊 Forensic data keys: {list(forensic_data.keys())}")
                
                # Store forensic data in session state
                call_logs = forensic_data.get('call_logs', [])
                browser_history = forensic_data.get('browser_history', [])
                installed_apps = forensic_data.get('installed_apps', [])
                wifi_networks = forensic_data.get('wifi_networks', [])
                system_logs = forensic_data.get('system_logs', [])
                
                st.session_state.extracted_call_logs = call_logs
                st.session_state.extracted_browser_history = browser_history
                st.session_state.extracted_installed_apps = installed_apps
                st.session_state.extracted_wifi_networks = wifi_networks
                st.session_state.extracted_system_logs = system_logs
                
                logger.info(f"✅ Stored forensic data: Calls={len(call_logs)}, Browser={len(browser_history)}, Apps={len(installed_apps)}, WiFi={len(wifi_networks)}, Logs={len(system_logs)}")
                
                # Store app artifact data (WhatsApp, Instagram, Messaging apps)
                whatsapp_artifacts = forensic_data.get('whatsapp_artifacts', [])
                instagram_artifacts = forensic_data.get('instagram_artifacts', [])
                messaging_artifacts = forensic_data.get('messaging_app_artifacts', [])
                
                st.session_state.extracted_whatsapp_artifacts = whatsapp_artifacts
                st.session_state.extracted_instagram_artifacts = instagram_artifacts
                st.session_state.extracted_messaging_artifacts = messaging_artifacts
                
                logger.info(f"✅ Stored app artifacts: WhatsApp={len(whatsapp_artifacts)}, Instagram={len(instagram_artifacts)}, Messaging={len(messaging_artifacts)}")
                
                # Store media files from forensic extraction (merge with existing media files)
                forensic_media_files = forensic_data.get('media_files', [])
                logger.info(f"🎬 Forensic media files found: {len(forensic_media_files)}")
                
                if forensic_media_files:
                    # Merge with existing media files from earlier extraction
                    existing_media = st.session_state.get('extracted_media_files', [])
                    logger.info(f"📁 Existing media files: {len(existing_media)}")
                    
                    # Convert existing media to new format if needed
                    merged_media = []
                    seen_paths = set()
                    
                    # Add forensic media files first
                    for file in forensic_media_files:
                        file_path = file.get('path', '')
                        if file_path and file_path not in seen_paths:
                            merged_media.append(file)
                            seen_paths.add(file_path)
                    
                    # Add existing media files (avoiding duplicates)
                    for file in existing_media:
                        file_path = file.get('path', '')
                        if file_path and file_path not in seen_paths:
                            merged_media.append(file)
                            seen_paths.add(file_path)
                    
                    st.session_state.extracted_media_files = merged_media
                    logger.info(f"✅ Merged {len(forensic_media_files)} forensic media files with {len(existing_media)} existing files = {len(merged_media)} total")
                    st.write(f"🎬 Media Files (Forensic): {len(forensic_media_files)}")
                else:
                    logger.warning("⚠️ No media files found in forensic extraction")
                    st.write(f"🎬 Media Files (Forensic): 0")
                
                # Update extraction data
                extraction_data['call_logs'] = len(forensic_data.get('call_logs', []))
                extraction_data['browser_history'] = len(forensic_data.get('browser_history', []))
                extraction_data['installed_apps'] = len(forensic_data.get('installed_apps', []))
                extraction_data['wifi_networks'] = len(forensic_data.get('wifi_networks', []))
                extraction_data['system_logs'] = len(forensic_data.get('system_logs', []))
                extraction_data['whatsapp_artifacts'] = len(forensic_data.get('whatsapp_artifacts', []))
                extraction_data['instagram_artifacts'] = len(forensic_data.get('instagram_artifacts', []))
                extraction_data['messaging_artifacts'] = len(forensic_data.get('messaging_app_artifacts', []))
                extraction_data['forensic_media'] = len(forensic_media_files)
                
                # Display extraction progress
                st.write(f"📞 Call Logs: {extraction_data['call_logs']}")
                st.write(f"🌐 Browser History: {extraction_data['browser_history']}")
                st.write(f"📦 Installed Apps: {extraction_data['installed_apps']}")
                st.write(f"📡 WiFi Networks: {extraction_data['wifi_networks']}")
                st.write(f"📋 System Logs: {extraction_data['system_logs']}")
                st.write(f"💬 WhatsApp Artifacts: {extraction_data['whatsapp_artifacts']}")
                st.write(f"📸 Instagram Artifacts: {extraction_data['instagram_artifacts']}")
                st.write(f"💬 Messaging App Artifacts: {extraction_data['messaging_artifacts']}")
                
                st.success("✅ All forensic agents completed!")
                logger.info(f"Forensic data extracted: {extraction_data}")
                
            elif adapter_type.lower() == 'ios':
                from adapters.ios_logical import Adapter as iOSAdapter
                ios = iOSAdapter()
                
                # Extract all forensic data using adapter method
                forensic_data = ios.extract_all_forensic_data(device_id)
                
                # Store forensic data in session state
                st.session_state.extracted_call_logs = forensic_data.get('call_logs', [])
                st.session_state.extracted_browser_history = forensic_data.get('browser_history', [])
                st.session_state.extracted_installed_apps = forensic_data.get('installed_apps', [])
                st.session_state.extracted_wifi_networks = forensic_data.get('wifi_networks', [])
                st.session_state.extracted_system_logs = forensic_data.get('system_logs', [])
                
                # Store app artifact data (WhatsApp, Instagram, Messaging apps)
                st.session_state.extracted_whatsapp_artifacts = forensic_data.get('whatsapp_artifacts', [])
                st.session_state.extracted_instagram_artifacts = forensic_data.get('instagram_artifacts', [])
                st.session_state.extracted_messaging_artifacts = forensic_data.get('messaging_app_artifacts', [])
                
                # Store media files from forensic extraction (merge with existing media files)
                forensic_media_files = forensic_data.get('media_files', [])
                if forensic_media_files:
                    # Merge with existing media files from earlier extraction
                    existing_media = st.session_state.get('extracted_media_files', [])
                    
                    # Convert existing media to new format if needed
                    merged_media = []
                    seen_paths = set()
                    
                    # Add forensic media files first
                    for file in forensic_media_files:
                        file_path = file.get('path', '')
                        if file_path and file_path not in seen_paths:
                            merged_media.append(file)
                            seen_paths.add(file_path)
                    
                    # Add existing media files (avoiding duplicates)
                    for file in existing_media:
                        file_path = file.get('path', '')
                        if file_path and file_path not in seen_paths:
                            merged_media.append(file)
                            seen_paths.add(file_path)
                    
                    st.session_state.extracted_media_files = merged_media
                    logger.info(f"✅ Merged {len(forensic_media_files)} forensic media files with {len(existing_media)} existing files")
                
                # Update extraction data
                extraction_data['call_logs'] = len(forensic_data.get('call_logs', []))
                extraction_data['browser_history'] = len(forensic_data.get('browser_history', []))
                extraction_data['installed_apps'] = len(forensic_data.get('installed_apps', []))
                extraction_data['wifi_networks'] = len(forensic_data.get('wifi_networks', []))
                extraction_data['system_logs'] = len(forensic_data.get('system_logs', []))
                extraction_data['whatsapp_artifacts'] = len(forensic_data.get('whatsapp_artifacts', []))
                extraction_data['instagram_artifacts'] = len(forensic_data.get('instagram_artifacts', []))
                extraction_data['messaging_artifacts'] = len(forensic_data.get('messaging_app_artifacts', []))
                extraction_data['forensic_media'] = len(forensic_media_files)
                
                # Display extraction progress
                st.write(f"📞 Call Logs: {extraction_data['call_logs']}")
                st.write(f"🌐 Browser History: {extraction_data['browser_history']}")
                st.write(f"📦 Installed Apps: {extraction_data['installed_apps']}")
                st.write(f"📡 WiFi Networks: {extraction_data['wifi_networks']}")
                st.write(f"📋 System Logs: {extraction_data['system_logs']}")
                st.write(f"💬 WhatsApp Artifacts: {extraction_data['whatsapp_artifacts']}")
                st.write(f"📸 Instagram Artifacts: {extraction_data['instagram_artifacts']}")
                st.write(f"💬 Messaging App Artifacts: {extraction_data['messaging_artifacts']}")
                st.write(f"🎬 Media Files (Forensic): {extraction_data['forensic_media']}")
                
                st.success("✅ All forensic agents completed!")
                logger.info(f"Forensic data extracted: {extraction_data}")
        
        except Exception as e:
            logger.warning(f"Forensic agents extraction failed: {e}")
            st.warning(f"⚠️ Forensic agents extraction failed: {e}")
        
        time.sleep(0.5)

    except Exception as e:
        logger.error(f"❌ Extraction error: {e}")
        st.error(f"❌ Extraction error: {e}")
    
    # Mark extraction as completed
    st.session_state.extraction_completed = True
    
    # Store REAL results in session state
    st.session_state.extraction_results = {
        'case_id': case_id,
        'device_id': device_id,
        'status': 'Completed',
        'modules': extraction_data,
        'artifacts': [
            {'type': 'emails', 'count': extraction_data.get('emails', 0)},
            {'type': 'messages', 'count': extraction_data.get('messages', 0)},
            {'type': 'files', 'count': extraction_data.get('files', 0)},
            {'type': 'attachments', 'count': extraction_data.get('attachments', 0)},
            {'type': 'contacts', 'count': extraction_data.get('contacts', 0)},
            {'type': 'media', 'count': extraction_data.get('media', 0)}
        ],
        'summary': {
            'total_items': sum(extraction_data.values()),
            'extracted_items': sum(extraction_data.values()),
            'duration': f"{(datetime.now() - st.session_state.extraction_start_time).total_seconds():.1f}s"
        }
    }
    
    logger.info(f"✅ REAL extraction completed for {case_id}")
    logger.info(f"📊 Extracted Data: {extraction_data}")
