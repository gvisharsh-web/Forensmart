"""
DEVICE SELECTOR UI - Device and Account Selection
Handles UI for selecting devices and cloud accounts for extraction

This module provides:
- Device type selector (Android, iOS, HDD, Google Drive, OneDrive, Email, Social Media)
- Device/Account list display
- Connection status indicator
- Device info display
- OAuth2 authentication UI
- IMAP configuration UI
"""

import logging
import streamlit as st
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# DEVICE SELECTOR UI
# ============================================================================

def render_device_selector():
    """Render device selector UI"""
    
    st.header("📱 Device & Account Selector")
    
    # Step 1: Select device type category
    st.subheader("Step 1: Select Device Type Category")
    
    category = st.radio("Category", 
                       ["Physical Devices", "Cloud & Email", "Social Media"],
                       horizontal=True)
    
    # Step 2: Select specific device/account
    st.subheader("Step 2: Select Device/Account")
    
    if category == "Physical Devices":
        render_physical_device_selector()
    elif category == "Cloud & Email":
        render_cloud_account_selector()
    else:  # Social Media
        render_social_media_selector()


def render_physical_device_selector():
    """Render physical device selector"""
    
    logger.info("🔍 Rendering physical device selector")
    
    device_type = st.selectbox("Device Type", 
                              ["Android (ADB)", "iOS (iTunes)", "Storage (HDD)"])
    
    st.info("💡 Please connect your device via USB")
    
    # Simulated device list
    devices = [
        {'id': 'device_001', 'name': 'Samsung Galaxy S21', 'status': 'connected'},
        {'id': 'device_002', 'name': 'iPhone 13', 'status': 'connected'},
        {'id': 'device_003', 'name': 'USB Storage', 'status': 'offline'}
    ]
    
    # Filter by device type
    if device_type == "Android (ADB)":
        filtered_devices = [d for d in devices if 'Samsung' in d['name']]
    elif device_type == "iOS (iTunes)":
        filtered_devices = [d for d in devices if 'iPhone' in d['name']]
    else:
        filtered_devices = [d for d in devices if 'USB' in d['name']]
    
    if not filtered_devices:
        st.warning(f"⚠️ No {device_type} devices found")
        return
    
    st.success(f"✅ Found {len(filtered_devices)} device(s)")
    
    # Display devices
    for device in filtered_devices:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"📱 {device['name']}")
            st.caption(f"ID: {device['id']}")
        
        with col2:
            if device['status'] == 'connected':
                st.success("✅ Connected")
            else:
                st.warning("⚠️ Offline")
        
        with col3:
            if st.button("Select", key=f"select_{device['id']}"):
                st.session_state.selected_device = device
                st.session_state.device_type = device_type
                st.success(f"✅ Selected: {device['name']}")
    
    # Show selected device info
    if st.session_state.get('selected_device'):
        st.divider()
        st.subheader("Selected Device Info")
        
        device = st.session_state.selected_device
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Device Name", device.get('name', 'N/A'))
            st.metric("Device ID", device.get('id', 'N/A'))
        
        with col2:
            st.metric("Status", device.get('status', 'N/A'))
            st.metric("Type", st.session_state.device_type)


def render_cloud_account_selector():
    """Render cloud account selector"""
    
    logger.info("🌐 Rendering cloud account selector")
    
    account_type = st.selectbox("Account Type", 
                               ["Google Drive", "OneDrive", "Email"])
    
    if account_type == "Google Drive":
        render_google_drive_selector()
    elif account_type == "OneDrive":
        render_onedrive_selector()
    else:  # Email
        render_email_selector()


def render_google_drive_selector():
    """Render Google Drive account selector"""
    
    logger.info("🔐 Rendering Google Drive selector")
    
    st.subheader("🔐 Google Drive Authentication")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🔐 Login with Google"):
            st.info("💡 Redirecting to Google OAuth2...")
            st.session_state.google_auth_token = "simulated_token_123"
            st.success("✅ Logged in to Google Drive")
    
    if st.session_state.get('google_auth_token'):
        st.divider()
        st.subheader("Google Drive Account Info")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Account", "user@gmail.com")
            st.metric("Storage Used", "5 GB / 100 GB")
        
        with col2:
            st.metric("Files", "150")
            st.metric("Folders", "12")
        
        if st.button("✅ Select This Account"):
            st.session_state.selected_account = {
                'type': 'google_drive',
                'email': 'user@gmail.com',
                'token': st.session_state.google_auth_token
            }
            st.success("✅ Selected Google Drive account")


def render_onedrive_selector():
    """Render OneDrive account selector"""
    
    logger.info("🔐 Rendering OneDrive selector")
    
    st.subheader("🔐 OneDrive Authentication")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🔐 Login with Microsoft"):
            st.info("💡 Redirecting to Microsoft OAuth2...")
            st.session_state.onedrive_auth_token = "simulated_token_456"
            st.success("✅ Logged in to OneDrive")
    
    if st.session_state.get('onedrive_auth_token'):
        st.divider()
        st.subheader("OneDrive Account Info")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Account", "user@outlook.com")
            st.metric("Storage Used", "10 GB / 1 TB")
        
        with col2:
            st.metric("Files", "250")
            st.metric("Folders", "25")
        
        if st.button("✅ Select This Account"):
            st.session_state.selected_account = {
                'type': 'onedrive',
                'email': 'user@outlook.com',
                'token': st.session_state.onedrive_auth_token
            }
            st.success("✅ Selected OneDrive account")


def render_email_selector():
    """Render Email account selector"""
    
    logger.info("📧 Rendering email selector")
    
    st.subheader("📧 Email Provider Selection")
    
    email_provider = st.selectbox("Email Provider", 
                                  ["Gmail", "Outlook", "Other (IMAP)"])
    
    if email_provider == "Gmail":
        st.subheader("🔐 Gmail Authentication")
        
        if st.button("🔐 Login with Google"):
            st.info("💡 Redirecting to Google OAuth2...")
            st.session_state.gmail_auth_token = "simulated_token_789"
            st.success("✅ Logged in to Gmail")
        
        if st.session_state.get('gmail_auth_token'):
            st.divider()
            st.subheader("Gmail Account Info")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Account", "user@gmail.com")
                st.metric("Emails", "150")
            
            with col2:
                st.metric("Folders", "4")
                st.metric("Attachments", "25")
            
            if st.button("✅ Select Gmail Account"):
                st.session_state.selected_account = {
                    'type': 'gmail',
                    'email': 'user@gmail.com',
                    'token': st.session_state.gmail_auth_token
                }
                st.success("✅ Selected Gmail account")
    
    elif email_provider == "Outlook":
        st.subheader("🔐 Outlook Authentication")
        
        if st.button("🔐 Login with Microsoft"):
            st.info("💡 Redirecting to Microsoft OAuth2...")
            st.session_state.outlook_auth_token = "simulated_token_101"
            st.success("✅ Logged in to Outlook")
        
        if st.session_state.get('outlook_auth_token'):
            st.divider()
            st.subheader("Outlook Account Info")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Account", "user@outlook.com")
                st.metric("Emails", "200")
            
            with col2:
                st.metric("Folders", "5")
                st.metric("Attachments", "40")
            
            if st.button("✅ Select Outlook Account"):
                st.session_state.selected_account = {
                    'type': 'outlook',
                    'email': 'user@outlook.com',
                    'token': st.session_state.outlook_auth_token
                }
                st.success("✅ Selected Outlook account")
    
    else:  # IMAP
        st.subheader("📧 IMAP Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            imap_server = st.text_input("IMAP Server", 
                                       placeholder="imap.gmail.com",
                                       key="imap_server")
            email_address = st.text_input("Email Address",
                                         placeholder="user@example.com",
                                         key="imap_email")
        
        with col2:
            password = st.text_input("Password", 
                                    type="password",
                                    key="imap_password")
            port = st.number_input("Port", 
                                  value=993,
                                  key="imap_port")
        
        if st.button("🔐 Connect"):
            st.info("💡 Connecting to IMAP server...")
            st.success("✅ Connected to IMAP server")
            
            st.session_state.selected_account = {
                'type': 'imap',
                'email': email_address,
                'server': imap_server,
                'port': port
            }
            st.success(f"✅ Selected IMAP account: {email_address}")


def render_social_media_selector():
    """Render social media account selector"""
    
    logger.info("📱 Rendering social media selector")
    
    social_app = st.selectbox("Social Media App", 
                             ["WhatsApp", "Instagram", "Telegram", "Facebook", "Snapchat"])
    
    if social_app == "WhatsApp":
        render_whatsapp_selector()
    elif social_app == "Instagram":
        render_instagram_selector()
    elif social_app == "Telegram":
        render_telegram_selector()
    elif social_app == "Facebook":
        render_facebook_selector()
    else:  # Snapchat
        render_snapchat_selector()


def render_whatsapp_selector():
    """Render WhatsApp selector"""
    
    st.subheader("💬 WhatsApp Extraction Options")
    
    extraction_method = st.radio("How to extract?",
                                ["From Phone (ADB/iTunes)",
                                 "From Cloud Backup",
                                 "From Local Backup"])
    
    if extraction_method == "From Phone (ADB/iTunes)":
        st.info("💡 Connect your phone via USB")
        device_type = st.selectbox("Device Type", ["Android", "iOS"])
        
        if st.button("🔗 Connect"):
            st.success("✅ Connected to WhatsApp on phone")
            st.session_state.selected_account = {
                'type': 'whatsapp',
                'method': 'phone',
                'device_type': device_type
            }
    
    elif extraction_method == "From Cloud Backup":
        st.info("💡 Login to your cloud account")
        
        if st.button("🔐 Login to Google Drive"):
            st.success("✅ Logged in to Google Drive")
            st.session_state.selected_account = {
                'type': 'whatsapp',
                'method': 'cloud_backup'
            }
    
    else:  # Local Backup
        backup_file = st.file_uploader("Upload WhatsApp backup file")
        
        if backup_file:
            st.success("✅ Backup file loaded")
            st.session_state.selected_account = {
                'type': 'whatsapp',
                'method': 'local_backup',
                'file': backup_file.name
            }


def render_instagram_selector():
    """Render Instagram selector"""
    
    st.subheader("📸 Instagram Extraction Options")
    
    extraction_method = st.radio("How to extract?",
                                ["From Phone (ADB/iTunes)",
                                 "From Cloud Account"])
    
    if extraction_method == "From Phone (ADB/iTunes)":
        device_type = st.selectbox("Device Type", ["Android", "iOS"])
        
        if st.button("🔗 Connect"):
            st.success("✅ Connected to Instagram on phone")
            st.session_state.selected_account = {
                'type': 'instagram',
                'method': 'phone',
                'device_type': device_type
            }
    
    else:  # Cloud Account
        if st.button("🔐 Login with Instagram"):
            st.success("✅ Logged in to Instagram")
            st.session_state.selected_account = {
                'type': 'instagram',
                'method': 'cloud_account'
            }


def render_telegram_selector():
    """Render Telegram selector"""
    
    st.subheader("📱 Telegram Extraction Options")
    
    extraction_method = st.radio("How to extract?",
                                ["From Phone (ADB/iTunes)",
                                 "From Cloud Account"])
    
    if extraction_method == "From Phone (ADB/iTunes)":
        device_type = st.selectbox("Device Type", ["Android", "iOS"])
        
        if st.button("🔗 Connect"):
            st.success("✅ Connected to Telegram on phone")
            st.session_state.selected_account = {
                'type': 'telegram',
                'method': 'phone',
                'device_type': device_type
            }
    
    else:  # Cloud Account
        if st.button("🔐 Login with Telegram"):
            st.success("✅ Logged in to Telegram")
            st.session_state.selected_account = {
                'type': 'telegram',
                'method': 'cloud_account'
            }


def render_facebook_selector():
    """Render Facebook selector"""
    
    st.subheader("👥 Facebook Extraction Options")
    
    extraction_method = st.radio("How to extract?",
                                ["From Phone (ADB/iTunes)",
                                 "From Cloud Account"])
    
    if extraction_method == "From Phone (ADB/iTunes)":
        device_type = st.selectbox("Device Type", ["Android", "iOS"])
        
        if st.button("🔗 Connect"):
            st.success("✅ Connected to Facebook on phone")
            st.session_state.selected_account = {
                'type': 'facebook',
                'method': 'phone',
                'device_type': device_type
            }
    
    else:  # Cloud Account
        if st.button("🔐 Login with Facebook"):
            st.success("✅ Logged in to Facebook")
            st.session_state.selected_account = {
                'type': 'facebook',
                'method': 'cloud_account'
            }


def render_snapchat_selector():
    """Render Snapchat selector"""
    
    st.subheader("👻 Snapchat Extraction Options")
    
    extraction_method = st.radio("How to extract?",
                                ["From Phone (ADB/iTunes)",
                                 "From Cloud Account"])
    
    if extraction_method == "From Phone (ADB/iTunes)":
        device_type = st.selectbox("Device Type", ["Android", "iOS"])
        
        if st.button("🔗 Connect"):
            st.success("✅ Connected to Snapchat on phone")
            st.session_state.selected_account = {
                'type': 'snapchat',
                'method': 'phone',
                'device_type': device_type
            }
    
    else:  # Cloud Account
        if st.button("🔐 Login with Snapchat"):
            st.success("✅ Logged in to Snapchat")
            st.session_state.selected_account = {
                'type': 'snapchat',
                'method': 'cloud_account'
            }


def get_selected_device() -> Optional[Dict[str, Any]]:
    """Get selected device from session state"""
    return st.session_state.get('selected_device')


def get_selected_account() -> Optional[Dict[str, Any]]:
    """Get selected account from session state"""
    return st.session_state.get('selected_account')
