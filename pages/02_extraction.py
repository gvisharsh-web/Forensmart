"""
Extraction Page
Handles case management and extraction workflows (Android, iOS, HDD)
"""

import streamlit as st
from pathlib import Path
import sys
import logging
import json
import base64
from datetime import datetime
from urllib.parse import quote_plus

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.approval.sync import ApprovalSync
from modules.consent.manager import get_consent_manager
from modules.shared.device_detector import DeviceDetector
from modules.extraction.orchestrator import DataExtractionOrchestrator
from ui_components.extraction_ui import render_extraction_tab

# Setup logging
logger = logging.getLogger(__name__)

def _create_whatsapp_link(phone: str, message: str) -> str:
    """Create WhatsApp sharing link"""
    sanitized = ''.join(filter(str.isdigit, phone))
    if not sanitized:
        return ""
    return f"https://wa.me/{sanitized}?text={quote_plus(message)}"

def _create_sms_link(phone: str, message: str) -> str:
    """Create SMS sharing link"""
    sanitized = ''.join(filter(str.isdigit, phone))
    if not sanitized:
        return ""
    return f"sms:+{sanitized}?body={quote_plus(message)}"

def _create_email_link(email: str, subject: str, message: str) -> str:
    """Create email sharing link"""
    return f"mailto:{email}?subject={quote_plus(subject)}&body={quote_plus(message)}"

def _generate_qr_code_url(data: str) -> str:
    """Generate QR code URL"""
    return f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={quote_plus(data)}"

def render_extraction_page():
    """Render extraction page"""
    
    st.markdown("# 📦 Data Extraction")
    st.markdown("Manage cases and extract data from devices")
    
    # Initialize session state
    if 'case_id' not in st.session_state:
        st.session_state['case_id'] = None
    
    if 'extraction_status' not in st.session_state:
        st.session_state['extraction_status'] = {}
    
    # Sidebar: Case Management
    st.sidebar.markdown("### 📋 Case Management")
    
    cm = get_consent_manager()
    
    # Create new case
    with st.sidebar.expander("➕ Create New Case"):
        case_id_input = st.text_input("Case ID", key="new_case_id")
        device_type = st.selectbox("Device Type", ["Android", "iOS", "HDD"])
        
        if st.button("Create Case"):
            if case_id_input:
                st.session_state['case_id'] = case_id_input
                st.success(f"✅ Case created: {case_id_input}")
                st.rerun()
            else:
                st.error("Please enter a Case ID")
    
    # Select existing case
    with st.sidebar.expander("📂 Select Case"):
        cases = list(cm.sessions.keys())
        if cases:
            selected_case = st.selectbox(
                "Available Cases",
                cases,
                key="select_case"
            )
            if st.button("Load Case"):
                st.session_state['case_id'] = selected_case
                st.rerun()
        else:
            st.info("No cases found")
    
    # Current case info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Current Case")
    
    case_id = st.session_state.get('case_id')
    
    if case_id:
        st.sidebar.metric("Case ID", case_id)
        
        # Check approval status
        approval = ApprovalSync.get_approval_status(case_id)
        if approval:
            status = approval.get('decision', 'pending').upper()
            if status == 'APPROVED':
                st.sidebar.success(f"✅ {status}")
            elif status == 'DENIED':
                st.sidebar.error(f"❌ {status}")
            else:
                st.sidebar.warning(f"⏳ {status}")
        else:
            st.sidebar.warning("⏳ Pending")
    else:
        st.sidebar.info("No case selected")
    
    # Main content
    if not case_id:
        st.info("👈 Select or create a case from the sidebar to begin")
        return
    
    # Check approval status - try multiple methods
    approval = ApprovalSync.get_approval_status(case_id)
    is_approved = False
    
    # Check if approved via ApprovalSync
    if approval and approval.get('decision') == 'approved':
        is_approved = True
    
    # Also check approval file directly as fallback
    if not is_approved:
        try:
            from modules.approval.utils import get_approvals_file
            approvals_file = get_approvals_file()
            if approvals_file.exists():
                approvals = json.loads(approvals_file.read_text())
                if case_id in approvals and approvals[case_id].get('status') == 'approved':
                    is_approved = True
        except Exception as e:
            logger.warning(f"Failed to check approval file: {e}")
    
    if not is_approved:
        st.warning("⏳ Awaiting approval from nominee")
        st.info("The nominee must approve this extraction request before you can proceed.")
        
        # Add refresh button to check approval status
        col_refresh = st.columns([1, 4])[0]
        with col_refresh:
            if st.button("🔄 Check Approval Status", use_container_width=True):
                # Clear cache and recheck
                try:
                    ApprovalSync.clear_cache(case_id)
                except Exception:
                    pass
                st.rerun()
        
        # Generate approval link
        with st.expander("📧 Generate & Share Approval Link", expanded=True):
            st.markdown("### Fill in nominee details to generate approval link:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                device_id = st.text_input("Device ID", placeholder="ABC123XYZ", key="approval_device_id")
                nominee_name = st.text_input("Nominee Name", placeholder="John Doe", key="approval_nominee")
            
            with col2:
                purpose = st.text_area("Purpose", placeholder="Extraction purpose...", height=80, key="approval_purpose")
                requested_level = st.selectbox("Requested Level", ["STANDARD", "FULL", "LEGAL"], key="approval_level")
            
            col1, col2 = st.columns(2)
            with col1:
                nominee_phone = st.text_input("Nominee Phone (for WhatsApp/SMS)", placeholder="+1234567890", key="approval_phone")
            with col2:
                nominee_email = st.text_input("Nominee Email", placeholder="nominee@example.com", key="approval_email")
            
            if st.button("🔗 Generate Approval Link", use_container_width=True, key="gen_approval_link"):
                if device_id:
                    # Create approval data
                    approval_data_dict = {
                        'case_id': case_id,
                        'device_id': device_id,
                        'purpose': purpose or "Data extraction for investigation",
                        'requested_level': requested_level,
                        'nominee_name': nominee_name or "Not specified",
                        'created_at': datetime.now().isoformat()
                    }
                    
                    # Encode approval data
                    encoded = base64.b64encode(json.dumps(approval_data_dict).encode()).decode()
                    
                    # Create approval link
                    base_url = "https://forensmart-m8fackxhwafzsu7tfvfccl.streamlit.app"
                    approval_link = f"{base_url}?data={encoded}"
                    
                    st.success("✅ Approval link generated!")
                    
                    # Display link
                    st.markdown("### 🔗 Approval Link")
                    st.code(approval_link, language="text")
                    
                    # Delivery options
                    st.markdown("### 📤 Delivery Options")
                    
                    delivery_col1, delivery_col2, delivery_col3 = st.columns(3)
                    
                    # WhatsApp
                    with delivery_col1:
                        if nominee_phone and nominee_phone.strip():
                            message = f"Hi {nominee_name or 'there'},\n\nPlease review and approve this ForenSmart extraction request:\n\n{approval_link}\n\nThank you!"
                            whatsapp_link = _create_whatsapp_link(nominee_phone, message)
                            if whatsapp_link:
                                st.markdown(f"[🟢 Share via WhatsApp]({whatsapp_link})")
                            else:
                                st.warning("Invalid phone number for WhatsApp")
                        else:
                            st.info("Enter phone number to enable WhatsApp sharing")
                    
                    # SMS
                    with delivery_col2:
                        if nominee_phone and nominee_phone.strip():
                            message = f"ForenSmart approval link: {approval_link}"
                            sms_link = _create_sms_link(nominee_phone, message)
                            if sms_link:
                                st.markdown(f"[📱 Send via SMS]({sms_link})")
                            else:
                                st.warning("Invalid phone number for SMS")
                        else:
                            st.info("Enter phone number to enable SMS sharing")
                    
                    # Email
                    with delivery_col3:
                        if nominee_email and nominee_email.strip():
                            subject = f"ForenSmart Approval Request - Case {case_id}"
                            message = f"Hi {nominee_name or 'there'},\n\nPlease review and approve this ForenSmart extraction request:\n\n{approval_link}\n\nPurpose: {purpose or 'Data extraction for investigation'}\n\nThank you!"
                            email_link = _create_email_link(nominee_email, subject, message)
                            st.markdown(f"[✉️ Send via Email]({email_link})")
                        else:
                            st.info("Enter email to enable email sharing")
                    
                    # QR Code
                    st.markdown("### 📲 QR Code")
                    qr_url = _generate_qr_code_url(approval_link)
                    st.image(qr_url, caption="Scan to open approval link", width=200)
                    
                    # Save to audit
                    try:
                        audit_dir = Path("audit/generated_links")
                        audit_dir.mkdir(parents=True, exist_ok=True)
                        link_file = audit_dir / f"{case_id}_link.json"
                        link_data = {
                            'case_id': case_id,
                            'link': approval_link,
                            'generated_at': datetime.now().isoformat(),
                            'nominee_name': nominee_name,
                            'nominee_phone': nominee_phone,
                            'nominee_email': nominee_email
                        }
                        link_file.write_text(json.dumps(link_data, indent=2))
                    except Exception as e:
                        logger.warning(f"Failed to save link to audit: {e}")
                else:
                    st.error("Please enter Device ID")
        
        return
    
    # Approval granted - show extraction options
    st.success("✅ **APPROVED** - Ready for extraction")
    
    st.divider()
    
    # Extraction tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📱 Android",
        "🍎 iOS",
        "💾 HDD",
        "📊 History"
    ])
    
    with tab1:
        st.markdown("### 📱 Android Extraction")
        render_extraction_tab("android", case_id)
    
    with tab2:
        st.markdown("### 🍎 iOS Extraction")
        render_extraction_tab("ios", case_id)
    
    with tab3:
        st.markdown("### 💾 HDD Extraction")
        render_extraction_tab("hdd", case_id)
    
    with tab4:
        st.markdown("### 📊 Extraction History")
        st.info("Extraction history and statistics will be displayed here")

def main():
    """Main function"""
    render_extraction_page()

if __name__ == "__main__":
    main()
