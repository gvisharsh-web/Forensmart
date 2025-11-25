"""
Consent Portal Page
Handles nominee approval/denial and activity logging
"""

import streamlit as st
from pathlib import Path
import json
import sys
from datetime import datetime
import logging
import re
from urllib.parse import quote_plus

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.approval.sync import ApprovalSync
from modules.approval.utils import get_approvals_file, save_approval_decision
from modules.consent.manager import get_consent_manager, ConsentAuditTrail
from modules.approval.redirect import ApprovalRedirect, ApprovalNotifier

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

def _log_approval_activity(case_id: str, decision: str, nominee_name: str = None) -> bool:
    """Log approval activity to activity log for display on main page"""
    try:
        activity_log_file = Path('audit/consent_portal/activity_log.json')
        activity_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        activities = {}
        if activity_log_file.exists():
            try:
                activities = json.loads(activity_log_file.read_text())
            except Exception:
                activities = {}
        
        # Add new activity with unique ID
        activity_id = f"{case_id}_{datetime.now().isoformat()}"
        activities[activity_id] = {
            'case_id': case_id,
            'decision': decision,
            'nominee_name': nominee_name or 'Unknown',
            'timestamp': datetime.now().isoformat(),
            'status': 'completed'
        }
        
        activity_log_file.write_text(json.dumps(activities, indent=2))
        logger.info(f"Activity logged: {case_id} - {decision} by {nominee_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to log activity: {e}")
        return False

def _render_approval_activity_log() -> None:
    """Display recent approval activity on consent portal main page"""
    st.markdown("## 📋 Recent Approval Activity")
    
    try:
        activity_log_file = Path('audit/consent_portal/activity_log.json')
        
        if activity_log_file.exists():
            activities = json.loads(activity_log_file.read_text())
            
            if activities:
                # Sort by timestamp (newest first)
                sorted_activities = sorted(
                    activities.items(),
                    key=lambda x: x[1].get('timestamp', ''),
                    reverse=True
                )
                
                # Display in table format
                activity_data = []
                for activity_id, activity in sorted_activities[:10]:  # Show last 10
                    decision_emoji = "✅" if activity.get('decision') == 'approved' else "❌"
                    activity_data.append({
                        'Case ID': activity.get('case_id'),
                        'Decision': f"{decision_emoji} {activity.get('decision', 'unknown').upper()}",
                        'Nominee': activity.get('nominee_name', 'Unknown'),
                        'Time': activity.get('timestamp', 'N/A')[:19]
                    })
                
                st.dataframe(activity_data, use_container_width=True, hide_index=True)
            else:
                st.info("No approval activity yet")
        else:
            st.info("No approval activity yet")
    except Exception as e:
        st.error(f"Failed to load activity log: {e}")

def _save_approval(case_id: str, decision: str, nominee_name: str = None, approval_link: str = None) -> bool:
    """Save approval decision"""
    try:
        success = save_approval_decision(case_id, decision, nominee_name, None)
        if success:
            approvals_file = get_approvals_file()
            approvals = {}
            if approvals_file.exists():
                try:
                    approvals = json.loads(approvals_file.read_text())
                except Exception:
                    approvals = {}
            
            if case_id in approvals:
                approvals[case_id]['approval_link'] = approval_link
                approvals_file.write_text(json.dumps(approvals, indent=2))
            
            cm = get_consent_manager()
            session = cm.get_session(case_id)
            if session:
                session.approval_status = decision
                session.approval_timestamp = datetime.now().isoformat()
                session.nominee_name = nominee_name
                session.approval_link = approval_link
                cm.persist_session(case_id)
            
            ConsentAuditTrail.record_approval(
                case_id=case_id,
                decision=decision,
                nominee_name=nominee_name or 'Unknown',
                device_id='CONSENT_PORTAL',
                purpose='Nominee approval via consent portal'
            )
            
            ApprovalNotifier.notify_approval(
                case_id=case_id,
                device_id='CONSENT_PORTAL',
                decision=decision,
                nominee_name=nominee_name,
                extraction_type="consent_portal"
            )
            
            # Log activity
            _log_approval_activity(case_id, decision, nominee_name)
            
            logger.info(f"✅ Approval saved for case {case_id}")
            return True
        else:
            st.error(f"Failed to save approval for case {case_id}")
            return False
    except Exception as e:
        logger.error(f"Failed to save approval: {e}")
        st.error(f"Failed to save approval: {e}")
        return False

def _save_approval_link(case_id: str, approval_link: str, nominee_name: str = None) -> bool:
    """Save approval link"""
    try:
        approvals_file = get_approvals_file()
        approvals = {}
        if approvals_file.exists():
            try:
                approvals = json.loads(approvals_file.read_text())
            except Exception:
                approvals = {}
        
        if case_id not in approvals:
            approvals[case_id] = {}
        
        current_status = approvals[case_id].get('status', 'pending')
        approvals[case_id].update({
            'approval_link': approval_link,
            'link_created_at': datetime.now().isoformat(),
            'nominee_name': nominee_name,
            'status': current_status
        })
        
        approvals_file.write_text(json.dumps(approvals, indent=2))
        return True
    except Exception as e:
        st.error(f"Failed to save approval link: {e}")
        return False

def render_consent_portal():
    """Render consent portal page"""
    
    st.markdown("# 🔐 ForenSmart Consent Portal")
    st.markdown("Review and approve data extraction requests")
    
    # Get query parameters
    params = st.query_params
    
    approval_data = None
    token = None
    
    # Try to get approval data from URL
    if 'data' in params:
        data_param = params.get('data')
        if isinstance(data_param, list):
            data_param = data_param[-1]
        try:
            import base64
            decoded = base64.b64decode(data_param).decode('utf-8')
            approval_data = json.loads(decoded)
        except Exception as e:
            logger.error(f"Failed to decode approval data: {e}")
    
    # Fallback to token-based lookup
    if not approval_data:
        token = params.get("unlock_token")
        if isinstance(token, list):
            token = token[-1]
    
    if not approval_data and not token:
        st.warning(
            "No approval data supplied. This page must be opened via the secure link "
            "shared by the investigator."
        )
        st.info(
            "Example: https://your-consent-app.streamlit.app/?unlock_token=TOKEN_HERE"
        )
        
        # Display recent approval activity on main page
        st.divider()
        _render_approval_activity_log()
        
        # Add approval link generator for investigators
        st.divider()
        st.markdown("## 📤 Generate & Share Approval Links")
        st.caption("Generate approval links and share them with nominees via WhatsApp, SMS, or Email")
        
        with st.form("generate_approval_link_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                case_id = st.text_input("Case ID", placeholder="CASE_001")
                device_id = st.text_input("Device ID", placeholder="ABC123XYZ")
                nominee_name = st.text_input("Nominee Name", placeholder="John Doe")
            
            with col2:
                purpose = st.text_area("Purpose", placeholder="Extraction purpose...", height=100)
                requested_level = st.selectbox("Requested Level", ["STANDARD", "FULL", "LEGAL"])
            
            col1, col2 = st.columns(2)
            with col1:
                nominee_phone = st.text_input("Nominee Phone (for WhatsApp/SMS)", placeholder="+1234567890")
            with col2:
                nominee_email = st.text_input("Nominee Email", placeholder="nominee@example.com")
            
            submitted = st.form_submit_button("Generate Approval Link", use_container_width=True)
        
        if submitted and case_id and device_id:
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
            import base64
            encoded = base64.b64encode(json.dumps(approval_data_dict).encode()).decode()
            
            # Use Streamlit Cloud URL for approval link
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
            
            # Save link to audit
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
        
        return
    
    # If we have embedded approval data, show approval form
    if approval_data:
        cm = get_consent_manager()
        case_id = approval_data.get('case_id')
        device_id = approval_data.get('device_id', 'UNKNOWN_DEVICE')
        purpose = approval_data.get('purpose', 'Investigator did not provide details.')
        requested_level_name = approval_data.get('requested_level', 'STANDARD')
        nominee_name = approval_data.get('nominee_name')
        
        st.markdown("# 🔐 ForenSmart Consent Approval")
        st.info("Review the request details below and choose whether to unlock data extraction.")
        
        st.markdown("### Case Information")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Case ID", case_id or "N/A")
            st.metric("Device ID", device_id)
        with col_b:
            st.metric("Requested Level", requested_level_name)
            st.metric("Nominee", nominee_name or "Not specified")
        
        st.markdown("### Purpose")
        st.write(purpose)
        
        st.markdown("### Your Decision")
        st.caption("Please confirm whether you approve or deny this extraction request.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button('✅ Yes, Approve', key='approve_btn', use_container_width=True):
                current_url = st.query_params.get('_url', 'N/A')
                if _save_approval(case_id, 'approved', nominee_name, approval_link=str(st.query_params)):
                    _save_approval_link(case_id, str(st.query_params), nominee_name)
                    
                    # Clear cache
                    try:
                        ApprovalSync.clear_cache(case_id)
                    except Exception:
                        pass
                    
                    st.success("✅ **Approval Granted** - Thank you for your consent. The investigator has been notified.")
                    st.caption(f"Nominee: {nominee_name or 'Not specified'}")
                    
                    # Detect device type from browser headers
                    is_mobile = False
                    try:
                        headers = st.context.headers if hasattr(st, 'context') else {}
                        user_agent = headers.get('user-agent', '').lower()
                        is_mobile = bool(re.search(r'mobile|android|iphone|ipad|ipod|windows phone', user_agent))
                    except Exception:
                        user_agent = st.query_params.get('user_agent', '').lower()
                        is_mobile = bool(re.search(r'mobile|android|iphone|ipad|ipod|windows phone', user_agent))
                    
                    # Redirect to extraction page
                    st.info("🔄 **Redirecting to extraction page...**")
                    st.markdown("""
                    The extraction page will now:
                    1. Recognize your approval
                    2. Show extraction options
                    3. Allow data extraction to proceed
                    
                    If you're not redirected automatically, click the link below.
                    """)
                    
                    import time
                    time.sleep(1)
                    
                    # Use Streamlit Cloud URL for redirect
                    dashboard_url = "https://forensmart-m8fackxhwafzsu7tfvfccl.streamlit.app"
                    redirect_url = f"{dashboard_url}?case_id={case_id}&auto_extract=true"
                    
                    st.markdown(
                        f"""
                        <script>
                        window.location.href = "{redirect_url}";
                        </script>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(f"[Click here if not redirected automatically]({redirect_url})")
                    
                    st.balloons()
                else:
                    st.error("Failed to save approval. Please try again.")
        
        with col2:
            if st.button('❌ No, Deny', key='deny_btn', use_container_width=True):
                current_url = st.query_params.get('_url', 'N/A')
                if _save_approval(case_id, 'denied', nominee_name, approval_link=str(st.query_params)):
                    _save_approval_link(case_id, str(st.query_params), nominee_name)
                    
                    # Clear cache
                    try:
                        ApprovalSync.clear_cache(case_id)
                    except Exception:
                        pass
                    
                    st.error("❌ **Request Denied** - Your decision has been recorded and the investigator has been notified.")
                    st.caption(f"Nominee: {nominee_name or 'Not specified'}")
                    
                    # Detect device type
                    is_mobile = False
                    try:
                        headers = st.context.headers if hasattr(st, 'context') else {}
                        user_agent = headers.get('user-agent', '').lower()
                        is_mobile = bool(re.search(r'mobile|android|iphone|ipad|ipod|windows phone', user_agent))
                    except Exception:
                        user_agent = st.query_params.get('user_agent', '').lower()
                        is_mobile = bool(re.search(r'mobile|android|iphone|ipad|ipod|windows phone', user_agent))
                    
                    if is_mobile:
                        st.info("📱 **Denial recorded. You can close this page now.**")
                    else:
                        st.info("You can close this page now.")
                else:
                    st.error("Failed to save denial. Please try again.")
        
        return

def main():
    """Main function"""
    render_consent_portal()

if __name__ == "__main__":
    main()
