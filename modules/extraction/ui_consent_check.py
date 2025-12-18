"""
Consent Check UI Component

Displays consent status and shows approval link if not yet approved.
Allows investigator to check approval status in real-time.
"""

import streamlit as st
from datetime import datetime
import qrcode
from io import BytesIO
from typing import Dict, Tuple


def render_consent_check() -> Tuple[bool, Dict]:
    """
    Render consent check UI component.
    
    Returns:
        Tuple[bool, Dict]: (is_approved, consent_details)
    """
    st.subheader("🔐 Consent Requirements")
    
    # Get case ID from session
    case_id = st.session_state.get('case_id')
    
    if not case_id:
        st.warning("⚠️ No case selected")
        return False, {}
    
    # Get consent details from database
    consent_details = get_consent_details(case_id)
    
    if not consent_details:
        st.error("❌ Case not found")
        return False, {}
    
    # Display required consent level
    st.info(f"📋 Required Consent Level: {consent_details['required_level']}")
    
    # Check if consent is already approved
    is_approved = consent_details['status'] == 'APPROVED'
    
    if is_approved:
        # ✅ CONSENT ALREADY APPROVED
        st.success("✅ Consent Approved")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Approved by:** {consent_details['nominee_name']}")
        with col2:
            st.write(f"**Approved at:** {consent_details['approved_at']}")
        
        # Show approval details
        with st.expander("📋 View Approval Details"):
            st.write(f"**Case ID:** {case_id}")
            st.write(f"**Approval Method:** {consent_details['approval_method']}")
            st.write(f"**Consent Level:** {consent_details['current_level']}")
            st.write(f"**Timestamp:** {consent_details['approved_at']}")
        
        return True, consent_details
    
    else:
        # ⏳ WAITING FOR APPROVAL
        st.warning("⏳ Consent Not Yet Approved")
        
        # Generate approval link
        approval_link = generate_approval_link(case_id)
        
        # Display approval link
        st.info("📧 Send this link to nominee:")
        st.code(approval_link, language="text")
        
        # Display QR code
        st.write("**Or scan QR code:**")
        qr_image = generate_qr_code(approval_link)
        st.image(qr_image, width=200)
        
        # Show status
        st.write("**Status:** ⏳ Waiting for approval...")
        
        # Show nominee info
        if consent_details.get('nominee_email'):
            st.info(f"📧 Nominee: {consent_details['nominee_email']}")
        
        # Auto-refresh button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Check Status"):
                # Refresh consent status
                updated_details = check_approval_status(case_id)
                if updated_details['status'] == 'APPROVED':
                    st.success("✅ Consent Approved!")
                    st.rerun()
                else:
                    st.info("⏳ Still waiting for approval...")
        
        with col2:
            if st.button("📋 View Case Details"):
                with st.expander("Case Details"):
                    st.write(f"**Case ID:** {case_id}")
                    st.write(f"**Device:** {consent_details['device_id']}")
                    st.write(f"**Modules:** {', '.join(consent_details['modules'])}")
                    st.write(f"**Created at:** {consent_details['created_at']}")
        
        return False, consent_details


def generate_approval_link(case_id: str, nominee_email: str = "nominee@example.com") -> str:
    """
    Generate approval link for nominee with hash verification.
    
    Hash-based approval system:
    - Primary: Hash verification (from approval link)
    - Fallback: Fallback hash verification
    - No PIN required
    
    Args:
        case_id: Case ID
        nominee_email: Nominee email
        
    Returns:
        str: Approval link with hash parameters
    """
    import hmac
    import hashlib
    import secrets
    from datetime import datetime, timedelta
    
    # Generate hash components
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.utcnow() + timedelta(hours=24)).isoformat()
    
    # Create data to hash
    data_to_hash = f"{case_id}:{nominee_email}:{expires_at}:{token}"
    
    # Generate HMAC-SHA256 hash
    secret_key = "forensmart-secret-key"
    approval_hash = hmac.new(
        secret_key.encode(),
        data_to_hash.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Build approval link with hash parameters
    base_url = "https://forensmart.streamlit.app"
    approval_link = (
        f"{base_url}/?case_id={case_id}"
        f"&hash={approval_hash}"
        f"&token={token}"
        f"&expires_at={expires_at}"
        f"&nominee_email={nominee_email}"
    )
    
    return approval_link


def generate_secure_token(case_id: str) -> str:
    """
    Generate secure token for approval link.
    
    Args:
        case_id: Case ID
        
    Returns:
        str: Secure token
    """
    import hashlib
    import secrets
    
    # Generate random token
    random_part = secrets.token_hex(16)
    
    # Hash with case_id
    token = hashlib.sha256(f"{case_id}{random_part}".encode()).hexdigest()
    
    return token[:32]  # Use first 32 chars


def generate_qr_code(url: str):
    """
    Generate QR code for approval link.
    
    Args:
        url: URL to encode
        
    Returns:
        PIL.Image: QR code image
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    return img


def get_consent_details(case_id: str) -> Dict:
    """
    Get consent details from database.
    
    Args:
        case_id: Case ID
        
    Returns:
        Dict: Consent details
    """
    # Simulated database lookup
    # In production, would query actual database
    
    consent_db = {
        'case_001': {
            'case_id': 'case_001',
            'nominee_name': 'Jane Doe',
            'nominee_email': 'jane@example.com',
            'device_id': 'device_123',
            'status': 'PENDING_CONSENT',  # or 'APPROVED'
            'required_level': 'LEGAL',
            'current_level': 'LEGAL',
            'approval_method': 'HASH',  # Hash-based verification (no PIN)
            'approved_at': None,
            'created_at': '2025-11-26 17:00:00',
            'modules': ['Communications', 'Media', 'Social Media']
        }
    }
    
    return consent_db.get(case_id, {})


def check_approval_status(case_id: str) -> Dict:
    """
    Check if consent has been approved.
    
    Args:
        case_id: Case ID
        
    Returns:
        Dict: Updated consent details
    """
    # In production, would query database
    # For now, simulating check
    
    consent_details = get_consent_details(case_id)
    
    # Simulated: Check if approved
    # In real app, would check database
    if st.session_state.get(f'consent_approved_{case_id}'):
        consent_details['status'] = 'APPROVED'
        consent_details['approved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return consent_details


def show_consent_summary() -> None:
    """Show consent summary in sidebar."""
    with st.sidebar:
        st.subheader("📋 Consent Status")
        
        case_id = st.session_state.get('case_id')
        if case_id:
            consent_details = get_consent_details(case_id)
            
            if consent_details['status'] == 'APPROVED':
                st.success(f"✅ Approved by {consent_details['nominee_name']}")
            else:
                st.warning("⏳ Pending approval")
                
                # Show quick link
                if st.button("📧 Send Link Again"):
                    approval_link = generate_approval_link(case_id)
                    st.code(approval_link)
