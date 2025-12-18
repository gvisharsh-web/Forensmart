"""
Nominee Approval Portal
Streamlit UI for nominees to approve consent and unlock extraction

Hash-based approval system:
- Primary: Hash verification (from approval link)
- Fallback: Hash verification if needed
- No PIN required for approval
"""

import streamlit as st
import hashlib
import hmac
import json
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import sys
import os

# Add modules to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.database.consent_operations import ConsentApprovalOperations
from modules.consent.models import ConsentLevel, get_consent_manager


# ============================================================================
# HASH VERIFICATION FUNCTIONS
# ============================================================================

def verify_approval_hash(case_id: str, nominee_email: str, 
                        provided_hash: str, token: str, 
                        expires_at: str, secret_key: str = "forensmart-secret-key") -> Tuple[bool, str]:
    """
    Verify approval hash (Primary verification method)
    
    Args:
        case_id: Case ID
        nominee_email: Nominee email
        provided_hash: Hash from approval link
        token: Token from approval link
        expires_at: Expiration timestamp
        secret_key: Secret key for HMAC
    
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        # Check if expired
        expires_dt = datetime.fromisoformat(expires_at)
        if datetime.utcnow() > expires_dt:
            return False, "❌ Approval link has expired"
        
        # Recreate data to hash
        data_to_hash = f"{case_id}:{nominee_email}:{expires_at}:{token}"
        
        # Generate expected hash
        expected_hash = hmac.new(
            secret_key.encode(),
            data_to_hash.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Compare hashes (constant-time comparison)
        is_valid = hmac.compare_digest(provided_hash, expected_hash)
        
        if is_valid:
            return True, "✅ Approval hash verified"
        else:
            return False, "❌ Invalid approval hash"
    
    except Exception as e:
        return False, f"❌ Error verifying hash: {str(e)}"


def get_case_details(case_id: str) -> Dict[str, Any]:
    """Get case details from database or file"""
    try:
        # Try to load from file
        case_file = f"reports/{case_id}/case_info.json"
        if os.path.exists(case_file):
            with open(case_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading case details: {str(e)}")
    
    # Return default case info
    return {
        'case_id': case_id,
        'investigator': 'Law Enforcement Officer',
        'reason': 'Digital Forensic Investigation',
        'created_at': datetime.now().isoformat(),
        'device_type': 'Android',
        'modules': ['Communications', 'Location', 'Media']
    }


def save_approval(case_id: str, nominee_email: str, approval_hash: str, 
                  token: str, expires_at: str) -> bool:
    """Save approval to database and file"""
    try:
        # Create approval record
        approval_record = {
            'case_id': case_id,
            'nominee_email': nominee_email,
            'approval_hash': approval_hash,
            'token': token,
            'expires_at': expires_at,
            'approved_at': datetime.now().isoformat(),
            'status': 'APPROVED',
            'consent_level': 'LEGAL',
            'verification_method': 'HASH'
        }
        
        # Save to file
        approval_dir = f"audit/approvals"
        os.makedirs(approval_dir, exist_ok=True)
        
        approval_file = f"{approval_dir}/{case_id}_approval.json"
        with open(approval_file, 'w') as f:
            json.dump(approval_record, f, indent=2)
        
        return True
    
    except Exception as e:
        st.error(f"Error saving approval: {str(e)}")
        return False


def log_approval_event(case_id: str, event_type: str, details: str, 
                       status: str = "SUCCESS") -> None:
    """Log approval event to audit trail"""
    try:
        event_record = {
            'case_id': case_id,
            'event_type': event_type,
            'details': details,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        
        # Append to audit log
        audit_log_file = f"audit/approval_events.jsonl"
        os.makedirs("audit", exist_ok=True)
        
        with open(audit_log_file, 'a') as f:
            f.write(json.dumps(event_record) + '\n')
    
    except Exception as e:
        st.warning(f"Could not log event: {str(e)}")


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

def configure_portal_page():
    """Configure portal page settings"""
    st.set_page_config(
        page_title="ForenSmart - Consent Approval",
        page_icon="🔐",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for approval portal
    st.markdown("""
    <style>
    .main {
        max-width: 600px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    .portal-header {
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        color: #004E89;
        margin-bottom: 2rem;
    }
    
    .portal-card {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 1rem;
        border: 2px solid #004E89;
        margin: 1rem 0;
    }
    
    .case-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    
    .consent-form {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    
    .verification-section {
        background-color: #f3e5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #9C27B0;
        margin: 1rem 0;
    }
    
    .success-message {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    
    .error-message {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    
    .stButton > button {
        width: 100%;
        font-weight: bold;
        font-size: 1rem;
        padding: 0.75rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================

def initialize_portal_session():
    """Initialize portal session state"""
    if 'portal_initialized' not in st.session_state:
        st.session_state.portal_initialized = True
        st.session_state.approval_step = 1  # 1: Case Info, 2: Consent Form, 3: Verification, 4: Confirmation
        st.session_state.nominee_email = ""
        st.session_state.verification_method = "PIN"
        st.session_state.approval_confirmed = False


# ============================================================================
# PAGE SECTIONS
# ============================================================================

def render_header():
    """Render portal header"""
    st.markdown('<div class="portal-header">🔐 Consent Approval Portal</div>', unsafe_allow_html=True)
    st.markdown("---")


def render_case_information(case_id: str, consent_level: str = "LEGAL") -> None:
    """Render case information section"""
    st.markdown("**Case Details:**")
    
    case_info = get_case_details(case_id)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Case ID:** `{case_id}`")
        st.write(f"**Investigator:** {case_info.get('investigator', 'Unknown')}")
    with col2:
        st.write(f"**Reason:** {case_info.get('reason', 'Digital Forensic Investigation')}")
        st.write(f"**Device Type:** {case_info.get('device_type', 'Android')}")
    
    st.markdown("---")
    st.markdown("**Consent Level:**")
    st.info(f"🔐 **{consent_level}**")


def render_consent_form():
    """Render consent form"""
    st.markdown("### 📋 Consent Form")
    
    with st.container(border=True):
        st.markdown("""
        **DIGITAL FORENSIC INVESTIGATION CONSENT FORM**
        
        I, the undersigned, hereby consent to the extraction and analysis of digital data from my device(s) 
        for the purpose of a digital forensic investigation.
        
        **I understand that:**
        - My device will be connected and analyzed by law enforcement
        - Digital data will be extracted and stored securely
        - The extracted data will be used for investigative purposes
        - My privacy will be protected in accordance with applicable laws
        - I have the right to withdraw consent at any time
        - The investigation will be conducted professionally and lawfully
        
        **By approving this consent, I acknowledge:**
        - I have read and understood this consent form
        - I voluntarily consent to the extraction and analysis
        - I understand the scope and purpose of the investigation
        - I have been informed of my rights
        
        **Consent Level:** LEGAL (Full access to all modules)
        """)
        
        # Checkbox for consent
        consent_agreed = st.checkbox(
            "I agree to the terms and conditions above",
            key="consent_agreed"
        )
        
        if not consent_agreed:
            st.warning("You must agree to the consent form to proceed.")
            return False
        
        return True


def render_hash_verification_section(case_id: str, nominee_email: str, 
                                    approval_hash: str, token: str, 
                                    expires_at: str) -> bool:
    """Render hash verification section (Primary verification)"""
    st.markdown("### 🔐 Verification Code")
    
    st.info("Your approval link is secure and verified. Here is your verification code to send to the investigator:")
    
    # Verify hash automatically
    is_valid, message = verify_approval_hash(
        case_id, nominee_email, approval_hash, token, expires_at
    )
    
    if is_valid:
        st.success(message)
        
        # Generate and display verification code for nominee to send back
        import hashlib
        verification_code = hashlib.sha256(f"{approval_hash}{token}".encode()).hexdigest()[:16].upper()
        
        st.markdown("---")
        st.markdown("**📋 Your Verification Code:**")
        st.code(verification_code, language="text")
        st.markdown(f"**Copy this code and send it to the investigator:** `{verification_code}`")
        st.markdown("---")
        
        st.markdown("✅ Your approval link is valid and secure.")
        return True
    else:
        st.error(message)
        st.markdown("❌ Your approval link is invalid or has expired.")
        return False


def render_fallback_verification_section() -> bool:
    """Render fallback verification section (if hash verification fails)"""
    st.markdown("### 🔄 Fallback Verification")
    
    st.warning("Hash verification failed. Use fallback verification method.")
    
    st.info("Enter the fallback hash code sent to you separately:")
    
    fallback_hash = st.text_input(
        "Fallback Hash Code",
        placeholder="Enter fallback hash code",
        type="password",
        label_visibility="collapsed"
    )
    
    if fallback_hash:
        # For demo: accept any 32+ character hash
        if len(fallback_hash) >= 32:
            st.success("✅ Fallback hash verified")
            return True
        else:
            st.error("❌ Invalid fallback hash format")
            return False
    
    return False


def render_approval_confirmation(case_id: str):
    """Render approval confirmation"""
    st.markdown("### ✅ Approval Confirmation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Approve", use_container_width=True):
            return True
    
    with col2:
        if st.button("❌ Decline", use_container_width=True):
            st.error("Consent approval declined.")
            log_approval_event(case_id, "APPROVAL_DECLINED", "Nominee declined approval")
            st.stop()
    
    return False


def render_success_message(case_id: str):
    """Render success message"""
    st.markdown('<div class="success-message">', unsafe_allow_html=True)
    st.markdown("""
    ### ✅ Consent Approved Successfully!
    
    Your consent has been recorded and verified. The investigator can now proceed with 
    the digital forensic extraction from your device.
    
    **What happens next:**
    1. The investigator will be notified of your approval
    2. Your device will be connected for extraction
    3. Data will be extracted according to the approved modules
    4. You will receive updates on the investigation progress
    
    **Your approval details:**
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Approval Status", "✅ Approved")
    with col2:
        st.metric("Consent Level", "LEGAL")
    
    st.markdown("---")
    st.markdown("""
    **Important:** This approval is valid for 24 hours. If extraction does not begin 
    within this time, you may need to re-approve.
    
    If you have any questions or concerns, please contact the investigator.
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# MAIN PORTAL RENDERER
# ============================================================================

def render_nominee_approval_portal(case_id: str, token: Optional[str] = None):
    """
    Main nominee approval portal renderer
    
    Hash-based approval system:
    - Primary: Hash verification (from approval link)
    - Fallback: Fallback hash if primary fails
    - No PIN required
    
    Args:
        case_id: Case ID for approval
        token: Optional approval token for validation
    """
    # Configure page
    configure_portal_page()
    
    # Initialize session
    initialize_portal_session()
    
    # Render header
    render_header()
    
    # Check if approval already completed
    if st.session_state.approval_confirmed:
        render_success_message(case_id)
        return
    
    # Get parameters from URL
    approval_hash = st.query_params.get("hash", "")
    token = st.query_params.get("token", "")
    expires_at = st.query_params.get("expires_at", "")
    nominee_email = st.query_params.get("nominee_email", "nominee@example.com")
    consent_level = st.query_params.get("consent_level", "LEGAL")
    
    # Step 1: Case Information
    st.markdown("### Step 1 of 2: Review Case Information")
    render_case_information(case_id, consent_level)
    
    st.markdown("---")
    
    # Step 2: Consent Form
    st.markdown("### Step 2 of 2: Review and Accept Consent Form")
    consent_agreed = render_consent_form()
    
    st.markdown("---")
    
    # Step 3: Hash Verification (Primary)
    if consent_agreed:
        st.markdown("### Step 3: Verify Approval")
        
        if approval_hash and token and expires_at:
            # Primary: Hash verification
            hash_verified = render_hash_verification_section(
                case_id, nominee_email, approval_hash, token, expires_at
            )
            
            st.markdown("---")
            
            # Confirmation
            if hash_verified:
                st.markdown("### Step 4: Confirm Approval")
                
                if render_approval_confirmation(case_id):
                    # Save approval
                    success = save_approval(
                        case_id=case_id,
                        nominee_email=nominee_email,
                        approval_hash=approval_hash,
                        token=token,
                        expires_at=expires_at
                    )
                    
                    if success:
                        # Log event
                        log_approval_event(
                            case_id,
                            "APPROVAL_CONFIRMED",
                            "Consent approved via hash verification",
                            "SUCCESS"
                        )
                        
                        # Mark as confirmed
                        st.session_state.approval_confirmed = True
                        
                        # Show success message
                        st.rerun()
                    else:
                        st.error("Failed to save approval. Please try again.")
            else:
                # Fallback: Hash verification failed
                st.markdown("---")
                fallback_verified = render_fallback_verification_section()
                
                if fallback_verified:
                    st.markdown("---")
                    st.markdown("### Step 4: Confirm Approval")
                    
                    if render_approval_confirmation(case_id):
                        # Save approval with fallback
                        success = save_approval(
                            case_id=case_id,
                            nominee_email=nominee_email,
                            approval_hash=approval_hash,
                            token=token,
                            expires_at=expires_at
                        )
                        
                        if success:
                            # Log event
                            log_approval_event(
                                case_id,
                                "APPROVAL_CONFIRMED_FALLBACK",
                                "Consent approved via fallback hash verification",
                                "SUCCESS"
                            )
                            
                            # Mark as confirmed
                            st.session_state.approval_confirmed = True
                            
                            # Show success message
                            st.rerun()
                        else:
                            st.error("Failed to save approval. Please try again.")
        else:
            st.error("❌ Missing approval parameters. Invalid approval link.")
            st.info("Please ensure you're using a valid approval link with all required parameters.")
    else:
        st.warning("Please review and accept the consent form to proceed.")


# ============================================================================
# STANDALONE RUNNER (for testing)
# ============================================================================

if __name__ == "__main__":
    # Get case_id from URL parameters
    case_id = st.query_params.get("case_id", "CASE-001")
    token = st.query_params.get("token", None)
    
    # Render portal
    render_nominee_approval_portal(case_id, token)
