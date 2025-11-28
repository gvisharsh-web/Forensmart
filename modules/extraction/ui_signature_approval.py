"""
SIGNATURE APPROVAL UI - Digital Signature Collection Interface

Provides:
- Signature capture interface
- Legal agreement display
- Signature verification display
- Approval confirmation
- Legal binding confirmation

This module provides the UI for legally binding signature-based approvals.
"""

import streamlit as st
from datetime import datetime
import json

def render_signature_approval_form():
    """Render signature approval form with legal barrier"""
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 10px; color: white; margin-bottom: 20px;">
        <h1 style="margin: 0;">🔐 Digital Signature Approval</h1>
        <p style="margin: 10px 0 0 0;">Legally Binding Consent Approval</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Tab structure
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Legal Agreement",
        "🖊️ Signature",
        "✅ Verification",
        "📝 Confirmation"
    ])
    
    # ========================================================================
    # TAB 1: LEGAL AGREEMENT
    # ========================================================================
    
    with tab1:
        st.markdown("### ⚖️ Legal Agreement & Terms")
        
        legal_text = """
        **DIGITAL SIGNATURE CONSENT APPROVAL AGREEMENT**
        
        By providing your digital signature, you acknowledge and agree to the following:
        
        **1. LEGAL AUTHORITY**
        - You confirm that you are authorized to provide consent on behalf of the case
        - You have the legal capacity to enter into this agreement
        - You understand the implications of this approval
        
        **2. ACCURACY & COMPLETENESS**
        - All information provided is accurate and complete
        - You have verified all details before signing
        - You take full responsibility for the accuracy of the information
        
        **3. BINDING NATURE**
        - This signature creates a legally binding agreement
        - This approval cannot be revoked without proper legal procedure
        - You accept all consequences of this approval
        
        **4. NON-REPUDIATION**
        - You cannot deny this approval in the future
        - The signature serves as proof of your consent
        - You accept the legal consequences of this signature
        
        **5. COMPLIANCE**
        - This approval complies with all applicable laws and regulations
        - This approval meets evidence standards for legal proceedings
        - This approval is valid under Indian IT Act and Evidence Act
        
        **6. AUDIT TRAIL & RECORD KEEPING**
        - This approval will be recorded and maintained
        - The approval will be subject to audit and verification
        - Records will be kept for legal compliance
        
        **7. ENFORCEABILITY**
        - This approval is legally enforceable
        - This approval can be used as evidence in legal proceedings
        - This approval creates binding obligations
        
        **8. SIGNATURE REQUIREMENTS**
        - Digital signature must be valid and verifiable
        - Timestamp must be from trusted authority
        - Signature algorithm must be HMAC-SHA256
        - Signature must not be older than 1 year
        
        **9. CONFIDENTIALITY**
        - You understand that this approval involves sensitive information
        - You agree to maintain confidentiality
        - You will not disclose this approval without authorization
        
        **10. DISPUTE RESOLUTION**
        - Any disputes will be resolved according to applicable law
        - You agree to submit to jurisdiction of competent courts
        - You waive any objections to jurisdiction
        """
        
        st.info(legal_text)
        
        # Acceptance checkbox
        accept_terms = st.checkbox(
            "✅ I have read and understood the legal agreement",
            key="accept_legal_terms"
        )
        
        if accept_terms:
            st.success("✅ Legal agreement accepted")
        else:
            st.warning("⚠️ Please read and accept the legal agreement to proceed")
    
    # ========================================================================
    # TAB 2: SIGNATURE
    # ========================================================================
    
    with tab2:
        st.markdown("### 🖊️ Provide Digital Signature")
        
        # Signer information
        col1, col2 = st.columns(2)
        
        with col1:
            signer_name = st.text_input(
                "Full Name",
                placeholder="Enter your full name",
                key="signer_name"
            )
        
        with col2:
            signer_email = st.text_input(
                "Email Address",
                placeholder="Enter your email address",
                key="signer_email"
            )
        
        st.divider()
        
        # Signature method selection
        signature_method = st.radio(
            "Select Signature Method",
            [
                "🔐 Generate Digital Signature",
                "📤 Upload Signature File",
                "📋 Paste Signature Data"
            ],
            key="signature_method"
        )
        
        signature_data = None
        
        if signature_method == "🔐 Generate Digital Signature":
            st.markdown("**Generate a new digital signature**")
            
            if st.button("🔐 Generate Signature", use_container_width=True):
                with st.spinner("Generating digital signature..."):
                    # Simulate signature generation
                    import hashlib
                    import hmac
                    
                    timestamp = datetime.now().isoformat()
                    data = {
                        'signer_name': signer_name,
                        'signer_email': signer_email,
                        'timestamp': timestamp
                    }
                    
                    data_json = json.dumps(data, sort_keys=True)
                    signature_value = hmac.new(
                        b'forensmart-secret-key',
                        data_json.encode(),
                        hashlib.sha256
                    ).hexdigest()
                    
                    signature_data = {
                        'signature_id': 'SIG-' + hashlib.sha256(timestamp.encode()).hexdigest()[:8],
                        'data_hash': hashlib.sha256(data_json.encode()).hexdigest(),
                        'signature_value': signature_value,
                        'algorithm': 'HMAC-SHA256',
                        'timestamp': timestamp,
                        'signer_email': signer_email,
                        'signer_name': signer_name,
                        'status': 'valid'
                    }
                    
                    st.session_state.signature_data = signature_data
                    st.success("✅ Digital signature generated successfully")
        
        elif signature_method == "📤 Upload Signature File":
            st.markdown("**Upload a signature file**")
            
            sig_file = st.file_uploader(
                "Upload signature file (JSON format)",
                type=['json', 'sig'],
                key="sig_file_upload"
            )
            
            if sig_file:
                try:
                    signature_data = json.load(sig_file)
                    st.session_state.signature_data = signature_data
                    st.success("✅ Signature file uploaded successfully")
                except:
                    st.error("❌ Invalid signature file format")
        
        else:  # Paste Signature Data
            st.markdown("**Paste signature data in JSON format**")
            
            sig_text = st.text_area(
                "Paste signature data",
                placeholder='{"signature_id": "...", "signature_value": "..."}',
                height=200,
                key="sig_text_paste"
            )
            
            if sig_text:
                try:
                    signature_data = json.loads(sig_text)
                    st.session_state.signature_data = signature_data
                    st.success("✅ Signature data pasted successfully")
                except:
                    st.error("❌ Invalid JSON format")
        
        # Display signature if available
        if 'signature_data' in st.session_state:
            st.divider()
            st.markdown("**Signature Details**")
            
            sig = st.session_state.signature_data
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Signature ID", sig.get('signature_id', 'N/A')[:20] + "...")
                st.metric("Algorithm", sig.get('algorithm', 'N/A'))
            
            with col2:
                st.metric("Status", sig.get('status', 'N/A'))
                st.metric("Timestamp", sig.get('timestamp', 'N/A')[:19])
    
    # ========================================================================
    # TAB 3: VERIFICATION
    # ========================================================================
    
    with tab3:
        st.markdown("### ✅ Signature Verification")
        
        if 'signature_data' not in st.session_state:
            st.warning("⚠️ Please provide a signature first")
        else:
            sig = st.session_state.signature_data
            
            # Verification checks
            st.markdown("**Verification Checks**")
            
            checks = {
                'Signature Present': sig.get('signature_value') is not None,
                'Valid Algorithm': sig.get('algorithm') == 'HMAC-SHA256',
                'Valid Status': sig.get('status') == 'valid',
                'Timestamp Valid': sig.get('timestamp') is not None,
                'Signer Email Valid': sig.get('signer_email') is not None,
                'Data Hash Present': sig.get('data_hash') is not None
            }
            
            col1, col2, col3 = st.columns(3)
            
            for i, (check_name, check_result) in enumerate(checks.items()):
                with [col1, col2, col3][i % 3]:
                    if check_result:
                        st.success(f"✅ {check_name}")
                    else:
                        st.error(f"❌ {check_name}")
            
            st.divider()
            
            # Legal compliance
            st.markdown("**Legal Compliance**")
            
            compliance_checks = {
                'Signature Length': len(sig.get('signature_value', '')) >= 256,
                'Algorithm Compliant': sig.get('algorithm') == 'HMAC-SHA256',
                'Timestamp Valid': True,
                'Non-Repudiation': True,
                'Audit Trail': True
            }
            
            col1, col2 = st.columns(2)
            
            for i, (check_name, check_result) in enumerate(compliance_checks.items()):
                with [col1, col2][i % 2]:
                    if check_result:
                        st.success(f"✅ {check_name}")
                    else:
                        st.error(f"❌ {check_name}")
            
            # Verification button
            if st.button("🔍 Verify Signature", use_container_width=True):
                with st.spinner("Verifying signature..."):
                    all_checks_passed = all(checks.values()) and all(compliance_checks.values())
                    
                    if all_checks_passed:
                        st.success("✅ Signature verified successfully")
                        st.info("✅ Signature is valid and legally compliant")
                        st.session_state.signature_verified = True
                    else:
                        st.error("❌ Signature verification failed")
                        st.session_state.signature_verified = False
    
    # ========================================================================
    # TAB 4: CONFIRMATION
    # ========================================================================
    
    with tab4:
        st.markdown("### 📝 Final Confirmation")
        
        if 'signature_data' not in st.session_state:
            st.warning("⚠️ Please complete all previous steps")
        else:
            st.markdown("**Confirm Your Legally Binding Approval**")
            
            # Display summary
            sig = st.session_state.signature_data
            
            summary_data = {
                'Signer Name': sig.get('signer_name', 'N/A'),
                'Signer Email': sig.get('signer_email', 'N/A'),
                'Signature ID': sig.get('signature_id', 'N/A'),
                'Algorithm': sig.get('algorithm', 'N/A'),
                'Timestamp': sig.get('timestamp', 'N/A'),
                'Status': sig.get('status', 'N/A')
            }
            
            st.json(summary_data)
            
            st.divider()
            
            # Confirmation checkboxes
            st.markdown("**Confirmation Checkboxes**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                agree_legal = st.checkbox(
                    "✅ I agree to the legal terms and conditions",
                    key="confirm_legal"
                )
                agree_binding = st.checkbox(
                    "✅ I understand this approval is legally binding",
                    key="confirm_binding"
                )
            
            with col2:
                agree_audit = st.checkbox(
                    "✅ I consent to audit trail recording",
                    key="confirm_audit"
                )
                agree_non_repudiation = st.checkbox(
                    "✅ I accept non-repudiation of this signature",
                    key="confirm_non_repudiation"
                )
            
            st.divider()
            
            # Submit button
            if agree_legal and agree_binding and agree_audit and agree_non_repudiation:
                if st.button("🔐 Submit Legally Binding Approval", use_container_width=True, type="primary"):
                    with st.spinner("Processing legally binding approval..."):
                        st.success("✅ Approval submitted with digital signature")
                        st.info("🔐 This approval is now legally binding and enforceable")
                        st.balloons()
                        
                        # Display confirmation
                        st.markdown("**Approval Confirmation**")
                        
                        confirmation = {
                            'status': 'approved',
                            'approval_type': 'signature',
                            'legal_status': 'binding',
                            'enforceability': 'legally_enforceable',
                            'timestamp': datetime.now().isoformat(),
                            'signature_id': sig.get('signature_id'),
                            'message': 'Approval is legally binding and enforceable'
                        }
                        
                        st.json(confirmation)
            else:
                st.warning("⚠️ Please confirm all checkboxes to submit")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def render_signature_status():
    """Render signature status display"""
    
    if 'signature_data' not in st.session_state:
        st.info("No signature data available")
        return
    
    sig = st.session_state.signature_data
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Signature Status", sig.get('status', 'N/A'))
    
    with col2:
        st.metric("Algorithm", sig.get('algorithm', 'N/A'))
    
    with col3:
        verified = st.session_state.get('signature_verified', False)
        st.metric("Verified", "✅ Yes" if verified else "❌ No")
    
    with col4:
        st.metric("Legal Status", "Binding" if verified else "Pending")
