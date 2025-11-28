"""
CONSENT UI MODULE - Streamlit UI Components
Handles consent forms, approval portals, and consent management UI

This module provides:
- Consent form rendering
- Approval portal UI
- Consent management dashboard
- Approval link generation UI
- Testing dashboard
"""

import streamlit as st
import os
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any, List
from modules.consent.models import (
    ConsentManager,
    ConsentLevel,
    ApprovalLinkGenerator,
    InstantApprovalSync,
    ConsentTestingLoopholes,
    get_consent_manager,
    MODULE_MIN_LEVELS,
    NotificationHandler
)


# ============================================================================
# CONSENT FORM RENDERING
# ============================================================================

def render_consent_form(
    case_id: str,
    investigator_name: str,
    device_info: str,
    on_approve: Callable[[str, ConsentLevel, str], None]
) -> None:
    """Render consent form for nominee"""
    
    st.markdown("# 📋 CONSENT APPROVAL FORM")
    st.markdown("---")
    
    # Case Information
    st.markdown("## 📁 Case Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Case ID:** {case_id}")
        st.write(f"**Investigator:** {investigator_name}")
    with col2:
        st.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Device:** {device_info}")
    
    st.markdown("---")
    
    # Consent Information
    st.markdown("## 🔐 What You're Consenting To")
    
    st.markdown("""
    By approving this consent form, you authorize the investigator to extract and analyze 
    data from your device. The data will be used solely for this investigation.
    
    **Your rights:**
    - You can withdraw consent at any time
    - Your data will be stored securely
    - Only authorized personnel will access your data
    - Your data will not be shared without your permission
    """)
    
    st.markdown("---")
    
    # Consent Level Selection
    st.markdown("## 📊 Consent Level")
    
    consent_level = st.radio(
        "Select the level of access you grant:",
        options=[
            ("🟡 STANDARD - Device + Location + Media + Security", "STANDARD"),
            ("🟠 LEGAL - All data including Communications", "LEGAL"),
            ("🔴 FULL - Complete access including System logs", "FULL")
        ],
        format_func=lambda x: x[0]
    )
    
    selected_level = consent_level[1]
    
    st.markdown("---")
    
    # Approval Method
    st.markdown("## ✅ Approval Method")
    
    approval_method = st.radio(
        "Choose how you want to approve:",
        options=["PIN Code", "Pattern", "Biometric"]
    )
    
    if approval_method == "PIN Code":
        pin = st.text_input("Enter 4-digit PIN:", type="password", max_chars=4)
        if len(pin) == 4 and pin.isdigit():
            if st.button("✅ Approve with PIN", use_container_width=True, type="primary"):
                on_approve(case_id, ConsentLevel[selected_level], "PIN")
                st.success("✅ Consent approved! Extraction starting...")
                st.balloons()
    
    elif approval_method == "Pattern":
        st.write("Draw pattern to approve (simulated)")
        if st.button("✅ Approve with Pattern", use_container_width=True, type="primary"):
            on_approve(case_id, ConsentLevel[selected_level], "PATTERN")
            st.success("✅ Consent approved! Extraction starting...")
            st.balloons()
    
    elif approval_method == "Biometric":
        st.write("Use biometric to approve (simulated)")
        if st.button("✅ Approve with Biometric", use_container_width=True, type="primary"):
            on_approve(case_id, ConsentLevel[selected_level], "BIOMETRIC")
            st.success("✅ Consent approved! Extraction starting...")
            st.balloons()


# ============================================================================
# APPROVAL LINK GENERATION UI
# ============================================================================

def render_approval_link_generator(
    case_id: str,
    approval_link_generator: ApprovalLinkGenerator
) -> None:
    """Render approval link generation UI"""
    
    st.markdown("## 🔗 Generate Approval Link")
    
    col1, col2 = st.columns(2)
    
    with col1:
        expiry_hours = st.slider("Link expiry (hours):", 1, 24, 1)
    
    with col2:
        if st.button("🔗 Generate Link", use_container_width=True):
            approval_link = approval_link_generator.generate_link(case_id, expiry_hours)
            st.success("✅ Approval link generated!")
            st.code(approval_link, language="text")
            
            # Copy to clipboard
            st.write("**Share this link with the nominee:**")
            st.info(approval_link)
    
    # Share options
    st.markdown("### 📧 Share Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📧 Send via Email", use_container_width=True):
            st.success("✅ Email sent to nominee!")
    
    with col2:
        if st.button("📱 Send via SMS", use_container_width=True):
            st.success("✅ SMS sent to nominee!")
    
    with col3:
        if st.button("🔗 Copy Link", use_container_width=True):
            st.success("✅ Link copied to clipboard!")


# ============================================================================
# CONSENT MANAGEMENT DASHBOARD
# ============================================================================

def render_consent_management_dashboard(
    consent_manager: ConsentManager
) -> None:
    """Render consent management dashboard"""
    
    st.markdown("## 🔐 Consent Management")
    
    tab1, tab2, tab3 = st.tabs(["Active Consents", "Audit Trail", "Revoke Consent"])
    
    with tab1:
        st.markdown("### 📋 Active Consents")
        
        if not consent_manager.sessions:
            st.info("No active consents")
        else:
            consent_data = []
            for case_id, session in consent_manager.sessions.items():
                consent_data.append({
                    "Case ID": case_id,
                    "Consent Level": session.level.name,
                    "Approved By": session.approved_by,
                    "Approval Method": session.approval_method,
                    "Timestamp": session.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                })
            
            import pandas as pd
            df = pd.DataFrame(consent_data)
            st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 Audit Trail")
        
        case_id = st.selectbox("Select case:", list(consent_manager.sessions.keys()))
        
        if case_id:
            audit_trail = consent_manager.get_audit_trail(case_id)
            
            if not audit_trail:
                st.info("No audit trail for this case")
            else:
                audit_data = []
                for trail in audit_trail:
                    audit_data.append({
                        "Event": trail.event,
                        "Actor": trail.actor,
                        "Role": trail.actor_role,
                        "Consent Level": trail.consent_level,
                        "Timestamp": trail.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "IP Address": trail.ip_address or "N/A"
                    })
                
                import pandas as pd
                df = pd.DataFrame(audit_data)
                st.dataframe(df, use_container_width=True)
    
    with tab3:
        st.markdown("### 🚫 Revoke Consent")
        
        case_id = st.selectbox("Select case to revoke:", list(consent_manager.sessions.keys()))
        
        if case_id:
            if st.button("🚫 Revoke Consent", use_container_width=True, type="secondary"):
                consent_manager.revoke_consent(case_id, "INVESTIGATOR")
                st.success(f"✅ Consent revoked for case {case_id}")
                st.rerun()


# ============================================================================
# TESTING DASHBOARD
# ============================================================================

def render_testing_loophole_toggles() -> Dict[str, bool]:
    """Render testing loophole toggle buttons"""
    
    if not ConsentTestingLoopholes.is_testing_mode():
        return {}
    
    st.markdown("## 🔧 TESTING LOOPHOLE TOGGLES")
    st.warning("⚠️ Testing mode enabled - Use toggles carefully")
    
    col1, col2, col3, col4 = st.columns(4)
    
    toggles = {}
    
    with col1:
        toggles['bypass_mode'] = st.toggle(
            "🔓 Bypass Mode",
            value=os.getenv('CONSENT_BYPASS_MODE', 'false').lower() == 'true',
            help="Skip consent checks entirely"
        )
    
    with col2:
        toggles['auto_approve'] = st.toggle(
            "✅ Auto-Approve",
            value=os.getenv('CONSENT_AUTO_APPROVE', 'false').lower() == 'true',
            help="Automatically approve all consents"
        )
    
    with col3:
        toggles['skip_audit'] = st.toggle(
            "📝 Skip Audit",
            value=os.getenv('CONSENT_SKIP_AUDIT', 'false').lower() == 'true',
            help="Skip audit trail logging"
        )
    
    with col4:
        toggles['instant_approval'] = st.toggle(
            "⚡ Instant Approval",
            value=os.getenv('CONSENT_INSTANT_APPROVAL', 'false').lower() == 'true',
            help="Instant approval without waiting"
        )
    
    return toggles


def render_testing_dashboard(
    consent_manager: ConsentManager,
    approval_link_generator: ApprovalLinkGenerator
) -> None:
    """Render testing dashboard with loopholes"""
    
    if not ConsentTestingLoopholes.is_testing_mode():
        st.error("❌ Testing mode not enabled")
        return
    
    st.markdown("# 🧪 TESTING DASHBOARD")
    st.warning("⚠️ This dashboard is only available in TESTING mode")
    
    # Render toggle buttons
    toggles = render_testing_loophole_toggles()
    
    st.markdown("---")
    
    st.markdown("---")
    
    # Testing Controls
    st.markdown("## 🎮 Testing Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ✅ Auto-Approve")
        case_id = st.text_input("Case ID:", key="auto_approve_case")
        if st.button("Auto-Approve Consent", use_container_width=True):
            if case_id:
                try:
                    ConsentTestingLoopholes.auto_approve_consent(consent_manager, case_id, 'LEGAL')
                    st.success(f"✅ Auto-approved consent for {case_id}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    with col2:
        st.markdown("### 🔗 Instant Link")
        case_id = st.text_input("Case ID:", key="instant_link_case")
        if st.button("Create Instant Link", use_container_width=True):
            if case_id:
                link = approval_link_generator.generate_link(case_id, 1)
                st.success(f"✅ Instant link created")
                st.code(link)
        
        st.markdown("### 📋 Mock Consent")
        consent_level = st.selectbox("Consent Level:", ["STANDARD", "LEGAL", "FULL"], key="mock_level")
    
    with col3:
        st.markdown("### 🔄 Reset All")
        if st.button("Reset All Consents", use_container_width=True, type="secondary"):
            ConsentTestingLoopholes.reset_all_consents(consent_manager)
            st.success("✅ All consents reset")
            st.rerun()
    
    st.markdown("---")
    
    # Mock Consent Management
    st.markdown("## 📋 Mock Consent Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ➕ Create Mock Consent")
        case_id = st.text_input("Case ID:", key="mock_case")
        consent_level = st.selectbox("Consent Level:", ["STANDARD", "LEGAL", "FULL"], key="mock_consent_level")
        if st.button("Create Mock", use_container_width=True):
            if case_id:
                ConsentTestingLoopholes.create_mock_consent(consent_manager, case_id, consent_level)
                st.success(f"✅ Mock consent created")
    
    with col2:
        st.markdown("### 📊 View Status")
        if st.button("Refresh Status", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # Testing Status
    st.markdown("## 📊 Testing Status")
    
    status_info = f"""
    **Testing Mode**: {os.getenv('TESTING', 'false')}
    **Bypass Mode**: {os.getenv('CONSENT_BYPASS_MODE', 'false')}
    **Auto-Approve**: {os.getenv('CONSENT_AUTO_APPROVE', 'false')}
    **Skip Audit**: {os.getenv('CONSENT_SKIP_AUDIT', 'false')}
    **Instant Approval**: {os.getenv('CONSENT_INSTANT_APPROVAL', 'false')}
    **Test User**: {os.getenv('TEST_USER_ID', 'test_user_123')}
    **Test Device**: {os.getenv('TEST_DEVICE_ID', 'test_device_123')}
    
    **Active Consents**: {len(consent_manager.sessions)}
    **Audit Trails**: {len(consent_manager.audit_trails)}
    """
    
    st.info(status_info)


# ============================================================================
# CONSENT STATUS DISPLAY
# ============================================================================

def render_consent_status(
    case_id: str,
    consent_manager: ConsentManager
) -> None:
    """Render consent status display"""
    
    session = consent_manager.get_session(case_id)
    
    if not session:
        st.warning(f"⚠️ No consent for case {case_id}")
        return
    
    st.markdown("## 🔐 Consent Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Consent Level", session.level.name)
    
    with col2:
        st.metric("Approved By", session.approved_by)
    
    with col3:
        st.metric("Approval Method", session.approval_method)
    
    with col4:
        st.metric("Timestamp", session.timestamp.strftime("%H:%M:%S"))
    
    # Module access
    st.markdown("### 📦 Module Access")
    
    module_access = []
    for module_name, min_level in MODULE_MIN_LEVELS.items():
        has_access = session.level >= min_level
        status = "✅ ALLOWED" if has_access else "❌ BLOCKED"
        module_access.append({
            "Module": module_name.replace('_', ' ').title(),
            "Required": min_level.name,
            "Status": status
        })
    
    import pandas as pd
    df = pd.DataFrame(module_access)
    st.dataframe(df, use_container_width=True)


# ============================================================================
# CONSENT PREVIEW BEFORE APPROVAL
# ============================================================================

def render_consent_preview(case_id: str, consent_level: ConsentLevel) -> bool:
    """Render consent preview before approval"""
    
    st.markdown("## 📋 Consent Preview")
    st.info("Please review the following consent details before approving")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Case Information")
        st.write(f"**Case ID**: {case_id}")
        st.write(f"**Consent Level**: {consent_level.name}")
        st.write(f"**Preview Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col2:
        st.markdown("### Consent Scope")
        
        scope_info = {
            ConsentLevel.STANDARD: [
                '✅ Device Information',
                '✅ Location Data',
                '✅ Security Settings',
                '✅ Media Files',
                '❌ Communications',
                '❌ System Files'
            ],
            ConsentLevel.LEGAL: [
                '✅ Device Information',
                '✅ Location Data',
                '✅ Security Settings',
                '✅ Media Files',
                '✅ Communications',
                '❌ System Files'
            ],
            ConsentLevel.FULL: [
                '✅ Device Information',
                '✅ Location Data',
                '✅ Security Settings',
                '✅ Media Files',
                '✅ Communications',
                '✅ System Files'
            ]
        }
        
        for item in scope_info.get(consent_level, []):
            st.write(item)
    
    st.markdown("### Consent Terms")
    
    with st.expander("📖 Full Terms & Conditions"):
        st.markdown("""
        **FORENSMART CONSENT AGREEMENT**
        
        By approving this consent, you authorize:
        
        1. **Data Collection**: Collection of device data as per consent level
        2. **Data Processing**: Processing of collected data for investigation
        3. **Data Storage**: Secure storage of data for case duration
        4. **Data Sharing**: Sharing with authorized investigators only
        5. **Data Retention**: Retention per legal requirements
        
        **Your Rights:**
        - Right to revoke consent at any time
        - Right to access collected data
        - Right to request data deletion
        - Right to receive notifications
        """)
    
    st.markdown("### Approval")
    
    col1, col2 = st.columns(2)
    
    with col1:
        agree = st.checkbox("I have read and agree to the consent terms")
    
    with col2:
        if agree:
            if st.button("✅ Approve Consent", use_container_width=True, type="primary"):
                st.success("✅ Consent approved successfully!")
                return True
        else:
            st.button("✅ Approve Consent", use_container_width=True, disabled=True)
    
    return False


# ============================================================================
# CONSENT MODIFICATION UI
# ============================================================================

def render_consent_modification(case_id: str) -> Optional[Dict[str, Any]]:
    """Render consent modification UI"""
    
    st.markdown("## ✏️ Modify Consent")
    
    consent_manager = get_consent_manager()
    session = consent_manager.get_session(case_id)
    
    if not session:
        st.error(f"❌ No consent session found for case {case_id}")
        return None
    
    st.info(f"Current Consent Level: **{session.level.name}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Upgrade Consent")
        higher_levels = [level for level in ConsentLevel if level.value > session.level.value]
        if higher_levels:
            new_level = st.selectbox(
                "Upgrade to level:",
                higher_levels,
                format_func=lambda x: x.name,
                key="upgrade_level"
            )
            
            if st.button("⬆️ Upgrade Consent", use_container_width=True):
                success = consent_manager.upgrade_consent_level(
                    case_id,
                    new_level,
                    actor="investigator@forensmart.com"
                )
                if success:
                    st.success(f"✅ Consent upgraded to {new_level.name}")
                else:
                    st.error("❌ Failed to upgrade consent")
        else:
            st.info("Already at highest consent level")
    
    with col2:
        st.markdown("### Downgrade Consent")
        lower_levels = [level for level in ConsentLevel if level.value < session.level.value]
        if lower_levels:
            new_level = st.selectbox(
                "Downgrade to level:",
                lower_levels,
                format_func=lambda x: x.name,
                key="downgrade_level"
            )
            
            if st.button("⬇️ Downgrade Consent", use_container_width=True):
                success = consent_manager.downgrade_consent_level(
                    case_id,
                    new_level,
                    actor="investigator@forensmart.com"
                )
                if success:
                    st.success(f"✅ Consent downgraded to {new_level.name}")
                else:
                    st.error("❌ Failed to downgrade consent")
        else:
            st.info("Already at lowest consent level")
    
    return {'case_id': case_id}


# ============================================================================
# CONSENT REVOCATION CONFIRMATION
# ============================================================================

def render_consent_revocation_confirmation(case_id: str) -> bool:
    """Render consent revocation confirmation dialog"""
    
    st.markdown("## ⚠️ Revoke Consent")
    st.warning("⚠️ This action cannot be undone!")
    
    consent_manager = get_consent_manager()
    session = consent_manager.get_session(case_id)
    
    if not session:
        st.error(f"❌ No consent session found for case {case_id}")
        return False
    
    st.markdown("### Revocation Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Case ID**: {case_id}")
        st.write(f"**Consent Level**: {session.level.name}")
        st.write(f"**Approved By**: {session.approved_by}")
    
    with col2:
        st.write(f"**Approval Date**: {session.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Approval Method**: {session.approval_method}")
    
    st.markdown("### Confirmation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confirm = st.checkbox("I understand this action cannot be undone")
    
    with col2:
        reason = st.text_input("Reason for revocation (optional):")
    
    with col3:
        if confirm:
            if st.button("🚫 Revoke Consent", use_container_width=True, type="secondary"):
                success = consent_manager.revoke_consent(case_id, "INVESTIGATOR")
                if success:
                    st.error(f"🚫 Consent revoked for case {case_id}")
                    NotificationHandler.notify_consent_revocation(
                        case_id,
                        nominee_email=session.approved_by
                    )
                    return True
                else:
                    st.error("❌ Failed to revoke consent")
        else:
            st.button("🚫 Revoke Consent", use_container_width=True, disabled=True)
    
    return False


# ============================================================================
# CONSENT EXPIRY WARNINGS
# ============================================================================

def render_consent_expiry_warnings(consent_manager: ConsentManager) -> None:
    """Render consent expiry warnings"""
    
    st.markdown("## ⏰ Consent Expiry Warnings")
    
    # Get expiring consents
    expiring_24h = consent_manager.get_expiring_consents(hours=24)
    expiring_7d = consent_manager.get_expiring_consents(hours=168)
    
    if not expiring_24h and not expiring_7d:
        st.success("✅ No expiring consents")
        return
    
    # 24-hour warnings
    if expiring_24h:
        st.error(f"🔴 {len(expiring_24h)} consent(s) expiring within 24 hours!")
        
        for consent in expiring_24h:
            with st.expander(f"⏰ {consent['case_id']} - Expires in {consent['hours_remaining']:.1f}h"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Case ID**: {consent['case_id']}")
                    st.write(f"**Level**: {consent['level']}")
                
                with col2:
                    st.write(f"**Expires At**: {consent['expires_at']}")
                    st.write(f"**Hours Remaining**: {consent['hours_remaining']:.1f}h")
                
                with col3:
                    if st.button(f"🔄 Extend {consent['case_id']}", use_container_width=True):
                        st.info(f"Consent extension initiated for {consent['case_id']}")
    
    # 7-day warnings
    if expiring_7d and not expiring_24h:
        st.warning(f"🟡 {len(expiring_7d)} consent(s) expiring within 7 days")


# ============================================================================
# BULK CONSENT OPERATIONS
# ============================================================================

def render_bulk_consent_operations(consent_manager: ConsentManager) -> None:
    """Render bulk consent operations UI"""
    
    st.markdown("## 📦 Bulk Consent Operations")
    
    tab1, tab2, tab3 = st.tabs(["Bulk Create", "Bulk Upgrade", "Bulk Revoke"])
    
    with tab1:
        st.markdown("### ➕ Bulk Create Consents")
        
        case_ids_text = st.text_area(
            "Enter case IDs (one per line):",
            height=100,
            placeholder="CASE-001\nCASE-002\nCASE-003"
        )
        
        consent_level = st.selectbox(
            "Consent Level:",
            [ConsentLevel.STANDARD, ConsentLevel.LEGAL, ConsentLevel.FULL],
            format_func=lambda x: x.name,
            key="bulk_create_level"
        )
        
        if st.button("➕ Create Bulk Consents", use_container_width=True):
            case_ids = [cid.strip() for cid in case_ids_text.split('\n') if cid.strip()]
            if case_ids:
                results = consent_manager.batch_create_sessions(
                    case_ids,
                    consent_level,
                    "bulk_operator@forensmart.com",
                    "BULK_OPERATION"
                )
                
                successful = sum(1 for v in results.values() if v)
                st.success(f"✅ {successful}/{len(case_ids)} consents created successfully")
                
                with st.expander("📊 Results"):
                    for case_id, success in results.items():
                        status = "✅" if success else "❌"
                        st.write(f"{status} {case_id}")
    
    with tab2:
        st.markdown("### ⬆️ Bulk Upgrade Consents")
        
        case_ids_text = st.text_area(
            "Enter case IDs (one per line):",
            height=100,
            placeholder="CASE-001\nCASE-002\nCASE-003",
            key="bulk_upgrade_cases"
        )
        
        new_level = st.selectbox(
            "Upgrade to Level:",
            [ConsentLevel.LEGAL, ConsentLevel.FULL],
            format_func=lambda x: x.name,
            key="bulk_upgrade_level"
        )
        
        if st.button("⬆️ Upgrade Bulk Consents", use_container_width=True):
            case_ids = [cid.strip() for cid in case_ids_text.split('\n') if cid.strip()]
            if case_ids:
                results = consent_manager.batch_upgrade_consents(
                    case_ids,
                    new_level,
                    "bulk_operator@forensmart.com"
                )
                
                successful = sum(1 for v in results.values() if v)
                st.success(f"✅ {successful}/{len(case_ids)} consents upgraded successfully")
    
    with tab3:
        st.markdown("### 🚫 Bulk Revoke Consents")
        
        case_ids_text = st.text_area(
            "Enter case IDs (one per line):",
            height=100,
            placeholder="CASE-001\nCASE-002\nCASE-003",
            key="bulk_revoke_cases"
        )
        
        if st.button("🚫 Revoke Bulk Consents", use_container_width=True, type="secondary"):
            case_ids = [cid.strip() for cid in case_ids_text.split('\n') if cid.strip()]
            if case_ids:
                results = consent_manager.batch_revoke_consents(
                    case_ids,
                    "bulk_operator@forensmart.com"
                )
                
                successful = sum(1 for v in results.values() if v)
                st.success(f"✅ {successful}/{len(case_ids)} consents revoked successfully")


# ============================================================================
# CONSENT TEMPLATES
# ============================================================================

def render_consent_templates() -> Optional[Dict[str, Any]]:
    """Render consent templates"""
    
    st.markdown("## 📝 Consent Templates")
    
    templates = {
        'Standard Investigation': {
            'level': ConsentLevel.STANDARD,
            'description': 'Basic device data for standard investigation',
            'modules': ['device_info', 'location', 'security', 'media']
        },
        'Legal Investigation': {
            'level': ConsentLevel.LEGAL,
            'description': 'Full data including communications for legal investigation',
            'modules': ['device_info', 'location', 'security', 'media', 'communications']
        },
        'Full Forensic Analysis': {
            'level': ConsentLevel.FULL,
            'description': 'Complete access including system files for forensic analysis',
            'modules': ['device_info', 'location', 'security', 'media', 'communications', 'system']
        }
    }
    
    selected_template = st.selectbox(
        "Select a template:",
        list(templates.keys())
    )
    
    template = templates[selected_template]
    
    st.markdown("### Template Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Consent Level**: {template['level'].name}")
        st.write(f"**Description**: {template['description']}")
    
    with col2:
        st.write(f"**Modules**: {', '.join(template['modules'])}")
    
    st.markdown("### Apply Template")
    
    case_id = st.text_input("Enter case ID to apply template:")
    
    if st.button("✅ Apply Template", use_container_width=True):
        if case_id:
            consent_manager = get_consent_manager()
            session = consent_manager.create_session(
                case_id,
                template['level'],
                "template_operator@forensmart.com",
                "TEMPLATE"
            )
            
            if session:
                st.success(f"✅ Template '{selected_template}' applied to {case_id}")
                return {'case_id': case_id, 'template': selected_template, 'level': template['level']}
            else:
                st.error("❌ Failed to apply template")
    
    return None
