"""
CONSENT APPROVAL MANAGEMENT - Streamlit UI

Provides:
- Send approval links to nominees
- Track approval status
- Verify approval responses
- Manage approval history
- View approval statistics
- Revoke/Extend approvals
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Consent Approval - ForenSmart",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# IMPORTS & INITIALIZATION
# ============================================================================

try:
    from modules.extraction.consent_approval_workflow import ConsentApprovalWorkflow
    from modules.shared.api import APIClient
    from modules.shared.database import DatabaseManager
    WORKFLOW_AVAILABLE = True
except ImportError:
    WORKFLOW_AVAILABLE = False
    st.error("Consent Approval Workflow not available")

# Initialize session state
if 'workflow' not in st.session_state:
    if WORKFLOW_AVAILABLE:
        api_client = APIClient()
        database_manager = DatabaseManager()
        database_manager.connect()
        st.session_state.workflow = ConsentApprovalWorkflow(api_client, database_manager)

if 'approval_results' not in st.session_state:
    st.session_state.approval_results = []

# ============================================================================
# MAIN HEADER
# ============================================================================

st.markdown('<div class="main-header">✅ Consent Approval Management</div>', unsafe_allow_html=True)

if not WORKFLOW_AVAILABLE:
    st.error("Consent Approval Workflow not available. Please check installation.")
    st.stop()

# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Send Approval Link",
    "Check Status",
    "Verify Response",
    "Approval History",
    "Statistics"
])

# ============================================================================
# TAB 1: SEND APPROVAL LINK
# ============================================================================

with tab1:
    st.markdown('<div class="section-header">Send Approval Link to Nominee</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Case Information**")
        case_id = st.text_input("Case ID", placeholder="CASE-001")
        
    with col2:
        st.markdown("**Nominee Information**")
        nominee_email = st.text_input("Nominee Email", placeholder="nominee@example.com")
        nominee_name = st.text_input("Nominee Name", placeholder="John Doe")
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Send Approval Link", use_container_width=True, key="send_link"):
            if not case_id or not nominee_email:
                st.error("Please fill in Case ID and Nominee Email")
            else:
                with st.spinner("Sending approval link..."):
                    result = st.session_state.workflow.send_approval_link(
                        case_id=case_id,
                        nominee_email=nominee_email,
                        nominee_name=nominee_name or "Nominee"
                    )
                    
                    if result.get('success'):
                        st.markdown('<div class="success-box">Approval link sent successfully!</div>', 
                                  unsafe_allow_html=True)
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**Approval Link**: {result['approval_link']}")
                        with col_b:
                            st.write(f"**Expires**: {result['expires_at']}")
                        
                        st.session_state.approval_results.append(result)
                    else:
                        st.markdown(f'<div class="error-box">Error: {result.get("error")}</div>', 
                                  unsafe_allow_html=True)
    
    with col2:
        if st.button("Clear Form", use_container_width=True):
            st.rerun()
    
    with col3:
        st.info("Link expires in 7 days")

# ============================================================================
# TAB 2: CHECK STATUS
# ============================================================================

with tab2:
    st.markdown('<div class="section-header">Check Approval Status</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        case_id_check = st.text_input("Enter Case ID to check status", placeholder="CASE-001", key="check_status")
    
    with col2:
        st.write("")
        st.write("")
        if st.button("Check Status", use_container_width=True):
            if not case_id_check:
                st.error("Please enter Case ID")
            else:
                with st.spinner("Checking status..."):
                    result = st.session_state.workflow.check_approval_status(case_id_check)
                    
                    if result.get('success'):
                        status = result.get('status', 'unknown').upper()
                        
                        # Color code status
                        if status == 'APPROVED':
                            st.markdown('<div class="success-box">Status: APPROVED</div>', 
                                      unsafe_allow_html=True)
                        elif status == 'REJECTED':
                            st.markdown('<div class="error-box">Status: REJECTED</div>', 
                                      unsafe_allow_html=True)
                        elif status == 'EXPIRED':
                            st.markdown('<div class="warning-box">Status: EXPIRED</div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="info-box">Status: PENDING</div>', 
                                      unsafe_allow_html=True)
                        
                        # Display details
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Status", status)
                        
                        with col_b:
                            st.metric("Nominee", result.get('nominee_email', 'N/A'))
                        
                        with col_c:
                            st.metric("Sent At", result.get('sent_at', 'N/A')[:10])
                        
                        st.divider()
                        
                        # Detailed info
                        st.write("**Detailed Information**")
                        info_df = pd.DataFrame([{
                            'Field': 'Case ID',
                            'Value': result.get('case_id')
                        }, {
                            'Field': 'Status',
                            'Value': status
                        }, {
                            'Field': 'Nominee Email',
                            'Value': result.get('nominee_email')
                        }, {
                            'Field': 'Sent At',
                            'Value': result.get('sent_at')
                        }, {
                            'Field': 'Expires At',
                            'Value': result.get('expires_at')
                        }, {
                            'Field': 'Approved At',
                            'Value': result.get('approved_at', 'N/A')
                        }, {
                            'Field': 'Is Expired',
                            'Value': 'Yes' if result.get('is_expired') else 'No'
                        }])
                        
                        st.dataframe(info_df, use_container_width=True, hide_index=True)
                    else:
                        st.markdown(f'<div class="error-box">Error: {result.get("error")}</div>', 
                                  unsafe_allow_html=True)

# ============================================================================
# TAB 3: VERIFY RESPONSE
# ============================================================================

with tab3:
    st.markdown('<div class="section-header">Verify Approval Response</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Approval Token**")
        approval_token = st.text_input("Enter Approval Token", placeholder="abc123...")
    
    with col2:
        st.markdown("**Response**")
        response = st.selectbox("Nominee Response", ["approved", "rejected"])
    
    with col3:
        st.write("")
        st.write("")
        if st.button("Verify Response", use_container_width=True):
            if not approval_token:
                st.error("Please enter Approval Token")
            else:
                with st.spinner("Verifying response..."):
                    result = st.session_state.workflow.verify_approval(approval_token, response)
                    
                    if result.get('success'):
                        if response == 'approved':
                            st.markdown('<div class="success-box">Approval Verified Successfully!</div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="warning-box">Rejection Recorded</div>', 
                                      unsafe_allow_html=True)
                        
                        st.write(f"**Case ID**: {result.get('case_id')}")
                        st.write(f"**Response**: {result.get('response').upper()}")
                        st.write(f"**Responded At**: {result.get('responded_at')}")
                    else:
                        st.markdown(f'<div class="error-box">Error: {result.get("error")}</div>', 
                                  unsafe_allow_html=True)

# ============================================================================
# TAB 4: APPROVAL HISTORY
# ============================================================================

with tab4:
    st.markdown('<div class="section-header">Approval History</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        case_id_history = st.text_input("Enter Case ID to view history", placeholder="CASE-001", key="history")
    
    with col2:
        st.write("")
        st.write("")
        if st.button("View History", use_container_width=True):
            if not case_id_history:
                st.error("Please enter Case ID")
            else:
                with st.spinner("Retrieving history..."):
                    result = st.session_state.workflow.get_approval_history(case_id_history)
                    
                    if result.get('success'):
                        st.write(f"**Total Requests**: {result.get('total_requests')}")
                        st.write(f"**Latest Status**: {result.get('latest_status').upper()}")
                        
                        st.divider()
                        
                        history = result.get('history', [])
                        if history:
                            # Create dataframe
                            history_data = []
                            for record in history:
                                history_data.append({
                                    'ID': record.get('id'),
                                    'Case ID': record.get('case_id'),
                                    'Nominee': record.get('nominee_email'),
                                    'Status': record.get('status', 'unknown').upper(),
                                    'Sent At': record.get('sent_at', 'N/A')[:10],
                                    'Expires At': record.get('expires_at', 'N/A')[:10]
                                })
                            
                            history_df = pd.DataFrame(history_data)
                            st.dataframe(history_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No approval history found")
                    else:
                        st.markdown(f'<div class="error-box">Error: {result.get("error")}</div>', 
                                  unsafe_allow_html=True)

# ============================================================================
# TAB 5: STATISTICS
# ============================================================================

with tab5:
    st.markdown('<div class="section-header">Approval Statistics</div>', unsafe_allow_html=True)
    
    if st.button("Refresh Statistics", use_container_width=True):
        with st.spinner("Loading statistics..."):
            result = st.session_state.workflow.get_approval_statistics()
            
            if result.get('success'):
                stats = result.get('statistics', {})
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Requests", stats.get('total_requests', 0))
                
                with col2:
                    st.metric("Approved", stats.get('approved', 0))
                
                with col3:
                    st.metric("Rejected", stats.get('rejected', 0))
                
                with col4:
                    st.metric("Approval Rate", stats.get('approval_rate', 'N/A'))
                
                st.divider()
                
                # Display additional stats
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Pending", stats.get('pending', 0))
                
                with col_b:
                    st.metric("Expired", stats.get('expired', 0))
                
                with col_c:
                    st.metric("Revoked", stats.get('revoked', 0))
                
                st.divider()
                
                # Summary
                st.markdown("**Summary**")
                summary_text = f"""
                - Total approval requests: {stats.get('total_requests', 0)}
                - Approved: {stats.get('approved', 0)}
                - Rejected: {stats.get('rejected', 0)}
                - Pending: {stats.get('pending', 0)}
                - Expired: {stats.get('expired', 0)}
                - Revoked: {stats.get('revoked', 0)}
                - Approval Rate: {stats.get('approval_rate', 'N/A')}
                """
                st.info(summary_text)
            else:
                st.markdown(f'<div class="error-box">Error: {result.get("error")}</div>', 
                          unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("Consent Approval Management v1.0 | Last Updated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
