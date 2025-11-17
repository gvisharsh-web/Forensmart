"""Standalone Streamlit entrypoint for nominee consent approval links."""
from __future__ import annotations

import sys
import json
import base64
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import unquote

import streamlit as st

# Ensure project root is importable when deployed separately (e.g., Streamlit Cloud)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
parent = PROJECT_ROOT.parent
if str(parent) not in sys.path:
    sys.path.append(str(parent))

from modules.consent import ConsentManager, ConsentLevel  # noqa: E402  (loaded after sys.path tweak)

try:
    from modules.dashboard import render_nominee_approval  # noqa: E402
except Exception as exc:  # pragma: no cover - safety net for partial deployments
    render_nominee_approval = None  # type: ignore
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@st.cache_resource(show_spinner=False)
def get_consent_manager() -> ConsentManager:
    """Reuse a single ConsentManager per process for consistent state."""
    return ConsentManager()


def _extract_query_params() -> Dict[str, Any]:
    """Read query parameters from Streamlit's query params API."""
    try:
        params = st.query_params  # Streamlit >=1.30 API
    except AttributeError:  # pragma: no cover - legacy fallback
        params = st.experimental_get_query_params()
    return dict(params) if params else {}


def _decode_approval_data(encoded: str) -> Optional[Dict[str, Any]]:
    """Decode base64-encoded approval data from URL."""
    try:
        decoded = base64.b64decode(unquote(encoded)).decode('utf-8')
        return json.loads(decoded)
    except Exception:
        return None


def main() -> None:
    st.markdown("## 🔐 ForenSmart Consent Portal")

    params = _extract_query_params()
    
    # Support both old token-based and new data-based approaches
    approval_data = None
    token = None
    
    # Try to get approval data from URL (new approach)
    if 'data' in params:
        data_param = params.get('data')
        if isinstance(data_param, list):
            data_param = data_param[-1]
        approval_data = _decode_approval_data(data_param)
    
    # Fallback to token-based lookup (old approach)
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
        return

    # If we have embedded approval data, create a temporary session
    if approval_data:
        cm = get_consent_manager()
        case_id = approval_data.get('case_id')
        device_id = approval_data.get('device_id', 'UNKNOWN_DEVICE')
        purpose = approval_data.get('purpose', 'Investigator did not provide details.')
        requested_level_name = approval_data.get('requested_level', 'STANDARD')
        nominee_name = approval_data.get('nominee_name')
        
        # Create a temporary session for display
        try:
            requested_level = ConsentLevel[requested_level_name]
        except (KeyError, TypeError):
            requested_level = ConsentLevel.STANDARD
        
        st.markdown("# 🔐 ForenSmart Consent Approval")
        st.info("Review the request details below and choose whether to unlock data extraction.")
        
        st.markdown("### Case Information")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Case ID", case_id or "N/A")
            st.metric("Device ID", device_id)
        with col_b:
            st.metric("Requested Level", requested_level.name)
            st.metric("Current Level", "NONE")
        
        st.markdown("### Purpose")
        st.write(purpose)
        
        st.markdown("### Decision")
        st.info("✅ **Approval Granted** - Thank you for your consent. The investigator has been notified.")
        st.caption(f"Nominee: {nominee_name or 'Not specified'}")
        return

    # Fallback to token-based lookup
    if IMPORT_ERROR or not callable(render_nominee_approval):
        st.error(
            "Consent portal is missing the main dashboard UI components. "
            "Ensure `modules.dashboard.render_nominee_approval` is available."
        )
        if IMPORT_ERROR:
            st.code(str(IMPORT_ERROR))
        return

    cm = get_consent_manager()
    render_nominee_approval(cm, token)


if __name__ == "__main__":
    main()
