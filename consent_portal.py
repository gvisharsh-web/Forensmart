"""Standalone Streamlit entrypoint for nominee consent approval links."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import streamlit as st

# Ensure project root is importable when deployed separately (e.g., Streamlit Cloud)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
parent = PROJECT_ROOT.parent
if str(parent) not in sys.path:
    sys.path.append(str(parent))

from modules.consent import ConsentManager  # noqa: E402  (loaded after sys.path tweak)

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


def _extract_unlock_token() -> Optional[str]:
    """Read ?unlock_token=... from Streamlit's query params API."""
    try:
        params = st.query_params  # Streamlit >=1.30 API
    except AttributeError:  # pragma: no cover - legacy fallback
        params = st.experimental_get_query_params()
    if not params:
        return None
    token = params.get("unlock_token")
    if isinstance(token, list):
        return token[-1]
    return token


def main() -> None:
    st.markdown("## 🔐 ForenSmart Consent Portal")

    if IMPORT_ERROR or not callable(render_nominee_approval):
        st.error(
            "Consent portal is missing the main dashboard UI components. "
            "Ensure `modules.dashboard.render_nominee_approval` is available."
        )
        if IMPORT_ERROR:
            st.code(str(IMPORT_ERROR))
        return

    token = _extract_unlock_token()
    if not token:
        st.warning(
            "No approval token supplied. This page must be opened via the secure link "
            "shared by the investigator."
        )
        st.info(
            "Example: https://your-consent-app.streamlit.app/?unlock_token=TOKEN_HERE"
        )
        return

    cm = get_consent_manager()
    render_nominee_approval(cm, token)


if __name__ == "__main__":
    main()
