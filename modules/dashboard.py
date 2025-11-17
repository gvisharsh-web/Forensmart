"""
ForenSmart Dashboard
====================

Main Streamlit dashboard that orchestrates:
- Consent management UI
- Data extraction UI with progress
- Media viewer integration
- Report generation scaffold
- Intelligence: Suspicious Message Classifier & Location Intelligence

Run with: streamlit run modules/dashboard.py
"""

import os
import json
import re
import sys
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from collections import Counter
from urllib.parse import quote_plus

# Ensure project root is in sys.path when executed via Streamlit
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st

# --- Modern UI Imports ---
from modules.extraction_ui import ( # pyright: ignore[reportMissingImports]
    render_extraction_tab,
    render_intelligence_tab,
    render_extraction_history,
    get_extraction_ui_manager
)
from modules.progress_ui import ProgressTracker, render_extraction_progress, render_live_artifact_feed # pyright: ignore[reportMissingImports]
# --- End Modern UI Imports ---

from modules.consent import ConsentManager, ConsentLevel, ConsentSession # pyright: ignore[reportMissingImports]
from modules.data_extraction_orchestrator import DataExtractionOrchestrator # pyright: ignore[reportMissingImports]
from modules.shared_utils import ( # pyright: ignore[reportMissingImports]
    ResultsRepository,
    MediaManifest,
    ProgressLogFormatter,
    format_system_checks,
    case_selection_options,
    persist_case_snapshot,
    render_consent_status,
    consent_otp_controls,
    render_vault_entries,
    capture_diagnostics_snapshot,
    AsyncJobRegistry,
)
from modules.storage_manager import StorageManager, StorageAnalytics # pyright: ignore[reportMissingImports]
from modules.storage_ui import render_storage_dashboard # pyright: ignore[reportMissingImports]
from modules.error_checker import ErrorChecker # pyright: ignore[reportMissingImports]

try:  # Streamlit internal helper (best-effort import)
    from streamlit.web.server.websocket_headers import _get_websocket_headers  # type: ignore
except Exception:  # pragma: no cover - optional dependency in non-Streamlit contexts
    _get_websocket_headers = None

# Optional integrations
try:
    # Expecting a UI function exported
    from modules import media_viewer as media_viewer_module # pyright: ignore[reportMissingImports]
except Exception:
    media_viewer_module = None

# ============================================================================
# Modern UI Configuration
# ============================================================================
st.set_page_config(
    page_title="ForenSmart - Modern Forensics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)
# App-wide singletons (kept in session state for Streamlit reruns)


def get_consent_manager() -> ConsentManager:
    if 'consent_manager' not in st.session_state:
        st.session_state['consent_manager'] = ConsentManager()
    return st.session_state['consent_manager']


def get_orchestrator(consent_manager: Optional[ConsentManager] = None) -> DataExtractionOrchestrator:
    if 'orchestrator' not in st.session_state:
        cm = consent_manager or get_consent_manager()
        st.session_state['orchestrator'] = DataExtractionOrchestrator(cm)
        st.session_state['job_registry'] = AsyncJobRegistry()
        st.session_state.setdefault('event_cache', {})
    return st.session_state['orchestrator']


def _kpi_card(label: str, value: Any, help_text: Optional[str] = None):
    st.metric(label=label, value=value, help=help_text)


def _load_results_json(case_id: str) -> Optional[Dict[str, Any]]:
    return ResultsRepository.load(case_id)


def _save_results_json(case_id: str, results: Dict[str, Any]):
    ResultsRepository.save(case_id, results)


def _run_system_checks() -> Dict[str, List[str]]:
    warnings: List[str] = []
    info: List[str] = []

    # OpenCellID
    if not os.getenv('OPENCELLID_KEY'):
        warnings.append('OpenCellID API key missing; location intelligence may lack tower lookups.')
    else:
        info.append('OpenCellID key detected.')

    # ADB / artifacts
    adb_dirs = ['artifacts', os.path.join('artifacts', 'default_case'), 'reports']
    missing_dirs = [d for d in adb_dirs if not os.path.exists(d)]
    if missing_dirs:
        warnings.append(f"Missing expected directories: {', '.join(missing_dirs)}")
    else:
        info.append('Artifacts and reports directories found.')

    try:
        from adapters.android_adb import AndroidADB  # type: ignore
        adb = AndroidADB()
        adb_report = format_system_checks(adb.device_summary())
        warnings.extend(adb_report['warnings'])
        info.extend(adb_report['info'])
    except Exception as exc:
        warnings.append(
            'ADB integration unavailable; ensure adapters/android_adb module is configured '
            f'(details: {exc}).'
        )

    return {'warnings': warnings, 'info': info}


def _run_storage_checks() -> Dict[str, Any]:
    """Run storage integrity checks with graceful fallback."""
    try:
        return ErrorChecker.check_storage_integrity()
    except Exception as exc:  # pragma: no cover - safety net for runtime issues
        return {
            'status': 'error',
            'errors': [f'Storage checks failed: {exc}'],
            'warnings': [],
            'info': [],
            'checks': {}
        }


PII_EMAIL_REGEX = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
PII_PHONE_REGEX = re.compile(r'\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?){2}\d{4}\b')
LOCATION_KEYS = {'lat', 'latitude', 'lon', 'lng', 'longitude'}


def _format_bytes(num: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024 or unit == 'TB':
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def _list_existing_reports(case_id: str) -> List[Dict[str, Any]]:
    if not case_id or not isinstance(case_id, str):
        return []
    reports_dir = os.path.join('reports', case_id)
    if not os.path.exists(reports_dir):
        return []

    entries: List[Dict[str, Any]] = []
    for filename in os.listdir(reports_dir):
        if not filename.lower().startswith('report_'):
            continue
        if not filename.lower().endswith(('.json', '.html', '.txt', '.pdf')):
            continue
        path = os.path.join(reports_dir, filename)
        try:
            stat = os.stat(path)
        except OSError:
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.json':
            mime = 'application/json'
        elif ext == '.html':
            mime = 'text/html'
        elif ext == '.txt':
            mime = 'text/plain'
        else:
            mime = 'application/pdf'
        entries.append({
            'name': filename,
            'path': path,
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'size': stat.st_size,
            'mime': mime
        })

    entries.sort(key=lambda item: item['modified'], reverse=True)
    return entries


def _sanitize_report_data(
    data: Any,
    redact_pii: bool = False,
    round_location: bool = False,
    parent_key: Optional[str] = None
) -> Any:
    if isinstance(data, dict):
        return {
            key: _sanitize_report_data(value, redact_pii, round_location, key)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [
            _sanitize_report_data(item, redact_pii, round_location, parent_key)
            for item in data
        ]
    if isinstance(data, (int, float)):
        if round_location and parent_key and parent_key.lower() in LOCATION_KEYS:
            return round(data, 2)
        return data
    if isinstance(data, str):
        result = data
        if round_location and parent_key and parent_key.lower() in LOCATION_KEYS:
            try:
                result = f"{round(float(result), 2):.2f}"
            except ValueError:
                pass
        if redact_pii:
            result = PII_EMAIL_REGEX.sub('[REDACTED]', result)
            result = PII_PHONE_REGEX.sub('[REDACTED]', result)
        return result
    return data


def _render_text_report(preview: Dict[str, Any]) -> str:
    metadata = preview.get('metadata', {})
    lines = [
        f"ForenSmart Report - Case {metadata.get('case_id', 'unknown')}",
        f"Generated: {metadata.get('generated_at', datetime.now().isoformat())}",
        f"Device: {metadata.get('device_id', 'N/A')}",
        f"Status: {metadata.get('status', 'unknown')}",
        f"Consent Level: {metadata.get('consent_level', 'unknown')}",
        "="*40,
        ""
    ]

    for section_name, section_data in preview.get('sections', {}).items():
        lines.append(f"## {section_name}")
        if section_name == 'Intelligence Summary' and isinstance(section_data, dict) and 'summary' in section_data:
            lines.append(section_data['summary'])
            lines.append("")
        elif isinstance(section_data, list) and section_data and isinstance(section_data[0], dict):
            for i, item in enumerate(section_data[:50]): # Limit items
                lines.append(f"--- Item {i+1} ---")
                for key, value in item.items():
                    lines.append(f"  {key}: {value}")
                lines.append("")
        elif isinstance(section_data, dict):
             for key, value in section_data.items():
                lines.append(f"### {key}")
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    for i, item in enumerate(value[:50]):
                        lines.append(f"--- Item {i+1} ---")
                        for k, v in item.items():
                            lines.append(f"  {k}: {v}")
                        lines.append("")
                else:
                    lines.append(json.dumps(value, indent=2, default=str))
        else:
            lines.append(json.dumps(section_data, indent=2, default=str))
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _check_adb_status() -> bool:
    """Check if ADB is available."""
    import subprocess
    try:
        result = subprocess.run(['adb', 'devices'], capture_output=True, timeout=2)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _pdf_escape(text: str) -> str:
    return text.replace('\\', r'\\').replace('(', r'\(').replace(')', r'\)')


def _build_pdf_lines(preview: Dict[str, Any]) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    metadata = preview.get('metadata', {})

    def add_line(text: str, font: str = 'F1', size: int = 12, indent: int = 0):
        # Basic wrapping for long lines
        max_len = 80 - (indent * 3)
        text = str(text) # Ensure text is a string
        while len(text) > max_len:
            split_pos = text.rfind(' ', 0, max_len)
            if split_pos == -1:
                split_pos = max_len
            lines.append({'text': text[:split_pos], 'font': font, 'size': size, 'indent': indent})
            text = text[split_pos:].lstrip()
        lines.append({'text': text, 'font': font, 'size': size, 'indent': indent})

    add_line(f"ForenSmart Report - Case {metadata.get('case_id', 'unknown')}", font='F2', size=18)
    add_line(f"Generated: {metadata.get('generated_at', datetime.now().isoformat())}", size=12)
    add_line(f"Device: {metadata.get('device_id', 'N/A')}", size=12)
    add_line(f"Status: {metadata.get('status', 'unknown')}", size=12)
    add_line(f"Consent Level: {metadata.get('consent_level', 'unknown')}", size=12)
    add_line("", size=12)

    summary = preview.get('sections', {}).get('Summary', {})
    if summary:
        add_line('Summary', font='F2', size=14)
        for key, value in summary.items():
            if isinstance(value, (list, dict)):
                val_str = json.dumps(value, indent=2, default=str)
                add_line(f"{key}:", indent=1, size=11)
                for v_line in val_str.splitlines():
                    add_line(v_line, indent=2, size=10)
            else:
                add_line(f"{key}: {value}", indent=1, size=11)
        add_line("", size=11)

    for section_name, section_data in preview.get('sections', {}).items():
        if section_name == 'Summary':
            continue
        add_line(section_name, font='F2', size=14)
        
        if section_name == 'Intelligence Summary' and isinstance(section_data, dict) and 'summary' in section_data:
            summary_text = section_data['summary']
            # Remove markdown for PDF
            summary_text = summary_text.replace('**', '').replace('### ', '').replace('- ', '')
            for line in summary_text.splitlines():
                if line.strip():
                    add_line(line, indent=1, size=11)
        elif isinstance(section_data, list) and section_data and isinstance(section_data[0], dict):
            for i, item in enumerate(section_data[:20]): # Limit items in report
                add_line(f"Item {i+1}", font='F2', size=12, indent=1)
                for key, value in item.items():
                    add_line(f"{key}: {str(value)}", indent=2, size=10)
                add_line("", size=5)
        elif isinstance(section_data, dict):
            for key, value in section_data.items():
                add_line(str(key).replace('_', ' ').title(), font='F2', size=12, indent=1)
                if isinstance(value, list) and value and isinstance(value[0], dict):
                     for i, item in enumerate(value[:20]): # Limit items
                        add_line(f"Item {i+1}", font='F1', size=11, indent=2)
                        for k, v in item.items():
                            add_line(f"{k}: {str(v)}", indent=3, size=10)
                        add_line("", size=5)
                else:
                    val_str = json.dumps(value, indent=2, default=str)
                    for v_line in val_str.splitlines():
                        add_line(v_line, indent=2, size=10)
                add_line("", size=5)
        else:
            section_json = json.dumps(section_data, indent=2, default=str)
            for line in section_json.splitlines():
                add_line(line, indent=1, size=11)
        
        add_line("", size=11)

    return lines


def _render_pdf_report(preview: Dict[str, Any]) -> bytes:
    pdf_lines = _build_pdf_lines(preview)
    if not pdf_lines:
        pdf_lines = [{'text': '', 'font': 'F1', 'size': 12, 'indent': 0}]

    y_position = 780.0
    content_ops: List[str] = []

    for line in pdf_lines:
        text = _pdf_escape(line['text'])
        size = line['size']
        font = line['font']
        indent = line['indent']
        if y_position < 60:
            content_ops.append('ET')
            break  # Simple single-page safeguard
        content_ops.extend([
            'BT',
            f"/{font} {size} Tf",
            f"1 0 0 1 {50 + indent * 18:.1f} {y_position:.1f} Tm",
            f"({text}) Tj",
            'ET'
        ])
        y_position -= max(size + 4, 12)

    content_stream = "\n".join(content_ops) + "\n"
    stream_bytes = content_stream.encode('latin-1', errors='replace')
    length = len(stream_bytes)

    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Contents 4 0 R /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >> endobj\n",
        b"4 0 obj << /Length %d >> stream\n" % length + stream_bytes + b"endstream endobj\n",
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        b"6 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> endobj\n"
    ]

    pdf_parts = [b"%PDF-1.4\n"]
    offsets = [0]
    current_offset = len(pdf_parts[0])

    for obj in objects:
        offsets.append(current_offset)
        pdf_parts.append(obj)
        current_offset += len(obj)

    xref_start = current_offset
    xref_lines = [f"xref\n0 {len(offsets)}\n", "0000000000 65535 f \n"]
    for offset in offsets[1:]:
        xref_lines.append(f"{offset:010d} 00000 n \n")
    xref_bytes = ''.join(xref_lines).encode('ascii')
    pdf_parts.append(xref_bytes)

    trailer = (
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    ).encode('ascii')
    pdf_parts.append(trailer)

    return b''.join(pdf_parts)


# --- Comms Analyzer Tab Integration ---
try:
    # Ensure this module is available for the intelligence tab
    from modules import comms_analyzer # pyright: ignore[reportMissingImports]
    HAS_COMMS_ANALYZER = True
except Exception:
    HAS_COMMS_ANALYZER = False

def render_comms_analyzer(cm: ConsentManager):
    st.markdown('## 📞 Communications Analyzer')
    case_id = st.session_state.get('case_id')
    if not case_id:
        st.info('Select or create a case from Overview')
        return
    if HAS_COMMS_ANALYZER:
        comms_analyzer.render_ui(case_id)
    else:
        st.warning('Comms Analyzer module not available.')


def render_dashboard_home(orchestrator: DataExtractionOrchestrator):
    """Render the main dashboard with a modern tabbed interface."""
    cm = get_consent_manager()
    sessions = cm.sessions
    for case_id in list(sessions.keys()):
        cm.ensure_device_id(case_id)
    total_sessions = len(sessions)
    active_requests = sum(1 for s in sessions.values()
                          if 'verification_sid' in s.metadata)
    primary_evidence = sum(1 for s in sessions.values() if s.primary_evidence)
    total_attempts = sum(s.sms_attempts for s in sessions.values())

    # Modern Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color: #00d4ff; font-size: 48px; margin: 0;'>🔍 ForenSmart</h1>
            <p style='color: #00d4ff; font-size: 16px; margin: 5px 0;'>Modern Digital Forensics Platform</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Quick stats
    st.markdown("### 📊 Dashboard Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Cases", total_sessions)
    with col2:
        st.metric("Total Artifacts", StorageAnalytics.get_total_storage_info().get('total_files', 'N/A'))
    with col3:
        st.metric("Extractions", len(get_extraction_ui_manager().extraction_history))
    with col4:
        st.metric("Reports Generated", len(_list_existing_reports(st.session_state.get('case_id') or '')))

    st.divider()

    # Main tabs
    (tab_case, tab_extraction, tab_intelligence, tab_media,
     tab_consent, tab_reports, tab_storage) = st.tabs([
        "🗂️ Case Management",
        "📱 Extraction",
        "🧠 Intelligence",
        "🖼️ Media Viewer",
        "🔐 Consent",
        "📑 Reports",
        "💾 Storage"
    ])

    with tab_case:
        render_case_management_tab(cm)

    # ========================================================================
    # Extraction Tab
    # ========================================================================
    with tab_extraction:
        case_id = st.session_state.get('case_id')
        consent_id = st.session_state.get('consent_id')
        if case_id:
            render_extraction_tab(case_id)
        else:
            st.info("Please select a case from the 'Case Management' tab.")

    # ========================================================================
    # Intelligence Tab
    # ========================================================================
    with tab_intelligence:
        case_id = st.session_state.get('case_id')
        consent_id = st.session_state.get('consent_id')
        if case_id:
            render_intelligence(cm, orchestrator)
        else:
            st.info("Please select a case from the 'Case Management' tab.")

    # ========================================================================
    # Media Viewer Tab
    # ========================================================================
    with tab_media:
        render_media(cm)

    # ========================================================================
    # Consent Tab
    # ========================================================================
    with tab_consent:
        render_consent(cm)

    # ========================================================================
    # Reports Tab
    # ========================================================================
    with tab_reports:
        render_reports()

    # ========================================================================
    # Storage Tab
    # ========================================================================
    with tab_storage:
        render_storage_dashboard()


def render_consent_status_sidebar(cm: ConsentManager):
    """
    DEPRECATED: This function is a stub to prevent NameError from old call sites.
    The sidebar consent status is now rendered in the main function.
    """
    pass


def _detect_streamlit_base_url() -> str:
    """Best-effort detection of the public Streamlit app URL via request headers."""
    headers = None
    try:
        # Use modern st.context.headers API (Streamlit >=1.30)
        headers = st.context.headers
    except Exception:  # pragma: no cover - fallback for older versions
        if callable(_get_websocket_headers):
            try:
                headers = _get_websocket_headers()
            except Exception:
                headers = None
    if not headers:
        return ''

    normalized = {str(k).lower(): str(v) for k, v in headers.items() if k and v}
    host = normalized.get('x-forwarded-host') or normalized.get('host')
    if not host:
        return ''

    proto = (normalized.get('x-forwarded-proto')
             or normalized.get('x-forwarded-protocol')
             or '').split(',')[0].strip().lower()
    if proto not in {'http', 'https'}:
        if 'https' in (normalized.get('cf-visitor') or '').lower() or normalized.get('x-arr-ssl'):
            proto = 'https'
        else:
            proto = 'http'

    base = f"{proto}://{host}".rstrip('/')

    prefix = normalized.get('x-forwarded-prefix') or normalized.get('x-forwarded-pathbase') or ''
    script_hint = normalized.get('x-forwarded-script-name') or normalized.get('x-streamlit-path') or ''
    extra_path = (prefix + script_hint).strip()
    if extra_path and not extra_path.startswith('/'):
        extra_path = '/' + extra_path
    extra_path = extra_path.rstrip('/')
    if extra_path:
        base = f"{base}{extra_path}"

    return base


def _get_default_approval_base_url() -> str:
    """Return cached auto-detected base URL for approval links."""
    cache_key = '_auto_approval_base_url'
    cached = st.session_state.get(cache_key)
    if cached is None:
        cached = _detect_streamlit_base_url()
        st.session_state[cache_key] = cached
    return cached or ''


def _build_approval_link(base_url: str, token: str, approval_data: Optional[Dict[str, Any]] = None) -> str:
    """Build approval link with embedded data (preferred) or token fallback."""
    import json
    import base64
    from urllib.parse import quote
    
    base = (base_url or '').strip() or _get_default_approval_base_url().strip()
    if not base:
        if approval_data:
            encoded = quote(base64.b64encode(json.dumps(approval_data).encode()).decode())
            return f"?data={encoded}"
        return f"?unlock_token={token}"
    
    separator = '&' if '?' in base else '?'
    if approval_data:
        encoded = quote(base64.b64encode(json.dumps(approval_data).encode()).decode())
        return f"{base}{separator}data={encoded}"
    return f"{base}{separator}unlock_token={token}"


def render_nominee_approval(cm: ConsentManager, token: str) -> None:
    st.markdown("# 🔐 ForenSmart Consent Approval")
    st.info("Review the request details below and choose whether to unlock data extraction.")

    record = cm.get_unlock_request_by_token(token)
    if not record:
        st.error("This approval link is invalid or no longer available. Contact the investigator for a new link.")
        return

    session: ConsentSession = record['session']
    unlock_meta = record['unlock_meta']

    st.markdown("### Case Information")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Case ID", session.case_id)
        st.metric("Device ID", session.device_id)
    with col_b:
        st.metric("Requested Level", unlock_meta.get('requested_level', session.level.name))
        st.metric("Current Level", session.level.name)

    st.markdown("### Purpose")
    st.write(unlock_meta.get('purpose') or "Investigator did not provide details.")

    status = unlock_meta.get('status')
    if status in {'verified', 'denied'}:
        if status == 'verified':
            st.success("✅ Approval already granted. No further action needed.")
        else:
            st.error(unlock_meta.get('last_error') or "Request was denied previously.")
        st.stop()

    with st.form("nominee_decision_form"):
        decision = st.radio(
            "Do you approve unlocking the device for the stated purpose?",
            ('approved', 'denied'),
            format_func=lambda v: 'Yes, approve' if v == 'approved' else 'No, deny'
        )
        message = st.text_area('Optional message to investigator')
        submit = st.form_submit_button('Submit decision')

    if submit:
        result = cm.respond_to_unlock_token(token, decision, message.strip() or None)
        if result.get('status') == 'verified':
            st.success('Thank you. The investigator has been notified that you approved the request.')
        elif result.get('status') == 'denied':
            st.warning('Decision recorded as DENIED. The investigator will be notified.')
        else:
            st.error(result.get('message', 'Unable to record your decision.'))
        st.stop()

    st.caption("Your decision takes effect immediately once submitted.")


def render_consent(cm: ConsentManager):
    st.markdown("## 🔐 Consent Management")
    case_id = st.session_state.get('case_id')
    if not case_id:
        st.info("Select or create a case from the 'Case Management' tab.")
        return
    session = cm.get_session(case_id)
    if not session:
        st.warning("No consent session found for this case. Create a new session from the 'Case Management' tab.")
        return

    unlock_status = {}
    unlock_fn = getattr(cm, 'get_unlock_status', None)
    if callable(unlock_fn):
        unlock_status = unlock_fn(case_id)

    detected_device = cm.ensure_device_id(case_id)
    device_label = cm.get_device_label(detected_device)
    st.markdown('#### Device Confirmation')
    col_dev1, col_dev2 = st.columns(2)
    with col_dev1:
        st.metric("Detected Device", device_label)
    with col_dev2:
        st.metric("Consent Level", session.level.name)
    refresh_col1, refresh_col2 = st.columns([1, 3])
    with refresh_col1:
        if st.button('🔄 Refresh device detection', key=f'{case_id}_refresh_device'):
            updated_device = cm.ensure_device_id(case_id)
            st.session_state['device_refresh_ts'] = datetime.now().isoformat()
            if updated_device and updated_device != 'UNKNOWN_DEVICE':
                st.success(f"Detected device: {cm.get_device_label(updated_device)}")
            else:
                st.warning('No authorised device detected. Ensure the handset is connected and authorised via ADB.')
            st.rerun()
    with refresh_col2:
        st.caption("Click refresh after connecting/authorising a device to update detection.")
    st.caption("Ask the nominee to confirm the device identifier above matches their handset before approving.")

    st.markdown('### Unlock Approval Workflow')
    st.caption('Generate a secure approval link for the nominee. Once they approve, extraction unlocks instantly—no OTPs required.')

    status = unlock_status.get('status')

    auto_base_url = _get_default_approval_base_url()
    
    # Initialize session state before creating widget
    if 'approval_base_url' not in st.session_state:
        st.session_state['approval_base_url'] = auto_base_url
    
    st.markdown('**Approval Portal URL**')
    st.warning('⚠️ **IMPORTANT:** Enter your PUBLIC Streamlit consent portal URL (e.g., https://forensmart-xxx.streamlit.app). Do NOT use localhost—nominees cannot access it on their phones.')
    
    # Callback to update session state before widget is instantiated
    def _use_detected_url():
        st.session_state['approval_base_url'] = auto_base_url
    
    base_url = st.text_input('Approval base URL', key='approval_base_url', placeholder='https://forensmart-xxx.streamlit.app')
    
    if auto_base_url and auto_base_url != 'http://localhost:8501':
        st.caption(f"✅ Detected app URL: {auto_base_url}")
        if st.button('Use detected URL', key=f'{case_id}_use_detected_url'):
            st.session_state['approval_base_url'] = auto_base_url
            st.rerun()
    else:
        st.caption('💡 Tip: Paste your public Streamlit consent portal URL above (from https://share.streamlit.io)')

    purpose = st.text_area('Purpose / Notes for nominee', value='Unlock required to proceed with communications extraction.')
    requested_level = st.selectbox(
        'Requested Consent Level',
        [ConsentLevel.STANDARD, ConsentLevel.FULL, ConsentLevel.LEGAL],
        format_func=lambda lvl: lvl.name.title()
    )
    nominee_name = st.text_input('Nominee Name (optional)')

    nominee_phone_key = f'{case_id}_nominee_phone'
    nominee_phone_default = session.nominee_phone or ''
    nominee_contact = st.text_input('Nominee Phone (for WhatsApp/SMS delivery)', value=nominee_phone_default, key=nominee_phone_key)
    if nominee_contact.strip() != (session.nominee_phone or ''):
        session.nominee_phone = nominee_contact.strip()
        cm.persist_session(case_id)

    approval_link = None
    token = None
    if st.button('Generate Approval Link', key=f'{case_id}_generate_link'):
        result = cm.create_unlock_approval(case_id, requested_level, purpose, nominee_name)
        if result.get('status') == 'pending':
            token = result.get('token')
            # Build link with embedded approval data for better UX
            approval_data = {
                'case_id': case_id,
                'device_id': detected_device or 'UNKNOWN_DEVICE',
                'purpose': purpose,
                'requested_level': requested_level.name,
                'nominee_name': nominee_name
            }
            approval_link = _build_approval_link(base_url.strip(), token, approval_data)
            st.session_state['latest_approval_link'] = approval_link
            st.success('Approval request created. Share the link below with the nominee.')
        else:
            st.error(result.get('message', 'Unable to create approval link.'))

    approval_link = approval_link or st.session_state.get('latest_approval_link')
    if approval_link:
        st.markdown('**Approval Link**')
        st.text_input('Copyable link', value=approval_link, key=f'{case_id}_link_display', disabled=True)
        if approval_link.startswith('http://') or approval_link.startswith('https://'):
            st.link_button('🔗 Open approval link', approval_link, type='secondary')
        else:
            st.caption('Provide a publicly reachable base URL above so the link becomes clickable for nominees.')
        share_message = f"Hi {nominee_name or 'there'}, please review and approve this ForenSmart extraction request: {approval_link}"
        sanitized_phone = ''.join(filter(str.isdigit, nominee_contact or ''))
        whatsapp_url = (f"https://wa.me/{sanitized_phone}?text={quote_plus(share_message)}"
                        if sanitized_phone else f"https://wa.me/?text={quote_plus(share_message)}")
        sms_url = (f"sms:+{sanitized_phone}?body={quote_plus(share_message)}"
                   if sanitized_phone else f"sms:?body={quote_plus(share_message)}")
        email_url = f"mailto:?subject={quote_plus('ForenSmart Approval Request')}&body={quote_plus(share_message)}"
        col_share1, col_share2, col_share3 = st.columns(3)
        with col_share1:
            st.markdown(f"<a href='{whatsapp_url}' target='_blank'>🟢 Share via WhatsApp</a>", unsafe_allow_html=True)
        with col_share2:
            st.markdown(f"<a href='{sms_url}' target='_blank'>📱 Send SMS</a>", unsafe_allow_html=True)
        with col_share3:
            st.markdown(f"<a href='{email_url}' target='_blank'>✉️ Email Link</a>", unsafe_allow_html=True)
        qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=180x180&data={quote_plus(approval_link)}"
        st.image(qr_url, caption='Scan to review', width=180)

    if status == 'pending':
        st.info("Awaiting nominee approval. Share the link and wait for their decision.")
    elif status == 'verified':
        verified_at = unlock_status.get('verified_at') or unlock_status.get('responded_at')
        st.success(f"Consent verified at {verified_at}")
    elif status == 'denied':
        st.error(unlock_status.get('last_error') or 'Nominee denied the request. Generate a new link to retry.')
    elif status == 'expired':
        st.warning('Last approval request expired. Generate a fresh link.')
    elif unlock_status.get('last_error'):
        st.warning(unlock_status['last_error'])

    st.caption('Nominee response history (latest)')
    history = unlock_status.get('decision_history', []) if isinstance(unlock_status, dict) else []
    if history:
        recent_rows = []
        for entry in history[-5:]:
            recent_rows.append({
                'Time': entry.get('timestamp'),
                'Action': entry.get('action'),
                'Message': entry.get('message', '')
            })
        st.table(recent_rows)
    else:
        st.info('No nominee decisions recorded yet.')

    st.caption('Unlock activity log')
    activity = cm.get_unlock_activity(case_id, limit=10)
    if activity:
        activity_rows = []
        for entry in activity:
            activity_rows.append({
                'Time': entry.get('timestamp'),
                'Action': entry.get('action'),
                'Details': entry.get('message') or entry.get('purpose') or entry.get('requested_level')
            })
        st.table(activity_rows)
    else:
        st.info('No unlock activity entries recorded yet.')

    last_status_key = f'{case_id}_last_unlock_status'
    prev_status = st.session_state.get(last_status_key)
    if status in {'verified', 'denied'} and status != prev_status:
        if status == 'verified':
            st.toast('Nominee approved the request. Extraction can continue.', icon='✅')
        else:
            st.toast('Nominee denied the request.', icon='⚠️')
    st.session_state[last_status_key] = status

    st.markdown('---')
    st.markdown('#### Session Summary')
    col_summary, col_level = st.columns([2, 1])
    with col_summary:
        st.write({
            'case_id': session.case_id,
            'device_id': session.device_id,
            'level': session.level.name,
            'last_verified': session.last_verified,
            'unlock_status': unlock_status.get('status'),
            'unlock_attempts': unlock_status.get('attempts', 0)
        })

    with col_level:
        st.caption('Manual Level Override')
        levels = list(ConsentLevel)
        try:
            default_index = levels.index(session.level)
        except ValueError:
            default_index = 0
        selected_level = st.selectbox(
            'Consent Level',
            levels,
            index=default_index,
            format_func=lambda lvl: f"{lvl.name.title()}" if isinstance(lvl, ConsentLevel) else str(lvl),
            key=f'{case_id}_level_select'
        )
        reason = st.text_input('Reason', key=f'{case_id}_level_reason')
        if st.button('Update Level', key=f'{case_id}_level_update'):
            result = cm.set_consent_level(case_id, selected_level, reason)
            msg = result.get('message', '')
            if result.get('status') == 'updated':
                st.success(msg)
                st.rerun()
            elif result.get('status') == 'noop':
                st.info(msg)
            else:
                st.error(msg)

    st.markdown('### Consent Snapshot')
    session = cm.get_session(case_id)
    status = render_consent_status(session)
    st.write(status['message'])
    if session:
        st.json(cm.get_session_summary(case_id))
    else:
        st.info('No consent session found. Initialize consent first.')

    st.markdown('#### Messaging Vault Entries')
    entries = cm.get_messaging_vault_entries(case_id)
    delete_entry_fn = getattr(cm, 'delete_messaging_secret', None)
    if entries:
        for entry in entries:
            vault_id = entry.get('vault_id')
            auth_type = entry.get('auth_type', 'PIN')
            created_at = entry.get('created_at')
            fallback = entry.get('fallback')
            label = f"Vault {vault_id} ({auth_type}) added {created_at}"
            if fallback:
                label += " • fallback"
            cols = st.columns([4, 1])
            with cols[0]:
                st.write(f'- {label}')
            with cols[1]:
                disabled = not callable(delete_entry_fn)
                if st.button('Delete', key=f'delete_{case_id}_{vault_id}', disabled=disabled):
                    if delete_entry_fn(case_id, vault_id):
                        st.success(f'Removed vault entry {vault_id}.')
                        st.rerun()
                    else:
                        st.error('Failed to remove vault entry. Refresh and try again.')
    else:
        st.info('No messaging vault secrets stored yet.')

    with st.form(f'{case_id}_vault_form'):
        st.caption('Add messaging secret (stored securely in vault)')
        secret_value = st.text_input('Secret (PIN / Pattern / Password)', type='password')
        auth_type = st.selectbox('Authentication type', ['PIN', 'PATTERN', 'PASSWORD'])
        submitted = st.form_submit_button('Store Secret')
        if submitted:
            if secret_value.strip():
                store_fn = getattr(cm, 'store_messaging_secret', None)
                if callable(store_fn):
                    vault_id = store_fn(case_id, session.device_id, secret_value.strip(), auth_type=auth_type)
                    if vault_id:
                        st.success(f'Secret stored under vault id {vault_id}.')
                    else:
                        st.error('Failed to store secret. Ensure consent session is active.')
                else:
                    st.warning('Current runtime does not support vault storage. Update ConsentManager to use messaging secrets.')
            else:
                st.warning('Secret cannot be empty.')

    st.markdown('#### Consent History (latest 10)')
    history_fn = getattr(cm, 'get_recent_history', None)
    history = history_fn(case_id, limit=10) if callable(history_fn) else []
    if history:
        st.table(history)
    else:
        st.info('No consent history recorded yet.')


def render_media(cm: ConsentManager):
    st.markdown('## 🖼️ Media Viewer')
    case_id = st.session_state.get('case_id')
    if not case_id:
        st.info('Select or create a case from Overview')
        return

    media_origin = st.session_state.get('media_origin')
    if media_origin and media_origin.get('case_id') == case_id:
        st.info(
            f"Opened from {media_origin.get('source', 'unknown source')} attachment."
        )
    elif media_origin:
        st.session_state.pop('media_origin')

    if media_viewer_module and hasattr(media_viewer_module, 'render_media_view'):
        try:
            media_viewer_module.render_media_view(case_id, cm)
        except Exception as e:
            st.error(f'Media viewer error: {e}')
    else:
        st.info(
            'Media viewer module not integrated. '
            'Expected modules/media_viewer.py to expose render_media_view(case_id).'
        )


def generate_ai_summary(findings: Dict[str, Any]) -> str:
    """
    Generates a human-readable summary of the intelligence findings.
    """
    summary_lines = []
    summary_lines.append("### AI-Generated Summary & Observations\n")

    if not findings:
        summary_lines.append("No intelligence findings were generated for this case.")
        return "\n".join(summary_lines)

    # Suspicious Messages Summary
    suspicious_messages = findings.get('suspicious_messages', [])
    if suspicious_messages:
        summary_lines.append(f"**Suspicious Communications:** Found {len(suspicious_messages)} messages flagged as potentially suspicious.")
        
        # Extract top keywords
        from collections import Counter
        import re
        texts = [msg.get('text', '') for msg in suspicious_messages]
        token_counter: Counter[str] = Counter()
        pattern = re.compile(r'[A-Za-z]{4,}')
        for text in texts:
            token_counter.update(tok.lower() for tok in pattern.findall(text))
        
        if token_counter:
            top_keywords = [kw for kw, count in token_counter.most_common(5)]
            summary_lines.append(f"- Key themes identified in these messages include: **{', '.join(top_keywords)}**.")

        summary_lines.append("- It is recommended to review these messages for context.\n")


    # Suspicious Calls Summary
    suspicious_calls = findings.get('suspicious_calls', [])
    if suspicious_calls:
        summary_lines.append(f"**Suspicious Call Patterns:** Detected {len(suspicious_calls)} calls with suspicious characteristics.")
        
        reasons = [reason for call in suspicious_calls for reason in call.get('reasons', [])]
        reason_counts = Counter(reasons)

        if reason_counts:
            top_reasons = [f"'{r}' ({c} times)" for r, c in reason_counts.most_common(3)]
            summary_lines.append(f"- Common suspicious patterns include: {', '.join(top_reasons)}.")
        
        summary_lines.append("- Further investigation into these call records is advised.\n")

    # Location Clusters Summary
    location_clusters = findings.get('location_clusters', [])
    if location_clusters:
        # Filter out noise cluster (-1)
        clusters = [c for c in location_clusters if c.get('cluster', -1) != -1]
        if clusters:
            summary_lines.append(f"**Location Analysis:** Identified {len(clusters)} significant location clusters where the device was frequently present.")
            
            clusters.sort(key=lambda x: x.get('count', 0), reverse=True)
            top_cluster = clusters[0]
            summary_lines.append(f"- The most significant cluster contains {top_cluster.get('count')} data points.")
            summary_lines.append("- This may indicate a primary location such as a home or workplace. Manual review of the map is recommended to determine the nature of this location.\n")

    if len(summary_lines) == 1: # Only title was added
        summary_lines.append("No specific intelligence findings to summarize. Run intelligence modules to generate insights.")

    return "\n".join(summary_lines)


def render_reports():
    st.markdown('## 📑 Report Generation')
    case_id = st.session_state.get('case_id')
    if not case_id:
        st.info('Select or create a case from Overview')
        return

    st.markdown('### Options')
    report_prefs = st.session_state.setdefault('_report_prefs', {})
    case_prefs = report_prefs.get(case_id, {})

    available_sections = ['Summary', 'Intelligence Summary', 'Communications', 'Location',
                          'Media', 'System', 'Security', 'Errors']
    scope_default = [s for s in case_prefs.get('sections', ['Summary', 'Intelligence Summary']) if s in available_sections]
    if not scope_default:
        scope_default = ['Summary', 'Intelligence Summary']
    scope = st.multiselect('Sections', available_sections, default=scope_default)

    redact_key = f'{case_id}_report_redact'
    round_key = f'{case_id}_report_round'
    format_key = f'{case_id}_report_format'
    st.session_state.setdefault(redact_key, case_prefs.get('redact_pii', False))
    st.session_state.setdefault(round_key, case_prefs.get('round_location', False))
    st.session_state.setdefault(format_key, case_prefs.get('format', 'TXT'))

    redact_pii = st.checkbox('Redact PII (emails/phones)', key=redact_key)
    round_location = st.checkbox('Round location to ~2 decimals (privacy)', key=round_key)
    output_fmt = st.selectbox('Format', ['TXT', 'PDF'], key=format_key)

    report_prefs[case_id] = {
        'sections': scope,
        'redact_pii': redact_pii,
        'round_location': round_location,
        'format': output_fmt
    }

    results = _load_results_json(case_id)
    if not results:
        st.warning(
            'No saved results found (expected reports/<case_id>/results.json). Generate extraction first.')
        return

    existing_reports = _list_existing_reports(case_id)
    if existing_reports:
        st.markdown('### Existing Reports')
        for entry in existing_reports[:10]:
            cols = st.columns([4, 2, 2])
            with cols[0]:
                st.write(f"**{entry['name']}**")
                st.caption(entry['modified'].strftime('%Y-%m-%d %H:%M:%S'))
            with cols[1]:
                st.caption(_format_bytes(entry['size']))
            with cols[2]:
                try:
                    with open(entry['path'], 'rb') as handle:
                        st.download_button(
                            'Download',
                            data=handle.read(),
                            file_name=entry['name'],
                            mime=entry['mime'],
                            key=f"download_{entry['name']}"
                        )
                except OSError as err:
                    st.warning(f"Cannot read {entry['name']}: {err}")
        if len(existing_reports) > 10:
            st.info(f"Showing 10 of {len(existing_reports)} reports.")

    # Sanitize data according to toggles
    sanitized_data = _sanitize_report_data(
        results,
        redact_pii=redact_pii,
        round_location=round_location
    )

    # Build a simple preview
    st.markdown('### Preview')
    preview: Dict[str, Any] = {
        'metadata': {
            'case_id': sanitized_data.get('case_id'),
            'device_id': sanitized_data.get('device_id'),
            'consent_level': sanitized_data.get('consent_level'),
            'status': sanitized_data.get('status'),
            'generated_at': datetime.now().isoformat()
        },
        'sections': {}
    }

    if 'Intelligence Summary' in scope:
        findings = sanitized_data.get('data', {}).get('intelligence_findings', {})
        summary_text = generate_ai_summary(findings)
        preview['sections']['Intelligence Summary'] = {'summary': summary_text}

    def add_if(name: str, key: str):
        if name in scope and key in sanitized_data.get('data', {}):
            preview['sections'][name] = sanitized_data['data'][key]

    add_if('Communications', 'communications')
    add_if('Location', 'location')
    add_if('Media', 'media')
    add_if('System', 'system')
    add_if('Security', 'security')

    # Always include summary
    if 'Summary' in scope:
        preview['sections']['Summary'] = {
            'modules_run': sanitized_data.get('modules_run'),
            'errors': sanitized_data.get('errors'),
            'duration_seconds': sanitized_data.get('duration_seconds')
        }

    st.json(preview)

    if 'Intelligence Summary' in preview['sections']:
        st.markdown("---")
        st.markdown("### Intelligence Summary Preview")
        st.markdown(preview['sections']['Intelligence Summary']['summary'])

    st.markdown('---')
    if st.button('Generate Report'):
        reports_dir = os.path.join('reports', case_id)
        os.makedirs(reports_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        if output_fmt == 'TXT':
            out = os.path.join(reports_dir, f'report_{ts}.txt')
            with open(out, 'w', encoding='utf-8') as handle:
                handle.write(_render_text_report(preview))
            st.success(f'Report saved: {out}')
        else:
            pdf_bytes = _render_pdf_report(preview)
            out = os.path.join(reports_dir, f'report_{ts}.pdf')
            with open(out, 'wb') as handle:
                handle.write(pdf_bytes)
            st.success(f'Report saved: {out}')


@st.cache_data(show_spinner=False, ttl=300)
def _load_intelligence_data(case_id: str) -> Dict[str, Any]:
    """Cache intelligence data loading to improve performance."""
    results = ResultsRepository.load(case_id) or {}
    data = results.get('data', {}) if isinstance(results, dict) else {}
    modules_run = results.get('modules_run', []) if isinstance(results, dict) else []
    return {'data': data, 'modules_run': modules_run}

def render_intelligence(cm: ConsentManager, orchestrator: DataExtractionOrchestrator):
    st.markdown('## 🧠 Intelligence')
    case_id = st.session_state.get('case_id')
    if not case_id:
        st.info('Select or create a case from Overview')
        return

    session = cm.get_session(case_id)
    if not session:
        st.warning('No consent session found. Initialize consent from the Consent tab.')
        return

    # Load cached intelligence data
    intel_data = _load_intelligence_data(case_id)
    data = intel_data.get('data', {})
    modules_run = intel_data.get('modules_run', [])

    def _module_meta(name: str) -> Dict[str, Any]:
        if isinstance(modules_run, list):
            for entry in modules_run:
                if isinstance(entry, dict) and entry.get('name') == name:
                    return entry
        return {}

    def _format_ts(value: Optional[str]) -> str:
        if not value:
            return '—'
        try:
            return datetime.fromisoformat(value).strftime('%Y-%m-%d %H:%M')
        except Exception:
            return str(value)

    comms_payload = data.get('communications') if isinstance(data, dict) else {}
    location_payload = data.get('location') if isinstance(data, dict) else {}

    def _len(value: Any) -> int:
        return len(value) if isinstance(value, list) else 0

    comms_counts = {
        'SMS': _len((comms_payload or {}).get('sms_messages')),
        'Calls': _len((comms_payload or {}).get('call_logs')),
        'WhatsApp': _len((comms_payload or {}).get('whatsapp_messages')),
        'Other': sum(
            _len((comms_payload or {}).get(key))
            for key in ['telegram_messages', 'snapchat_messages', 'instagram_messages']
        ),
    }
    comms_total = sum(comms_counts.values())
    comms_last = _format_ts(_module_meta('communications').get('extracted_at'))

    location_counts = {
        'GPS': _len((location_payload or {}).get('gps_coordinates')),
        'WiFi': _len((location_payload or {}).get('wifi_networks')),
        'Cell': _len((location_payload or {}).get('cell_towers')),
    }
    location_total = sum(location_counts.values())
    location_last = _format_ts(_module_meta('location').get('extracted_at'))

    st.markdown('### Intelligence Summary')
    summary_cols = st.columns(3)
    summary_cols[0].metric('Loaded communications', comms_total, help=', '.join(f"{k}: {v}" for k, v in comms_counts.items()))
    summary_cols[1].metric('Comms last run', comms_last)
    summary_cols[2].metric('Location last run', location_last)

    if location_total:
        loc_detail = ', '.join(f"{k}: {v}" for k, v in location_counts.items())
        st.caption(f"Location artifacts detected • {loc_detail}")
    else:
        st.caption('No location artifacts saved yet. Run the Location extraction module to populate analytics.')

    tabs = st.tabs(['Location Intelligence', 'Comms Analyzer'])

    # Cache hints to avoid repeated calls
    @st.cache_data(show_spinner=False, ttl=300)
    def _get_hints(case_id: str) -> tuple:
        hint_fn = getattr(orchestrator, 'provide_data_or_hint', None)
        comms_hint = hint_fn(case_id, 'comms') if callable(hint_fn) else {'status': 'unknown'}
        location_hint = hint_fn(case_id, 'location') if callable(hint_fn) else {'status': 'unknown'}
        return comms_hint, location_hint

    comms_hint, location_hint = _get_hints(case_id)

    def _navigate_to_extraction(target_modules: List[str]) -> None:
        st.session_state['nav'] = 'Extraction'
        st.session_state['retry_modules'] = target_modules
        st.rerun()

    def _run_module(module_name: str, label: str) -> None:
        device_id = session.device_id
        if not device_id:
            st.error('No device bound to this case. Verify consent and device pairing before running extraction.')
            return
        try:
            with st.spinner(f'Running {label} extraction…'):
                orchestrator.extract_all_data(
                    case_id,
                    device_id,
                    progress_callback=None,
                    modules_override=[module_name],
                )
            st.success(f'{label} extraction completed. Refreshing intelligence data…')
            st.rerun()
        except Exception as exc:
            st.error(f'{label} extraction failed: {exc}')

    with tabs[1]: # Comms Analyzer
        try:
            import modules.comms_analyzer as sc # pyright: ignore[reportMissingImports]
            status = comms_hint.get('status', 'unknown')
            if status == 'ok':
                st.caption(f"{comms_total} communications loaded • {', '.join(f'{k}: {v}' for k, v in comms_counts.items())}")
                sc.render_ui(case_id)
            else:
                st.warning('No communications data detected for this case.')
                missing_msg = comms_hint.get('message')
                if missing_msg:
                    st.info(missing_msg)
                action_cols = st.columns(2)
                with action_cols[0]:
                    if st.button('Run Communications extraction', key=f'run_comms_{case_id}'):
                        _run_module('communications', 'Communications')
                with action_cols[1]:
                    if st.button('Open Extraction tab', key=f'nav_comms_{case_id}'):
                        _navigate_to_extraction(['communications'])
        except Exception as e:
            st.error(f'Comms analyzer error: {e}')

    with tabs[0]: # Location Intelligence
        try:
            import modules.location_intelligence as li # pyright: ignore[reportMissingImports]
            status = location_hint.get('status', 'unknown')
            if status == 'ok':
                if location_total:
                    st.caption(f"Location datasets loaded • {', '.join(f'{k}: {v}' for k, v in location_counts.items())}")
                li.render_ui(case_id)
            else:
                st.warning('No location data detected for this case.')
                missing_msg = location_hint.get('message')
                if missing_msg:
                    st.info(missing_msg)
                action_cols = st.columns(2)
                with action_cols[0]:
                    if st.button('Run Location extraction', key=f'run_location_{case_id}'):
                        _run_module('location', 'Location')
                with action_cols[1]:
                    if st.button('Open Extraction tab', key=f'nav_location_{case_id}'):
                        _navigate_to_extraction(['location'])
        except Exception as e:
            st.error(f'Location intelligence error: {e}')


def delete_case_artifacts(case_id: str, cm: ConsentManager) -> tuple[bool, str, Dict[str, Any]]:
    """Delete case artifacts and consent session via storage manager."""
    success, message, info = StorageManager.delete_entire_case(case_id, consent_manager=cm)
    if success:
        # cache last deletion details for user visibility post-rerun
        st.session_state.setdefault('last_deletion_details', {})
        st.session_state['last_deletion_details'][case_id] = info
    return success, message, info


def render_case_management_tab(cm: ConsentManager):
    """Renders the tab for creating and selecting cases."""
    st.markdown("## 🗂️ Case Management")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Create New Case")
        with st.form("new_case_form"):
            subject_name = st.text_input("Subject Name")
            subject_id = st.text_input("Subject ID / Phone Number (optional)")
            consent_level_str = st.selectbox(
                "Initial Consent Level",
                [level.name for level in ConsentLevel if level != ConsentLevel.NONE]
            )
            submitted = st.form_submit_button("✅ Create Case")

            if submitted:
                if not subject_name:
                    st.error("Subject Name is required to create a case.")
                else:
                    consent_level = ConsentLevel[consent_level_str]
                    
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', subject_name.replace(' ', '_'))
                    new_case_id = f"CASE-{safe_name}-{timestamp}"

                    try:
                        cm.create_session(new_case_id, subject_id)
                        cm.set_consent_level(new_case_id, consent_level, "Initial Case Creation")
                        st.session_state['case_id'] = new_case_id
                        st.success(f"✅ Case created successfully: {new_case_id}")
                        st.rerun()
                    except ValueError:
                        st.warning(f"A case with a similar ID might already exist. Try a different name.")
                    except Exception as e:
                        st.error(f"Failed to create case: {e}")

    with col2:
        st.markdown("### Active Cases")
        render_active_cases_widget(cm)

def render_active_cases_widget(consent_manager: ConsentManager):
    """Renders a widget for displaying and selecting active cases."""
    active_cases = [consent_manager.get_session_summary(case_id) for case_id in consent_manager.sessions]
    active_cases = [c for c in active_cases if c]

    if 'confirm_delete_case_id' not in st.session_state:
        st.session_state['confirm_delete_case_id'] = None

    if active_cases:
        st.info(f"Found {len(active_cases)} existing case(s). Select one to proceed.")
        for case in active_cases[:10]: # Show max 10
            case_id = case.get('case_id')
            if not case_id:
                continue

            with st.container():
                col_a, col_b, col_c = st.columns([3, 1, 1])
                with col_a:
                    st.markdown(f"**{case_id}**")
                    st.caption(f"Level: {case.get('level', '')} | Device: {case.get('device_id', 'N/A')}")
                with col_b:
                    if st.button("Select", key=f"select_{case_id}"):
                        st.session_state['case_id'] = case_id
                        st.session_state['confirm_delete_case_id'] = None # Clear confirmation on selection
                        st.rerun()
                with col_c:
                    if st.button("Delete", key=f"delete_{case_id}"):
                        st.session_state['confirm_delete_case_id'] = case_id
                        st.rerun()

                if st.session_state.get('confirm_delete_case_id') == case_id:
                    st.warning(f"Are you sure you want to delete case **{case_id}**? This action cannot be undone.")
                    col_confirm, col_cancel = st.columns(2)
                    with col_confirm:
                        if st.button("Yes, delete permanently", key=f"confirm_delete_{case_id}"):
                            success, message, info = delete_case_artifacts(case_id, consent_manager)
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                            
                            st.session_state['confirm_delete_case_id'] = None
                            if st.session_state.get('case_id') == case_id:
                                st.session_state['case_id'] = None
                            st.rerun()
                    with col_cancel:
                        if st.button("Cancel", key=f"cancel_delete_{case_id}"):
                            st.session_state['confirm_delete_case_id'] = None
                            st.rerun()

                st.divider()
        if len(active_cases) > 10:
            st.caption("Showing the 10 most recent cases.")
    else:
        st.info("No active cases found. Create a new case to begin.")

def main():

    cm = get_consent_manager()
    orchestrator = get_orchestrator(cm)

    # Nominee approval deep link handling
    params = st.query_params
    token_value = params.get('unlock_token')
    if token_value:
        render_nominee_approval(cm, token_value)
        return

    # Sidebar navigation
    if 'nav' not in st.session_state:
        st.session_state['nav'] = 'Overview'


    if '_system_report' not in st.session_state:
        st.session_state['_system_report'] = _run_system_checks()
    if '_storage_report' not in st.session_state:
        st.session_state['_storage_report'] = _run_storage_checks()

    system_report = st.session_state['_system_report']
    storage_report = st.session_state['_storage_report']
    # Initialize session state
    if 'case_id' not in st.session_state:
        st.session_state['case_id'] = None



    # --- Modern Sidebar ---
    with st.sidebar:
        st.markdown("### 📋 Case Selection")
        case_id_input = st.text_input(
            "Case ID",
            value=st.session_state.get('case_id', ''),
            key='case_id_input'
        )
        if case_id_input:
            st.session_state['case_id'] = case_id_input

        st.markdown("### 🔐 Consent Status")
        case_id = st.session_state.get('case_id')
        if case_id:
            session = cm.get_session(case_id)
            if session and session.level != ConsentLevel.NONE:
                st.success(f"✅ Consent Active\n\nCase: {case_id}\n\nLevel: {session.level.name}")
            else:
                st.warning("⚠️ No Consent\n\nCapture consent to proceed")
        else:
            st.warning("⚠️ No Consent\n\nSelect a case to see status")

        st.markdown("### 🔧 System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ADB", "✅" if _check_adb_status() else "❌")
        with col2:
            st.metric("Storage", "✅" if storage_report.get('status') == 'ok' else "⚠️")

        with st.expander("System Checks Details", expanded=False):
            if not system_report['warnings'] and not system_report['info']:
                st.success("All system checks passed.")
            else:
                for warning in system_report['warnings']:
                    st.warning(warning)
                for info in system_report['info']:
                    st.info(info)

        st.divider()

        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

        if st.button("📋 Extraction History", use_container_width=True):
            st.session_state['view_history'] = True

    # Route to appropriate view
    if st.session_state.get('view_history'):
        render_extraction_history()
        if st.button("← Back to Dashboard"):
            st.session_state['view_history'] = False
            st.rerun()
    else:
        render_dashboard_home(orchestrator)


if __name__ == '__main__':
    main()
