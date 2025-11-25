"""
Suspicious Message Classifier (Clean Module)

Provides:
- Model loading helpers (TF-IDF based)
- Scoring utilities for message texts
- Streamlit UI panel to review suspicious messages for a selected case

Looks for models/suspicious_tfidf_auto.pkl or models/suspicious_tfidf.pkl.
Reads messages from reports/<case_id>/results.json if available, with CSV upload fallback.
"""
import os
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import streamlit as st

try:
    import joblib  # type: ignore
except Exception:
    joblib = None

# NEW: Import audit trail for suspicious classifications
try:
    from modules.consent.portal import ConsentAuditTrail
except ImportError:
    ConsentAuditTrail = None  # Optional dependency


def _load_results_messages(case_id: str) -> List[Dict[str, Any]]:
    path = os.path.join('reports', case_id, 'results.json')
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            comms = data.get('data', {}).get('communications', {})
            msgs = comms.get('sms_messages', [])
            return msgs if isinstance(msgs, list) else []
        except Exception:
            return []
    return []


def load_model(prefer_auto: bool = True) -> Optional[Any]:
    """Load a suspicious TF-IDF model pipeline. Returns None if not available."""
    candidates = []
    if prefer_auto:
        candidates = [
            os.path.join('models', 'suspicious_tfidf_auto.pkl'),
            os.path.join('models', 'suspicious_tfidf.pkl'),
        ]
    else:
        candidates = [
            os.path.join('models', 'suspicious_tfidf.pkl'),
            os.path.join('models', 'suspicious_tfidf_auto.pkl'),
        ]

    for p in candidates:
        if os.path.exists(p) and joblib:
            try:
                return joblib.load(p)
            except Exception:
                continue
    return None


def score_messages(model: Any, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score messages using the provided model. Returns list of messages with 'score' field."""
    if not model or not messages:
        return messages
    texts = [m.get('message', '') for m in messages]
    if not texts:
        return messages

    scores_list = []
    try:
        # Assume model is a pipeline with decision_function or predict_proba
        if hasattr(model, 'decision_function'):
            scores = model.decision_function(texts)
            # Normalize to [0,1] if necessary
            import numpy as np  # type: ignore
            if hasattr(scores, 'shape'):
                # For binary, scores could be 1-d
                arr = scores if len(
                    getattr(scores, 'shape', [])) == 1 else scores[:, 0]
            else:
                arr = scores
            # min-max normalize
            a = np.array(arr)
            mn, mx = a.min(), a.max()
            if mx - mn > 1e-9:
                norm = (a - mn) / (mx - mn)
            else:
                norm = a * 0 + 0.5
            scores_list = norm.tolist()
        elif hasattr(model, 'predict_proba'):
            probs = model.predict_proba(texts)
            import numpy as np  # type: ignore
            p = np.array(probs)
            # Use positive class probability if binary
            if p.ndim == 2 and p.shape[1] > 1:
                pos = p[:, 1]
            else:
                pos = p[:, 0] if p.ndim == 2 else p
            scores_list = pos.tolist()
        else:
            # Fallback predict -> label 0/1, set score=label
            preds = model.predict(texts)
            scores_list = [float(p) for p in preds]
        
        for i, msg in enumerate(messages):
            msg['score'] = float(scores_list[i])

    except Exception:
        for i, msg in enumerate(messages):
            msg['score'] = 0.0

    return messages


def get_suspicion_level(score: float) -> str:
    """Convert numeric score to suspicion level."""
    if score >= 0.8:
        return "CRITICAL"
    elif score >= 0.6:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    elif score >= 0.2:
        return "LOW"
    else:
        return "MINIMAL"


def get_suspicion_color(score: float) -> str:
    """Get color for suspicion level."""
    level = get_suspicion_level(score)
    colors = {
        "CRITICAL": "#FF0000",  # Red
        "HIGH": "#FF6B00",      # Orange
        "MEDIUM": "#FFD700",    # Gold
        "LOW": "#90EE90",       # Light Green
        "MINIMAL": "#00AA00"    # Green
    }
    return colors.get(level, "#808080")


def render_ui(case_id: str):
    st.markdown('### [CLASSIFIER] Suspicious Message Classifier')

    # Load messages
    messages = _load_results_messages(case_id)

    col1, col2 = st.columns(2)
    with col1:
        st.caption('Messages from extraction (if available)')
        st.write(
            f"Loaded {len(messages)} messages from reports/{case_id}/results.json")
    with col2:
        uploaded = st.file_uploader(
            'Upload communications file (CSV/TXT/JSON/XLSX)',
            type=['csv', 'txt', 'json', 'xlsx'],
            key=f'{case_id}_suspicious_uploader')
        extra_messages: List[Dict[str, Any]] = []
        if uploaded:
            import pandas as pd  # type: ignore
            name = uploaded.name.lower()
            try:
                if name.endswith('.csv'):
                    df = pd.read_csv(uploaded)
                    if 'message' in df.columns:
                        extra_messages = df.to_dict('records')
                    else:
                        st.error('CSV must contain a "message" column')
                elif name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded)
                    if 'message' in df.columns:
                        extra_messages = df.to_dict('records')
                    else:
                        st.error('XLSX must contain a "message" column')
                elif name.endswith('.json'):
                    import json
                    payload = json.load(uploaded)
                    if isinstance(payload, list):
                        if payload and isinstance(payload[0], dict):
                            extra_messages = [item for item in payload if item.get('message')]
                        else:
                            extra_messages = [{'message': str(item)} for item in payload if item]
                    elif isinstance(payload, dict):
                        msgs_from_json = payload.get('messages') or payload.get('data') or []
                        if isinstance(msgs_from_json, list):
                            extra_messages = [
                                item if isinstance(item, dict) else {'message': str(item)}
                                for item in msgs_from_json if (item.get('message') if isinstance(item, dict) else item)
                            ]
                    else:
                        st.warning('Unsupported JSON structure. Expected list or { messages: [...] }.')
                elif name.endswith('.txt'):
                    uploaded.seek(0)
                    content = uploaded.read().decode('utf-8', errors='ignore')
                    extra_messages = [{'message': line.strip()} for line in content.splitlines() if line.strip()]
                else:
                    st.warning('Unsupported file type.')
            except Exception as exc:
                st.error(f'Failed to process uploaded file: {exc}')

    all_messages = messages + extra_messages
    if not all_messages:
        st.info(
            'No messages available to score. Provide extraction results or upload CSV.')
        return

    model = load_model(prefer_auto=True)
    if not model:
        st.warning(
            'No model found in models/. Place suspicious_tfidf_auto.pkl or suspicious_tfidf.pkl')
        return

    threshold = st.slider('Suspicion threshold', 0.0, 1.0, 0.7, 0.01, key=f'{case_id}_suspicion_threshold')
    topn = st.number_input('Show top N', min_value=5, max_value=200, value=20, key=f'{case_id}_show_top_n')

    if st.button('Classify'):
        with st.spinner('Scoring messages...'):
            results = score_messages(model, all_messages)
        results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        suspicious = [r for r in results if r.get('score', 0.0) >= threshold]

        st.success(
            f"Found {len(suspicious)} suspicious messages (threshold {threshold:.2f})")
        
        # Show suspicion level distribution
        level_counts = {}
        for r in suspicious:
            level = get_suspicion_level(r.get('score', 0.0))
            level_counts[level] = level_counts.get(level, 0) + 1
        
        if level_counts:
            st.markdown("#### Suspicion Level Distribution")
            dist_cols = st.columns(len(level_counts))
            for idx, (level, count) in enumerate(sorted(level_counts.items(), key=lambda x: ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].index(x[0]))):
                with dist_cols[idx]:
                    st.metric(level, count)
        
        for r in suspicious[:int(topn)]:
            score = r.get('score', 0.0)
            level = get_suspicion_level(score)
            
            # Display with level indicator
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**Score:** {score:.2f}")
            with col2:
                st.markdown(f"**Level:** `{level}`")
            with col3:
                st.caption(f"Threshold: {threshold:.2f}")
            
            contact = r.get('contact') or r.get('contact_name') or '(Unknown)'
            number = r.get('number') or r.get('phone') or ''
            contact_key = f"show_number_{contact}_{number}_{score}"
            if st.button(contact, key=contact_key):
                st.info(f"Number: {number}")
            st.write(r.get('message', ''))
            st.markdown('---')

        # Export
        export = st.button('Export Suspicious JSON')
        if export:
            out = {
                'case_id': case_id,
                'generated_at': datetime.now().isoformat(),
                'threshold': threshold,
                'count': len(suspicious),
                'items': suspicious
            }
            st.download_button('Download JSON', json.dumps(
                out, indent=2), file_name=f'suspicious_{case_id}.json', mime='application/json')