"""
Comms Analyzer (renamed from Suspicious Classifier)

Clean module for text scoring of communications extracted from a case.
- Loads TF-IDF model pipeline if available
- Scores messages and provides a streamlined UI
- Avoids duplicated/spammy UI across modules
"""
import os
import json
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

try:
    import joblib  # type: ignore
except Exception:
    joblib = None

try:
    from pyvis.network import Network
except ImportError:
    Network = None


from modules.shared_utils import (
    ArtifactPathBuilder,
    adb_root_access_message,
    parse_sms_dump,
    parse_calls_dump,
)

try:
    from adapters.android_adb import AndroidADB  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AndroidADB = None  # type: ignore


def _save_intelligence_findings(case_id: str, key: str, data: Any):
    """Saves intelligence findings to the case's results.json."""
    results_path = os.path.join('reports', case_id, 'results.json')
    if not os.path.exists(results_path):
        return

    try:
        with open(results_path, 'r') as f:
            main_results = json.load(f)

        main_results.setdefault('data', {}).setdefault('intelligence_findings', {})[key] = data

        with open(results_path, 'w') as f:
            json.dump(main_results, f, indent=2, default=str)
        
        st.toast(f"{key.replace('_', ' ').title()} findings saved to case report.")
    except Exception as e:
        st.error(f"Failed to save intelligence findings: {e}")


@st.cache_data(show_spinner=False, ttl=300)
def _load_results_comms(case_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load and cache communications data with 5-minute TTL for performance."""
    return _load_results_comms_raw(case_id)

@st.cache_data(show_spinner=False, ttl=300)
def _process_call_records_cached(case_id: str, call_count: int) -> Dict[str, Any]:
    """Cache processed call records metadata to avoid reprocessing."""
    return {'processed': True, 'count': call_count}


def _load_results_comms_raw(case_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load multiple communications sources from extraction results."""
    path = os.path.join('reports', case_id, 'results.json')
    sources: Dict[str, List[Dict[str, Any]]] = {
        'sms': [], 'calls': [], 'whatsapp': [], 'telegram': [], 'snapchat': [], 'instagram': []
    }
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            comms = data.get('data', {}).get('communications', {})
            # SMS
            sms = comms.get('sms_messages', [])
            if isinstance(sms, list):
                sources['sms'] = sms
            # Calls
            calls = comms.get('call_logs', [])
            if isinstance(calls, list):
                sources['calls'] = calls
            # App messages (if your extractor stores them)
            wa = comms.get('whatsapp_messages', [])
            if isinstance(wa, list):
                sources['whatsapp'] = wa
            tg = comms.get('telegram_messages', [])
            if isinstance(tg, list):
                sources['telegram'] = tg
            sc = comms.get('snapchat_messages', [])
            if isinstance(sc, list):
                sources['snapchat'] = sc
            ig = comms.get('instagram_messages', [])
            if isinstance(ig, list):
                sources['instagram'] = ig
        except Exception:
            return sources

    # Fallback to recent ADB dumps if available
    dumps_dir = ArtifactPathBuilder.resolve(case_id, 'android', 'provider_dumps')
    sms_dump = os.path.join(dumps_dir, 'sms_dump.txt')
    if os.path.exists(sms_dump):
        parsed_sms = parse_sms_dump(sms_dump)
        if parsed_sms:
            sources['sms'] = parsed_sms
    call_dump = os.path.join(dumps_dir, 'calllog_dump.txt')
    if os.path.exists(call_dump):
        parsed_calls = parse_calls_dump(call_dump)
        if parsed_calls:
            sources['calls'] = parsed_calls
    return sources


def load_model(prefer_auto: bool = True) -> Optional[Any]:
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


@st.cache_resource(show_spinner=False)
def _load_model_cached(prefer_auto: bool = True) -> Optional[Any]:
    return load_model(prefer_auto)


def score_messages(model: Any, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score messages using the provided model. Returns list of messages with 'score' field."""
    if not model or not messages:
        return messages
    texts = [m.get('text', '') for m in messages]
    if not texts:
        return messages

    scores_list = []
    try:
        # Assume model is a pipeline with decision_function or predict_proba
        if hasattr(model, 'decision_function'):
            scores = model.decision_function(texts)
            # Normalize to [0,1] if necessary
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


def _adb_pull_communications(case_id: str) -> Dict[str, Any]:
    """Attempt to refresh communications artifacts directly from the device via ADB."""
    if AndroidADB is None:
        return {'status': 'error', 'message': 'ADB module not available in runtime.'}

    adb = AndroidADB()
    summary = adb.device_summary()
    if not summary.get('installed'):
        return {'status': 'error', 'message': adb_root_access_message(summary, 'Communications extraction')}
    if not summary.get('connected'):
        return {'status': 'error', 'message': adb_root_access_message(summary, 'Communications extraction')}

    dumps_dir = ArtifactPathBuilder.resolve(case_id, 'android', 'provider_dumps', ensure_dir=True)
    try:
        dumps = adb.dump_content_providers(case_id, dumps_dir)
    except Exception as exc:
        return {'status': 'error', 'message': f'Provider dump failed: {exc}'}

    if not dumps:
        return {
            'status': 'empty',
            'message': adb_root_access_message(summary, 'SMS/call content providers')
        }

    sms_dump = os.path.join(dumps_dir, 'sms_dump.txt')
    sms = parse_sms_dump(sms_dump) if os.path.exists(sms_dump) else []

    call_dump = os.path.join(dumps_dir, 'calllog_dump.txt')
    calls = parse_calls_dump(call_dump) if os.path.exists(call_dump) else []

    for row in calls:
        duration = row.get('duration')
        if isinstance(duration, str) and duration.isdigit():
            row['duration'] = int(duration)

    return {
        'status': 'ok',
        'message': f"Provider dumps saved under {dumps_dir}",
        'sms': sms,
        'calls': calls,
        'path': dumps_dir,
        'files': dumps,
    }


_NLP_MODEL: Optional[Any] = None


def _extract_entities(text: str) -> List[Dict[str, str]]:
    """Extract entities using spaCy if available, otherwise use simple keyword matching."""
    entities = []
    global _NLP_MODEL
    try:
        import spacy
        if _NLP_MODEL is None:
            _NLP_MODEL = spacy.load('en_core_web_sm')
        doc = _NLP_MODEL(text)
        entities = [{'text': ent.text, 'type': ent.label_} for ent in doc.ents]
    except Exception:
        # Fallback to simple keyword matching
        keywords = ['location', 'address', 'meet',
            'time', 'date', 'phone', 'email']
        for kw in keywords:
            if kw.lower() in text.lower():
                entities.append({'text': kw, 'type': 'KEYWORD'})
    return entities


def _collect_base_records(sources: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    base_records: List[Dict[str, Any]] = []

    def _append_messages(items: List[Dict[str, Any]], source: str, text_keys: Tuple[str, ...] = ('message', 'text', 'body')) -> None:
        for message in items:
            if not isinstance(message, dict):
                continue
            
            found_text = False
            for key in text_keys:
                if message.get(key):
                    if 'text' not in message:
                        message['text'] = message[key]
                    found_text = True
                    break
            
            if found_text:
                message['source'] = source
                base_records.append(message)

    _append_messages(sources.get('sms', []), 'SMS', ('body', 'message', 'text'))
    _append_messages(sources.get('whatsapp', []), 'WhatsApp')
    _append_messages(sources.get('telegram', []), 'Telegram')
    _append_messages(sources.get('snapchat', []), 'Snapchat')
    _append_messages(sources.get('instagram', []), 'Instagram')
    _append_messages(sources.get('calls', []), 'Call', text_keys=('note', 'transcript', 'text'))

    return base_records


def _top_keywords(texts: List[str], limit: int = 6) -> List[Tuple[str, int]]:
    token_counter: Counter[str] = Counter()
    pattern = re.compile(r'[A-Za-z]{4,}')
    for text in texts:
        token_counter.update(tok.lower() for tok in pattern.findall(text))
    return token_counter.most_common(limit)


def render_ui(case_id: str):
    st.markdown('### 📡 Comms Analyzer')

    sources = _load_results_comms(case_id)
    base_records = _collect_base_records(sources)
    messages_to_score = [r for r in base_records if r.get('source') != 'Call']
    call_records = sources.get('calls', [])
    source_counts = Counter(record['source'] for record in base_records)

    tabs = st.tabs(["Messages", "Call Logs", "Suspicious Calls", "Network Graph"])

    # --- Messages Tab ---
    with tabs[0]:
        use_nlp = st.checkbox('Enable NLP analysis (requires spaCy)', value=False, key=f'nlp_{case_id}')
        col1, col2 = st.columns(2)
        with col1:
            st.caption('Messages from extraction (if available)')
            st.write(f"Loaded {len(messages_to_score)} messages from reports/{case_id}/results.json")
        with col2:
            uploaded = st.file_uploader(
                'Upload communications file (CSV/TXT/JSON/XLSX)',
                type=['csv', 'txt', 'json', 'xlsx'],
                key=f'{case_id}_comms_uploader')
            extra_messages: List[Dict[str, Any]] = []
            if uploaded:
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
                                extra_messages = [{'text': str(item), 'source': 'Uploaded'} for item in payload if item]
                        elif isinstance(payload, dict):
                            msgs_from_json = payload.get('messages') or payload.get('data') or []
                            if isinstance(msgs_from_json, list):
                                extra_messages = [
                                    item if isinstance(item, dict) else {'text': str(item), 'source': 'Uploaded'}
                                    for item in msgs_from_json if (item.get('message') if isinstance(item, dict) else item)
                                ]
                        else:
                            st.warning('Unsupported JSON structure. Expected list or { messages: [...] }.')
                    elif name.endswith('.txt'):
                        uploaded.seek(0)
                        content = uploaded.read().decode('utf-8', errors='ignore')
                        extra_messages = [{'text': line.strip(), 'source': 'Uploaded'} for line in content.splitlines() if line.strip()]
                    else:
                        st.warning('Unsupported file type.')
                except Exception as exc:
                    st.error(f'Failed to process uploaded file: {exc}')

        stat_cols = st.columns(3)
        stat_cols[0].metric('Messages ready', len(messages_to_score) + len(extra_messages))
        stat_cols[1].metric('Sources detected', len(source_counts))
        top_source = source_counts.most_common(1)
        stat_cols[2].metric('Top source', top_source[0][0] if top_source else '—', help=', '.join(f"{src}: {count}" for src, count in source_counts.most_common(4)))
        if source_counts:
            st.caption('Source mix: ' + ', '.join(f"{src}: {count}" for src, count in source_counts.most_common()))

        adb_feedback = st.empty()
        if st.button('Pull latest communications via ADB', key=f'adb_pull_{case_id}'):
            with st.spinner('Collecting communications via ADB…'):
                result = _adb_pull_communications(case_id)
            status = result.get('status')
            msg = result.get('message', '')
            if status == 'ok':
                adb_feedback.success(msg)
                _load_results_comms.clear()
                st.rerun()
            elif status == 'empty':
                adb_feedback.info(msg)
            else:
                adb_feedback.error(msg)

        all_messages = messages_to_score + extra_messages
        if not all_messages:
            st.info(
                'No messages available to score. Provide extraction results or upload a file.')
            return

        model = _load_model_cached(prefer_auto=True)
        if not model:
            st.warning(
                'No model found in models/. Place suspicious_tfidf_auto.pkl or suspicious_tfidf.pkl')
            return

        threshold = st.slider('Suspicion threshold', 0.0, 1.0, 0.7, 0.01, key=f'thresh_{case_id}')
        topn = st.number_input('Show top N', min_value=5, max_value=200, value=20, key=f'topn_{case_id}')

        auto_run = st.checkbox(
            'Auto-analyze on load (if data available)', value=True, key=f'autoan_{case_id}')

        def run_analysis():
            with st.spinner('Scoring communications...'):
                results = score_messages(model, all_messages)

            results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
            suspicious = [r for r in results if r.get('score', 0.0) >= threshold]
            
            _save_intelligence_findings(case_id, 'suspicious_messages', suspicious)

            st.success(
                f"Found {len(suspicious)} suspicious items (threshold {threshold:.2f})"
            )

            if suspicious:
                keyword_snapshot = _top_keywords([r['text'] for r in suspicious[: int(topn)]])
                if keyword_snapshot:
                    st.markdown('#### 🔑 Common keywords')
                    st.caption(', '.join(f"{word} ({count})" for word, count in keyword_snapshot))

            for r in suspicious[:int(topn)]:
                st.markdown(f"**Score:** {r.get('score', 0.0):.2f}")
                contact = r.get('contact') or r.get('name') or r.get('address') or r.get('sender') or '(Unknown)'
                st.markdown(f"**From:** {contact}")
                st.write(r['text'])
                attachments = r.get('attachments') or []
                for idx, path in enumerate(attachments):
                    if st.button(f"Open attachment {idx + 1}", key=f"att_{idx}_{r.get('score', 0.0)}"):
                        st.session_state['media_origin'] = {
                            'case_id': case_id,
                            'source': r.get('source', 'Communications')
                        }
                        st.session_state['current_media'] = path
                        st.session_state['nav'] = 'Media'
                        st.rerun()
                st.markdown('---')

            if suspicious and use_nlp:
                st.markdown('#### 🔍 NLP Analysis')
                entity_counts: Counter[Tuple[str, str]] = Counter()
                for r in suspicious[:int(topn)]:
                    entities = _extract_entities(r['text'])
                    if entities:
                        for ent in entities:
                            entity_counts[(ent['text'], ent['type'])] += 1
                if entity_counts:
                    top_entities = entity_counts.most_common(12)
                    cols = st.columns(3)
                    for idx, ((text, ent_type), count) in enumerate(top_entities):
                        cols[idx % 3].markdown(f'`{text}` • {ent_type} ({count})')

            if st.button('Export JSON', key=f'export_json_{case_id}'):
                out = {
                    'case_id': case_id,
                    'generated_at': datetime.now().isoformat(),
                    'threshold': threshold,
                    'count': len(suspicious),
                    'items': suspicious
                }
                st.download_button(
                    'Download JSON',
                    json.dumps(out, indent=2, default=str),
                    file_name=f'comms_{case_id}.json',
                    mime='application/json'
                )

        if auto_run and all_messages:
            run_analysis()
        elif st.button('Analyze', key=f'analyze_{case_id}'):
            run_analysis()

    # --- Call Logs Tab ---
    with tabs[1]:
        st.markdown('### 📞 Call Logs')
        if not call_records:
            st.info('No call logs available for this case.')
        else:
            df_calls = pd.DataFrame(call_records)
            st.dataframe(df_calls)
            st.download_button('Download Call Logs (CSV)', df_calls.to_csv(index=False), file_name=f'calls_{case_id}.csv')

    # --- Suspicious Calls Tab ---
    with tabs[2]:
        st.markdown('### 🚩 Suspicious Call Classifier')
        if not call_records:
            st.info('No call logs available for this case.')
        else:
            df_calls = pd.DataFrame(call_records)
            suspicious = []
            # --- Customizable Parameters ---
            st.markdown('#### Suspicious Call Detection Rules')
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                LONG_DURATION = st.number_input('Long call threshold (seconds)', min_value=60, max_value=3600*2, value=600, step=30, key='long_call')
            with col2:
                NIGHT_START = st.number_input('Night start hour', min_value=0, max_value=23, value=22, key='night_start')
            with col3:
                NIGHT_END = st.number_input('Night end hour', min_value=0, max_value=23, value=7, key='night_end')
            with col4:
                RAPID_REPEAT_MIN = st.number_input('Rapid repeat min calls/hr', min_value=2, max_value=10, value=3, step=1, key='rapid_repeat')
            FREQUENT_UNKNOWN_MIN = st.number_input('Frequent unknown min calls', min_value=2, max_value=20, value=5, step=1, key='freq_unknown')
            NIGHT_HOURS = (NIGHT_START, NIGHT_END)
            # --- Preprocess (Optimized) ---
            if 'timestamp' in df_calls.columns:
                df_calls['timestamp'] = pd.to_datetime(df_calls['timestamp'], errors='coerce', format='%Y-%m-%dT%H:%M:%S', exact=False)
            elif 'date' in df_calls.columns:
                df_calls['timestamp'] = pd.to_datetime(df_calls['date'], errors='coerce', format='%Y-%m-%dT%H:%M:%S', exact=False)
            else:
                df_calls['timestamp'] = pd.NaT
            df_calls['duration'] = pd.to_numeric(df_calls.get('duration', 0), errors='coerce').fillna(0).astype(int)
            df_calls['number'] = df_calls.get('number', df_calls.get('phone', 'UNKNOWN')).astype(str)
            
            for idx, row in df_calls.iterrows():
                reasons = []
                if row['duration'] >= LONG_DURATION:
                    reasons.append(f'Long call ({row["duration"]}s)')
                if pd.notnull(row['timestamp']):
                    t = row['timestamp'].time()
                    if (t.hour >= NIGHT_HOURS[0] or t.hour < NIGHT_HOURS[1]):
                        reasons.append('Odd hour')
                if not row.get('contact') or row.get('contact') in ('', 'UNKNOWN', None):
                    reasons.append('Unknown number')
                suspicious.append({'idx': idx, 'number': row['number'], 'timestamp': row.get('timestamp'), 'duration': row['duration'], 'reasons': reasons})
            
            if 'number' in df_calls and 'timestamp' in df_calls:
                df_sorted = df_calls.sort_values(['number', 'timestamp'])
                for num, group in df_sorted.groupby('number'):
                    times = group['timestamp'].dropna().sort_values()
                    for i in range(len(times) - RAPID_REPEAT_MIN + 1):
                        window = times.iloc[i:i+RAPID_REPEAT_MIN]
                        if (window.max() - window.min()).total_seconds() <= 3600:
                            for idx in group.iloc[i:i+RAPID_REPEAT_MIN].index:
                                suspicious[idx]['reasons'].append('Rapid repeat')
            
            unknown_counts = df_calls[df_calls.apply(lambda row: not row.get('contact') or row.get('contact') in ('', 'UNKNOWN', None), axis=1)]['number'].value_counts()
            for num, count in unknown_counts.items():
                if count >= FREQUENT_UNKNOWN_MIN:
                    for idx in df_calls[df_calls['number'] == num].index:
                        suspicious[idx]['reasons'].append(f'Frequent unknown ({count})')
            
            suspicious_df = pd.DataFrame([s for s in suspicious if s['reasons']])
            if not suspicious_df.empty:
                _save_intelligence_findings(case_id, 'suspicious_calls', suspicious_df.to_dict('records'))
            
            # --- Contact Enrichment ---
            if 'contact' in df_calls.columns:
                contact_map = df_calls[df_calls['contact'].notna() & (df_calls['contact'] != '')].groupby('number')['contact'].first().to_dict()
                suspicious_df['contact_name'] = suspicious_df['number'].map(contact_map).fillna('(Unknown)')
            else:
                suspicious_df['contact_name'] = '(Unknown)'

            direction_col = next((col for col in ['direction', 'type', 'call_type'] if col in df_calls.columns), None)
            if direction_col:
                suspicious_df['direction'] = suspicious_df['idx'].map(df_calls[direction_col])
            else:
                suspicious_df['direction'] = 'unknown'

            st.markdown('#### 📊 Call Patterns Visualization')
            if not df_calls['timestamp'].isnull().all():
                df_calls['hour'] = df_calls['timestamp'].dt.hour
                fig1 = px.histogram(df_calls, x='hour', nbins=24, title='Call Count by Hour', labels={'hour':'Hour of Day'})
                st.plotly_chart(fig1, use_container_width=True)
            
            fig2 = px.histogram(df_calls, x='duration', nbins=30, title='Call Duration Distribution (seconds)')
            st.plotly_chart(fig2, use_container_width=True)
            
            if not suspicious_df.empty:
                exploded = suspicious_df.explode('reasons')
                fig3 = px.histogram(exploded, x='reasons', title='Suspicious Call Reasons')
                st.plotly_chart(fig3, use_container_width=True)

            if not suspicious_df.empty:
                all_reasons = set(r for reasons in suspicious_df['reasons'] for r in reasons)
                selected_reason = st.selectbox('Filter by suspicious reason', ['All'] + sorted(all_reasons), key='reason_filter')
                filtered_df = suspicious_df
                if selected_reason != 'All':
                    filtered_df = suspicious_df[suspicious_df['reasons'].apply(lambda x: selected_reason in x)]
                
                all_directions = suspicious_df['direction'].unique()
                selected_direction = st.selectbox('Filter by call direction/type', ['All'] + sorted(all_directions), key='dir_filter')
                if selected_direction != 'All':
                    filtered_df = filtered_df[filtered_df['direction'] == selected_direction]
            else:
                filtered_df = suspicious_df

            if suspicious_df.empty:
                st.success('No suspicious calls detected based on current rules.')
            else:
                filtered_df['reasons_str'] = filtered_df['reasons'].apply(lambda x: ', '.join(set(x)))
                display_df = filtered_df[['contact_name', 'number', 'timestamp', 'duration', 'direction', 'reasons_str']].copy()
                display_df.columns = ['Contact Name', 'Phone Number', 'Timestamp', 'Duration (s)', 'Direction', 'Reasons']
                st.dataframe(display_df, use_container_width=True)
                
                st.markdown('#### 📋 Call Details')
                for idx, row in filtered_df.iterrows():
                    contact_name = row['contact_name'] or '(Unknown)'
                    phone_number = row['number']
                    with st.expander(f"📞 {contact_name} - {phone_number}"):
                        st.write(f"**Contact:** {contact_name}")
                        st.write(f"**Number:** {phone_number}")
                        st.write(f"**Duration:** {row['duration']}s")
                        st.write(f"**Direction:** {row['direction']}")
                        st.write(f"**Timestamp:** {row['timestamp']}")
                        st.write(f"**Reasons:** {row['reasons_str']}")
                
                st.download_button('Download Suspicious Calls (CSV)', filtered_df.to_csv(index=False), file_name=f'suspicious_calls_{case_id}.csv')
                
                pdf_buffer = io.BytesIO()
                c = canvas.Canvas(pdf_buffer, pagesize=letter)
                c.drawString(30, 750, f"Suspicious Calls Report - Case {case_id}")
                y = 730
                for idx, row in filtered_df.iterrows():
                    line = f"{row['timestamp']} | {row['number']} | {row['contact_name']} | {row['duration']}s | {row['direction']} | {row['reasons_str']}"
                    c.drawString(30, y, line[:120])
                    y -= 15
                    if y < 40:
                        c.showPage()
                        y = 750
                c.save()
                st.download_button('Export PDF Report', pdf_buffer.getvalue(), file_name=f'suspicious_calls_{case_id}.pdf', mime='application/pdf')
                
                if 'recording' in df_calls.columns:
                    st.markdown('#### 🎧 Linked Call Recordings (if available)')
                    for idx, row in filtered_df.iterrows():
                        rec_path = df_calls.loc[row['idx']].get('recording')
                        if rec_path and isinstance(rec_path, str) and rec_path.strip():
                            st.audio(rec_path, format='audio/wav')
                            st.caption(f"Recording for call: {row['number']} at {row['timestamp']}")

    # --- Network Graph Tab ---
    with tabs[3]:
        st.markdown('### 🕸️ Communication Network Graph')
        if Network is None:
            st.error("Pyvis library not installed. Please run: pip install pyvis")
            return

        if not base_records and not call_records:
            st.info('No communication data available to build a graph.')
            return

        if st.button('Generate Network Graph', key=f'generate_net_graph_{case_id}'):
            with st.spinner('Building communication graph...'):
                net = create_network_graph(base_records, call_records)
                try:
                    net.save_graph('tmp/conversation_graph.html')
                    with open('tmp/conversation_graph.html', 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    components.html(html_content, height=600, scrolling=True)
                    st.success('Graph generated successfully.')
                except Exception as e:
                    st.error(f"Failed to generate or display the graph: {e}")


def create_network_graph(messages: List[Dict[str, Any]], calls: List[Dict[str, Any]]) -> 'Network':
    """Creates a pyvis network graph from messages and calls."""
    net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white', notebook=True)
    net.barnes_hut()

    contacts = set()
    edges = Counter()

    for msg in messages:
        sender = msg.get('sender') or msg.get('address') or msg.get('contact') or 'Unknown'
        receiver = msg.get('receiver') or 'Device Owner'
        if sender != 'Unknown':
            contacts.add(sender)
            contacts.add(receiver)
            edges[(sender, receiver)] += 1

    for call in calls:
        caller = call.get('contact') or call.get('number') or 'Unknown'
        callee = 'Device Owner'
        if caller != 'Unknown':
            contacts.add(caller)
            contacts.add(callee)
            edges[(caller, callee)] += 1

    for contact in contacts:
        net.add_node(contact, label=contact, title=contact)

    for (source, target), weight in edges.items():
        net.add_edge(source, target, value=weight)

    return net
