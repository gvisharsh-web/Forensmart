import os  # required for FORNSMART_DEV_MODE env toggle
import streamlit as st
from adapters.runner import AdapterRunner
from pathlib import Path as _P
import pathlib
from report_integrity_module import finalize_report_integrity


# --- Top-level Tabs (Main + Adapters) ---
# --- DEV DEBUG: show FORNSMART_DEV_MODE env var and apply dev defaults automatically ---
import os as _os_debug

_dev_mode_env = _os_debug.getenv("FORNSMART_DEV_MODE", None)
st.info(f"DEV DEBUG: FORNSMART_DEV_MODE={_dev_mode_env}")

try:
    # call helper if available to apply defaults
    if "_apply_dev_mode_defaults" in globals():
        _apply_dev_mode_defaults(st)
except Exception as _e:
    st.error("DEV DEBUG: failed to apply dev mode defaults: " + str(_e))
    # --- end dev debug ---
    tabs_top = st.tabs(["Main App", "Adapters"])
    # Main App tab contains a note to continue using the existing UI below
    with tabs_top[0]:
        st.markdown("## Main App")
        st.info(
            "The original application UI continues below. Use the 'Adapters' tab to access extraction controls."
        )
    # Adapters tab: render adapter controls isolated from the rest of the UI
    with tabs_top[1]:
        # Apply developer mode defaults if enabled
        if "_apply_dev_mode_defaults" in globals():
            _apply_dev_mode_defaults(st)
        st.markdown("## Extraction Adapters (Top-level)")
        consent_id = st.session_state.get("consent_id")
        if not consent_id:
            st.warning(
                "No consent_id linked in session. Capture consent in the Consent area before running intrusive extractions."
            )
        st.subheader("Android")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Run Android Adapter (test)"):
                if not consent_id:
                    st.warning("No consent_id; capture consent first.")
                else:
                    out_dir = f"./artifacts/{consent_id}/android"
                    try:
                        runner = AdapterRunner(
                            adapter_module="adapters.android_adb", adapter_cls="Adapter"
                        )
                        res = runner.run_python_adapter(
                            out_dir, ["all"], {"consent_id": consent_id}
                        )
                        st.write(res)
                    except Exception as e:
                        st.error(f"Adapter run failed: {e}")
            if st.button("List Android Artifacts"):
                base = _P(f"./artifacts/{consent_id}/android")
                if not base.exists():
                    st.info("No artifacts yet for this consent_id.")
                else:
                    files = [
                        str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()
                    ]
                    st.write(files)
        with col2:
            st.subheader("iOS")
            if st.button("Run iOS Adapter (test)"):
                if not consent_id:
                    st.warning("No consent_id; capture consent first.")
                else:
                    ok, msg = (
                        verify_consent_id(consent_id)
                        if "verify_consent_id" in globals()
                        else (True, "")
                    )
                    if not ok:
                        st.error("Consent verification failed: " + msg)
                    else:
                        out_dir = f"./artifacts/{consent_id}/ios"
                        try:
                            runner = AdapterRunner(
                                adapter_module="adapters.ios_logical",
                                adapter_cls="Adapter",
                            )
                            res = runner.run_python_adapter(
                                out_dir, ["all"], {"consent_id": consent_id}
                            )
                            st.write(res)
                        except Exception as e:
                            st.error(f"iOS adapter run failed: {e}")
            st.subheader("HDD")
            if st.button("Run HDD Imager (simulate)"):
                if not consent_id:
                    st.warning("No consent_id; capture consent first.")
                else:
                    ok, msg = (
                        verify_consent_id(consent_id)
                        if "verify_consent_id" in globals()
                        else (True, "")
                    )
                    if not ok:
                        st.error("Consent verification failed: " + msg)
                    else:
                        out_dir = f"./artifacts/{consent_id}/hdd"
                        try:
                            runner = AdapterRunner(
                                adapter_module="adapters.hdd_imager",
                                adapter_cls="Adapter",
                            )
                            res = runner.run_python_adapter(
                                out_dir, ["simulate"], {"consent_id": consent_id}
                            )
                            st.write(res)
                        except Exception as e:
                            st.error(f"HDD adapter run failed: {e}")
except Exception:
    pass


# --- Consent verification helper ---
def verify_consent_id(consent_id: str):
    """Return (True, message) if consent verification passes, else (False, message)."""
    from pathlib import Path
    import subprocess
    import sys
    import json

    if not consent_id:
        return False, "No consent_id provided."
    consent_folder = Path(f"./consent_records/{consent_id}")
    if not consent_folder.exists():
        return False, f"Consent folder not found: {consent_folder}"
    verifier = Path("verify_consent.py")
    if verifier.exists():
        try:
            cp = subprocess.run(
                [sys.executable, str(verifier), str(consent_folder)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if cp.returncode == 0:
                return True, "Verification script passed."
            else:
                return False, f"Verifier reported failure: {cp.stdout}\\n{cp.stderr}"
        except Exception as e:
            return False, f"Failed to run verifier: {e}"
    else:
        # Inline minimal check
        cj = consent_folder / "consent.json"
        sigfile = consent_folder / "signature.b64"
        if not cj.exists():
            return False, "consent.json not found"
        md = {}
        try:
            md = json.loads(cj.read_text())
        except Exception:
            pass
        has_hash = "artifact_hash" in md
        has_sig = ("signature" in md) or sigfile.exists()
        if has_hash and has_sig:
            return True, "Basic presence checks OK (artifact_hash + signature present)."
        return False, "Missing artifact_hash or signature; cannot verify."


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import joblib
import datetime
import hashlib
from fpdf import FPDF
from io import BytesIO
import pydeck as pdk
from geopy.distance import geodesic
import time
import json

# ================== CONFIG ==================
st.set_page_config(page_title="ForenSmart", layout="wide")
st.title("🔍 ForenSmart - Next-Gen Forensic AI Suite")

# Create models folder if not exists
if not os.path.exists("models"):
    os.makedirs("models")

# ================== CASE METADATA ==================
st.sidebar.header("📂 Case Metadata")

# ---- INSERT: Adapters & Consent Capture (added by assistant) ----
with st.sidebar.expander("🔌 Data Adapters & Consent", expanded=True):
    st.markdown(
        "Configure data adapters (ingest connectors) and capture investigator consent / chain-of-custody notes."
    )
    AVAILABLE_ADAPTERS = [
        "Local CSV",
        "Google Drive (link)",
        "iCloud (link)",
        "ADB (USB)",
        "WhatsApp Export (.txt/.zip)",
        "Telegram Export",
        "Manual JSON",
        "SFTP",
    ]
    st.session_state.setdefault("selected_adapters", [])
    st.session_state.setdefault("adapter_config", {})
    st.session_state.setdefault("adapters_last_sync", None)
    st.session_state.setdefault("adapters_discrepancy", False)

    sel = st.multiselect(
        "Select active adapters",
        AVAILABLE_ADAPTERS,
        default=st.session_state.get("selected_adapters", []),
    )
    cols = st.columns([3, 1])
    with cols[0]:
        adapter_notes = st.text_area(
            "Adapter notes / endpoint config (JSON or plain text)",
            value=st.session_state.get("adapter_config", {}).get("notes", ""),
            height=80,
        )
    with cols[1]:
        if st.button("Save Adapters"):
            prev = set(st.session_state.get("selected_adapters", []))
            new = set(sel)
            st.session_state["adapter_config"] = {
                "notes": adapter_notes,
                "saved_at": str(datetime.datetime.utcnow()),
            }
            st.session_state["selected_adapters"] = list(sel)
            st.session_state["adapters_last_sync"] = (
                datetime.datetime.utcnow().isoformat() + "Z"
            )
            if prev != new:
                st.session_state["adapters_discrepancy"] = True
                st.warning(
                    "Adapter set changed — discrepancy flagged. Please review ingestion steps."
                )
            else:
                st.session_state["adapters_discrepancy"] = False
            st.success("Adapters saved to session state.")

    st.markdown("---")
    st.write("Current adapters:", st.session_state.get("selected_adapters", []))
    if st.session_state.get("adapters_last_sync"):
        st.write("Last adapters save (UTC):", st.session_state["adapters_last_sync"])

    if st.session_state.get("adapters_discrepancy", False):
        st.error(
            "⚠️ Adapter discrepancy detected. Consider re-running extraction or recording justification below."
        )
        justification = st.text_area(
            "Discrepancy justification (investigator notes):",
            value=st.session_state.get("adapter_config", {}).get(
                "discrepancy_just", ""
            ),
            height=80,
        )
        if st.button("Record justification"):
            st.session_state["adapter_config"]["discrepancy_just"] = justification
            st.success("Justification saved.")

    # Consent capture
    st.markdown("### Investigator Consent / Chain-of-Custody")
    if "consent_record" not in st.session_state:
        st.session_state["consent_record"] = {
            "consent_given": False,
            "by": None,
            "timestamp": None,
            "notes": None,
        }
    ccol1, ccol2 = st.columns([3, 2])
    with ccol1:
        consent = st.checkbox(
            "I confirm I have authorization to analyze this data (Give Consent)",
            value=st.session_state["consent_record"].get("consent_given", False),
        )
    with ccol2:
        initials = st.text_input(
            "Investigator initials",
            value=st.session_state["consent_record"].get("by") or "",
        )
    consent_notes = st.text_area(
        "Consent / custody notes (optional)",
        value=st.session_state["consent_record"].get("notes", ""),
        height=80,
    )
    if st.button("Record Consent"):
        ts = datetime.datetime.utcnow().isoformat() + "Z"
        st.session_state["consent_record"] = {
            "consent_given": bool(consent),
            "by": initials.strip() or None,
            "timestamp": ts if consent else None,
            "notes": consent_notes,
        }
        if "case_audit" not in st.session_state:
            st.session_state["case_audit"] = []
        st.session_state["case_audit"].append(
            {
                "event": "consent_recorded",
                "consent_given": bool(consent),
                "by": initials.strip() or None,
                "timestamp": ts,
                "notes": consent_notes,
            }
        )
        # persist consent to disk for chain-of-custody (consent_records/<case_id>/consent.json)
        try:
            cr_dir = os.path.join(
                "consent_records", st.session_state.get("case_id", "unknown_case")
            )
            pathlib.Path(cr_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(cr_dir, "consent.json"), "w") as f:
                json.dump(st.session_state["consent_record"], f, indent=2)
            cid = hashlib.sha256(
                json.dumps(st.session_state["consent_record"], sort_keys=True).encode()
            ).hexdigest()
            with open(os.path.join(cr_dir, "consent_id.txt"), "w") as f:
                f.write(cid)
            st.session_state["consent_record"]["consent_id"] = cid
            st.success(f"Consent recorded at {ts} (UTC). consent_id={cid}")
        except Exception as e:
            st.warning(f"Failed to persist consent to disk: {e}")

    st.markdown("#### Current consent record")
    st.write(st.session_state.get("consent_record", {}))
# ---- END INSERT ----


st.session_state["case_id"] = st.sidebar.text_input(
    "Case ID", value=f"CASE-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
)
st.session_state["device_id"] = st.sidebar.text_input("Device ID / Serial")
st.session_state["investigator"] = st.sidebar.text_input("Investigator Name")
st.session_state["acquisition_date"] = st.sidebar.date_input(
    "Acquisition Date", value=datetime.date.today()
)

# ================== TABS ==================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📍 Location Intelligence",
        "💬 Suspicious Messages",
        "🔑 Password Strength",
        "📊 Tabular ML Training",
        "🖼️ Image ML Training",
        "📄 Unified Report",
    ]
)

# ======================================================
# 1️⃣ LOCATION INTELLIGENCE TAB
# ======================================================
with tab1:
    st.subheader("📍 Location Intelligence - GPS, Cell, Link Mode")

    mode = st.radio(
        "Choose Mode",
        ["USB GPS Logs", "Link Mode", "Cell Tower Fallback", "Manual Input"],
    )

    # Utility: hash calculator
    def compute_hash(data):
        if isinstance(data, str):
            return hashlib.sha256(data.encode()).hexdigest()
        elif isinstance(data, bytes):
            return hashlib.sha256(data).hexdigest()
        elif isinstance(data, dict) or isinstance(data, list):
            return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        else:
            return "N/A"

    # AI Timeline Summary
    def generate_timeline_summary(
        df, lat_col="lat", lon_col="lon", time_col="timestamp", score_col=None
    ):
        if df.empty:
            return "No records"
        df = df.sort_values(by=time_col).reset_index(drop=True)
        summary = []
        for i in range(len(df) - 1):
            t1, t2 = df.loc[i, time_col], df.loc[i + 1, time_col]
            p1, p2 = (df.loc[i, lat_col], df.loc[i, lon_col]), (
                df.loc[i + 1, lat_col],
                df.loc[i + 1, lon_col],
            )
            gap = (t2 - t1).total_seconds() / 60
            dist = geodesic(p1, p2).km
            speed = (dist / (gap / 60)) if gap > 0 else 0
            if gap > 30:
                summary.append(f"⏳ GPS gap {int(gap)} mins between {t1} and {t2}")
            elif speed > 200:
                summary.append(
                    f"🚨 Impossible travel: {dist:.1f} km in {int(gap)} mins (~{speed:.1f} km/h)"
                )
            else:
                summary.append(
                    f"📍 Moved {dist:.1f} km between {t1} and {t2} (avg {speed:.1f} km/h)"
                )
        return "\n".join(summary)

    # Example USB GPS Mode
    if mode == "USB GPS Logs":
        gps_file = st.file_uploader("Upload GPS Data (CSV)", type=["csv"])
        if gps_file:
            gps_df = pd.read_csv(gps_file)
            gps_df["timestamp"] = pd.to_datetime(gps_df["timestamp"])
            gps_df = gps_df.rename(columns={"latitude": "lat", "longitude": "lon"})

            if st.button("▶️ Playback + Summary"):
                for t in pd.date_range(
                    gps_df["timestamp"].min(), gps_df["timestamp"].max(), freq="30min"
                ):
                    window = gps_df[gps_df["timestamp"] <= t]
                    if not window.empty:
                        st.write(f"🕒 {t}")
                        st.map(window)
                        time.sleep(1)

                summary = generate_timeline_summary(gps_df)
                st.subheader("📝 AI Timeline Summary")
                st.text(summary)
                st.session_state["location_summary"] = summary

# ======================================================
# 2️⃣ SUSPICIOUS MESSAGES TAB
# ======================================================
with tab2:
    st.subheader("💬 Suspicious Message Detection")
    msg = st.text_area("Paste a message (supports multi-language):")
    if msg:
        # Simple placeholder risk scoring
        risk_score = np.random.rand()
        explanation = "Contains keywords that may indicate threat"
        st.write(f"Suspicion Score: {risk_score:.2f}")
        st.session_state["suspicious_messages"] = [
            {
                "timestamp": str(datetime.datetime.now()),
                "sender": "Unknown",
                "text": msg,
                "score": risk_score,
                "explanation": explanation,
            }
        ]

# ======================================================
# 3️⃣ PASSWORD TAB
# ======================================================
with tab3:
    st.subheader("🔑 Password Strength")
    password = st.text_input("Enter password:", type="password")
    if password:
        if len(password) < 6:
            result = "Weak ❌"
        elif password.isalpha() or password.isnumeric():
            result = "Medium ⚠️"
        else:
            result = "Strong ✅"
        st.write(f"Password strength: {result}")
        st.session_state["password_eval"] = f"Password strength: {result}"

# ======================================================
# 6️⃣ UNIFIED REPORT TAB
# ======================================================
with tab6:
    st.subheader("📄 Generate Unified Case Report")

    def export_unified_case_report(metadata):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "ForenSmart - Unified Case Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", size=12)
        for k, v in metadata.items():
            pdf.cell(0, 10, f"{k}: {v}", ln=True)
        pdf.ln(10)

        if "location_summary" in st.session_state:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "📍 Location Analysis", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, st.session_state["location_summary"])
            pdf.multi_cell(
                0, 8, f"Hash: {compute_hash(st.session_state['location_summary'])}"
            )
            pdf.ln(5)

        if "suspicious_messages" in st.session_state:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "💬 Suspicious Messages", ln=True)
            pdf.set_font("Arial", size=12)
            for msg in st.session_state["suspicious_messages"]:
                pdf.multi_cell(
                    0,
                    8,
                    f"{msg['timestamp']} - {msg['text']} (Score {msg['score']:.2f})",
                )
                pdf.multi_cell(0, 8, f"Explanation: {msg['explanation']}")
            pdf.multi_cell(
                0, 8, f"Hash: {compute_hash(st.session_state['suspicious_messages'])}"
            )
            pdf.ln(5)

        if "password_eval" in st.session_state:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "🔑 Password Strength", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, st.session_state["password_eval"])
            pdf.multi_cell(
                0, 8, f"Hash: {compute_hash(st.session_state['password_eval'])}"
            )
            pdf.ln(5)

        # Master Case Hash
        combined = {
            "location_summary": st.session_state.get("location_summary"),
            "suspicious_messages": st.session_state.get("suspicious_messages"),
            "password_eval": st.session_state.get("password_eval"),
        }
        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(0, 8, f"🔒 Master Case Hash: {compute_hash(combined)}")

        pdf_output = BytesIO()
        pdf.output(pdf_output, "F")
        pdf_output.seek(0)
        return pdf_output

    metadata = {
        "Case ID": st.session_state["case_id"],
        "Device ID": st.session_state["device_id"],
        "Investigator": st.session_state["investigator"],
        "Acquisition Date": str(st.session_state["acquisition_date"]),
    }

    if st.button("📄 Generate Unified Report"):
        pdf_file = export_unified_case_report(metadata)
        st.download_button(
            "⬇️ Download Case Report",
            data=pdf_file,
            file_name=f"{st.session_state['case_id']}_report.pdf",
            mime="application/pdf",
        )
import streamlit as st
import pandas as pd
import re
import time
import datetime

# ======================================================
# # ---------------- Location Intelligence (FULL) ----------------
with tab1:
    import os
    import tempfile
    import re
    import datetime as dt
    import folium
    from streamlit_folium import st_folium
    from shapely.geometry import Point, Polygon
    import matplotlib.pyplot as plt

    st.subheader(
        "📍 Location Intelligence — USB / Link / Cell / Manual / Hotspots / Zones"
    )

    mode = st.radio(
        "Choose Mode",
        ["USB GPS Logs", "Link Mode (multi)", "Cell Tower Fallback", "Manual Input"],
    )

    # helper: compute SHA256
    def compute_hash(data):
        import json
        import hashlib

        if data is None:
            return "N/A"
        if isinstance(data, (dict, list)):
            b = json.dumps(data, sort_keys=True).encode()
        elif isinstance(data, bytes):
            b = data
        else:
            b = str(data).encode()
        return hashlib.sha256(b).hexdigest()

    # helper: simple timeline summary
    def generate_timeline_summary(
        df, lat_col="lat", lon_col="lon", time_col="timestamp"
    ):
        if df is None or df.empty:
            return "No records"
        df = df.sort_values(by=time_col).reset_index(drop=True)
        parts = []
        for i in range(len(df) - 1):
            t1 = df.loc[i, time_col]
            t2 = df.loc[i + 1, time_col]
            p1 = (df.loc[i, lat_col], df.loc[i, lon_col])
            p2 = (df.loc[i + 1, lat_col], df.loc[i + 1, lon_col])
            gap_mins = (t2 - t1).total_seconds() / 60.0
            dist_km = geodesic(p1, p2).km
            speed = (dist_km / (gap_mins / 60.0)) if gap_mins > 0 else 0
            if gap_mins > 30:
                parts.append(
                    f"⏳ GPS gap of {int(gap_mins)} mins between {t1} and {t2}"
                )
            elif speed > 200:
                parts.append(
                    f"🚨 Impossible travel: {dist_km:.1f} km in {int(gap_mins)} mins (~{speed:.1f} km/h)"
                )
            else:
                parts.append(
                    f"📍 Moved {dist_km:.1f} km between {t1} and {t2} (avg {speed:.1f} km/h)"
                )
        return "\n".join(parts)

    # helper: show a pydeck heatmap (visual only)
    def show_pydeck_heatmap(df, lat_col="lat", lon_col="lon"):
        if df is None or df.empty:
            st.info("No points to show heatmap.")
            return
        df2 = df.copy()
        df2["weight"] = 1
        try:
            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/dark-v10",
                    initial_view_state=pdk.ViewState(
                        latitude=df2[lat_col].mean(),
                        longitude=df2[lon_col].mean(),
                        zoom=11,
                        pitch=40,
                    ),
                    layers=[
                        pdk.Layer(
                            "HeatmapLayer",
                            data=df2,
                            get_position=[lon_col, lat_col],
                            get_weight="weight",
                            radiusPixels=60,
                        )
                    ],
                )
            )
        except Exception as e:
            st.warning(
                "Pydeck heatmap failed (mapbox key/pydeck). Falling back to scatter snapshot. "
                + str(e)
            )

    # ---------------- USB GPS Logs mode ----------------
    if mode == "USB GPS Logs":
        gps_file = st.file_uploader(
            "Upload GPS data (CSV with timestamp, latitude, longitude)", type=["csv"]
        )
        gps_df = None
        if gps_file:
            gps_df = pd.read_csv(gps_file)
            if "timestamp" in gps_df.columns:
                gps_df["timestamp"] = pd.to_datetime(gps_df["timestamp"])
            else:
                gps_df["timestamp"] = pd.Timestamp.now()
            # normalize columns
            if "latitude" in gps_df.columns and "longitude" in gps_df.columns:
                gps_df = gps_df.rename(columns={"latitude": "lat", "longitude": "lon"})
            elif "lat" in gps_df.columns and "lon" in gps_df.columns:
                pass
            else:
                st.error("GPS CSV must include latitude/longitude or lat/lon columns.")
                gps_df = None

        # time window + map + heatmap snapshot
        if gps_df is not None and not gps_df.empty:
            min_t, max_t = gps_df["timestamp"].min(), gps_df["timestamp"].max()
            time_window = st.slider(
                "Select time window",
                min_value=min_t.to_pydatetime(),
                max_value=max_t.to_pydatetime(),
                value=(min_t.to_pydatetime(), max_t.to_pydatetime()),
                format="YYYY-MM-DD HH:mm",
            )
            filt = gps_df[
                (gps_df["timestamp"] >= pd.to_datetime(time_window[0]))
                & (gps_df["timestamp"] <= pd.to_datetime(time_window[1]))
            ]

            st.markdown("#### Map (selected window)")
            st.map(filt[["lat", "lon"]])

            # pydeck heatmap (visual) + Matplotlib snapshot saved for PDF
            show_pydeck_heatmap(filt)
            # Matplotlib snapshot (saved to session_state for PDF)
            if not filt.empty:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(filt["lon"], filt["lat"], c="red", alpha=0.5)
                ax.set_title("Suspicion Points (snapshot)")
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(tmp.name, bbox_inches="tight")
                plt.close(fig)
                st.session_state["heatmap_snapshot"] = tmp.name
                st.image(tmp.name, caption="Heatmap snapshot (saved for report)")

            # playback + AI timeline summary
            if st.button("▶️ Playback + Generate Summary"):
                for t in pd.date_range(min_t, max_t, freq="30min"):
                    window = gps_df[gps_df["timestamp"] <= t]
                    if not window.empty:
                        st.write(f"🕒 {t}")
                        st.map(window[["lat", "lon"]])
                        time.sleep(0.6)
                summary = generate_timeline_summary(filt)
                st.subheader("📝 AI Timeline Summary")
                st.text(summary)
                st.session_state["location_summary"] = summary
                st.success(
                    "Location summary saved to session_state['location_summary']."
                )

            # ---------------- Crime hotspots upload (csv/xlsx/json/image) ----------------
            st.markdown("## Crime Scene Hotspots (file or image)")
            crime_file = st.file_uploader(
                "Upload hotspots (CSV/XLSX/JSON) or map image (JPG/PNG)",
                type=["csv", "xlsx", "xls", "json", "jpg", "png"],
                key="crime_file",
            )

            manual_hotspots = (
                []
            )  # used if user adds coordinates from an image during this run

            if crime_file:
                ext = os.path.splitext(crime_file.name)[1].lower()
                if ext == ".csv":
                    crime_df = pd.read_csv(crime_file)
                elif ext in [".xlsx", ".xls"]:
                    crime_df = pd.read_excel(crime_file)
                elif ext == ".json":
                    crime_df = pd.read_json(crime_file)
                elif ext in [".jpg", ".png"]:
                    st.image(
                        crime_file,
                        caption="Uploaded crime map image (use manual input below to add points)",
                    )
                    crime_df = None
                else:
                    st.error("Unsupported hotspot file.")
                    crime_df = None

                # if a tabular file provided, plot hotspots and check overlaps
                if crime_df is not None and not crime_df.empty:
                    if {"latitude", "longitude"}.issubset(set(crime_df.columns)) or {
                        "lat",
                        "lon",
                    }.issubset(set(crime_df.columns)):
                        if "latitude" in crime_df.columns:
                            crime_df_plot = crime_df.rename(
                                columns={"latitude": "lat", "longitude": "lon"}
                            )
                        else:
                            crime_df_plot = crime_df.rename(
                                columns={"lat": "lat", "lon": "lon"}
                            )
                        st.markdown("### Hotspot points (from file)")
                        st.map(crime_df_plot[["lat", "lon"]])

                        # check overlaps (≤ 200 m)
                        overlaps = []
                        for _, g in gps_df.iterrows():
                            for _, c in crime_df_plot.iterrows():
                                d = geodesic(
                                    (g["lat"], g["lon"]), (c["lat"], c["lon"])
                                ).meters
                                if d <= 200:
                                    overlaps.append(
                                        {
                                            "timestamp": g["timestamp"],
                                            "hotspot_name": c.get("name", "Unknown"),
                                            "distance_m": round(d, 1),
                                        }
                                    )
                        if overlaps:
                            st.markdown("### 🚨 Overlaps with hotspots (≤200m)")
                            st.dataframe(pd.DataFrame(overlaps))
                        else:
                            st.success("No overlaps with uploaded hotspots.")
                        # save for report
                        st.session_state["crime_hotspots_table"] = (
                            crime_df_plot.to_dict("records")
                        )
                        st.session_state["crime_hotspots_hash"] = compute_hash(
                            crime_df_plot.to_dict("records")
                        )
                    else:
                        st.warning(
                            "Hotspot file must contain 'latitude' and 'longitude' (or 'lat'/'lon')."
                        )
                else:
                    # file was image OR no dataframe produced
                    st.info(
                        "If you uploaded an image, enter hotspot coordinates manually below (they will be saved)."
                    )

                    # manual hotspot entry (useful when image uploaded)
                    st.markdown("### Manual hotspot entry (for uploaded image)")
                    mh_lat = st.number_input(
                        "Hotspot latitude", format="%.6f", key="mh_lat"
                    )
                    mh_lon = st.number_input(
                        "Hotspot longitude", format="%.6f", key="mh_lon"
                    )
                    mh_name = st.text_input(
                        "Hotspot name", value="Manual Hotspot", key="mh_name"
                    )
                    if st.button("➕ Add manual hotspot"):
                        manual_hotspots.append(
                            {"name": mh_name, "lat": mh_lat, "lon": mh_lon}
                        )
                        st.success(f"Added hotspot {mh_name} ({mh_lat},{mh_lon})")
                        # map update + check overlap
                        st.map(
                            pd.DataFrame(manual_hotspots).rename(
                                columns={"lat": "lat", "lon": "lon"}
                            )[["lat", "lon"]]
                        )
                        # check overlap with GPS
                        over = []
                        for _, g in gps_df.iterrows():
                            for c in manual_hotspots:
                                d = geodesic(
                                    (g["lat"], g["lon"]), (c["lat"], c["lon"])
                                ).meters
                                if d <= 200:
                                    over.append(
                                        {
                                            "timestamp": g["timestamp"],
                                            "hotspot_name": c["name"],
                                            "distance_m": round(d, 1),
                                        }
                                    )
                        if over:
                            st.markdown("### 🚨 Overlaps with manual hotspots (≤200m)")
                            st.dataframe(pd.DataFrame(over))
                        st.session_state["crime_hotspots_manual"] = manual_hotspots

            # ---------------- Polygon drawing (folium + streamlit-folium) ----------------
            st.markdown("## ✏️ Draw crime zones on map (polygon/circle/rectangle)")
            try:
                # center map on GPS trace mean
                center_lat = float(gps_df["lat"].mean())
                center_lon = float(gps_df["lon"].mean())
            except Exception:
                center_lat, center_lon = 20.5937, 78.9629

            m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
            folium.plugins.Draw(export=True).add_to(m)
            folium.TileLayer("openstreetmap").add_to(m)
            output = st_folium(m, width=800, height=450)

            if output and "all_drawings" in output and output["all_drawings"]:
                st.success("Crime zone(s) drawn — processing...")
                zones = []
                matches = []
                for idx, feat in enumerate(output["all_drawings"], start=1):
                    geom = feat.get("geometry", {})
                    dtype = geom.get("type", "")
                    if dtype == "Polygon":
                        coords = geom["coordinates"][
                            0
                        ]  # list of [lon,lat] tuples in GeoJSON
                        # convert to (lat, lon) pairs for plotting logic consistency
                        coords_latlon = [(c[1], c[0]) for c in coords]
                        poly = Polygon(
                            [(c[0], c[1]) for c in coords]
                        )  # shapely expects (x=lon,y=lat)
                        zones.append(
                            {"zone_id": f"Zone {idx}", "coords": coords}
                        )  # keep GeoJSON coords for report
                        # check GPS points inside polygon
                        for _, g in gps_df.iterrows():
                            pt = Point(g["lon"], g["lat"])
                            if poly.contains(pt):
                                matches.append(
                                    {
                                        "timestamp": str(g["timestamp"]),
                                        "zone_id": f"Zone {idx}",
                                        "lat": g["lat"],
                                        "lon": g["lon"],
                                    }
                                )
                # save zones & matches into session_state
                st.session_state["crime_zones"] = {"zones": zones, "matches": matches}
                # create and save snapshot image (matplotlib)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(
                    gps_df["lon"],
                    gps_df["lat"],
                    c="blue",
                    alpha=0.6,
                    label="Suspect GPS",
                )
                if matches:
                    md = pd.DataFrame(matches)
                    ax.scatter(
                        md["lon"], md["lat"], c="red", marker="x", s=80, label="Matches"
                    )
                for z in zones:
                    # coords are GeoJSON lon/lat pairs — convert to lists for plotting
                    coords = z["coords"]
                    xs = [c[0] for c in coords]  # lon
                    ys = [c[1] for c in coords]  # lat
                    ax.plot(xs, ys, "g-", linewidth=2, label=z["zone_id"])
                ax.set_title("Crime Zones & GPS Trace (snapshot)")
                ax.legend()
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(tmpf.name, bbox_inches="tight")
                plt.close(fig)
                st.session_state["crime_zones_map"] = tmpf.name
                st.image(tmpf.name, caption="Crime zone snapshot (saved for report)")
                if matches:
                    st.markdown("### 🚨 Matches detected inside drawn zones")
                    st.dataframe(pd.DataFrame(matches))
                else:
                    st.success("No GPS points fall inside drawn zones.")

    # ---------------- Link Mode (multi links) ----------------
    elif mode == "Link Mode (multi)":
        st.markdown(
            "Paste one or more shared GPS links (one per line). Example: https://maps.google.com/?q=12.9716,77.5946"
        )
        links_area = st.text_area("Links (newline separated)", height=140)
        link_rows = []
        if links_area.strip():
            for line in links_area.splitlines():
                m = re.search(r"([-+]?\d{1,2}\.\d+),\s*([-+]?\d{1,3}\.\d+)", line)
                if m:
                    lat = float(m.group(1))
                    lon = float(m.group(2))
                    link_rows.append(
                        {
                            "timestamp": dt.datetime.now(),
                            "lat": lat,
                            "lon": lon,
                            "source_text": line.strip(),
                        }
                    )
            if link_rows:
                link_df = pd.DataFrame(link_rows)
                st.map(link_df[["lat", "lon"]])
                # save to session state for later report
                st.session_state["link_points"] = link_df.to_dict("records")
                st.session_state["link_points_hash"] = compute_hash(
                    link_df.to_dict("records")
                )

                # check against uploaded hotspots (if present in session_state) and drawn zones
                hits = []
                # check hotspots table
                if "crime_hotspots_table" in st.session_state:
                    for _, p in link_df.iterrows():
                        for c in st.session_state["crime_hotspots_table"]:
                            d = geodesic(
                                (p["lat"], p["lon"]), (c["lat"], c["lon"])
                            ).meters
                            if d <= 200:
                                hits.append(
                                    {
                                        "timestamp": p["timestamp"],
                                        "match_type": "hotspot_file",
                                        "hotspot": c.get("name", "Unknown"),
                                        "distance_m": round(d, 1),
                                    }
                                )
                # check manual hotspots
                if "crime_hotspots_manual" in st.session_state:
                    for _, p in link_df.iterrows():
                        for c in st.session_state["crime_hotspots_manual"]:
                            d = geodesic(
                                (p["lat"], p["lon"]), (c["lat"], c["lon"])
                            ).meters
                            if d <= 200:
                                hits.append(
                                    {
                                        "timestamp": p["timestamp"],
                                        "match_type": "hotspot_manual",
                                        "hotspot": c.get("name", "Manual"),
                                        "distance_m": round(d, 1),
                                    }
                                )
                # check drawn zones
                if "crime_zones" in st.session_state:
                    for _, p in link_df.iterrows():
                        pt = Point(p["lon"], p["lat"])
                        for z in st.session_state["crime_zones"].get("zones", []):
                            poly = Polygon(z["coords"])
                            if poly.contains(pt):
                                hits.append(
                                    {
                                        "timestamp": p["timestamp"],
                                        "match_type": "drawn_zone",
                                        "zone_id": z["zone_id"],
                                        "lat": p["lat"],
                                        "lon": p["lon"],
                                    }
                                )
                if hits:
                    st.markdown("### 🚨 Link points matched hotspots/zones")
                    st.dataframe(pd.DataFrame(hits))
                    st.session_state["link_hits"] = hits
                else:
                    st.success("No matches for provided links.")

    # ---------------- Cell Tower Fallback ----------------
    elif mode == "Cell Tower Fallback":
        cell_file = st.file_uploader(
            "Upload cell tower logs (CSV with cellid,lac,mcc,mnc,timestamp)",
            type=["csv"],
        )
        if cell_file:
            cell_df = pd.read_csv(cell_file)
            # placeholder: in prod, call OpenCellID or other DB to convert to lat/lon
            # for now map to a dummy location or allow user to provide a towers->coords file
            if {"latitude", "longitude"}.issubset(cell_df.columns):
                cell_df = cell_df.rename(
                    columns={"latitude": "lat", "longitude": "lon"}
                )
                st.map(cell_df[["lat", "lon"]])
            else:
                st.info("Cell tower file missing lat/long; showing placeholder.")
                cell_df["lat"], cell_df["lon"] = 12.9716, 77.5946
                st.map(cell_df[["lat", "lon"]])
            st.session_state["cell_tower_points"] = cell_df.to_dict("records")
            st.session_state["cell_tower_hash"] = compute_hash(
                cell_df.to_dict("records")
            )

    # ---------------- Manual Input ----------------
    elif mode == "Manual Input":
        sub = st.radio("Manual input type", ["Add Coordinate", "Enter Timestamp Only"])
        if sub == "Add Coordinate":
            mlat = st.number_input("Latitude", format="%.6f", key="man_lat")
            mlon = st.number_input("Longitude", format="%.6f", key="man_lon")
            mname = st.text_input("Label (optional)", key="man_name")
            if st.button("➕ Add as manual point"):
                pt = {
                    "timestamp": str(dt.datetime.now()),
                    "lat": mlat,
                    "lon": mlon,
                    "label": mname,
                }
                if "manual_points" not in st.session_state:
                    st.session_state["manual_points"] = []
                st.session_state["manual_points"].append(pt)
                st.success(
                    "Manual point added and saved to session_state['manual_points']."
                )
                st.map(pd.DataFrame(st.session_state["manual_points"])[["lat", "lon"]])
        else:
            ts = st.text_input("Enter timestamp (YYYY-MM-DD HH:MM:SS)")
            try:
                parsed = pd.to_datetime(ts)
                st.success(f"Parsed: {parsed}")
                st.session_state["manual_timestamp_entry"] = str(parsed)
            except Exception:
                if ts:
                    st.error("Invalid timestamp format. Use YYYY-MM-DD HH:MM:SS")
# ======================================================
# 6️⃣ UNIFIED REPORT (UPDATED: cover page + snapshots + hashes)
# ======================================================
with tab6:
    import datetime
    from fpdf import FPDF
    from io import BytesIO
    import json
    import hashlib
    import os

    st.subheader("📄 Unified Case Report (Cover + Evidence + Hashes)")

    # ---------- helper: SHA256 ----------
    def compute_hash(data):
        if data is None:
            return "N/A"
        if isinstance(data, (dict, list)):
            b = json.dumps(data, sort_keys=True).encode()
        elif isinstance(data, str):
            b = data.encode()
        elif isinstance(data, bytes):
            b = data
        else:
            try:
                b = str(data).encode()
            except:
                return "N/A"
        return hashlib.sha256(b).hexdigest()

    # ---------- Collect metadata ----------
    case_id = st.session_state.get(
        "case_id", f"CASE-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    device_id = st.session_state.get("device_id", "Unknown Device")
    investigator = st.session_state.get("investigator", "Unknown Investigator")
    acquisition_date = st.session_state.get(
        "acquisition_date", str(datetime.date.today())
    )

    # optional abstract / summary input
    abstract = st.text_area(
        "Write a short case abstract / summary (this goes on cover page)", height=120
    )

    # ---------- Build PDF ----------
    def export_unified_report_pdf():
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # --- Cover Page ---
        pdf.add_page()
        pdf.set_font("Arial", "B", 20)
        pdf.cell(0, 12, "ForenSmart - Unified Case Report", ln=True, align="C")
        pdf.ln(6)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, f"Case ID: {case_id}", ln=True)
        pdf.cell(0, 8, f"Device ID: {device_id}", ln=True)
        pdf.cell(0, 8, f"Investigator: {investigator}", ln=True)
        pdf.cell(0, 8, f"Acquisition Date: {acquisition_date}", ln=True)
        pdf.ln(6)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, "Case Abstract", ln=True)
        pdf.set_font("Arial", size=11)
        if abstract and len(abstract) > 0:
            # wrap abstract text
            pdf.multi_cell(0, 7, abstract)
        else:
            pdf.multi_cell(0, 7, "No abstract provided.")
        pdf.ln(6)

        # ---------- Section: Location Summary ----------
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "📍 Location Intelligence", ln=True)
        pdf.set_font("Arial", size=11)
        loc_summary = st.session_state.get(
            "location_summary", "No location summary available."
        )
        pdf.multi_cell(0, 7, loc_summary)
        pdf.ln(2)
        loc_hash = compute_hash(loc_summary)
        pdf.set_font("Arial", "I", 9)
        pdf.multi_cell(0, 6, f"Hash (SHA256): {loc_hash}")
        pdf.ln(4)

        # ---------- Section: Heatmap Snapshot ----------
        heatmap_path = st.session_state.get("heatmap_snapshot")
        if heatmap_path and os.path.exists(heatmap_path):
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "🔥 Suspicion Heatmap Snapshot", ln=True)
            # ensure image fits
            try:
                pdf.image(heatmap_path, x=15, w=180)
            except Exception:
                pdf.multi_cell(0, 7, f"[Heatmap image available at: {heatmap_path}]")
            pdf.ln(4)
            heat_hash = compute_hash(open(heatmap_path, "rb").read())
            pdf.set_font("Arial", "I", 9)
            pdf.multi_cell(0, 6, f"Hash (SHA256) of heatmap snapshot: {heat_hash}")
            pdf.ln(4)

        # ---------- Section: Crime Hotspots (file) ----------
        if "crime_hotspots_table" in st.session_state:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "📌 Crime Hotspots (Uploaded File)", ln=True)
            pdf.set_font("Arial", size=10)
            try:
                for rec in st.session_state["crime_hotspots_table"]:
                    name = rec.get("name", "Unknown")
                    lat = rec.get("lat") or rec.get("latitude")
                    lon = rec.get("lon") or rec.get("longitude")
                    pdf.multi_cell(0, 6, f"- {name}: ({lat}, {lon})")
            except Exception:
                pdf.multi_cell(0, 6, "Hotspots table present but failed to render.")
            pdf.ln(2)
            pdf.set_font("Arial", "I", 9)
            pdf.multi_cell(
                0,
                6,
                f"Hash (SHA256): {st.session_state.get('crime_hotspots_hash','N/A')}",
            )
            pdf.ln(4)

        # ---------- Section: Manual Hotspots ----------
        if "crime_hotspots_manual" in st.session_state:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "📌 Manual Hotspots (added from image)", ln=True)
            pdf.set_font("Arial", size=10)
            try:
                for rec in st.session_state["crime_hotspots_manual"]:
                    pdf.multi_cell(
                        0,
                        6,
                        f"- {rec.get('name','Manual')}: ({rec.get('lat')}, {rec.get('lon')})",
                    )
            except Exception:
                pdf.multi_cell(0, 6, "Manual hotspots present but failed to render.")
            pdf.ln(2)
            pdf.set_font("Arial", "I", 9)
            pdf.multi_cell(
                0,
                6,
                f"Hash (SHA256): {compute_hash(st.session_state.get('crime_hotspots_manual'))}",
            )
            pdf.ln(4)

        # ---------- Section: Drawn Crime Zones ----------
        if "crime_zones" in st.session_state:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "✏️ Drawn Crime Zones", ln=True)
            pdf.set_font("Arial", size=10)
            zones_obj = st.session_state.get("crime_zones", {})
            zones = zones_obj.get("zones", [])
            matches = zones_obj.get("matches", [])
            if zones:
                for z in zones:
                    # print first few coords (avoid massive dumps)
                    coords = z.get("coords", [])
                    pdf.multi_cell(
                        0,
                        6,
                        f"- {z.get('zone_id','Zone')}: {len(coords)} points, sample: {coords[:3]}",
                    )
            else:
                pdf.multi_cell(0, 6, "No drawn zones found.")
            pdf.ln(2)
            pdf.set_font("Arial", size=10)
            if matches:
                pdf.multi_cell(0, 6, "🚨 Matches detected within zones:")
                for m in matches:
                    pdf.multi_cell(
                        0,
                        6,
                        f"  - {m.get('timestamp')} | {m.get('zone_id')} @ ({m.get('latitude')}, {m.get('longitude')})",
                    )
            else:
                pdf.multi_cell(0, 6, "No GPS matches inside drawn zones.")
            pdf.ln(2)
            pdf.set_font("Arial", "I", 9)
            pdf.multi_cell(
                0, 6, f"Hash (SHA256) of drawn zones object: {compute_hash(zones_obj)}"
            )
            pdf.ln(4)

            # include zone snapshot image if present
            cz_map = st.session_state.get("crime_zones_map")
            if cz_map and os.path.exists(cz_map):
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, "📍 Crime Zones Map Snapshot", ln=True)
                try:
                    pdf.image(cz_map, x=15, w=180)
                except Exception:
                    pdf.multi_cell(0, 7, f"[Crime zones image at: {cz_map}]")
                pdf.ln(4)
                pdf.set_font("Arial", "I", 9)
                pdf.multi_cell(
                    0,
                    6,
                    f"Hash (SHA256) of zone snapshot: {compute_hash(open(cz_map,'rb').read())}",
                )
                pdf.ln(4)

        # ---------- Section: Link Points (Link Mode) ----------
        if "link_points" in st.session_state:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "🔗 Link Mode Points", ln=True)
            pdf.set_font("Arial", size=10)
            try:
                for rec in st.session_state.get("link_points", []):
                    pdf.multi_cell(
                        0,
                        6,
                        f"- {rec.get('timestamp')}: ({rec.get('lat')}, {rec.get('lon')})",
                    )
            except Exception:
                pdf.multi_cell(0, 6, "Link points present but failed to render.")
            pdf.ln(2)
            pdf.set_font("Arial", "I", 9)
            pdf.multi_cell(
                0, 6, f"Hash (SHA256): {st.session_state.get('link_points_hash','N/A')}"
            )
            pdf.ln(4)

        # ---------- Section: Cell Tower Points ----------
        if "cell_tower_points" in st.session_state:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "📡 Cell Tower Points", ln=True)
            pdf.set_font("Arial", size=10)
            try:
                for rec in st.session_state.get("cell_tower_points", []):
                    pdf.multi_cell(
                        0,
                        6,
                        f"- {rec.get('timestamp','')}: ({rec.get('lat')},{rec.get('lon')})",
                    )
            except Exception:
                pdf.multi_cell(0, 6, "Cell tower data present but failed to render.")
            pdf.ln(2)
            pdf.set_font("Arial", "I", 9)
            pdf.multi_cell(
                0, 6, f"Hash (SHA256): {st.session_state.get('cell_tower_hash','N/A')}"
            )
            pdf.ln(4)

        # ---------- Section: Suspicious Messages ----------
        if "suspicious_messages" in st.session_state:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "💬 Suspicious Messages", ln=True)
            pdf.set_font("Arial", size=10)
            try:
                for m in st.session_state.get("suspicious_messages", []):
                    # some messages may have different fields
                    ts = m.get("timestamp", "")
                    sender = m.get("sender", "")
                    text = m.get("text", "")
                    score = m.get(
                        "suspicion_score", m.get("score", m.get("offense_score", 0.0))
                    )
                    pdf.multi_cell(
                        0, 6, f"- [{ts}] {sender}: {text} (score: {score:.2f})"
                    )
                    # include explainability tokens if present
                    if "top_tokens" in m:
                        try:
                            tokens = m.get("top_tokens", [])
                            # convert tokens to string if tuple list
                            tokens_str = ", ".join(
                                [
                                    t[0] if isinstance(t, (list, tuple)) else str(t)
                                    for t in tokens
                                ][:8]
                            )
                            pdf.multi_cell(0, 6, f"    Top tokens: {tokens_str}")
                        except Exception:
                            pass
            except Exception:
                pdf.multi_cell(
                    0, 6, "Suspicious messages present but failed to render."
                )
            pdf.ln(2)
            pdf.set_font("Arial", "I", 9)
            pdf.multi_cell(
                0,
                6,
                f"Hash (SHA256): {compute_hash(st.session_state.get('suspicious_messages'))}",
            )
            pdf.ln(4)

            # ---------- Section: Social Media Evidence ----------
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "📱 Social Media Evidence (Auto-parsed)", ln=True)
        pdf.set_font("Arial", size=11)
        parsed_dir = os.path.join(
            "uploads", st.session_state.get("case_id", "unknown_case"), "parsed"
        )
        if os.path.exists(parsed_dir):
            try:
                files = sorted(
                    [
                        f
                        for f in os.listdir(parsed_dir)
                        if os.path.isfile(os.path.join(parsed_dir, f))
                    ]
                )
                if not files:
                    pdf.multi_cell(0, 6, "No parsed files found in parsed directory.")
                else:
                    for fn in files:
                        fp = os.path.join(parsed_dir, fn)
                        pdf.set_font("Arial", "B", 11)
                        pdf.multi_cell(0, 6, "- Parsed file: " + fn)
                        try:
                            with open(fp, "r", encoding="utf-8") as fr:
                                data = json.load(fr)
                            if isinstance(data, dict) and "messages" in data:
                                pdf.set_font("Arial", size=10)
                                pdf.multi_cell(
                                    0,
                                    6,
                                    "  Messages parsed: "
                                    + str(len(data.get("messages", []))),
                                )

                                for sm in data.get("messages", [])[:3]:
                                    txt = (
                                        sm.get("text")
                                        if isinstance(sm, dict)
                                        else str(sm)
                                    )
                                    pdf.multi_cell(
                                        0, 6, "    • " + (txt[:200] if txt else "")
                                    )
                            else:
                                preview = json.dumps(data)[:500]
                                pdf.set_font("Arial", size=10)
                                pdf.multi_cell(0, 6, "  Preview: " + preview)
                        except Exception:
                            pdf.multi_cell(
                                0, 6, "  [Failed to load parsed file preview]"
                            )
                        try:
                            with open(fp, "rb") as fh:
                                file_hash = compute_hash(fh.read())
                            pdf.set_font("Arial", "I", 9)
                            pdf.multi_cell(0, 6, "  SHA256: " + file_hash)
                        except Exception:
                            pass
                        pdf.ln(2)
            except Exception:
                pdf.multi_cell(
                    0, 6, "Parsed directory exists but failed to enumerate files."
                )
        else:
            pdf.multi_cell(0, 6, "No parsed social media artifacts found.")
        pdf.ln(4)

        # ---------- Section: Password Evaluation ----------
        if "password_eval" in st.session_state:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "🔑 Password Evaluation", ln=True)
            pdf.set_font("Arial", size=10)
            try:
                pdf.multi_cell(0, 6, str(st.session_state.get("password_eval")))
            except Exception:
                pdf.multi_cell(0, 6, "Password eval present but failed to render.")
            pdf.ln(2)
            pdf.set_font("Arial", "I", 9)
            pdf.multi_cell(
                0,
                6,
                f"Hash (SHA256): {compute_hash(st.session_state.get('password_eval'))}",
            )
            pdf.ln(6)

        # ---------- Footer: master case hash + generation metadata ----------
        combined = {
            "location_summary": st.session_state.get("location_summary"),
            "crime_hotspots": st.session_state.get("crime_hotspots_table"),
            "manual_hotspots": st.session_state.get("crime_hotspots_manual"),
            "crime_zones": st.session_state.get("crime_zones"),
            "link_points": st.session_state.get("link_points"),
            "cell_tower": st.session_state.get("cell_tower_points"),
            "suspicious_messages": st.session_state.get("suspicious_messages"),
            "password_eval": st.session_state.get("password_eval"),
        }
        master_hash = compute_hash(combined)
        pdf.set_font("Arial", "B", 11)
        pdf.multi_cell(0, 8, f"🔒 Master Case Hash: {master_hash}")
        pdf.set_font("Arial", "I", 9)
        pdf.multi_cell(
            0,
            6,
            f"Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by ForenSmart",
        )

        # return bytes
        out = BytesIO()
        pdf.output(out, "F")
        out.seek(0)
        return out

    # ---------- Generate & Download ----------
    if st.button("📄 Generate Unified PDF Report"):
        pdf_bytes = export_unified_report_pdf()
        # --- Stage B: finalize report integrity ---
        consent_id = st.session_state.get("consent_id")
        investigator = st.session_state.get("investigator")
        if consent_id and "finalize_report_integrity" in globals():
            try:
                updated_pdf, master_hash = finalize_report_integrity(
                    consent_id, out_bytesio.getvalue(), investigator
                )
                # replace out_bytesio content if applicable (store hash in session)
                st.session_state["master_report_hash"] = master_hash
                out_bytesio = (
                    io.BytesIO(updated_pdf)
                    if isinstance(updated_pdf, (bytes, bytearray))
                    else out_bytesio
                )
            except Exception as _e:
                st.warning(f"Report integrity step failed: {_e}")
        filename = f"{case_id}_ForenSmart_Report.pdf"
        st.download_button(
            "⬇️ Download Report PDF",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
        )
        st.success(
            "Report generated — hashes included for each section and master case hash created."
        )
# ---------------------------
# Comms Analyzer (New Feature Model)
# ---------------------------
with st.sidebar:
    st.header("Comms Analyzer")
    mode = st.radio("Select Extraction Mode", ["USB Mode", "Link Mode"])

# ---------------------------
# Tab Structure
# ---------------------------
tabs = st.tabs(
    [
        "Data Input",
        "Preprocessing",
        "Message Intelligence",
        "Behavioral Network",
        "Psych & Sentiment",
        "Multimedia",
        "OSINT Integration",
        "Evidence & Report",
    ]
)

# --- Consent banner shown across tabs ---
try:
    current_consent = st.session_state.get("consent_id")
    if current_consent:
        st.info(f"Current consent_id: {current_consent}")
    else:
        st.info(
            "No consent_id linked in session. Capture consent for intrusive extractions."
        )
except Exception:
    pass


# ---------------------------
# Tab 1: Data Input
# ---------------------------
with tabs[0]:
    st.subheader("Data Input & Extraction")

    if mode == "USB Mode":
        st.info(
            "📱 Please enable **USB Debugging** on your device:\n"
            "1. Go to Settings → About Phone → tap *Build Number* 7 times to enable Developer Options.\n"
            "2. Go to Developer Options → Enable **USB Debugging** and **ADB Debugging**.\n"
            "3. Connect your phone via USB."
        )
        st.button("Check Connected Device")
        # Placeholder: ADB-based extraction logic goes here

    elif mode == "Link Mode":
        st.info(
            "🌐 To use Link Mode:\n"
            "1. Export your WhatsApp/Telegram/Signal chats to cloud (Google Drive/iCloud).\n"
            "2. Copy the link or connect your account via API.\n"
            "3. Paste the link below for analysis."
        )
        link = st.text_input("Paste Chat Export Link or API Token")
        st.button("Fetch & Extract Data")
        # Placeholder: Cloud/API extraction logic goes here

# ---------------------------
# Tabs 2 → 7: Feature Placeholders
# ---------------------------
with tabs[1]:
    st.subheader("Preprocessing & Multilingual Pipeline")
    st.write("🔄 Language detection, translation, code-switch detection…")

with tabs[2]:
    st.subheader("Message Intelligence")
    st.write("🔎 Keyword search, topic modeling, coded language detection…")

with tabs[3]:
    st.subheader("Behavioral & Network Analysis")
    st.write("📈 Conversation graphs, anomalies, temporal patterns…")

with tabs[4]:
    st.subheader("Psychological & Sentiment Profiling")
    st.write("🧠 Sentiment, emotion, stylometry, deception markers…")

with tabs[5]:
    st.subheader("Multimedia & Metadata Analysis")
    st.write("🖼️ OCR, CLIP embeddings, audio diarization, EXIF extraction…")

with tabs[6]:
    st.subheader("OSINT & Cross-Linking")
    st.write("🌍 External DB lookups, leaked dataset checks…")

# ---------------------------
# Tab 8: Evidence & Report
# ---------------------------
with tabs[7]:
    st.subheader("Evidence & Reporting")

    if st.button("Run Full Analysis Pipeline"):
        st.success(
            "✅ Running all modules (Preprocessing → Intelligence → Profiling → OSINT → Report)..."
        )
        # Here, trigger all tabs' processing at once
        st.info("📊 Dashboard generated below")
        # Placeholder for dashboard summary of results
        st.warning("📄 Do you want to generate a Unified Report with all results?")

        if st.button("Generate Unified Report"):
            st.success("Report generated! Ready for download.")
# ---------------------------
# Comms Analyzer ML Pipeline (paste into app.py under Comms Analyzer tab)
# ---------------------------
import hashlib
import os
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# NLP / ML imports with graceful failover
try:
    from langdetect import detect
except Exception:
    detect = None

try:
    from transformers import pipeline, AutoTokenizer

    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTE = True
except Exception:
    HAS_SENTE = False

try:
    from bertopic import BERTopic

    HAS_BERTOPIC = True
except Exception:
    HAS_BERTOPIC = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import nltk

# Ensure nltk data exists (download if not present)
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except Exception:
    nltk.download("stopwords")

# VADER fallback for sentiment (English)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    vader_analyzer = SentimentIntensityAnalyzer()
except Exception:
    vader_analyzer = None


# Utilities
def sha256_hash(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def detect_language_safe(text):
    if not text or not isinstance(text, str):
        return "unknown"
    if detect:
        try:
            return detect(text)
        except Exception:
            return "unknown"
    else:
        # Very naive fallback: check for common non-ascii chars
        if any(ord(c) > 127 for c in text):
            return "non-en"
        return "en"


# Optional translation using Marian (transformers) if available
TRANSLATOR = None
if HAS_TRANSFORMERS:
    try:
        # Use M2M100 or Marian if available — lazy init only when needed
        pass
    except Exception:
        TRANSLATOR = None


def translate_text(text, src=None, tgt="en"):
    """Translate text into target language if transformers-based translator available.
    If not available, returns original text."""
    if not text:
        return text
    if HAS_TRANSFORMERS:
        # Lazy-init a small MarianMT or m2m if model not loaded
        global TRANSLATOR
        if TRANSLATOR is None:
            try:
                # using Helsinki-NLP Marian multilingual as a reasonable fallback
                model_name = "Helsinki-NLP/opus-mt-mul-en"
                TRANSLATOR = pipeline("translation", model=model_name, device=-1)
            except Exception:
                TRANSLATOR = None
        if TRANSLATOR:
            try:
                out = TRANSLATOR(text, max_length=512)
                if (
                    isinstance(out, list)
                    and len(out) > 0
                    and "translation_text" in out[0]
                ):
                    return out[0]["translation_text"]
            except Exception:
                return text
    # fallback: no translation available
    return text


# Embeddings loader (sentence-transformers)
EMBED_MODEL = None


def get_embeddings(texts):
    global EMBED_MODEL
    if not texts:
        return None
    if HAS_SENTE:
        if EMBED_MODEL is None:
            try:
                EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # compact & fast
            except Exception:
                EMBED_MODEL = None
        if EMBED_MODEL:
            return EMBED_MODEL.encode(texts, show_progress_bar=False)
    # fallback: simple Bag-of-Words vectors via TF-IDF (dense)
    tfidf = TfidfVectorizer(max_features=1024, stop_words="english")
    X = tfidf.fit_transform(texts).toarray()
    return X


# Keyword extraction (TF-IDF top-N)
def extract_top_keywords(texts, top_n=20):
    if not texts:
        return {}
    vec = TfidfVectorizer(
        max_df=0.8, min_df=1, ngram_range=(1, 2), stop_words="english"
    )
    X = vec.fit_transform(texts)
    sums = X.sum(axis=0)
    terms = vec.get_feature_names_out()
    term_freq = [(term, sums[0, idx]) for idx, term in enumerate(terms)]
    term_freq = sorted(term_freq, key=lambda x: x[1], reverse=True)[:top_n]
    return dict(term_freq)


# Topic modeling (BERTopic if available, else NMF)
def topic_modeling(texts, n_topics=8):
    if not texts:
        return {"topics": [], "topic_info": []}
    if HAS_BERTOPIC:
        try:
            topic_model = BERTopic(
                n_gram_range=(1, 3), calculate_probabilities=False, verbose=False
            )
            topics, _ = topic_model.fit_transform(texts)
            info = topic_model.get_topic_info()
            # get top keywords per topic
            topics_dict = {}
            for t in set(topics):
                if t == -1:
                    continue
                kw = topic_model.get_topic(t)
                topics_dict[t] = [k for k, _ in kw]
            return {
                "topics": topics,
                "topic_info": info.to_dict(),
                "topics_dict": topics_dict,
            }
        except Exception:
            pass
    # fallback NMF on TF-IDF
    try:
        tfidf = TfidfVectorizer(
            max_df=0.95, min_df=2, max_features=2000, stop_words="english"
        )
        X = tfidf.fit_transform(texts)
        nmf = NMF(n_components=min(n_topics, 10), random_state=0)
        W = nmf.fit_transform(X)
        H = nmf.components_
        feature_names = tfidf.get_feature_names_out()
        topics = []
        topics_dict = {}
        for topic_idx, topic in enumerate(H):
            top_features_ind = topic.argsort()[: -10 - 1 : -1]
            top_features = [feature_names[i] for i in top_features_ind]
            topics_dict[topic_idx] = top_features
        # Assign each doc the top topic
        doc_topics = W.argmax(axis=1).tolist()
        return {"topics": doc_topics, "topic_info": {}, "topics_dict": topics_dict}
    except Exception:
        return {"topics": [], "topic_info": [], "topics_dict": {}}


# Sentiment / emotion (transformers classifier if possible, else VADER for English)
SENT_PIPE = None
if HAS_TRANSFORMERS:
    try:
        SENT_PIPE = pipeline("sentiment-analysis", return_all_scores=True)
    except Exception:
        SENT_PIPE = None


def sentiment_analysis(texts):
    if not texts:
        return {}
    agg = Counter()
    per_doc = []
    for t in texts:
        # simple short-circuit
        if SENT_PIPE:
            try:
                res = SENT_PIPE(t[:512])  # limit to first 512 chars
                # res is list of dicts sometimes; pick highest label
                if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                    # huggingface multi-return format
                    best = sorted(res[0], key=lambda x: x["score"], reverse=True)[0]
                    label = best["label"]
                elif (
                    isinstance(res, list)
                    and isinstance(res[0], dict)
                    and "label" in res[0]
                ):
                    label = res[0]["label"]
                else:
                    label = "neutral"
                agg[label] += 1
                per_doc.append({"text": t, "label": label})
                continue
            except Exception:
                pass
        # fallback VADER for English
        if vader_analyzer:
            try:
                vs = vader_analyzer.polarity_scores(t)
                if vs["compound"] >= 0.05:
                    lab = "POSITIVE"
                elif vs["compound"] <= -0.05:
                    lab = "NEGATIVE"
                else:
                    lab = "NEUTRAL"
                agg[lab] += 1
                per_doc.append({"text": t, "label": lab})
                continue
            except Exception:
                pass
        # ultimate fallback
        agg["UNKNOWN"] += 1
        per_doc.append({"text": t, "label": "UNKNOWN"})
    # normalize into percentages
    total = sum(agg.values())
    dist = {k: (v / total) * 100 for k, v in agg.items()} if total > 0 else {}
    return {"distribution": dist, "per_doc": per_doc}


# Conversation graph builder
def build_conversation_graph(messages):
    """
    messages: list of dicts with at least {'sender':..., 'receiver':..., 'text':...}
    returns: path to saved image and in-memory graph object
    """
    if not messages:
        return None, None
    G = nx.DiGraph()
    for m in messages:
        s = m.get("sender", "unknown")
        r = m.get("receiver", "group") or "group"
        # increase edge weight
        if G.has_edge(s, r):
            G[s][r]["weight"] += 1
        else:
            G.add_edge(s, r, weight=1)
    # simplify labels by degree and create a simple matplotlib visualization
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_nodes(
        G, pos, node_size=[200 + 2000 * G.degree(n) for n in G.nodes()]
    )
    nx.draw_networkx_edges(
        G, pos, width=[0.5 + (w * 0.3) for w in weights], arrowsize=12
    )
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.axis("off")
    out_path = os.path.join("tmp", "conversation_graph.png")
    os.makedirs("tmp", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path, G


# Stylometry: simple pairwise similarity by tf-idf cosine between authors
def stylometry_similarity(messages):
    """
    messages: list of dicts with {'sender':..., 'text':...}
    returns: matrix/dict of similarity between senders
    """
    by_sender = defaultdict(list)
    for m in messages:
        s = m.get("sender", "unknown")
        t = m.get("text", "")
        by_sender[s].append(t)
    senders = list(by_sender.keys())
    docs = [" ".join(by_sender[s]) for s in senders]
    if not docs:
        return {}
    tfidf = TfidfVectorizer(max_features=2048, stop_words="english")
    X = tfidf.fit_transform(docs)
    sim = cosine_similarity(X)
    sim_dict = {}
    for i, s in enumerate(senders):
        sim_dict[s] = {}
        for j, t in enumerate(senders):
            sim_dict[s][t] = float(sim[i, j])
    return {"senders": senders, "similarity": sim_dict}


# ---------------------------
# Integration UI (inside the Comms Analyzer tab)
# ---------------------------
with st.expander("Comms Analyzer — Run / Utilities", expanded=True):
    st.write(
        "Use these controls to run the Comms Analyzer ML pipeline on the currently loaded chat data."
    )
    # The UI expects chat data to be present in st.session_state["chats"]
    # Format: st.session_state["chats"] = list of messages where message is dict: {'sender','receiver','text','time'}
    st.write("Loaded messages:", len(st.session_state.get("chats", [])))
    if st.button("Run Comms Analyzer Pipeline"):
        msgs = st.session_state.get("chats", [])
        texts = [m.get("text", "") for m in msgs if m.get("text", "").strip()]

        # 1) Language detection (per message) + per-chat dominant language
        lang_counts = Counter()
        detected_langs = []
        for t in texts:
            lang = detect_language_safe(t)
            detected_langs.append(lang)
            lang_counts[lang] += 1
        dominant_lang = lang_counts.most_common(1)[0][0] if lang_counts else "unknown"
        st.session_state["comms_lang_counts"] = dict(lang_counts)
        st.session_state["comms_dominant_lang"] = dominant_lang

        # 2) Optional: translate non-English messages into English for uniform analysis
        translated_texts = []
        for i, t in enumerate(texts):
            if detected_langs[i] != "en" and detected_langs[i] != "unknown":
                # attempt translate
                try:
                    translated = translate_text(t, src=detected_langs[i], tgt="en")
                    translated_texts.append(translated)
                except Exception:
                    translated_texts.append(t)
            else:
                translated_texts.append(t)
        st.session_state["comms_translated_texts"] = translated_texts

        # 3) Keyword extraction (top keywords across chats)
        top_keywords = extract_top_keywords(translated_texts, top_n=25)
        st.session_state["top_keywords"] = top_keywords

        # 4) Topic modeling
        topics_res = topic_modeling(translated_texts, n_topics=8)
        st.session_state["comms_topics"] = topics_res

        # 5) Embeddings (for clustering / similarity)
        embeddings = get_embeddings(translated_texts)
        st.session_state["comms_embeddings"] = embeddings  # may be ndarray or None

        # 6) Sentiment / Emotion
        sent_res = sentiment_analysis(translated_texts)
        st.session_state["sentiment_summary"] = sent_res.get("distribution", {})
        st.session_state["sentiment_per_message"] = sent_res.get("per_doc", [])

        # 7) Conversation graph
        graph_img, G = build_conversation_graph(msgs)
        if graph_img:
            st.session_state["conversation_graph"] = graph_img

        # 8) Stylometry / authorship similarity
        stylometry = stylometry_similarity(msgs)
        st.session_state["stylometry"] = stylometry

        # 9) Suspicious message heuristic example (regex/keyword + sentiment + short messages)
        suspicious = []
        suspect_keywords = set(
            [
                "bomb",
                "attack",
                "explode",
                "transfer",
                "kill",
                "meet",
                "drop",
                "pickup",
                "send",
                "wire",
            ]
        )
        for m in msgs:
            t = m.get("text", "").lower()
            score = 0
            if any(k in t for k in suspect_keywords):
                score += 1
            if len(t.split()) < 6 and len(t) > 0:
                score += 0.5
            # negative sentiment add weight
            label = None
            if "sentiment_per_message" in st.session_state:
                # try to find corresponding sentiment entry
                pass
            if score >= 1:
                suspicious.append(
                    {
                        "sender": m.get("sender"),
                        "time": m.get("time"),
                        "text": m.get("text"),
                        "score": score,
                    }
                )
        st.session_state["suspicious_messages"] = suspicious

        # 10) Evidence hashing for chat text blocks (store per-message hash and master)
        evidence_hashes = []
        for m in msgs:
            h = sha256_hash(m.get("text", ""))
            evidence_hashes.append(
                {"sender": m.get("sender"), "time": m.get("time"), "hash": h}
            )
        st.session_state["comms_evidence_hashes"] = evidence_hashes
        st.session_state["comms_master_hash"] = sha256_hash(
            "".join([e["hash"] for e in evidence_hashes])
        )

        st.success(
            "Comms Analyzer pipeline finished. Results saved to st.session_state keys: "
            "top_keywords, comms_topics, comms_embeddings, sentiment_summary, conversation_graph, stylometry, suspicious_messages, comms_evidence_hashes, comms_master_hash"
        )

# Quick buttons: run only keywords / sentiment / graph if needed
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Extract Top Keywords Only"):
        texts = st.session_state.get("chats", [])
        texts = [m.get("text", "") for m in texts]
        st.session_state["top_keywords"] = extract_top_keywords(texts)
        st.success("Top keywords extracted.")
with col2:
    if st.button("Build Conversation Graph Only"):
        msgs = st.session_state.get("chats", [])
        graph_img, G = build_conversation_graph(msgs)
        if graph_img:
            st.session_state["conversation_graph"] = graph_img
            st.success("Conversation graph generated.")
        else:
            st.warning("No messages to build graph.")
with col3:
    if st.button("Run Sentiment Only"):
        texts = [m.get("text", "") for m in st.session_state.get("chats", [])]
        sent_res = sentiment_analysis(texts)
        st.session_state["sentiment_summary"] = sent_res.get("distribution", {})
        st.session_state["sentiment_per_message"] = sent_res.get("per_doc", [])
        st.success("Sentiment analysis complete.")
# ---------------------------
# Comms Analyzer: Trainable Suspicious Classifier + Codeword Detector + Dashboard Wiring
# Paste this after your existing Comms Analyzer pipeline in app.py
# ---------------------------
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import json


# --- Helper: persist/load a simple sklearn pipeline into session_state ---
def save_suspicious_model(model):
    st.session_state["suspicious_model"] = model


def load_suspicious_model():
    return st.session_state.get("suspicious_model", None)


# --- Labeling UI for supervised suspicious-message classifier ---
with st.expander(
    "Label Messages for Suspicious Classifier (trainable)", expanded=False
):
    st.write(
        "Label some messages as Suspicious / Not Suspicious to build the classifier. Minimum ~20 examples recommended (balanced)."
    )
    msgs = st.session_state.get("chats", [])
    # Prepare a DataFrame display for a random sample of messages to label
    sample_size = min(40, max(10, int(len(msgs) * 0.3)))
    if "susp_label_index" not in st.session_state:
        st.session_state["susp_label_index"] = 0
        st.session_state["susp_labels"] = []  # list of dicts {text,label,sender,time}
    if len(msgs) == 0:
        st.info("No messages loaded in st.session_state['chats'] to label.")
    else:
        # show a paginated sample to label
        start = st.session_state["susp_label_index"]
        end = min(start + 5, len(msgs))
        cols = st.columns(1)
        for i in range(start, end):
            m = msgs[i]
            text = m.get("text", "")
            st.markdown(
                f"**Msg #{i}** — *{m.get('sender','unknown')}* @ {m.get('time','')}"
            )
            choice = st.radio(
                f"Label message {i}",
                ["Unlabeled", "Not Suspicious", "Suspicious"],
                key=f"sublabel_{i}",
            )
            if choice != "Unlabeled":
                # store label into session_state list if not already present or if changed
                existing = [
                    x for x in st.session_state["susp_labels"] if x["index"] == i
                ]
                if existing:
                    existing[0].update(
                        {
                            "label": 1 if choice == "Suspicious" else 0,
                            "text": text,
                            "sender": m.get("sender"),
                            "time": m.get("time"),
                        }
                    )
                else:
                    st.session_state["susp_labels"].append(
                        {
                            "index": i,
                            "text": text,
                            "label": 1 if choice == "Suspicious" else 0,
                            "sender": m.get("sender"),
                            "time": m.get("time"),
                        }
                    )
        nav1, nav2, nav3 = st.columns([1, 1, 2])
        with nav1:
            if st.button("Prev", key="susp_prev"):
                st.session_state["susp_label_index"] = max(
                    0, st.session_state["susp_label_index"] - 5
                )
        with nav2:
            if st.button("Next", key="susp_next"):
                st.session_state["susp_label_index"] = min(
                    len(msgs) - 1, st.session_state["susp_label_index"] + 5
                )
        with nav3:
            if st.button("Clear Labels", key="susp_clear"):
                st.session_state["susp_labels"] = []
                st.success("Cleared temporary labels (will not affect saved model).")

    st.markdown("---")
    # Show current labeled counts
    lab_df = pd.DataFrame(st.session_state.get("susp_labels", []))
    if not lab_df.empty:
        st.write("Labels collected:", lab_df.shape[0])
        st.dataframe(lab_df[["index", "sender", "time", "label", "text"]])

    # Train button
    if st.button("Train/Update Suspicious Classifier"):
        lab_df = pd.DataFrame(st.session_state.get("susp_labels", []))
        if lab_df.shape[0] < 10:
            st.warning("Please label at least 10 messages (balanced) before training.")
        else:
            X = lab_df["text"].astype(str).tolist()
            y = lab_df["label"].astype(int).tolist()
            pipeline = Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=5000, ngram_range=(1, 2), stop_words="english"
                        ),
                    ),
                    ("clf", LogisticRegression(max_iter=1000)),
                ]
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            report = classification_report(y_test, preds, output_dict=True)
            st.write("Training complete — validation report:")
            st.text(classification_report(y_test, preds))
            save_suspicious_model(pipeline)
            st.success(
                "Suspicious classifier saved in session_state['suspicious_model']."
            )

# --- Codeword Detector: editable dictionary + embedding similarity fallback ---
with st.expander(
    "Codeword Detector (dictionary + embedding-similarity)", expanded=False
):
    # default dictionary (you can expand)
    default_codewords = st.session_state.get(
        "codeword_dict",
        {
            "drop": ["drop", "dp"],
            "pickup": ["pickup", "pk"],
            "wire": ["wire", "transfer", "send cash"],
            "meet": ["meet", "mtg", "pickup"],
            "bomb": ["bomb", "explode", "blow up"],
        },
    )
    st.write("Edit codeword dictionary (key: intent, value: list of terms):")
    cw_json = st.text_area(
        "codeword_dict (JSON)",
        value=json.dumps(default_codewords, indent=2),
        height=140,
    )
    try:
        parsed = json.loads(cw_json)
        st.session_state["codeword_dict"] = parsed
    except Exception:
        st.warning("Invalid JSON — keep previous dictionary.")
    st.write(
        "Run detection to scan messages for dictionary matches and embedding-similarity hits (if embeddings exist)."
    )
    if st.button("Run Codeword Detection"):
        codew = st.session_state.get("codeword_dict", {})
        msgs = st.session_state.get("chats", [])
        texts = [m.get("text", "") for m in msgs]
        hits = []
        # build flat set of code terms
        term_map = {}
        for intent, terms in codew.items():
            for t in terms:
                term_map[t.lower()] = intent
        # exact substring / token match pass
        for i, m in enumerate(msgs):
            t = m.get("text", "").lower()
            for term, intent in term_map.items():
                if term in t:
                    hits.append(
                        {
                            "index": i,
                            "sender": m.get("sender"),
                            "time": m.get("time"),
                            "term": term,
                            "intent": intent,
                            "method": "dict_match",
                            "text": m.get("text"),
                        }
                    )
        # embedding-sim fallback
        emb_matrix = st.session_state.get("comms_embeddings", None)
        if emb_matrix is not None and len(texts) > 0:
            # compute embeddings for code terms (use get_embeddings)
            term_list = list(term_map.keys())
            term_embs = get_embeddings(term_list)
            if term_embs is not None:
                # ensure emb_matrix rows align with texts order
                # get similarity between each message and each term
                try:
                    from sklearn.metrics.pairwise import cosine_similarity

                    if isinstance(emb_matrix, np.ndarray):
                        sim = cosine_similarity(emb_matrix, term_embs)
                        threshold = 0.70
                        for i in range(sim.shape[0]):
                            for j in range(sim.shape[1]):
                                if sim[i, j] >= threshold:
                                    hits.append(
                                        {
                                            "index": i,
                                            "sender": msgs[i].get("sender"),
                                            "time": msgs[i].get("time"),
                                            "term": term_list[j],
                                            "intent": term_map[term_list[j]],
                                            "method": "embedding_sim",
                                            "score": float(sim[i, j]),
                                            "text": msgs[i].get("text"),
                                        }
                                    )
                except Exception:
                    pass
        st.session_state["codeword_hits"] = hits
        st.success(
            f"Codeword detection finished — {len(hits)} hits stored in st.session_state['codeword_hits']."
        )

# --- Use Suspicious Classifier to predict on all messages ---
with st.expander("Apply Suspicious Classifier / Heuristics", expanded=True):
    st.write(
        "Use the trained classifier (if any) + heuristic rules to generate a scored suspicious_messages list."
    )
    if st.button("Run Suspicion Scoring (Classifier + Heuristics)"):
        msgs = st.session_state.get("chats", [])
        texts = [m.get("text", "") for m in msgs]
        model = load_suspicious_model()
        preds = []
        probs = []
        if model is not None:
            try:
                probs_all = model.predict_proba(texts)
                preds = model.predict(texts)
            except Exception:
                # fallback: no prob support
                preds = model.predict(texts)
                probs_all = None
        else:
            probs_all = None

        # start with classifier output, then add heuristics & codeword hits
        suspicious_out = []
        cw_hits = {h["index"]: h for h in st.session_state.get("codeword_hits", [])}
        suspect_keywords = set(
            [
                "bomb",
                "attack",
                "explode",
                "transfer",
                "kill",
                "meet",
                "drop",
                "pickup",
                "send",
                "wire",
            ]
        )
        for i, m in enumerate(msgs):
            t = m.get("text", "", "")
            score = 0.0
            reason = []
            # classifier
            if model is not None:
                try:
                    label = int(model.predict([t])[0])
                    score += 1.0 * label
                    if probs_all is not None:
                        # try to get probability for label 1
                        try:
                            p = float(probs_all[i][1])
                            score += p  # add probability as score
                            reason.append(f"classifier_prob={p:.2f}")
                        except Exception:
                            pass
                except Exception:
                    pass
            # heuristic keyword hits
            if any(k in t.lower() for k in suspect_keywords):
                score += 0.8
                reason.append("keyword")
            # short urgent messages
            if len(t.split()) <= 5 and len(t) > 0:
                score += 0.3
                reason.append("short_msg")
            # codeword hit
            if i in cw_hits:
                score += 1.2
                reason.append(f"codeword:{cw_hits[i]['term']}")
            if score >= 1.0:
                suspicious_out.append(
                    {
                        "index": i,
                        "sender": m.get("sender"),
                        "time": m.get("time"),
                        "text": t,
                        "score": float(score),
                        "reasons": reason,
                    }
                )
        # sort by score desc
        suspicious_out = sorted(suspicious_out, key=lambda x: x["score"], reverse=True)
        st.session_state["suspicious_messages"] = suspicious_out
        st.success(
            f"Suspicion scoring complete — {len(suspicious_out)} messages flagged (stored in st.session_state['suspicious_messages'])."
        )


# --- Dashboard wiring: richer plots for Comms Analyzer (hooks into Unified Dashboard)
# These write into session_state keys which your Unified Dashboard reads:
def render_comms_charts():
    # Top keywords bar chart
    top_keywords = st.session_state.get("top_keywords", {})
    if top_keywords:
        kw_items = list(top_keywords.items())[:20]
        labels = [k for k, _ in kw_items]
        vals = [v for _, v in kw_items]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(range(len(labels))[::-1], vals[::-1])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels[::-1])
        ax.set_title("Top Keywords")
        plt.tight_layout()
        st.pyplot(fig)

    # Sentiment pie (already stored in sentiment_summary)
    sentiment = st.session_state.get("sentiment_summary", {})
    if sentiment:
        labels = list(sentiment.keys())
        vals = list(sentiment.values())
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.pie(vals, labels=labels, autopct="%1.1f%%")
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

    # Suspicious messages: per-sender counts + table
    suspicious = st.session_state.get("suspicious_messages", [])
    if suspicious:
        df = pd.DataFrame(suspicious)
        st.markdown("**Suspicious Messages (top 50)**")
        st.dataframe(df.head(50))
        # counts by sender
        counts = df["sender"].value_counts().to_dict()
        if counts:
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.bar(counts.keys(), counts.values())
            ax.set_title("Suspicious Messages by Sender")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

    # Stylometry similarity heatmap
    styl = st.session_state.get("stylometry", {})
    if styl and "similarity" in styl:
        sim = styl["similarity"]
        senders = styl["senders"]
        mat = np.array([[sim[a][b] for b in senders] for a in senders])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(mat, vmin=0, vmax=1)
        ax.set_xticks(range(len(senders)))
        ax.set_xticklabels(senders, rotation=45, ha="right")
        ax.set_yticks(range(len(senders)))
        ax.set_yticklabels(senders)
        ax.set_title("Stylometry Similarity (cosine)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        st.pyplot(fig)


# expose a button to render charts inline (or dashboard will call this)
if st.button("Render Comms Charts Now"):
    render_comms_charts()
    st.success(
        "Comms charts rendered (top_keywords, sentiment, suspicious messages, stylometry)."
    )

# ensure Unified Dashboard picks these up (store a small summary)
st.session_state["comms_summary"] = {
    "num_messages": len(st.session_state.get("chats", [])),
    "num_suspicious": len(st.session_state.get("suspicious_messages", [])),
    "num_codeword_hits": len(st.session_state.get("codeword_hits", [])),
    "dominant_lang": st.session_state.get("comms_dominant_lang", "unknown"),
}
# ---------------------------
# Upgraded Suspicious Classifier (TF-IDF + Transformer fine-tuning)
# Paste into Comms Analyzer section of app.py
# ---------------------------
import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# Try to import Transformers + datasets; set flags if available
HAS_TRANSFORMERS = False
HAS_DATASETS = False
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

try:
    from datasets import Dataset, load_metric

    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False


# Helper: save/load sklearn pipeline (persist to session_state and disk)
def save_tfidf_model(pipeline, path="models/suspicious_tfidf.pkl"):
    import joblib

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    st.session_state["suspicious_model"] = pipeline
    st.session_state["suspicious_model_type"] = "tfidf"
    st.success(f"TF-IDF suspicious model saved to {path}")


def load_tfidf_model(path="models/suspicious_tfidf.pkl"):
    import joblib

    if os.path.exists(path):
        model = joblib.load(path)
        st.session_state["suspicious_model"] = model
        st.session_state["suspicious_model_type"] = "tfidf"
        return model
    return st.session_state.get("suspicious_model", None)


# Transformer save/load helpers
def save_transformer_model(model, tokenizer, out_dir="models/suspicious_transformer"):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    st.session_state["suspicious_transformer_dir"] = out_dir
    st.session_state["suspicious_model_type"] = "transformer"
    st.success(f"Transformer model saved to {out_dir}")


def load_transformer_model(out_dir="models/suspicious_transformer"):
    if not HAS_TRANSFORMERS:
        return None, None
    if os.path.exists(out_dir):
        tokenizer = AutoTokenizer.from_pretrained(out_dir)
        model = AutoModelForSequenceClassification.from_pretrained(out_dir)
        st.session_state["suspicious_model_type"] = "transformer"
        st.session_state["suspicious_transformer_dir"] = out_dir
        return model, tokenizer
    return None, None


# Quick weak-supervision / pseudo-label generator:
def generate_pseudo_labels(messages, codeword_hits=None, suspect_keywords=None):
    """
    messages: list of dicts {'text','sender','time'...}
    codeword_hits: st.session_state.get("codeword_hits", [])
    suspect_keywords: optional set
    returns: DataFrame with columns ['text','label','source']
    label: 1 suspicious, 0 non-suspicious (pseudo)
    """
    if suspect_keywords is None:
        suspect_keywords = set(
            [
                "bomb",
                "attack",
                "explode",
                "transfer",
                "kill",
                "meet",
                "drop",
                "pickup",
                "send",
                "wire",
            ]
        )
    pseudo = []
    cw_idx_set = set()
    if codeword_hits:
        for h in codeword_hits:
            cw_idx_set.add(h["index"])
    for i, m in enumerate(messages):
        t = (m.get("text") or "").strip()
        if not t:
            continue
        label = None
        sources = []
        # codeword hit -> suspicious
        if i in cw_idx_set:
            label = 1
            sources.append("codeword")
        # suspect keyword -> suspicious
        if any(k in t.lower() for k in suspect_keywords):
            label = 1 if label is None else label
            sources.append("keyword")
        # length heuristics -> maybe suspicious
        if len(t.split()) <= 4 and len(t) > 0:
            if label is None:
                label = 0  # short messages are ambiguous; default to non-suspicious
                sources.append("short")
        # negative sentiment adds weight; use existing sentiment_per_message if available
        # fallback: none
        if label is None:
            label = 0
        pseudo.append(
            {"text": t, "label": int(label), "sources": ",".join(sources), "index": i}
        )
    df = pd.DataFrame(pseudo)
    return df


# TF-IDF trainer (fast)
def train_tfidf_classifier(texts, labels):
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10000, ngram_range=(1, 2), stop_words="english"
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipeline.fit(texts, labels)
    return pipeline


# Transformer fine-tuning trainer
def train_transformer_classifier(
    texts,
    labels,
    model_name="distilbert-base-uncased",
    out_dir="models/suspicious_transformer",
    num_train_epochs=1,
    batch_size=8,
    lr=2e-5,
):
    """
    texts: list[str], labels: list[int]
    returns: trained model, tokenizer
    """
    if not HAS_TRANSFORMERS or not HAS_DATASETS:
        raise RuntimeError("transformers/datasets not available in environment")

    # Prepare dataset
    df = pd.DataFrame({"text": texts, "label": labels})
    ds = Dataset.from_pandas(df)
    ds = ds.train_test_split(test_size=0.15, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=256
        )

    ds = ds.map(tokenize_fn, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=max(1, batch_size),
        per_device_eval_batch_size=max(1, batch_size),
        evaluation_strategy="epoch",
        num_train_epochs=max(1, int(num_train_epochs)),
        learning_rate=lr,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        fp16=False,
    )

    (
        load_metric("accuracy")
        if hasattr(__import__("datasets"), "load_metric")
        else None
    )

    def compute_metrics(eval_pred):
        logits, labels_eval = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels_eval, preds)),
            "f1": float(f1_score(labels_eval, preds, zero_division=0)),
            "precision": float(precision_score(labels_eval, preds, zero_division=0)),
            "recall": float(recall_score(labels_eval, preds, zero_division=0)),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # save model + tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return model, tokenizer, trainer


# Inference wrappers
def predict_with_tfidf(model, texts):
    preds = model.predict(texts)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)
    else:
        probs = None
    return preds, probs


def predict_with_transformer_dir(out_dir, texts, batch_size=16):
    if not HAS_TRANSFORMERS:
        raise RuntimeError("Transformers not available")
    tokenizer = AutoTokenizer.from_pretrained(out_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(out_dir)
    model.eval()
    enc = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():

        outputs = model(**enc)
        logits = outputs.logits.cpu().numpy()
        preds = np.argmax(logits, axis=-1)
        # softmax prob for class 1
        from scipy.special import softmax

        probs = softmax(logits, axis=1)
    return preds, probs


# ---------------------------
# Streamlit UI: upgraded trainer / controls
# ---------------------------
with st.expander("Upgraded Suspicious Classifier (Train / Evaluate)", expanded=True):
    st.write(
        "Choose training mode. Transformer training requires `transformers` and `datasets` packages installed."
    )
    # ensure we have some labeled data or pseudo labels
    labeled_df = pd.DataFrame(st.session_state.get("susp_labels", []))
    # allow pseudo-label generation
    if st.button("Generate Pseudo-Labels (weak supervision)"):
        pseudo = generate_pseudo_labels(
            st.session_state.get("chats", []),
            codeword_hits=st.session_state.get("codeword_hits", []),
        )
        # Merge pseudo labels with any human labels (human labels take precedence)
        human = (
            labeled_df[["text", "label"]].dropna()
            if not labeled_df.empty
            else pd.DataFrame(columns=["text", "label"])
        )
        if not human.empty:
            # remove human-labeled texts from pseudo
            pseudo = pseudo[~pseudo["text"].isin(human["text"])]
        combined = pd.concat(
            [human, pseudo[["text", "label"]].rename(columns={"label": "label"})],
            ignore_index=True,
        )
        st.session_state["susp_pseudo_df"] = combined
        st.write("Pseudo-labels generated — sample:")
        st.dataframe(combined.head(20))

    # Choose model type
    model_choice = st.selectbox(
        "Model type",
        options=(
            ["tfidf", "transformer"] if HAS_TRANSFORMERS and HAS_DATASETS else ["tfidf"]
        ),
        index=0,
    )
    st.markdown("**Training Data Source**")
    data_sel = st.selectbox(
        "Use labeled examples from:",
        options=["Human labels (susp_labels)", "Pseudo-labeled (generated)"],
    )
    if data_sel == "Human labels (susp_labels)":
        df_train = pd.DataFrame(st.session_state.get("susp_labels", []))
    else:
        df_train = st.session_state.get("susp_pseudo_df", pd.DataFrame())

    if df_train is None or df_train.empty:
        st.warning(
            "No labeled data available. Label messages in the Labeler (earlier) or generate pseudo-labels first."
        )
    else:
        st.write("Training examples:", df_train.shape[0])
        # hyperparameters
        if model_choice == "tfidf":
            c1, c2 = st.columns(2)
            with c1:
                test_size = st.slider("Test size (fraction)", 0.05, 0.5, 0.2, 0.05)
            with c2:
                random_state = st.number_input("Random seed", value=42, step=1)
            if st.button("Train TF-IDF Suspicious Classifier"):
                texts = df_train["text"].astype(str).tolist()
                labels = df_train["label"].astype(int).tolist()
                try:
                    model = train_tfidf_classifier(texts, labels)
                    save_tfidf_model(model)
                    # Evaluate
                    X_train, X_test, y_train, y_test = train_test_split(
                        texts,
                        labels,
                        test_size=test_size,
                        random_state=random_state,
                        stratify=labels if len(set(labels)) > 1 else None,
                    )
                    preds = model.predict(X_test)
                    st.text(classification_report(y_test, preds))
                    st.success("TF-IDF model trained and saved.")
                except Exception as e:
                    st.error(f"TF-IDF training failed: {e}")
        else:
            # transformer branch
            c1, c2, c3 = st.columns(3)
            with c1:
                model_name = st.text_input(
                    "Transformer base model", value="distilbert-base-uncased"
                )
            with c2:
                num_epochs = st.number_input(
                    "Epochs", min_value=1, max_value=10, value=1
                )
            with c3:
                batch_size = st.selectbox("Batch size", [4, 8, 16], index=1)
            out_dir = st.text_input("Output dir", value="models/suspicious_transformer")
            if st.button("Train Transformer Suspicious Classifier"):
                if not HAS_TRANSFORMERS or not HAS_DATASETS:
                    st.error(
                        "Transformers/datasets not installed. Install required packages to use transformer training."
                    )
                else:
                    texts = df_train["text"].astype(str).tolist()
                    labels = df_train["label"].astype(int).tolist()
                    try:
                        st.info(
                            "Training transformer (this may be slow on CPU). Training progress will stream below."
                        )
                        t0 = time.time()
                        model, tokenizer, trainer = train_transformer_classifier(
                            texts,
                            labels,
                            model_name=model_name,
                            out_dir=out_dir,
                            num_train_epochs=num_epochs,
                            batch_size=batch_size,
                        )
                        save_transformer_model(model, tokenizer, out_dir)
                        t1 = time.time()
                        st.success(
                            f"Transformer training finished in {int(t1-t0)}s and saved to {out_dir}"
                        )
                        # Evaluate on holdout stored in trainer.state if available
                        if hasattr(trainer, "evaluate"):
                            eval_res = trainer.evaluate()
                            st.write("Evaluation on held-out set:", eval_res)
                    except Exception as e:
                        st.error(f"Transformer training failed: {e}")

# ---------------------------
# Apply the selected model to all messages and update st.session_state["suspicious_messages"]
# ---------------------------
with st.expander("Apply Suspicious Model to Messages (Run Inference)", expanded=True):
    st.write(
        "Run inference using the selected model in session_state. Transformer inference will use saved dir if present."
    )
    model_type_now = st.session_state.get("suspicious_model_type", "tfidf")
    st.write("Current selected model type:", model_type_now)
    if st.button("Run Suspicion Inference (All messages)"):
        msgs = st.session_state.get("chats", [])
        texts = [(m.get("text") or "") for m in msgs]
        results = []
        if model_type_now == "tfidf":
            model = st.session_state.get("suspicious_model", None)
            if model is None:
                # try to load
                model = load_tfidf_model()
            if model is None:
                st.error("No TF-IDF model found. Train one first.")
            else:
                preds, probs = predict_with_tfidf(model, texts)
                for i, p in enumerate(preds):
                    prob = float(probs[i][1]) if probs is not None else None
                    if int(p) == 1 or (prob is not None and prob >= 0.5):
                        results.append(
                            {
                                "index": i,
                                "sender": msgs[i].get("sender"),
                                "time": msgs[i].get("time"),
                                "text": texts[i],
                                "score": float(prob) if prob is not None else 1.0,
                                "method": "tfidf",
                            }
                        )
        elif model_type_now == "transformer":
            out_dir = st.session_state.get(
                "suspicious_transformer_dir", "models/suspicious_transformer"
            )
            if os.path.exists(out_dir) and HAS_TRANSFORMERS:
                try:
                    import torch
                    from scipy.special import softmax

                    tokenizer = AutoTokenizer.from_pretrained(out_dir, use_fast=True)
                    model = AutoModelForSequenceClassification.from_pretrained(out_dir)
                    model.eval()
                    batch = []
                    batch_indices = []
                    for i, t in enumerate(texts):
                        batch.append(t)
                        batch_indices.append(i)
                        # process in batches of 32
                        if len(batch) >= 32 or i == len(texts) - 1:
                            enc = tokenizer(
                                batch,
                                truncation=True,
                                padding=True,
                                return_tensors="pt",
                            )
                            with torch.no_grad():
                                outputs = model(**enc)
                                logits = outputs.logits.cpu().numpy()
                                probs = softmax(logits, axis=1)
                                preds = np.argmax(logits, axis=1)
                                for local_idx, msg_idx in enumerate(batch_indices):
                                    p = int(preds[local_idx])
                                    prob = float(probs[local_idx][1])
                                    if p == 1 or prob >= 0.5:
                                        results.append(
                                            {
                                                "index": msg_idx,
                                                "sender": msgs[msg_idx].get("sender"),
                                                "time": msgs[msg_idx].get("time"),
                                                "text": texts[msg_idx],
                                                "score": prob,
                                                "method": "transformer",
                                            }
                                        )
                            batch = []
                            batch_indices = []
                except Exception as e:
                    st.error(f"Transformer inference failed: {e}")
            else:
                st.error(
                    "No transformer saved model found. Train and save first or switch to tfidf."
                )
        # enrich results with codeword hits if available
        cw_hits = {h["index"]: h for h in st.session_state.get("codeword_hits", [])}
        # merge duplicates: prefer higher score
        merged = {}
        for r in results:
            idx = r["index"]
            if idx in cw_hits:
                r["codeword"] = cw_hits[idx]["term"]
                r["method"] += "+codeword"
                r["score"] = max(r.get("score", 0), 1.0)
            if idx in merged:
                if r.get("score", 0) > merged[idx].get("score", 0):
                    merged[idx] = r
            else:
                merged[idx] = r
        final = sorted(
            list(merged.values()), key=lambda x: x.get("score", 0), reverse=True
        )
        st.session_state["suspicious_messages"] = final
        st.success(
            f"Inference complete — {len(final)} suspicious messages flagged and stored in st.session_state['suspicious_messages']."
        )

# ---------------------------
# Utility: Export/Import trained models & labeled data
# ---------------------------
with st.expander("Export / Import Model + Labeled Data", expanded=False):
    if st.button("Download labeled examples (CSV)"):
        labels = pd.DataFrame(st.session_state.get("susp_labels", []))
        if not labels.empty:
            csv = labels.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV", csv, file_name="suspicious_labels.csv", mime="text/csv"
            )
        else:
            st.warning("No labeled data available.")
    if st.button("Export TF-IDF Model (joblib)"):
        # try to save
        mdl = st.session_state.get("suspicious_model", None)
        if mdl:
            save_tfidf_model(mdl, path="models/exported_suspicious_tfidf.pkl")
            with open("models/exported_suspicious_tfidf.pkl", "rb") as f:
                st.download_button(
                    "Download TF-IDF model",
                    f,
                    file_name="exported_suspicious_tfidf.pkl",
                )
        else:
            st.warning("No TF-IDF model to export.")
    if st.button("Export Transformer (zip)"):
        d = st.session_state.get("suspicious_transformer_dir", None)
        if d and os.path.exists(d):
            import shutil

            shutil.make_archive("models/suspicious_transformer_export", "zip", d)
            with open("models/suspicious_transformer_export.zip", "rb") as f:
                st.download_button(
                    "Download Transformer model (zip)",
                    f,
                    file_name="suspicious_transformer_export.zip",
                )
        else:
            st.warning("No transformer saved model to export.")

# ---------------------------
# Semi-supervised Trainer (Self-training + Active Learning)
# Paste into Comms Analyzer section in app.py (after existing classifier code)
# ---------------------------
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Fallback trainer if not defined earlier
try:
    train_tfidf_classifier  # function defined earlier in your code
except Exception:
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    def train_tfidf_classifier(texts, labels):
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=10000, ngram_range=(1, 2), stop_words="english"
                    ),
                ),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
        pipeline.fit(texts, labels)
        return pipeline


# Fallback save function
def save_tfidf_model_local(model, path="models/suspicious_tfidf_auto.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    st.session_state["suspicious_model"] = model
    st.session_state["suspicious_model_type"] = "tfidf"
    st.success(f"Auto-trained TF-IDF model saved to {path}")


# Utility to get texts and mask of unlabeled
def _prepare_datasets():
    msgs = st.session_state.get("chats", [])
    texts = [(m.get("text") or "").strip() for m in msgs]
    # human labels map: index -> label
    human_labels = {}
    for h in st.session_state.get("susp_labels", []):
        idx = h.get("index")
        if idx is not None:
            human_labels[int(idx)] = int(h.get("label", 0))
    labeled_indices = set(human_labels.keys())
    unlabeled_indices = [i for i in range(len(texts)) if i not in labeled_indices]
    return texts, human_labels, labeled_indices, unlabeled_indices


# Main semi-supervised loop UI
with st.expander(
    "Semi-supervised Trainer (self-training + active learning)", expanded=True
):
    st.write(
        "This runs iterative pseudo-labeling on high-confidence predictions and surfaces low-confidence items for human labeling."
    )

    # Parameters
    cols = st.columns(4)
    with cols[0]:
        n_rounds = st.number_input("Rounds", min_value=1, max_value=10, value=3, step=1)
    with cols[1]:
        high_conf = st.slider("High-confidence threshold", 0.6, 0.99, 0.90, 0.01)
    with cols[2]:
        low_conf = st.slider(
            "Low-confidence threshold (active pool)", 0.01, 0.59, 0.4, 0.01
        )
    with cols[3]:
        sample_for_human = st.number_input(
            "Active sample size per round", min_value=1, max_value=200, value=20, step=1
        )

    st.markdown(
        "**Workflow:** start with any human-labeled messages (susp_labels) and optional pseudo labels. The loop will:"
    )
    st.markdown(
        "1) Train base TF-IDF classifier on current labels.  2) Predict on unlabeled.  3) Add predictions with prob >= high_conf as pseudo-labeled.  4) Collect low-confidence messages (prob in [low_conf, high_conf)) into an active pool for human labeling.  5) Repeat for n_rounds."
    )

    if st.button("Run Semi-supervised Loop"):
        t0 = time.time()
        texts, human_labels, labeled_idxs, unlabeled_idxs = _prepare_datasets()
        if not human_labels:
            st.warning(
                "No human labels found in st.session_state['susp_labels']. It's recommended to have at least a few (10+) human labels. You can still run, pseudo-labels will be used."
            )
        # Build initial training set: human labels + any pseudo stored earlier
        pseudo_df = st.session_state.get("susp_pseudo_df", pd.DataFrame())
        train_texts = []
        train_labels = []
        # add human labels first (strong)
        for idx, lab in human_labels.items():
            train_texts.append(texts[idx])
            train_labels.append(lab)
        # add pseudo-labeled examples (if any and not duplicate of human)
        if isinstance(pseudo_df, pd.DataFrame) and not pseudo_df.empty:
            for _, r in pseudo_df.iterrows():
                t = str(r.get("text", ""))
                if t not in train_texts:
                    train_texts.append(t)
                    train_labels.append(int(r.get("label", 0)))
        # If still empty, warn and stop
        if len(train_texts) < 1:
            st.error(
                "No training data available (no human labels and no pseudo labels). Label some messages first."
            )
        else:
            # Keep track of newly added pseudo indices
            all_new_pseudo = set()
            model = None
            for rnd in range(1, int(n_rounds) + 1):
                st.info(
                    f"Round {rnd} — training model on {len(train_texts)} examples..."
                )
                # train model
                model = train_tfidf_classifier(train_texts, train_labels)
                # predict probs on unlabeled
                unlabeled_idxs = [
                    i
                    for i in range(len(texts))
                    if i not in labeled_idxs and i not in all_new_pseudo
                ]
                if not unlabeled_idxs:
                    st.success("No unlabeled examples left.")
                    break
                unl_texts = [texts[i] for i in unlabeled_idxs]
                try:
                    probs = None
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(unl_texts)
                        pos_probs = probs[:, 1]
                    else:
                        # fallback: use decision_function
                        try:
                            dec = model.decision_function(unl_texts)
                            # map to [0,1] with logistic
                            pos_probs = 1 / (1 + np.exp(-dec))
                        except Exception:
                            pos_probs = np.zeros(len(unl_texts))
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    break

                # select high-confidence positives and negatives
                high_pos_idx_local = [
                    unlabeled_idxs[i] for i, p in enumerate(pos_probs) if p >= high_conf
                ]
                high_neg_idx_local = [
                    unlabeled_idxs[i]
                    for i, p in enumerate(pos_probs)
                    if p <= (1 - high_conf)
                ]
                # Add to pseudo labelled set
                added = 0
                for idx in high_pos_idx_local:
                    train_texts.append(texts[idx])
                    train_labels.append(1)
                    all_new_pseudo.add(idx)
                    added += 1
                for idx in high_neg_idx_local:
                    train_texts.append(texts[idx])
                    train_labels.append(0)
                    all_new_pseudo.add(idx)
                    added += 1

                st.write(
                    f"Round {rnd}: added {added} high-confidence pseudo-labels (pos:{len(high_pos_idx_local)} neg:{len(high_neg_idx_local)})"
                )
                # Build active pool of low-confidence for human labeling
                low_pool_local = [
                    unlabeled_idxs[i]
                    for i, p in enumerate(pos_probs)
                    if (p > (1 - high_conf) and p < high_conf)
                    and (p >= low_conf and p <= (1 - low_conf) or True)
                ]
                # sample limited size for this round
                if low_pool_local:
                    sample = low_pool_local[:sample_for_human]
                    # Store active sample for human labeling (append to st.session_state['susp_active_pool'])
                    existing_pool = st.session_state.get("susp_active_pool", [])
                    for sidx in sample:
                        if sidx not in existing_pool:
                            existing_pool.append(int(sidx))
                    st.session_state["susp_active_pool"] = existing_pool
                    st.write(
                        f"Round {rnd}: added {len(sample)} examples to active labeling pool (st.session_state['susp_active_pool'])."
                    )

            # end rounds
            st.success(
                f"Semi-supervised loop finished. Added total pseudo-labeled items: {len(all_new_pseudo)}"
            )
            # Save the final model and persist
            if model is not None:
                try:
                    save_tfidf_model_local(
                        model, path="models/suspicious_tfidf_auto.pkl"
                    )
                    st.session_state["susp_pseudo_added_count"] = len(all_new_pseudo)
                    # Optionally evaluate on a holdout of human labels if available
                    human_texts = [texts[i] for i in list(human_labels.keys())]
                    human_y = [human_labels[i] for i in list(human_labels.keys())]
                    if human_texts:
                        preds = model.predict(human_texts)
                        st.text("Evaluation on current human-labeled set:")
                        st.text(classification_report(human_y, preds))
                except Exception as e:
                    st.error(f"Failed to save model: {e}")
            t1 = time.time()
            st.write(
                f"Finished in {int(t1-t0)}s. Pseudo-labeled count: {len(all_new_pseudo)}"
            )

# Expose Active pool for human labeling
with st.expander(
    "Active Learning Pool — label these to improve the model", expanded=False
):
    pool = st.session_state.get("susp_active_pool", [])
    msgs = st.session_state.get("chats", [])
    if not pool:
        st.info(
            "Active pool is empty. Run semi-supervised loop or train to populate low-confidence examples."
        )
    else:
        st.write(f"{len(pool)} messages awaiting human review.")
        # show up to 30 examples to label here
        sample = pool[: min(50, len(pool))]
        for idx in sample:
            m = msgs[int(idx)]
            st.markdown(
                f"**Msg #{idx}** — *{m.get('sender','unknown')}* @ {m.get('time','')}"
            )
            choice = st.radio(
                f"Label active msg {idx}",
                ["Unlabeled", "Not Suspicious", "Suspicious"],
                key=f"active_label_{idx}",
            )
            if choice != "Unlabeled":
                # add or update label in st.session_state['susp_labels']
                found = False
                labels_list = st.session_state.get("susp_labels", [])
                for item in labels_list:
                    if item.get("index") == idx:
                        item.update(
                            {
                                "label": 1 if choice == "Suspicious" else 0,
                                "text": m.get("text", ""),
                                "sender": m.get("sender"),
                                "time": m.get("time"),
                            }
                        )
                        found = True
                if not found:
                    labels_list.append(
                        {
                            "index": int(idx),
                            "text": m.get("text", ""),
                            "label": 1 if choice == "Suspicious" else 0,
                            "sender": m.get("sender"),
                            "time": m.get("time"),
                        }
                    )
                st.session_state["susp_labels"] = labels_list
                # remove from active pool automatically
                new_pool = [
                    p for p in st.session_state.get("susp_active_pool", []) if p != idx
                ]
                st.session_state["susp_active_pool"] = new_pool
                st.success(
                    f"Labeled message #{idx} as {choice} and removed from active pool."
                )

# Quick utility to export active pool indices and pseudo suggestions
with st.expander("Export / Inspect semi-supervised artifacts", expanded=False):
    if st.button("Show indices of pseudo-labeled examples (session)"):
        st.write(
            "Pseudo-added-count:", st.session_state.get("susp_pseudo_added_count", 0)
        )
    if st.button("Download active pool indices (CSV)"):
        pool = st.session_state.get("susp_active_pool", [])
        df = pd.DataFrame({"index": pool})
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download active_pool.csv",
            csv,
            file_name="active_pool.csv",
            mime="text/csv",
        )


# ---- INSERT: Auto-parse social media databases (WhatsApp, Telegram, Messenger) ----
import sqlite3
from PIL import ExifTags


def parse_whatsapp_db(db_path):
    # Parse a non-encrypted WhatsApp msgstore.db SQLite and return list of messages.
    msgs = []
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cur.fetchall()]
        if "messages" in tables:
            try:
                cur.execute(
                    "SELECT _id, key_remote_jid, key_from_me, data, timestamp FROM messages ORDER BY timestamp ASC"
                )
                rows = cur.fetchall()
                for row in rows:
                    _id, jid, from_me, data, ts = row
                    text = data if data is not None else ""
                    # whatsapp timestamps often in milliseconds
                    ts_val = ts / 1000 if ts else None
                    msgs.append(
                        {
                            "id": _id,
                            "sender": jid,
                            "from_me": from_me,
                            "text": text,
                            "timestamp": ts_val,
                        }
                    )
            except Exception:
                # schema mismatch or other issue
                pass
        conn.close()
    except Exception as e:
        return {"error": str(e), "messages": []}
    return {"messages": msgs}


def parse_telegram_db(db_path):
    msgs = []
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cur.fetchall()]
        for t in ["messages", "message", "msgs"]:
            if t in tables:
                try:
                    cur.execute(f"SELECT * FROM {t} LIMIT 1000")
                    rows = cur.fetchall()
                    for i, r in enumerate(rows):
                        msgs.append({"row": i, "data": str(r)})
                except:
                    pass
        conn.close()
    except Exception as e:
        return {"error": str(e), "messages": []}
    return {"messages": msgs}


def extract_exif_from_image(img_path):
    info = {}
    try:
        img = Image.open(img_path)
        exif = getattr(img, "_getexif", lambda: None)()
        if exif:
            for tag, value in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                info[decoded] = value
    except Exception as e:
        info = {"error": str(e)}
    return info


def auto_parse_social_media(adb_out_dir):
    # Auto-detect common app DBs and parse them. Returns a dict of parsed outputs.
    parsed = {}
    if not adb_out_dir or not os.path.exists(adb_out_dir):
        return parsed
    # WhatsApp: look for msgstore*.db in tree
    wa_dbs = []
    for root, dirs, files in os.walk(adb_out_dir):
        for fn in files:
            if (fn.startswith("msgstore") and fn.endswith(".db")) or (
                fn.startswith("wa") and fn.endswith(".db")
            ):
                wa_dbs.append(os.path.join(root, fn))
    if wa_dbs:
        parsed["whatsapp"] = []
        for db in wa_dbs:
            res = parse_whatsapp_db(db)
            parsed["whatsapp"].append({"db": db, "result": res})
            out_dir = os.path.join(
                "uploads", st.session_state.get("case_id", "unknown_case"), "parsed"
            )
            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(db) + ".parsed.json")
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump(res, fw, indent=2, ensure_ascii=False)
            if isinstance(res, dict) and "messages" in res and res["messages"]:
                st.session_state.setdefault("chats", [])
                for m in res["messages"]:
                    st.session_state["chats"].append(
                        {
                            "sender": m.get("sender"),
                            "time": m.get("timestamp"),
                            "text": m.get("text") or "",
                            "source_db": db,
                        }
                    )
    # Telegram: look for cache*.db or cache4.db
    tg_dbs = []
    for root, dirs, files in os.walk(adb_out_dir):
        for fn in files:
            if ("cache" in fn and fn.endswith(".db")) or (
                "telegram" in fn.lower() and fn.endswith(".db")
            ):
                tg_dbs.append(os.path.join(root, fn))
    if tg_dbs:
        parsed["telegram"] = []
        for db in tg_dbs:
            res = parse_telegram_db(db)
            parsed["telegram"].append({"db": db, "result": res})
            out_dir = os.path.join(
                "uploads", st.session_state.get("case_id", "unknown_case"), "parsed"
            )
            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(db) + ".parsed.json")
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump(res, fw, indent=2, ensure_ascii=False)
            if isinstance(res, dict) and "messages" in res and res["messages"]:
                st.session_state.setdefault("chats", [])
                for m in res["messages"]:
                    st.session_state["chats"].append(
                        {
                            "sender": m.get("sender") if isinstance(m, dict) else None,
                            "time": None,
                            "text": m.get("text") if isinstance(m, dict) else str(m),
                            "source_db": db,
                        }
                    )
    # EXIF extraction for images
    images = []
    for root, dirs, files in os.walk(adb_out_dir):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(root, fn))
    if images:
        parsed["images"] = []
        for img in images:
            ex = extract_exif_from_image(img)
            parsed["images"].append({"file": img, "exif": ex})
            out_dir = os.path.join(
                "uploads", st.session_state.get("case_id", "unknown_case"), "parsed"
            )
            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(img) + ".exif.json")
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump(ex, fw, indent=2, ensure_ascii=False)
    return parsed


# ---- END INSERT: Auto-parse social media databases ----

# === FORNSMART ADB CONSENT PATCH START ===
# Helper functions to manage adb device consent and session-state in Streamlit.
# Call `require_adb_ui(st)` from your Adapters tab UI to show the consent/debug area,
# and call `require_adb_ready_or_abort()` before starting an extraction.
import json
import time
import hashlib


def get_adb_devices():
    """Return a dict with 'devices' list and raw output, or {'error': msg} on failure."""
    try:
        out = subprocess.check_output(
            ["adb", "devices", "-l"], stderr=subprocess.STDOUT
        ).decode(errors="ignore")
    except Exception as e:
        return {"error": str(e)}
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    devices = []
    for l in lines[1:]:  # skip header line if present
        parts = l.split()
        if len(parts) >= 2:
            dev_id = parts[0]
            state = parts[1]
            devices.append({"id": dev_id, "state": state, "raw": l})
    return {"devices": devices, "raw": out}


# Streamlit UI helper (safe to import; will only run when called)
def require_adb_ui(st):
    """Render ADB consent / status UI inside the Adapters tab. Accepts the streamlit module as `st`."""
    if "adb_last_devices_raw" not in st.session_state:
        st.session_state.adb_last_devices_raw = ""
    if "adb_consent_ok" not in st.session_state:
        st.session_state.adb_consent_ok = False
    if "adb_last_device_id" not in st.session_state:
        st.session_state.adb_last_device_id = None
    if "extraction_consent_given" not in st.session_state:
        st.session_state.extraction_consent_given = False

    info = get_adb_devices()
    if "error" in info:
        st.error("ADB error: " + info["error"])
        return
    raw = info.get("raw", "")
    devices = info.get("devices", [])
    st.text_area("adb output (debug)", raw, height=120)
    # compare raw to last raw to detect device change
    if raw != st.session_state.adb_last_devices_raw:
        st.session_state.adb_last_devices_raw = raw
        st.session_state.adb_consent_ok = False

    if not devices:
        st.warning("No devices detected. Connect a device or start an emulator.")
        if st.button("Re-check device"):
            # noop; rerun will refresh
            pass
        return

    dev = devices[0]
    st.write(f"Device: {dev['id']}  state: {dev['state']}")
    if dev["state"] != "device":
        st.session_state.adb_consent_ok = False
        st.error(
            "Device not authorized for ADB. On the device, accept the RSA prompt and re-run."
        )
        if st.button("Re-check device"):
            pass
    else:
        st.session_state.adb_consent_ok = True
        st.success("Device authorized for ADB. Ready to run adapters.")
        if not st.session_state.extraction_consent_given:
            agree = st.checkbox(
                "I confirm I have permission to extract data from this device (consent granted).",
                key="extraction_consent_checkbox",
            )
            if agree:
                st.session_state.extraction_consent_given = True
        else:
            st.info("Extraction consent previously given in this session.")


# Guard to call before extraction
def require_adb_ready_or_abort(st):
    if not st.session_state.get("adb_consent_ok", False):
        st.error(
            "Cannot run adapter: device not authorized. Please accept ADB prompt on the device and press 'Re-check device'."
        )
        st.stop()
    if not st.session_state.get("extraction_consent_given", False):
        st.error("Extraction consent not given in UI.")
        st.stop()


# End of patch
# === FORNSMART ADB CONSENT PATCH END ===


# === FORNSMART_CONSENT_OVERRIDE START ===
# Developer override to bypass extraction consent (useful for automated testing).
# WARNING: this bypass will allow adapters to run without the user checkbox. Keep it disabled in production.
def require_adb_ui_with_override(st):
    """Use this in place of require_adb_ui(st) when you want a visible developer override option."""
    # call the regular UI first
    try:
        require_adb_ui(st)
    except Exception:
        # fall back silently if original helper is not available
        pass

    # developer options section (hidden by default)
    show_dev = st.checkbox(
        "Show developer options (advanced)", key="dev_options_toggle"
    )
    if show_dev:
        st.markdown(
            "**Developer override:** skip the extraction consent checkbox (unsafe). Use only in dev/testing."
        )
        # Option A: environment / secret based override
        override_key = st.text_input(
            "Enter override key to enable skip (keeps session state):",
            "",
            type="password",
            key="dev_override_key_input",
        )
        # Default override key for local test only. You can change this to an env var or st.secrets lookup.
        REAL_OVERRIDE_KEY = "FORNSMART-OVERRIDE"
        if override_key and override_key == REAL_OVERRIDE_KEY:
            st.success(
                "Developer override accepted — extraction consent bypass enabled for this session."
            )
            st.session_state.extraction_consent_given = True
            st.session_state.adb_consent_ok = True
        elif override_key:
            st.error("Invalid override key.")


# Guard wrapper that respects the override flag in session_state
def require_adb_ready_or_abort_with_override(st):
    # If a developer override flag was set earlier in session_state, allow run
    if st.session_state.get("dev_skip_consent", False):
        return
    # Otherwise use the normal guard
    require_adb_ready_or_abort(st)


# === FORNSMART_CONSENT_OVERRIDE END ===


# === FORNSMART DEV MODE HOOK ===
# If environment variable FORNSMART_DEV_MODE is set to "1", enable developer consent bypass automatically.
def _apply_dev_mode_defaults(st):
    try:
        if os.getenv("FORNSMART_DEV_MODE", "0") == "1":
            st.session_state.adb_consent_ok = True
            st.session_state.extraction_consent_given = True
            # also mark a flag for downstream guards
            st.session_state.dev_skip_consent = True
    except Exception:
        # if session_state not available yet, ignore
        pass


# To use: call _apply_dev_mode_defaults(st) at the start of your Adapters tab render function
# Example:
#   _apply_dev_mode_defaults(st)
#   require_adb_ui_with_override(st)
# === END FORNSMART DEV MODE HOOK ===
