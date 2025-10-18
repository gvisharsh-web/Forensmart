import streamlit as st
import json
import pathlib
import datetime
import hashlib

st.title("Forensmart Consent Diagnostic Tool (embedded)")

st.markdown(
    "This diagnostic overlay helps identify why `consent_id` is not reflected in the Adapters tab. It does not change original adapter logic; it only reads files and session state and lets you force actions."
)

# Show session_state summary
st.header("Session state")
try:
    keys = list(st.session_state.keys())
    st.write("Keys in st.session_state:", keys)
    st.write({k: st.session_state.get(k) for k in keys})
except Exception as e:
    st.error("Failed reading st.session_state: " + str(e))

# Show artifacts directory listing and latest consent.json if any
st.header("Artifacts / consent files")
artifacts_path = pathlib.Path("artifacts")
consent_files = []
if artifacts_path.exists():
    for p in artifacts_path.rglob("consent.json"):
        try:
            mtime = p.stat().st_mtime
            consent_files.append((p, mtime))
        except Exception:
            consent_files.append((p, 0))
consent_files = sorted(consent_files, key=lambda x: x[1])
if not consent_files:
    st.info("No artifacts/*/consent.json files found.")
else:
    st.write(f"Found {len(consent_files)} consent.json files (oldest → newest):")
    for p, m in consent_files:
        st.write(
            f"- {p}  (mtime: {datetime.datetime.utcfromtimestamp(m).isoformat()}Z)"
        )
    latest_path = consent_files[-1][0]
    st.subheader("Latest consent.json content")
    try:
        data = json.load(open(latest_path, "r", encoding="utf-8"))
        st.json(data)
    except Exception as e:
        st.error("Failed to read latest consent.json: " + str(e))

# Also check consent_records folder
st.header("consent_records folder check")
cr_path = pathlib.Path("consent_records")
cr_found = []
if cr_path.exists():
    for p in cr_path.rglob("consent.json"):
        cr_found.append(p)
if cr_found:
    st.write(
        f"Found {len(cr_found)} consent.json in consent_records (showing first 5):"
    )
    for p in cr_found[:5]:
        st.write("-", p)
else:
    st.info("No consent_records/*/consent.json found.")

# Display any consent.sha256 files near those consent.json files
st.header("SHA checks for consent files")
sha_reports = []
for p, _ in consent_files:
    sha_path = p.parent / "consent.sha256"
    exists = sha_path.exists()
    content = sha_path.read_text(encoding="utf-8").strip() if exists else None
    try:
        raw = json.dumps(
            json.load(open(p, "r", encoding="utf-8")), sort_keys=True
        ).encode("utf-8")
        actual = hashlib.sha256(raw).hexdigest()
    except Exception as e:
        actual = f"error: {e}"
    sha_reports.append(
        {
            "path": str(p),
            "sha_path": str(sha_path),
            "sha_exists": exists,
            "sha_content": content,
            "actual_sha": actual,
        }
    )
st.table(sha_reports)

# Provide controls to force load a consent into session_state
st.header("Force actions")
col1, col2 = st.columns(2)
with col1:
    if st.button("Force load latest consent into session_state"):
        if not consent_files:
            st.warning("No consent files to load from artifacts.")
        else:
            try:
                latest = consent_files[-1][0]
                data = json.load(open(latest, "r", encoding="utf-8"))
                cid = data.get("consent_id") or data.get("id")
                if cid:
                    st.session_state["consent_id"] = cid
                    st.success(f"Set st.session_state['consent_id'] = {cid}")
                else:
                    st.error("consent.json missing consent_id key.")
            except Exception as e:
                st.error("Failed to load latest consent: " + str(e))
with col2:
    if st.button("Clear consent_id from session_state"):
        if "consent_id" in st.session_state:
            del st.session_state["consent_id"]
            st.success("Removed consent_id from session_state.")
        else:
            st.info("consent_id not present in session_state.")

# Button to write a small debug trace file in artifacts/
st.header("Write diagnostic trace file")
if st.button("Write diagnostic trace to artifacts/debug_trace.txt"):
    try:
        outdir = pathlib.Path("artifacts") / (
            "session_debug_" + datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        )
        outdir.mkdir(parents=True, exist_ok=True)
        trace_path = outdir / "debug_trace.txt"
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write("session_state keys:\\n")
            for k in st.session_state.keys():
                f.write(f"{k}: {repr(st.session_state.get(k))}\\n")
            f.write("\\nLatest consent files:\\n")
            for p, m in consent_files:
                f.write(f"{p} (mtime: {m})\\n")
        st.success(f"Wrote trace to {trace_path}")
    except Exception as e:
        st.error("Failed to write trace: " + str(e))

st.markdown("---")
st.info(
    "Use the Force actions and then switch to the Adapters tab in your main app (or re-run) to see whether adapters reflect the session_state changes. If the adapters still do not see the consent_id, copy the content of the 'Latest consent.json content' shown above and paste it here so I can inspect it."
)
