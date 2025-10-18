@echo off
:: ------------------------------------------------------------------
:: ForenSmart Stage B Patcher (Windows Batch Version)
:: ------------------------------------------------------------------
echo 🔧 ForenSmart Stage B Patcher Starting...

set APP_FILE=app_patched.py
set MODULE_FILE=report_integrity_module.py
set BACKUP_FILE=app_patched.py.backup.%date:~-4%%date:~4,2%%date:~7,2%T%time:~0,2%%time:~3,2%%time:~6,2%

if not exist "%APP_FILE%" (
  echo ❌ Error: %APP_FILE% not found in current directory.
  pause
  exit /b
)

copy "%APP_FILE%" "%BACKUP_FILE%"
echo 📦 Backup created: %BACKUP_FILE%

:: Create Stage B module
(
echo import os, json, hashlib, datetime
echo from pathlib import Path
echo.
echo def collect_manifests(consent_id):
echo.    manifests = []
echo.    paths = [Path(f"artifacts/{consent_id}"), Path(f"consent_records/{consent_id}/manifests")]
echo.    for base in paths:
echo.        if base.exists():
echo.            for mf in base.rglob("manifest.json"):
echo.                try:
echo.                    data = json.loads(mf.read_text())
echo.                    manifests.append({"path": str(mf), "data": data})
echo.                except Exception as e:
echo.                    manifests.append({"path": str(mf), "error": str(e)})
echo.    return manifests
echo.
echo def compute_master_hash(manifests):
echo.    sha = hashlib.sha256()
echo.    for m in sorted(manifests, key=lambda x: x.get("path","")):
echo.        sha.update(json.dumps(m, sort_keys=True).encode())
echo.    return sha.hexdigest()
echo.
echo def embed_report_hash(pdf_bytes, master_hash):
echo.    marker = (
echo.        f"\n\n--- REPORT INTEGRITY ---\n"
echo.        f"Digital Hash Signature: {master_hash}\n"
echo.        f"Generated: {datetime.datetime.utcnow().isoformat()}Z\n"
echo.    )
echo.    return pdf_bytes + marker.encode()
echo.
echo def write_report_signature(consent_id, master_hash, investigator):
echo.    base = Path(f"consent_records/{consent_id}")
echo.    base.mkdir(parents=True, exist_ok=True)
echo.    sig = {
echo.        "case_id": consent_id,
echo.        "consent_id": consent_id,
echo.        "master_hash": master_hash,
echo.        "signed_by": investigator or "Unknown",
echo.        "generated_on": datetime.datetime.utcnow().isoformat() + "Z"
echo.    }
echo.    (base / "report_signature.json").write_text(json.dumps(sig, indent=2))
echo.    (base / "report_integrity.log").write_text(f"Report integrity: {sig['generated_on']}\nHash: {master_hash}\n")
echo.    return sig
echo.
echo def finalize_report_integrity(consent_id, pdf_bytes, investigator):
echo.    manifests = collect_manifests(consent_id)
echo.    master_hash = compute_master_hash(manifests)
echo.    updated_pdf = embed_report_hash(pdf_bytes, master_hash)
echo.    sig = write_report_signature(consent_id, master_hash, investigator)
echo.    return updated_pdf, master_hash
) > "%MODULE_FILE%"

echo 🧩 Created module: %MODULE_FILE%

:: Add import line if missing
find /C "from report_integrity_module import finalize_report_integrity" "%APP_FILE%" >nul
if errorlevel 1 (
  powershell -Command "(Get-Content '%APP_FILE%') -replace 'import pathlib', 'import pathlib`nfrom report_integrity_module import finalize_report_integrity' | Set-Content '%APP_FILE%'"
  echo ✅ Linked module import added to %APP_FILE%
) else (
  echo ℹ️ Import already present.
)

:: Inject integrity call safely using PowerShell
powershell -Command "$lines = Get-Content '%APP_FILE%'; $found = $false; for ($i=0; $i -lt $lines.Count; $i++) { if ($lines[$i] -match 'def export_unified_report_pdf') { for ($j=$i; $j -lt $lines.Count; $j++) { if ($lines[$j] -match 'return' -and $lines[$j] -match 'pdf') { $indent = ($lines[$j] -match '^\s*') | Out-Null; $spaces = '    '; $inject = @('$spaces# --- Stage B: finalize report integrity ---', '$spacesconsent_id = st.session_state.get(''consent_id'')', '$spacesinvestigator = st.session_state.get(''investigator'')', '$spacesif consent_id and ''finalize_report_integrity'' in globals():', '$spaces    try:', '$spaces        updated_pdf, master_hash = finalize_report_integrity(consent_id, out_bytesio.getvalue(), investigator)', '$spaces        st.session_state[''master_report_hash''] = master_hash', '$spaces        out_bytesio = io.BytesIO(updated_pdf)', '$spaces    except Exception as _e:', '$spaces        st.warning(f''Report integrity step failed: {_e}'')'); $lines = $lines[0..$j] + $inject + $lines[$j+1..($lines.Count-1)]; $found = $true; break } } } }; if ($found) { $lines | Set-Content '%APP_FILE%' }"

:: Syntax check
python -m py_compile "%MODULE_FILE%" "%APP_FILE%"
if %errorlevel%==0 (
  echo ✅ Syntax OK for both %APP_FILE% and %MODULE_FILE%
) else (
  echo ❌ Syntax error detected. Check above output.
  pause
  exit /b
)

echo 🎉 Stage B integration complete!
echo Run your app with: streamlit run %APP_FILE%
pause
