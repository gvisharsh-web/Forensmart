
import os, json, hashlib, datetime
from pathlib import Path

def collect_manifests(consent_id):
    manifests = []
    # look in artifacts and consent_records manifests
    paths = []
    if consent_id:
        paths.append(Path(f"artifacts/{consent_id}"))
        paths.append(Path(f"consent_records/{consent_id}/manifests"))
    else:
        paths.append(Path("artifacts"))
        paths.append(Path("consent_records"))
    for base in paths:
        if base.exists():
            for mf in base.rglob("manifest.json"):
                try:
                    data = json.loads(mf.read_text())
                    manifests.append({"path": str(mf), "data": data})
                except Exception as e:
                    manifests.append({"path": str(mf), "error": str(e)})
    return manifests

def compute_master_hash(manifests):
    sha = hashlib.sha256()
    # deterministically sort by path
    for m in sorted(manifests, key=lambda x: x.get("path","")):
        sha.update(json.dumps(m, sort_keys=True).encode())
    return sha.hexdigest()

def embed_report_hash(pdf_bytes, master_hash):
    # Append a clear marker and the hash to the end of the PDF bytes.
    try:
        marker = f"\n\n--- REPORT INTEGRITY ---\nDigital Hash Signature: {master_hash}\nGenerated: {datetime.datetime.utcnow().isoformat()}Z\n"
        return pdf_bytes + marker.encode("utf-8")
    except Exception as e:
        raise RuntimeError(f"embed_report_hash failed: {e}")

def write_report_signature(consent_id, master_hash, investigator):
    try:
        base = Path(f"consent_records/{consent_id}")
        base.mkdir(parents=True, exist_ok=True)
        sig = {
            "case_id": consent_id,
            "consent_id": consent_id,
            "master_hash": master_hash,
            "signed_by": investigator or "Unknown",
            "generated_on": datetime.datetime.utcnow().isoformat() + "Z"
        }
        (base / "report_signature.json").write_text(json.dumps(sig, indent=2))
        (base / "report_integrity.log").write_text(f"Report integrity: {sig['generated_on']}\nHash: {master_hash}\n")
        return sig
    except Exception as e:
        raise RuntimeError(f"write_report_signature failed: {e}")

def finalize_report_integrity(consent_id, pdf_bytes, investigator):
    manifests = collect_manifests(consent_id)
    master_hash = compute_master_hash(manifests)
    updated_pdf = embed_report_hash(pdf_bytes, master_hash)
    sig = write_report_signature(consent_id, master_hash, investigator)
    return updated_pdf, master_hash
