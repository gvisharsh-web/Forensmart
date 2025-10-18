# adapters/runner.py
import os, json, subprocess, hashlib, time, traceback
from pathlib import Path
from typing import Dict, Any, List

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

class AdapterRunner:
    """
    Simple runner that loads adapter module and runs extract().
    For python adapters it imports and calls directly.
    """

    def __init__(self, adapter_module: str, adapter_cls: str = None, options: Dict[str,Any] = None):
        self.adapter_module = adapter_module
        self.adapter_cls = adapter_cls
        self.options = options or {}

    def run_python_adapter(self, out_dir: str, artifact_list: List[str], metadata: Dict[str,Any]):
        # dynamic import
        try:
            mod = __import__(self.adapter_module, fromlist=['*'])
            cls = getattr(mod, self.adapter_cls) if self.adapter_cls else getattr(mod, 'Adapter')
            adapter = cls(self.options)
        except Exception as e:
            return {"ok": False, "error": f"import_error: {e}", "trace": traceback.format_exc()}

        probe = {}
        try:
            probe = adapter.probe()
        except Exception as e:
            probe = {"ok": False, "error": f"probe_failed: {e}", "trace": traceback.format_exc()}

        # ensure out_dir exists
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        res = {}
        try:
            res = adapter.extract(artifact_list, out_dir, metadata)
        except Exception as e:
            res = {"ok": False, "error": f"extract_failed: {e}", "trace": traceback.format_exc()}

        # write meta for each produced file
        try:
            for p in Path(out_dir).rglob('*'):
                if p.is_file() and not str(p).endswith('.meta.json'):
                    data = p.read_bytes()
                    meta = {
                        "consent_id": metadata.get("consent_id"),
                        "adapter": getattr(adapter, "name", self.adapter_module),
                        "artifact_hash": sha256_bytes(data),
                        "created_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                    }
                    Path(str(p) + ".meta.json").write_text(json.dumps(meta))
        except Exception as e:
            # best-effort meta writes
            pass

        return {"ok": True, "probe": probe, "result": res}

if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--class", dest="cls")
    ap.add_argument("--out", required=True)
    ap.add_argument("--consent", required=True)
    ap.add_argument("--artifacts", default="all")
    args = ap.parse_args()
    artifacts = args.artifacts.split(",") if args.artifacts != "all" else ["all"]
    runner = AdapterRunner(args.module, args.cls)
    r = runner.run_python_adapter(args.out, artifacts, {"consent_id": args.consent})
    print(json.dumps(r, indent=2))
