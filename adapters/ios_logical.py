# adapters/ios_logical.py
import os, subprocess, json, shutil, tempfile
from pathlib import Path
from adapters.interface import AdapterBase


class Adapter(AdapterBase):
    name = "ios_logical"

    def probe(self):
        # Probe using libimobiledevice (if available)
        try:
            ideviceinfo = shutil.which("ideviceinfo") is not None
            if not ideviceinfo:
                return {
                    "ok": False,
                    "error": "libimobiledevice tools not found (ideviceinfo) - running in simulated mode",
                }
            out = subprocess.check_output(
                ["idevice_id", "-l"], stderr=subprocess.STDOUT
            ).decode()
            devices = [l.strip() for l in out.splitlines() if l.strip()]
            return {"ok": True, "devices": devices}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def extract(self, artifact_list, out_dir, metadata):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {"extracted": []}
        # If libimobiledevice tools are present, try a logical backup via idevicebackup2
        if shutil.which("idevicebackup2") is None:
            # Simulate: create Contacts.vcf and Messages.json
            sim_dir = out_dir / "simulated"
            sim_dir.mkdir(parents=True, exist_ok=True)
            (sim_dir / "Contacts.vcf").write_text(
                "BEGIN:VCARD\nFN:Alice\nTEL:+1111111\nEND:VCARD\n"
            )
            (sim_dir / "Messages.json").write_text(
                json.dumps([{"from": "+1111111", "body": "Hi"}])
            )
            summary["extracted"].extend(
                [
                    str(Path("simulated") / "Contacts.vcf"),
                    str(Path("simulated") / "Messages.json"),
                ]
            )
            return summary
        try:
            # real extraction path (best-effort)
            target = out_dir / "backup"
            subprocess.check_call(["idevicebackup2", "backup", str(target)])
            # list files pulled
            files = [
                str(p.relative_to(out_dir)) for p in target.rglob("*") if p.is_file()
            ]
            summary["extracted"].extend(files)
        except Exception as e:
            summary["error"] = str(e)
        return summary

    def status(self):
        return {"name": self.name, "status": "idle"}

    def abort(self):
        pass
