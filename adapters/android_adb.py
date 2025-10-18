# adapters/android_adb.py
import os, subprocess, json, shutil, tempfile, sqlite3
from pathlib import Path
from adapters.interface import AdapterBase

class Adapter(AdapterBase):
    name = "android_adb"

    def probe(self):
        try:
            out = subprocess.check_output(["adb", "devices", "-l"], stderr=subprocess.STDOUT).decode()
            devices = [l for l in out.splitlines() if l.strip() and "device" in l]
            return {"ok": True, "raw": out, "devices": devices}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _simulate_contacts_sms(self, out_dir: Path):
        # Create simulated contacts and sms sqlite-like JSON files for testing.
        contacts = out_dir / "contacts2.db.json"
        sms = out_dir / "mmssms.db.json"
        contacts.write_text(json.dumps([{"id":1,"name":"Alice Example","number":"+1111111"},{"id":2,"name":"Bob Example","number":"+2222222"}]))
        sms.write_text(json.dumps([{"id":1,"thread_id":1,"address":"+1111111","body":"Hello world","date":"2025-01-01T00:00:00Z"}]))
        # Also create vcard and sms json outputs for easy consumption
        vcf = out_dir / "contacts.vcf"
        vcf.write_text("BEGIN:VCARD\nVERSION:3.0\nFN:Alice Example\nTEL:+1111111\nEND:VCARD\n")
        msgs = out_dir / "sms_messages.json"
        msgs.write_text(json.dumps([{"from":"+1111111","body":"Hello world","date":"2025-01-01T00:00:00Z"}]))
        return [str(contacts.name), str(sms.name), str(vcf.name), str(msgs.name)]

    def _pull_path(self, remote_path: str, local_target: Path):
        try:
            subprocess.check_call(["adb", "pull", remote_path, str(local_target)])
            return True, None
        except Exception as e:
            return False, str(e)

    def _parse_sqlite_contacts(self, db_path: Path, out_dir: Path):
        # Best-effort: attempt to read common contacts tables and export VCF and CSV
        try:
            con = sqlite3.connect(str(db_path))
            cur = con.cursor()
            # try a few common fields
            rows = []
            try:
                cur.execute("SELECT _id, display_name FROM contacts")
            except:
                try:
                    cur.execute("SELECT _id, display_name FROM raw_contacts")
                except:
                    # fallback: try any text fields
                    cur.execute("PRAGMA table_info(contacts)")
                    rows = []
            for r in cur.fetchmany(1000):
                rows.append(r)
            # Create a simple CSV and VCF output (best-effort)
            csv_out = out_dir / (db_path.stem + "_contacts.csv")
            vcf_out = out_dir / (db_path.stem + "_contacts.vcf")
            with open(csv_out, "w") as fcsv, open(vcf_out, "w") as fvcf:
                fcsv.write("id,name\n")
                for r in rows:
                    fid = r[0] if len(r)>0 else ""
                    name = r[1] if len(r)>1 else ""
                    fcsv.write(f"{fid},{name}\n")
                    fvcf.write("BEGIN:VCARD\nVERSION:3.0\n")
                    fvcf.write(f"FN:{name}\n")
                    fvcf.write("END:VCARD\n")
            return [str(csv_out.name), str(vcf_out.name)]
        except Exception as e:
            return []

    def _parse_sqlite_sms(self, db_path: Path, out_dir: Path):
        try:
            con = sqlite3.connect(str(db_path))
            cur = con.cursor()
            rows = []
            try:
                cur.execute("SELECT _id, thread_id, address, body, date FROM sms")
            except:
                try:
                    cur.execute("SELECT _id, address, body, date FROM sms")
                except:
                    rows = []
            for r in cur.fetchmany(5000):
                rows.append(r)
            out = out_dir / (db_path.stem + "_sms.json")
            msgs = []
            for r in rows:
                msgs.append({"id": r[0] if len(r)>0 else None, "address": r[2] if len(r)>2 else None, "body": r[3] if len(r)>3 else None, "date": r[4] if len(r)>4 else None})
            out.write_text(json.dumps(msgs))
            return [str(out.name)]
        except Exception as e:
            return []

    def extract(self, artifact_list, out_dir, metadata):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {"extracted": []}
        adb_path = shutil.which("adb")
        if not adb_path:
            sim_dir = out_dir / "simulated"
            sim_dir.mkdir(parents=True, exist_ok=True)
            produced = self._simulate_contacts_sms(sim_dir)
            summary["extracted"].extend([str(Path("simulated") / p) for p in produced])
            return summary

        try:
            candidates = {
                "contacts": ["/data/data/com.android.providers.contacts/databases/contacts2.db"],
                "sms": ["/data/data/com.android.providers.telephony/databases/mmssms.db"]
            }
            for artifact in artifact_list or ["contacts","sms","photos"]:
                if artifact == "contacts":
                    for rp in candidates["contacts"]:
                        local = out_dir / "contacts2.db"
                        ok, err = self._pull_path(rp, local)
                        if ok and local.exists():
                            # try parsing sqlite to vcf/csv
                            parsed = self._parse_sqlite_contacts(local, out_dir)
                            summary["extracted"].append(str(local.name))
                            summary["extracted"].extend(parsed)
                            break
                elif artifact == "sms":
                    for rp in candidates["sms"]:
                        local = out_dir / "mmssms.db"
                        ok, err = self._pull_path(rp, local)
                        if ok and local.exists():
                            parsed = self._parse_sqlite_sms(local, out_dir)
                            summary["extracted"].append(str(local.name))
                            summary["extracted"].extend(parsed)
                            break
                elif artifact == "photos" or artifact == "all":
                    try:
                        target = out_dir / "DCIM"
                        subprocess.check_call(["adb", "pull", "/sdcard/DCIM", str(target)])
                        files = [str(p.relative_to(out_dir)) for p in target.rglob("*") if p.is_file()]
                        summary["extracted"].extend(files)
                    except Exception as e:
                        pass
            if not summary["extracted"]:
                (out_dir / "no_artifacts_found.txt").write_text("No artifacts extracted (permissions/tooling may be limited).")
                summary["extracted"].append("no_artifacts_found.txt")
        except Exception as e:
            summary["error"] = str(e)
        return summary

    def status(self):
        return {"name": self.name, "status": "idle"}

    def abort(self):
        pass
