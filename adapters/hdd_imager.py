# adapters/hdd_imager.py
import os, subprocess, json, shutil, hashlib, time
from pathlib import Path
from adapters.interface import AdapterBase

def sha256_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()

class Adapter(AdapterBase):
    name = "hdd_imager"

    def probe(self):
        # For now, probe local block devices (linux: /dev/sd*)
        devices = []
        try:
            for p in Path("/dev").glob("sd*"):
                devices.append(str(p))
        except Exception:
            pass
        return {"ok": True, "devices": devices}

    def extract(self, artifact_list, out_dir, metadata):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        summary = {"images": []}
        # This adapter will not run dangerous commands automatically. It provides a template to call imaging tools.
        # For safety, we simulate an image when no real device provided.
        try:
            if not artifact_list or artifact_list == ["simulate"]:
                # create small dummy image file
                img = out_dir / "simulated.img"
                with open(img, "wb") as f:
                    f.write(b"SIMULATEDDISKIMAGE")
                h = sha256_bytes(img.read_bytes())
                summary["images"].append({"path": str(img.name), "sha256": h})
            else:
                # user-provided device path expected in artifact_list[0], e.g. ['/dev/sda']
                dev = artifact_list[0]
                imgname = out_dir / (Path(dev).name + ".img")
                # Dangerous: user must run imaging command externally; here we just note intent.
                summary["note"] = f"Imaging of {dev} must be performed with dc3dd or similar. Place image at {imgname}."
            return summary
        except Exception as e:
            return {"error": str(e)}

    def status(self):
        return {"name": self.name, "status": "idle"}

    def abort(self):
        pass
