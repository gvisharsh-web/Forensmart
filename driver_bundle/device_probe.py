# device_probe.py
# Lightweight probe utility to detect connected devices and tools availability.
import subprocess
import sys
import json
import platform
import shutil


def run_cmd(cmd):
    try:
        cp = subprocess.run(
            cmd, capture_output=True, text=True, shell=False, timeout=10
        )
        return {"rc": cp.returncode, "out": cp.stdout.strip(), "err": cp.stderr.strip()}
    except Exception as e:
        return {"rc": -1, "out": "", "err": str(e)}


def detect_adb():
    adb = shutil.which("adb")
    if not adb:
        return {"found": False, "note": "adb not in PATH"}
    r = run_cmd([adb, "version"])
    devices = run_cmd([adb, "devices", "-l"])
    return {
        "found": True,
        "version": r.get("out", ""),
        "devices": devices.get("out", ""),
        "raw": devices,
    }


def detect_fastboot():
    fb = shutil.which("fastboot")
    if not fb:
        return {"found": False}
    v = run_cmd([fb, "devices"])
    return {"found": True, "devices": v.get("out", "")}


def detect_idevice():
    ideviceinfo = shutil.which("ideviceinfo")
    idevice_id = shutil.which("idevice_id")
    if not idevice_id:
        return {"found": False, "note": "libimobiledevice not installed"}
    r = run_cmd([idevice_id, "-l"])
    return {"found": True, "devices": r.get("out", "")}


def detect_mtp_windows():
    # On Windows, check if any MTP device is present via PowerShell Get-PnpDevice filtered by 'MTP' in name
    if platform.system().lower() != "windows":
        return {"found": False, "note": "Windows-only check"}
    try:
        cmd = [
            "powershell",
            "-Command",
            "Get-PnpDevice -Status OK | Where-Object { $_.FriendlyName -like '*MTP*' -or $_.FriendlyName -like '*Android*' } | Select-Object -Property FriendlyName,InstanceId | ConvertTo-Json",
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        return {"found": True, "json": cp.stdout.strip(), "rc": cp.returncode}
    except Exception as e:
        return {"found": False, "err": str(e)}


def list_usb_ids():
    # Cross-platform check: on Windows use 'wmic', on linux use 'lsusb' if present
    if platform.system().lower() == "windows":
        try:
            cmd = ["wmic", "path", "Win32_USBControllerDevice", "get", "Dependent"]
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            return {"platform": "windows", "out": cp.stdout.strip()}
        except Exception as e:
            return {"platform": "windows", "err": str(e)}
    else:
        lsusb = shutil.which("lsusb")
        if not lsusb:
            return {"platform": "unix", "note": "lsusb not available"}
        cp = run_cmd([lsusb])
        return {"platform": "unix", "out": cp.get("out", "")}


def detect_disks():
    # List physical drives on Windows via wmic or on unix via /dev
    import platform
    import subprocess

    if platform.system().lower() == "windows":
        try:
            cp = subprocess.run(
                ["wmic", "diskdrive", "get", "model,serialnumber,caption"],
                capture_output=True,
                text=True,
                timeout=8,
            )
            return {"platform": "windows", "out": cp.stdout.strip()}
        except Exception as e:
            return {"platform": "windows", "err": str(e)}
    else:
        import glob

        devs = glob.glob("/dev/sd*")[:10]
        return {"platform": "unix", "devices": devs}


def run_probe():
    out = {}
    out["python"] = sys.executable
    out["platform"] = platform.platform()
    out["adb"] = detect_adb()
    out["fastboot"] = detect_fastboot()
    out["idevice"] = detect_idevice()
    out["mtp_windows"] = detect_mtp_windows()
    out["usb_ids"] = list_usb_ids()
    out["disks"] = detect_disks()
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    res = run_probe()
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        for k, v in res.items():
            print(f"--- {k} ---")
            print(v)
            print()
