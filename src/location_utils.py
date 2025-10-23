# src/location_utils.py
import os
import json
import hashlib
import tempfile
from pathlib import Path
from typing import Union, Dict, List
import pandas as pd

def compute_hash(data: Union[str, bytes, dict, list]) -> str:
    """
    Compute SHA256 hex digest for strings, bytes, dicts/lists (JSON canonicalized).
    Returns 'N/A' on None.
    """
    if data is None:
        return "N/A"
    if isinstance(data, (dict, list)):
        b = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    elif isinstance(data, str):
        b = data.encode("utf-8")
    elif isinstance(data, bytes):
        b = data
    else:
        b = str(data).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def normalize_gps_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize an input GPS DataFrame to have columns:
      - 'timestamp' (pd.Timestamp)
      - 'lat' (float)
      - 'lon' (float)
    Accepts columns named 'latitude'/'longitude' or 'lat'/'lon' (case-insensitive).
    Raises ValueError if lat/lon columns cannot be found.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "lat", "lon"])

    df = df.copy()
    # lowercase columns map for robust matching
    cols_lower = {c.lower(): c for c in df.columns}

    # map lat/lon
    if "latitude" in cols_lower and "longitude" in cols_lower:
        df = df.rename(columns={cols_lower["latitude"]: "lat", cols_lower["longitude"]: "lon"})
    elif "lat" in cols_lower and "lon" in cols_lower:
        # normalize case
        df = df.rename(columns={cols_lower["lat"]: "lat", cols_lower["lon"]: "lon"})
    else:
        raise ValueError("GPS CSV must include 'latitude'/'longitude' or 'lat'/'lon' columns.")

    # normalize timestamp
    if "timestamp" in {c.lower(): c for c in df.columns}:
        # find original column name for timestamp
        ts_col = [c for c in df.columns if c.lower() == "timestamp"][0]
        df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
        # if parsing fails entirely, fill current time
        if df["timestamp"].isnull().all():
            df["timestamp"] = pd.Timestamp.now()
    else:
        df["timestamp"] = pd.Timestamp.now()

    # ensure lat/lon numeric
    df = df.dropna(subset=["lat", "lon"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df = df[["timestamp", "lat", "lon"]].reset_index(drop=True)
    return df


def save_snapshot_image(fig_or_bytes, out_dir: str, filename_prefix: str = "snapshot") -> Dict[str,str]:
    """
    Save a matplotlib figure (or bytes) into artifacts under out_dir.
    Returns dict: {"path": <fullpath>, "sha256": <hex>}.

    - fig_or_bytes: either a matplotlib.figure.Figure object, or bytes (PNG) or file-like object.
    - out_dir: directory path where file will be saved (created if needed).
    - filename_prefix: prefix added to filename; timestamp appended automatically.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    import time

    ts = int(time.time())
    filename = f"{filename_prefix}_{ts}.png"
    out_path = os.path.join(out_dir, filename)

    # If a matplotlib Figure provided, save it
    try:
        # delayed import to avoid heavy deps unless used
        from matplotlib.figure import Figure
        if isinstance(fig_or_bytes, Figure):
            fig_or_bytes.savefig(out_path, bbox_inches="tight")
        else:
            # treat as bytes-like or file-like
            if hasattr(fig_or_bytes, "read"):
                data = fig_or_bytes.read()
                with open(out_path, "wb") as f:
                    f.write(data)
            elif isinstance(fig_or_bytes, (bytes, bytearray)):
                with open(out_path, "wb") as f:
                    f.write(fig_or_bytes)
            else:
                # unknown type: try writing string representation
                with open(out_path, "wb") as f:
                    f.write(str(fig_or_bytes).encode("utf-8"))
    except Exception:
        # Fallback: write bytes if possible
        if isinstance(fig_or_bytes, (bytes, bytearray)):
            with open(out_path, "wb") as f:
                f.write(fig_or_bytes)
        else:
            # create an empty PNG placeholder
            with open(out_path, "wb") as f:
                f.write(b"")

    # compute hash
    try:
        with open(out_path, "rb") as fh:
            data = fh.read()
            sha = hashlib.sha256(data).hexdigest()
    except Exception:
        sha = "N/A"

    return {"path": out_path, "sha256": sha}


# small helper: convert list-of-dicts to canonical JSON and write file
def write_json_atomic(obj: Union[Dict, List], out_path: str) -> None:
    """
    Atomically write JSON to out_path (safe write: write temp -> rename).
    """
    out_dir = os.path.dirname(out_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=out_dir, suffix=".tmp")
    try:
        tmp.write(json.dumps(obj, sort_keys=True, indent=2).encode("utf-8"))
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, out_path)
    finally:
        if os.path.exists(tmp.name):
            try:
                os.remove(tmp.name)
            except Exception:
                pass
