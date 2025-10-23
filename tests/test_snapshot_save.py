# tests/test_snapshot_save.py
import os
import tempfile
from src.location_utils import save_snapshot_image, compute_hash

def test_save_snapshot_bytes_and_hash(tmp_path):
    out_dir = tmp_path / "artifacts"
    data = b"\x89PNG\r\n\x1a\n"  # minimal PNG header bytes
    res = save_snapshot_image(data, str(out_dir), filename_prefix="testimg")
    assert os.path.exists(res["path"])
    assert isinstance(res["sha256"], str) and len(res["sha256"]) == 64
    # file content hash should match computed hash
    with open(res["path"], "rb") as f:
        content = f.read()
    assert compute_hash(content) == res["sha256"]
