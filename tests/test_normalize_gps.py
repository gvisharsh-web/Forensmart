# tests/test_normalize_gps.py
import pandas as pd
from src.location_utils import normalize_gps_df

def test_normalize_latlon_columns():
    df = pd.DataFrame({
        "latitude":[12.9716, 12.9750],
        "longitude":[77.5946, 77.5950],
        "timestamp":["2025-10-23 10:00:00", "2025-10-23 10:10:00"]
    })
    out = normalize_gps_df(df)
    assert "lat" in out.columns and "lon" in out.columns and "timestamp" in out.columns
    assert len(out) == 2
    assert out["lat"].iloc[0] == 12.9716
