r"""
Location Intelligence (Clean Module)

Provides:
- Load GPS/location data from extraction results
- Simple clustering (DBSCAN) and visualization scaffolding
- Streamlit UI to explore locations for a selected case

Reads from reports/<case_id>/results.json -> data.location.gps_coordinates
"""
import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import datetime
import logging
from urllib.parse import urlparse, parse_qs

import streamlit as st

try:
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    from sklearn.cluster import DBSCAN  # type: ignore
except Exception:
    pd = None
    np = None
    DBSCAN = None

from modules.shared.utils import ArtifactPathBuilder, adb_root_access_message

try:
    from adapters.android_adb import AndroidADB  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AndroidADB = None  # type: ignore

# NEW: Import audit trail for intelligence findings
try:
    from modules.consent.portal import ConsentAuditTrail
except ImportError:
    ConsentAuditTrail = None  # Optional dependency


def _save_intelligence_findings(case_id: str, key: str, data: Any):
    """Saves intelligence findings to the case's results.json."""
    results_path = os.path.join('reports', case_id, 'results.json')
    if not os.path.exists(results_path):
        return

    try:
        with open(results_path, 'r') as f:
            main_results = json.load(f)

        main_results.setdefault('data', {}).setdefault('intelligence_findings', {})[key] = data

        with open(results_path, 'w') as f:
            json.dump(main_results, f, indent=2, default=str)
        
        # NEW: Record intelligence findings in audit trail
        if ConsentAuditTrail:
            try:
                ConsentAuditTrail.record_approval(
                    case_id=case_id,
                    decision=f"intelligence_{key}",
                    nominee_name="System",
                    device_id="INTELLIGENCE",
                    purpose=f"Location intelligence findings: {key.replace('_', ' ').title()}"
                )
            except Exception as audit_error:
                logging.warning(f"Failed to record intelligence audit trail: {audit_error}")
        
        st.toast(f"{key.replace('_', ' ').title()} findings saved to case report.")
    except Exception as e:
        st.error(f"Failed to save intelligence findings: {e}")


def _load_results_locations(case_id: str) -> List[Dict[str, Any]]:
    path = os.path.join('reports', case_id, 'results.json')
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            loc = data.get('data', {}).get('location', {})
            gps = loc.get('gps_coordinates', [])
            return gps if isinstance(gps, list) else []
        except Exception:
            return []
    # Fallback to adb artifact cache if available
    gps_path = ArtifactPathBuilder.resolve(case_id, 'android', 'location', 'gps_locations.json')
    if os.path.exists(gps_path):
        try:
            with open(gps_path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def _load_cell_towers(case_id: str) -> List[Dict[str, Any]]:
    path = os.path.join('reports', case_id, 'results.json')
    towers: List[Dict[str, Any]] = []
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            location_block = data.get('data', {}).get('location', {})
            tower_entries = location_block.get('cell_towers', [])
            if isinstance(tower_entries, list):
                towers = [entry for entry in tower_entries if isinstance(entry, dict)]
        except Exception:
            towers = []
    if towers:
        return towers

    tower_path = ArtifactPathBuilder.resolve(case_id, 'android', 'location', 'cell_towers.json')
    if os.path.exists(tower_path):
        try:
            with open(tower_path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            if isinstance(data, list):
                towers = [entry for entry in data if isinstance(entry, dict)]
        except Exception:
            pass
    return towers


def _adb_pull_location(case_id: str) -> Dict[str, Any]:
    if AndroidADB is None:
        return {'status': 'error', 'message': 'ADB module not available.'}

    adb = AndroidADB()
    summary = adb.device_summary()
    if not summary.get('installed'):
        return {'status': 'error', 'message': adb_root_access_message(summary, 'Location extraction')}
    if not summary.get('connected'):
        return {'status': 'error', 'message': adb_root_access_message(summary, 'Location extraction')}

    loc_dir = ArtifactPathBuilder.resolve(case_id, 'android', 'location', ensure_dir=True)
    try:
        dumps = adb.extract_location_data(case_id, loc_dir)
    except Exception as exc:
        return {'status': 'error', 'message': f'Location extraction failed: {exc}'}

    if not dumps:
        return {'status': 'empty', 'message': adb_root_access_message(summary, 'Location services')}

    return {
        'status': 'ok',
        'message': f"Location artifacts saved under {loc_dir}",
        'path': loc_dir,
        'files': dumps
    }


class OpenCellIDResolver:
    """Resolves cell tower IDs to approximate locations using OpenCellID."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENCELLID_KEY')
        self.cache_file = os.path.join('cache', 'cell_towers.json')
        self.cache = {}
        self._load_cache()

    def _load_cache(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                try:
                    self.cache = json.load(f)
                except json.JSONDecodeError:
                    pass

    def resolve_cell_tower(self, mcc: int, mnc: int, cell_id: int) -> Optional[Dict]:
        cache_key = f"{mcc}-{mnc}-{cell_id}"
        if self.api_key and cache_key not in self.cache:
            try:
                import requests
                url = (
                    f"https://opencellid.org/cell/get?"
                    f"key={self.api_key}&mcc={mcc}"
                    f"&mnc={mnc}&cellid={cell_id}&format=json"
                )
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('lat') and data.get('lon'):
                        self.cache[cache_key] = {
                            'latitude': data['lat'],
                            'longitude': data['lon'],
                            'accuracy': data.get('range', 1000),
                            'source': 'opencellid'
                        }
            except Exception:
                pass
        return self.cache.get(cache_key)

    def _save_cache(self):
        """Persist cell tower cache."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception:
            pass


def cluster_locations(df: 'pd.DataFrame', eps_m: float = 200.0, min_samples: int = 5) -> 'pd.DataFrame':
    if pd is None or np is None or DBSCAN is None:
        return df.assign(cluster=-1)

    try:
        k = 111_320.0
        X = df[['latitude', 'longitude']].to_numpy(dtype=float)
        X_m = np.column_stack([
            X[:, 0] * k,
            X[:, 1] * k * np.cos(np.radians(X[:, 0].mean()))
        ])

        try:
            db = DBSCAN(eps=eps_m, min_samples=min_samples).fit(X_m)
            df['cluster'] = db.labels_
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Clustering failed, using fallback: %s", exc
            )
            df['cluster'] = (
                df['latitude'].round(3).astype(str)
                + '_'
                + df['longitude'].round(3).astype(str)
            ).astype('category').cat.codes

        return df
    except Exception:
        return df.assign(cluster=-1)


def _parse_tower_query(value: str) -> Optional[Tuple[int, int, int]]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        if value.startswith('http'):
            parsed = urlparse(value)
            query = parse_qs(parsed.query)
            def _pick(*keys):
                for key in keys:
                    if key in query and query[key]:
                        return query[key][0]
                return None
            mcc = _pick('mcc')
            mnc = _pick('mnc')
            cell = _pick('cellid', 'cid', 'cell')
            if mcc and mnc and cell:
                return int(mcc), int(mnc), int(cell)
        digits = [part for part in re.split(r'[^0-9]+', value) if part]
        if len(digits) >= 3:
            return int(digits[0]), int(digits[1]), int(digits[2])
    except Exception:
        return None
    return None


def _render_tower_recovery(case_id: str, cell_resolver: OpenCellIDResolver) -> None:
    st.markdown('#### 📶 Tower Recovery (GPS Fallback)')
    st.caption('Paste a GPS sharing link (e.g., WhatsApp), an OpenCellID URL, or manually enter coordinates when GPS fixes are unavailable.')
    query = st.text_input(
        'Tower query (OpenCellID URL, MCC-MNC-CI, or shared GPS link)',
        key=f'{case_id}_tower_query',
        placeholder='https://opencellid.org/cell/get?mcc=404&mnc=10&cellid=21021 or 404-10-21021'
    )
    col_resolve, col_info = st.columns([1, 2])
    with col_resolve:
        resolve_clicked = st.button('Resolve Tower', key=f'{case_id}_tower_resolve')
    with col_info:
        st.markdown('[Open OpenCellID search](https://opencellid.org/#zoom=16)')

    manual_col1, manual_col2 = st.columns(2)
    with manual_col1:
        manual_lat = st.text_input('Manual latitude', key=f'{case_id}_manual_lat', placeholder='12.9716')
    with manual_col2:
        manual_lon = st.text_input('Manual longitude', key=f'{case_id}_manual_lon', placeholder='77.5946')
    plot_manual = st.button('Plot Manual Coordinates', key=f'{case_id}_plot_manual')

    if resolve_clicked:
        parsed = _parse_tower_query(query)
        if not parsed:
            st.error('Unable to parse tower query. Provide OpenCellID link or MCC-MNC-CI (e.g., 404-10-21021).')
            return
        mcc, mnc, cell_id = parsed
        with st.spinner('Resolving tower location…'):
            result = cell_resolver.resolve_cell_tower(mcc, mnc, cell_id)
        if not result:
            st.warning('No tower location found. Ensure API key is configured or try another tower.')
            return
        st.success('Tower resolved via OpenCellID cache/API.')
        st.json({
            'mcc': mcc,
            'mnc': mnc,
            'cell_id': cell_id,
            'latitude': result['latitude'],
            'longitude': result['longitude'],
            'accuracy_m': result.get('accuracy'),
            'source': result.get('source', 'opencellid')
        })
        if pd is not None:
            tower_df = pd.DataFrame([
                {
                    'latitude': result['latitude'],
                    'longitude': result['longitude']
                }
            ])
            st.map(tower_df)
        else:
            st.info(f"Approximate location: ({result['latitude']}, {result['longitude']})")

    if plot_manual:
        try:
            lat = float(manual_lat.strip()) if manual_lat else None
            lon = float(manual_lon.strip()) if manual_lon else None
        except Exception:
            lat = lon = None
        if lat is None or lon is None:
            st.error('Enter valid numeric latitude and longitude values.')
        else:
            st.success('Plotted manual coordinates.')
            payload = {
                'latitude': lat,
                'longitude': lon,
                'source': 'manual_entry'
            }
            st.json(payload)
            if pd is not None:
                st.map(pd.DataFrame([{'latitude': lat, 'longitude': lon}]))
            else:
                st.info(f"Location: ({lat}, {lon})")


def _render_cell_tower_mode(
    case_id: str,
    towers: List[Dict[str, Any]],
    cell_resolver: OpenCellIDResolver,
    *,
    auto_trigger: bool = False
) -> None:
    if not towers:
        if auto_trigger:
            st.warning('No cell tower artifacts available for fallback mode.')
        return

    if auto_trigger:
        st.info('GPS stream inactive – switching to cell tower fallback mode.')
    else:
        st.caption('Cell tower fallback mode enabled.')

    resolved_rows: List[Dict[str, Any]] = []
    for entry in towers:
        try:
            mcc = entry.get('mcc') or entry.get('mobileCountryCode')
            mnc = entry.get('mnc') or entry.get('mobileNetworkCode')
            ci = entry.get('ci') or entry.get('cell_id') or entry.get('cellId')
            if mcc is None or mnc is None or ci is None:
                continue
            mcc_i, mnc_i, ci_i = int(mcc), int(mnc), int(ci)
            result = cell_resolver.resolve_cell_tower(mcc_i, mnc_i, ci_i)
            if result:
                resolved_rows.append({
                    'mcc': mcc_i,
                    'mnc': mnc_i,
                    'cell_id': ci_i,
                    'latitude': result['latitude'],
                    'longitude': result['longitude'],
                    'accuracy_m': result.get('accuracy'),
                    'source': result.get('source', 'opencellid')
                })
        except Exception:
            continue

    if not resolved_rows:
        st.warning('Cell towers present but none could be resolved to coordinates. Ensure OpenCellID key is set.')
        return

    st.dataframe(resolved_rows, use_container_width=True)
    if pd is not None:
        map_df = pd.DataFrame([
            {
                'latitude': row['latitude'],
                'longitude': row['longitude']
            }
            for row in resolved_rows
        ])
        st.map(map_df)
    else:
        coords = ', '.join(
            f"({row['latitude']:.4f}, {row['longitude']:.4f})" for row in resolved_rows
        )
        st.info(f"Resolved tower coordinates: {coords}")


def render_ui(case_id: str):
    st.markdown('### 🗺️ Location Intelligence')

    if pd is None:
        st.warning(
            'pandas/sklearn not available. '
            'Install dependencies to enable clustering.'
        )

    from modules.consent.models import ConsentManager
    consent_mgr = ConsentManager()
    cell_resolver = OpenCellIDResolver(consent_mgr.get_opencellid_key(case_id))

    points = _load_results_locations(case_id)
    towers = _load_cell_towers(case_id)
    if not points:
        st.info('No location data found in reports/<case_id>/results.json')
        if st.button('Pull latest location data via ADB'):
            with st.spinner('Collecting location data via ADB…'):
                adb_result = _adb_pull_location(case_id)
            status = adb_result.get('status')
            msg = adb_result.get('message', '')
            if status == 'ok':
                st.success(msg)
                points = _load_results_locations(case_id)
                towers = _load_cell_towers(case_id)
            elif status == 'empty':
                st.info(msg)
            else:
                st.error(msg)
        if not points:
            _render_cell_tower_mode(case_id, towers, cell_resolver, auto_trigger=True)
            _render_tower_recovery(case_id, cell_resolver)
            return

    # Convert to DataFrame
    df = pd.DataFrame(points) if pd else None

    if df is not None and not df.empty:
        st.caption(f"Loaded {len(df)} GPS points")

        # Optional time filtering if timestamp present
        if 'timestamp' in df.columns:
            st.markdown('#### Time Filter')
            try:
                df['timestamp'] = pd.to_datetime(
                    df['timestamp'], errors='coerce')
                min_ts, max_ts = df['timestamp'].min(), df['timestamp'].max()
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    start = st.datetime_input(
                        'Start time', value=min_ts.to_pydatetime() if pd.notnull(min_ts) else None)
                with col_t2:
                    end = st.datetime_input(
                        'End time', value=max_ts.to_pydatetime() if pd.notnull(max_ts) else None)
                if start and end:
                    mask = (df['timestamp'] >= pd.to_datetime(start)) & (
                        df['timestamp'] <= pd.to_datetime(end))
                    df = df[mask]
                    st.caption(
                        f"Filtered to {len(df)} points in selected window")
            except Exception:
                pass

        if not df.empty and {'latitude', 'longitude'}.issubset(df.columns):
            st.map(df[['latitude', 'longitude']])

        st.markdown('#### Clustering')
        eps = st.slider('Neighborhood radius (meters)', 50, 2000, 300, 50)
        min_samples = st.slider('Min samples per cluster', 3, 50, 5, 1)

        if st.button('Run Clustering') and pd is not None and not df.empty:
            df_clustered = cluster_locations(
                df.copy(),
                eps_m=float(eps),
                min_samples=int(min_samples)
            )
            st.dataframe(df_clustered[['latitude', 'longitude', 'cluster']])

            # Cluster summary
            if 'cluster' in df_clustered.columns:
                counts = df_clustered['cluster'].value_counts().reset_index()
                counts.columns = ['cluster', 'count']
                st.markdown('#### Cluster Summary')
                st.table(counts)
                cluster_summary = counts.to_dict('records')
                _save_intelligence_findings(case_id, 'location_clusters', cluster_summary)


            # Export
            out_json = {
                'case_id': case_id,
                'generated_at': datetime.datetime.now().isoformat(),
                'eps_m': float(eps),
                'min_samples': min_samples,
                'points': df_clustered.to_dict(orient='records')
            }
            st.download_button(
                'Forensics Export',
                json.dumps(out_json, indent=2),
                file_name=f'clusters_{case_id}.json',
                mime='application/json'
            )

        # Initialize resolver
        _render_tower_recovery(case_id, cell_resolver)

        # Geofence scaffold
        st.markdown('#### Geofence (Polygon)')
        with st.expander('Define geofence polygon (paste list of {latitude, longitude} or GeoJSON)'):
            poly_text = st.text_area(
                'Polygon coordinates JSON list',
                height=150,
                placeholder='[{"latitude":12.34,"longitude":56.78},...]'
            )
            geojson_file = st.file_uploader(
                'Or upload GeoJSON (Feature with Polygon)', type=['json', 'geojson'])

            polygon: List[Tuple[float, float]] = []
            try:
                if poly_text:
                    import json as _json
                    coords = _json.loads(poly_text)
                    polygon = [
                        (float(c['latitude']), float(c['longitude']))
                        for c in coords
                    ]
                elif geojson_file is not None:
                    import json as _json
                    gj = _json.loads(geojson_file.getvalue())
                    # Basic extraction of first polygon coords
                    coords = gj.get('features', [{}])[0].get(
                        'geometry', {}).get('coordinates', [])
                    if coords and isinstance(coords[0], list):
                        ring = coords[0]
                        polygon = [
                            (float(lon), float(lat))
                            for lon, lat in ring
                        ]
            except Exception:
                st.error('Invalid polygon/GeoJSON format')
                polygon = []

            if polygon:
                st.caption(f"Polygon with {len(polygon)} points loaded")
                # point-in-polygon (ray casting) fallback

                def _pip(pt: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
                    x, y = pt[1], pt[0]  # lon, lat
                    inside = False
                    n = len(poly)
                    for i in range(n):
                        y1, x1 = poly[i][0], poly[i][1]
                        y2, x2 = poly[(i + 1) % n][0], poly[i][1]
                        if (x1 > x) != (x2 > x):
                            xinters = (y2 - y1) * (x - x1) / (x2 - x1 + 1e-12) + y1
                            if y < xinters:
                                inside = not inside
                    return inside

                if not df.empty and {'latitude', 'longitude'}.issubset(df.columns):
                    df['_inside'] = df.apply(
                        lambda r: _pip((r['latitude'], r['longitude']), polygon),
                        axis=1
                    )
                    inside_df = df[df['_inside']]
                    st.success(
                        f"Found {len(inside_df)} points"
                    )
                    if not inside_df.empty:
                        st.map(inside_df[['latitude', 'longitude']])
                        st.dataframe(inside_df.drop(columns=['_inside']))

        if 'cell_towers' in df.columns and not df['cell_towers'].isnull().all():
            st.markdown('#### 📡 Cell Tower Resolution')
            if st.checkbox('Resolve cell towers to locations'):
                with st.spinner('Resolving cell towers...'):
                    df['resolved_location'] = df['cell_towers'].apply(
                        lambda x: (
                            cell_resolver.resolve_cell_tower(
                                x.get('mcc'),
                                x.get('mnc'),
                                x.get('ci')
                            ) if isinstance(x, dict) else None
                        )
                    )
                resolved = df[~df['resolved_location'].isnull()]
                if not resolved.empty:
                    locations = resolved['resolved_location'].apply(
                        lambda x: {'lat': x['latitude'], 'lon': x['longitude']}
                    )
                    st.map(locations)
    else:
        st.info('No location data found in reports/<case_id>/results.json')
        towers = _load_cell_towers(case_id)
        _render_cell_tower_mode(case_id, towers, cell_resolver, auto_trigger=True)
        _render_tower_recovery(case_id, cell_resolver)