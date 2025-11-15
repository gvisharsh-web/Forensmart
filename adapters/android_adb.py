"""
Android ADB Adapter

Provides a minimal interface for connectivity, root detection, database pulling,
and content provider dumps to support extraction in the orchestrator.

This implementation uses subprocess to call adb if present on PATH. All operations
are best-effort and return informative results instead of raising when possible.
"""
import os
import subprocess
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Callable

class AndroidADB:
    _DEFAULT_CANDIDATES = [
        'adb',
        os.path.join('driver_bundle', 'platform-tools', 'adb.exe'),
        os.path.join('driver_bundle', 'platform-tools', 'adb'),
        os.path.join('platform-tools', 'adb.exe'),
        os.path.join('platform-tools', 'adb'),
    ]

    def __init__(self, adb_path: Optional[str] = None):
        self.adb_path = self._resolve_adb_path(adb_path)

    def _resolve_adb_path(self, override: Optional[str]) -> str:
        candidates: List[str] = []
        if override:
            candidates.append(override)

        env_path = os.getenv('ADB_PATH')
        if env_path:
            candidates.append(env_path)

        for candidate in self._DEFAULT_CANDIDATES:
            candidates.append(candidate)

        for path in candidates:
            if not path:
                continue
            expanded = os.path.abspath(path) if os.path.sep in path else path
            if os.path.isfile(expanded) or path == 'adb':
                return expanded
        return 'adb'

    def is_installed(self) -> bool:
        code, _, _ = self._run([self.adb_path, 'version'])
        return code == 0

    def _run(self, args: List[str], timeout: int = 30, max_retries: int = 3) -> Tuple[int, str, str]:
        last_error = None
        for attempt in range(max_retries):
            try:
                proc = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                )
                out, err = proc.communicate(timeout=timeout * (attempt + 1))  # Exponential backoff
                
                # Check for common ADB errors that can be retried
                if 'device not found' in err.lower() or 'device offline' in err.lower():
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Short delay before retry
                        continue
                
                return proc.returncode, out or '', err or ''
                
            except subprocess.TimeoutExpired:
                last_error = f"Command timed out after {timeout * (attempt + 1)}s"
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                    
        return 1, '', f"Failed after {max_retries} attempts: {last_error}"

    def _prepare(self, base_args: List[str], device_id: Optional[str] = None) -> List[str]:
        cmd = [self.adb_path]
        if device_id:
            cmd.extend(['-s', device_id])
        cmd.extend(base_args)
        return cmd

    def list_devices(self) -> List[Dict[str, str]]:
        """Return attached devices with their connection status."""
        code, out, _ = self._run([self.adb_path, 'devices'])
        if code != 0:
            return []

        devices: List[Dict[str, str]] = []
        for raw_line in out.splitlines()[1:]:  # Skip header line
            line = raw_line.strip()
            if not line:
                continue
            if '\t' in line:
                serial, status = line.split('\t', 1)
            else:
                serial, status = line, ''
            devices.append({'serial': serial, 'status': status})
        return devices

    def get_default_device(self) -> Optional[Dict[str, str]]:
        """Return the first authorised device entry, if any."""
        for device in self.list_devices():
            if device.get('status') == 'device':
                return device
        return None

    def is_connected(self, device_id: Optional[str] = None) -> bool:
        devices = self.list_devices()
        if device_id:
            return any(d.get('serial') == device_id and d.get('status') == 'device' for d in devices)
        return any(d.get('status') == 'device' for d in devices)

    def device_summary(self) -> Dict[str, Any]:
        """Return diagnostic information about adb state."""
        installed = self.is_installed()
        devices = self.list_devices()
        default = self.get_default_device()
        return {
            'installed': installed,
            'devices': devices,
            'default_device': default,
            'connected': bool(default),
            'path': self.adb_path,
        }

    def pull_directory(
        self,
        remote_path: str,
        local_dir: str,
        *,
        device_id: Optional[str] = None,
        recursive: bool = True,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Tuple[bool, str]:
        """
        Pull a remote directory to a local destination with progress reporting.
        
        Args:
            remote_path: Path on device to pull from
            local_dir: Local directory to save files to
            device_id: Optional device ID
            recursive: Whether to pull directories recursively
            progress_callback: Optional callback for progress updates (filename, current, total)
            
        Returns:
            Tuple of (success, message)
        """
        if not remote_path:
            return False, "No remote path specified"

        remote = remote_path.rstrip('/')
        os.makedirs(local_dir, exist_ok=True)

        # First, get list of files to transfer
        list_cmd = self._prepare(['shell', 'find', remote, '-type', 'f'], device_id)
        code, out, err = self._run(list_cmd, timeout=60)
        
        if code != 0:
            return False, f"Failed to list remote directory: {err or out}"
            
        remote_files = [f.strip() for f in out.splitlines() if f.strip()]
        total_files = len(remote_files)
        
        if not remote_files:
            return False, "No files found in remote directory"
            
        # Pull files one by one with progress
        success_count = 0
        for i, remote_file in enumerate(remote_files, 1):
            try:
                rel_path = os.path.relpath(remote_file, remote)
                local_path = os.path.join(local_dir, rel_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                if progress_callback:
                    progress_callback(rel_path, i, total_files)
                    
                pull_cmd = self._prepare(['pull', remote_file, local_path], device_id)
                code, out, err = self._run(pull_cmd, timeout=300)
                
                if code != 0 or not os.path.exists(local_path):
                    logger.warning(f"Failed to pull {remote_file}: {err or out}")
                else:
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Error pulling {remote_file}: {str(e)}")
                
        success = success_count > 0
        msg = f"Pulled {success_count} of {total_files} files"
        return success, msg

    def is_rooted(self) -> bool:
        # Attempt to run id as root
        code, out, err = self._run([self.adb_path, 'shell', 'id'])
        if code != 0:
            return False
        return 'uid=0(' in out

    def pull_databases(self, case_id: str, out_dir: str) -> Dict[str, str]:
        """
        Pull common SQLite DBs from the device with verification.
        
        Args:
            case_id: Case ID for logging
            out_dir: Local directory to save databases
            
        Returns:
            Dictionary mapping logical names to local file paths
        """
        os.makedirs(out_dir, exist_ok=True)
        pulled: Dict[str, str] = {}
        
        # Common DB paths (may vary by device)
        paths = {
            'sms_db': '/data/data/com.android.providers.telephony/databases/mmssms.db',
            'calllog_db': '/data/data/com.android.providers.contacts/databases/calllog.db',
            'contacts_db': '/data/data/com.android.providers.contacts/databases/contacts2.db',
            'whatsapp_msg': '/data/data/com.whatsapp/databases/msgstore.db',
            'whatsapp_wa': '/data/data/com.whatsapp/databases/wa.db',
        }
        
        # First check if we have root access
        root_check = self._run([self.adb_path, 'shell', 'su -c "echo root_check"'])
        has_root = root_check[0] == 0 and 'root_check' in root_check[1]
        
        for key, remote in paths.items():
            local = os.path.join(out_dir, f'{key}.sqlite')
            
            # Check if file exists on device
            check_cmd = [self.adb_path, 'shell', 'ls', '-la', remote]
            code, out, _ = self._run(check_cmd)
            
            if code != 0 or 'No such file' in out:
                logger.debug(f"Database not found: {remote}")
                continue
                
            # Try to pull with root if available
            pull_success = False
            if has_root:
                # Create temp file in /data/local/tmp
                tmp_file = f'/data/local/tmp/{os.path.basename(remote)}'
                # Copy with root
                self._run([self.adb_path, 'shell', f'su -c "cp {remote} {tmp_file} && chmod 666 {tmp_file}"'])
                # Pull the temp file
                code, _, _ = self._run([self.adb_path, 'pull', tmp_file, local], timeout=60)
                if code == 0 and os.path.exists(local):
                    pull_success = True
                # Clean up
                self._run([self.adb_path, 'shell', 'rm', '-f', tmp_file])
            
            # Fallback to non-root pull
            if not pull_success:
                code, _, _ = self._run([self.adb_path, 'pull', remote, local], timeout=60)
                pull_success = code == 0 and os.path.exists(local)
            
            # Verify SQLite file
            if pull_success and self._verify_sqlite(local):
                pulled[key] = local
                logger.info(f"Successfully pulled database: {key} -> {local}")
            else:
                logger.warning(f"Failed to pull/verify database: {remote}")
                if os.path.exists(local):
                    os.remove(local)
                    
        return pulled
        
    def _verify_sqlite(self, path: str) -> bool:
        """Basic SQLite file verification"""
        try:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return False
                
            # Check SQLite header (first 16 bytes should be "SQLite format 3\0")
            with open(path, 'rb') as f:
                header = f.read(16)
                if header != b'SQLite format 3\x00':
                    return False
                    
            return True
            
        except Exception:
            return False

    def dump_content_providers(self, case_id: str, out_dir: str) -> Dict[str, str]:
        """Fallback to content queries for SMS/Calls. Writes CSV-like dumps."""
        os.makedirs(out_dir, exist_ok=True)
        dumped: Dict[str, str] = {}
        # SMS content provider
        sms_out = os.path.join(out_dir, 'sms_dump.txt')
        code, out, err = self._run([self.adb_path, 'shell', 'content', 'query', '--uri', 'content://sms'])
        if code == 0 and out:
            with open(sms_out, 'w', encoding='utf-8') as f:
                f.write(out)
            dumped['sms_dump'] = sms_out
        # Call log content provider
        calls_out = os.path.join(out_dir, 'calllog_dump.txt')
        code, out, err = self._run([self.adb_path, 'shell', 'content', 'query', '--uri', 'content://call_log/calls'])
        if code == 0 and out:
            with open(calls_out, 'w', encoding='utf-8') as f:
                f.write(out)
            dumped['calllog_dump'] = calls_out
        return dumped

    def extract_location_data(self, case_id: str, out_dir: str) -> Dict[str, str]:
        """Extract GPS, WiFi, and cell tower data from dumpsys location output."""
        os.makedirs(out_dir, exist_ok=True)
        results = {}
        
        # Get full location data dump
        code, out, _ = self._run([self.adb_path, 'shell', 'dumpsys', 'location'])
        if code != 0:
            return results
            
        # Parse GPS locations
        gps_data = []
        if 'Last Known Locations:' in out:
            locations_section = out.split('Last Known Locations:')[1].split('\n\n')[0]
            for line in locations_section.splitlines():
                if 'Provider: gps' in line:
                    parts = line.split()
                    try:
                        gps_data.append({
                            'latitude': float(parts[parts.index('lat')+1]),
                            'longitude': float(parts[parts.index('lon')+1]),
                            'timestamp': int(parts[parts.index('time')+1]),
                            'accuracy': float(parts[parts.index('acc')+1]) if 'acc' in parts else None
                        })
                    except (ValueError, IndexError):
                        continue
        
        # Parse WiFi networks
        wifi_data = []
        if 'Wifi Scan Results:' in out:
            wifi_section = out.split('Wifi Scan Results:')[1].split('\n\n')[0]
            for line in wifi_section.splitlines():
                if 'BSSID:' in line:
                    parts = line.split()
                    wifi_data.append({
                        'bssid': parts[parts.index('BSSID:')+1],
                        'ssid': parts[parts.index('SSID:')+1] if 'SSID:' in parts else None,
                        'rssi': int(parts[parts.index('RSSI:')+1]) if 'RSSI:' in parts else None
                    })
        
        # Parse cell towers
        cell_data = []
        if 'Cell Infos:' in out:
            cell_section = out.split('Cell Infos:')[1].split('\n\n')[0]
            for line in cell_section.splitlines():
                if 'CellIdentityLte' in line:
                    parts = line.replace('=', ' ').split()
                    cell_data.append({
                        'mcc': int(parts[parts.index('mMcc')+1]) if 'mMcc' in parts else None,
                        'mnc': int(parts[parts.index('mMnc')+1]) if 'mMnc' in parts else None,
                        'ci': int(parts[parts.index('mCi')+1]) if 'mCi' in parts else None,
                        'tac': int(parts[parts.index('mTac')+1]) if 'mTac' in parts else None
                    })
        
        # Save results
        gps_out = os.path.join(out_dir, 'gps_locations.json')
        with open(gps_out, 'w') as f:
            json.dump(gps_data, f)
        results['gps_coordinates'] = gps_out
        
        wifi_out = os.path.join(out_dir, 'wifi_networks.json')
        with open(wifi_out, 'w') as f:
            json.dump(wifi_data, f)
        results['wifi_networks'] = wifi_out
        
        cell_out = os.path.join(out_dir, 'cell_towers.json')
        with open(cell_out, 'w') as f:
            json.dump(cell_data, f)
        results['cell_towers'] = cell_out
        
        return results
