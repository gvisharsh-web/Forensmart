import subprocess
import os
from typing import Dict, Any, Optional, List

class AndroidADB:
    def __init__(self):
        self.adb_path = self._find_adb()

    def _find_adb(self) -> Optional[str]:
        # In a real scenario, you would search the PATH or use a configured path
        return "adb"

    def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        if not self.adb_path:
            raise FileNotFoundError("ADB executable not found.")
        return subprocess.run([self.adb_path] + cmd, capture_output=True, text=True, check=False)

    def device_summary(self) -> Dict[str, Any]:
        try:
            result = self._run_command(["devices"])
            if result.returncode != 0:
                return {'installed': True, 'connected': False, 'devices': [], 'error': result.stderr}
            
            lines = result.stdout.strip().split('\n')[1:]
            devices_list = []
            for line in lines:
                parts = line.split('\t')
                if len(parts) == 2:
                    serial = parts[0]
                    status = parts[1]
                    devices_list.append({'serial': serial, 'status': status})
            return {'installed': True, 'connected': bool(devices_list), 'devices': devices_list}
        except FileNotFoundError:
            return {'installed': False, 'connected': False, 'devices': [], 'error': 'ADB executable not found.'}

    def _is_device_authorized(self, device_id: Optional[str]) -> bool:
        try:
            summary = self.device_summary()
            if not summary.get('installed', False):
                logger.error("ADB is not installed or not in PATH")
                return False
                
            devices = summary.get('devices', [])
            if not devices:
                logger.error("No devices found. Make sure USB debugging is enabled and device is connected")
                return False
                
            if device_id is None:
                # If no specific device ID is provided, check if there's exactly one device
                if len(devices) == 1:
                    return devices[0].get('status') == 'device'
                logger.error("Multiple devices found but no device_id specified")
                return False
                
            # Check if the specified device is connected and authorized
            for device in devices:
                if device['serial'] == device_id:
                    if device['status'] == 'device':
                        return True
                    logger.error(f"Device {device_id} is not in 'device' state. Current state: {device['status']}")
                    return False
                    
            logger.error(f"Device {device_id} not found in connected devices")
            return False
            
        except Exception as e:
            logger.error(f"Error checking device authorization: {str(e)}")
            return False

    def pull_databases(self, case_id: str, dest_dir: str, device_id: Optional[str] = None) -> Dict[str, str]:
        try:
            os.makedirs(dest_dir, exist_ok=True)
            
            if not self._is_device_authorized(device_id):
                raise RuntimeError("Device not authorized. Please accept the RSA prompt on the handset.")
                
            # Get the actual device ID to use
            if device_id is None:
                devices = self.device_summary().get('devices', [])
                if len(devices) == 1:
                    device_id = devices[0]['serial']
                else:
                    raise RuntimeError("Multiple devices found. Please specify device_id.")
            
            # Common database paths on Android
            db_paths = {
                'sms': "/data/data/com.android.providers.telephony/databases/mmssms.db",
                'calllog': "/data/data/com.android.providers.contacts/databases/contacts2.db",
                'contacts': "/data/data/com.android.providers.contacts/databases/contacts2.db",
                'whatsapp': "/data/data/com.whatsapp/databases/msgstore.db"
            }
            
            results = {}
            for db_type, remote_path in db_paths.items():
                local_path = os.path.join(dest_dir, f"{db_type}.db")
                try:
                    # First check if the file exists on the device
                    check_cmd = ["shell", "ls", remote_path]
                    if device_id:
                        check_cmd.insert(0, "-s")
                        check_cmd.insert(1, device_id)
                        
                    check_result = self._run_command(check_cmd)
                    if check_result.returncode != 0:
                        logger.warning(f"Database not found: {remote_path}")
                        continue
                        
                    # Pull the database file
                    pull_cmd = ["pull", remote_path, local_path]
                    if device_id:
                        pull_cmd.insert(0, "-s")
                        pull_cmd.insert(1, device_id)
                        
                    pull_result = self._run_command(pull_cmd)
                    if pull_result.returncode == 0:
                        results[db_type] = local_path
                        logger.info(f"Successfully pulled {db_type} database to {local_path}")
                    else:
                        logger.error(f"Failed to pull {db_type} database: {pull_result.stderr}")
                        
                except Exception as e:
                    logger.error(f"Error pulling {db_type} database: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in pull_databases: {str(e)}")
            raise
        open(sms_db_path, 'a').close()
        open(calllog_db_path, 'a').close()
        return {"sms_db": sms_db_path, "calllog_db": calllog_db_path}

    def dump_content_providers(self, case_id: str, dest_dir: str, device_id: Optional[str] = None) -> Dict[str, str]:
        if not self._is_device_authorized(device_id):
            raise RuntimeError("Device not authorized. Please accept the RSA prompt on the handset.")
        # This is a placeholder. A real implementation would dump content providers.
        sms_dump_path = os.path.join(dest_dir, "sms_dump.txt")
        calllog_dump_path = os.path.join(dest_dir, "calllog_dump.txt")
        with open(sms_dump_path, "w") as f:
            f.write("Simulated SMS dump")
        with open(calllog_dump_path, "w") as f:
            f.write("Simulated call log dump")
        return {"sms_dump": sms_dump_path, "calllog_dump": calllog_dump_path}

    def extract_location_data(self, case_id: str, dest_dir: str, device_id: Optional[str] = None) -> Dict[str, str]:
        if not self._is_device_authorized(device_id):
            raise RuntimeError("Device not authorized. Please accept the RSA prompt on the handset.")
        # This is a placeholder.
        gps_path = os.path.join(dest_dir, "gps.json")
        with open(gps_path, "w") as f:
            f.write("[]")
        return {"gps_coordinates": gps_path}

    def pull_directory(self, remote_path: str, local_path: str, device_id: Optional[str] = None) -> bool:
        if not self._is_device_authorized(device_id):
            raise RuntimeError("Device not authorized. Please accept the RSA prompt on the handset.")
        # This is a placeholder.
        os.makedirs(local_path, exist_ok=True)
        # Create a dummy file to simulate a pull
        dummy_file = os.path.join(local_path, "dummy.txt")
        with open(dummy_file, "w") as f:
            f.write("dummy")
        return True

    def list_devices(self) -> List[Dict[str, Any]]:
        summary = self.device_summary()
        return summary.get('devices', [])
