# adapters/ios_logical.py
import subprocess
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from adapters.interface import AdapterBase

# Import validators
try:
    from modules.shared.validators import validate_device_id, validate_file_path
    VALIDATORS_AVAILABLE = True
except ImportError:
    VALIDATORS_AVAILABLE = False

logger = logging.getLogger(__name__)


class Adapter(AdapterBase):
    name = "ios_logical"

    def probe(self):
        # Probe using libimobiledevice (if available)
        try:
            ideviceinfo = shutil.which("ideviceinfo") is not None
            if not ideviceinfo:
                logger.warning("⚠️ libimobiledevice tools not found (ideviceinfo) - running in simulated mode")
                return {
                    "ok": False,
                    "error": "libimobiledevice tools not found (ideviceinfo) - running in simulated mode",
                }
            out = subprocess.check_output(
                ["idevice_id", "-l"], stderr=subprocess.STDOUT,
                timeout=10
            ).decode()
            # Use consistent variable name 'line'
            devices = [line.strip() for line in out.splitlines() if line.strip()]
            
            # ✅ Validate each device_id
            valid_devices = []
            for device in devices:
                if validate_device_id(device):
                    valid_devices.append(device)
                else:
                    logger.warning(f"⚠️ Invalid device ID format: {device}")
            
            logger.info(f"✅ Found {len(valid_devices)} valid iOS devices")
            return {"ok": True, "devices": valid_devices}
        except subprocess.TimeoutExpired:
            logger.error("❌ Device probe timeout")
            return {"ok": False, "error": "Device probe timeout"}
        except Exception as e:
            logger.error(f"❌ Error probing devices: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}

    def extract(self, artifact_list, out_dir, metadata):
        """Extract artifacts from iOS device"""
        try:
            out_dir = Path(out_dir)
            
            # ✅ Validate output directory path
            if not validate_file_path(str(out_dir)):
                logger.error(f"❌ Invalid output directory: {out_dir}")
                return {"extracted": [], "error": "Invalid output directory"}
            
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
            
            # real extraction path (best-effort)
            target = out_dir / "backup"
            subprocess.check_call(["idevicebackup2", "backup", str(target)])
            # list files pulled
            files = [
                str(p.relative_to(out_dir)) for p in target.rglob("*") if p.is_file()
            ]
            summary["extracted"].extend(files)
            return summary
        except Exception as e:
            logger.error(f"❌ Error extracting iOS artifacts: {e}", exc_info=True)
            return {"extracted": [], "error": str(e)}

    # ============================================================================
    # FORENSIC AGENTS - Advanced Data Extraction for iOS
    # ============================================================================
    
    def extract_call_logs(self, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract call logs from iOS device via backup"""
        call_logs = []
        
        try:
            # ✅ Validate device_id if provided
            if device_id and not validate_device_id(device_id):
                logger.warning(f"⚠️ Invalid device ID: {device_id}")
            
            # Use idevicebackup2 to extract call history
            result = subprocess.run(
                ['idevicebackup2', 'unback', '--system', 'CallHistory.storedata'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if line.strip():
                        call_logs.append({'data': line.strip(), 'source': 'backup'})
                logger.info(f"✅ Extracted {len(call_logs)} call logs from iOS backup")
            else:
                logger.warning(f"⚠️ Could not extract call logs: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.error("❌ Call log extraction timeout")
        except Exception as e:
            logger.error(f"❌ Error extracting call logs: {e}", exc_info=True)
        
        return call_logs
    
    def extract_browser_history(self, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract browser history from Safari and other browsers"""
        browser_history = []
        
        try:
            # Safari history from backup
            result = subprocess.run(
                ['idevicebackup2', 'unback', '--system', 'Safari'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout:
                files = result.stdout.split('\n')
                for file in files:
                    if file.strip():
                        browser_history.append({'source': 'Safari', 'path': file.strip()})
            
            # Chrome history from backup
            result = subprocess.run(
                ['idevicebackup2', 'unback', '--system', 'Chrome'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout:
                files = result.stdout.split('\n')
                for file in files:
                    if file.strip():
                        browser_history.append({'source': 'Chrome', 'path': file.strip()})
        except Exception as e:
            browser_history.append({'error': str(e), 'source': 'error'})
        
        return browser_history
    
    def extract_installed_apps(self, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract list of installed apps from iOS device"""
        installed_apps = []
        
        try:
            # Get app list using ideviceinstaller
            result = subprocess.run(
                ['ideviceinstaller', '-l'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if line.strip() and '\t' in line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            app_id = parts[0].strip()
                            app_name = parts[1].strip() if len(parts) > 1 else app_id
                            installed_apps.append({
                                'package': app_id,
                                'name': app_name,
                                'type': 'app'
                            })
        except Exception as e:
            installed_apps.append({'error': str(e), 'source': 'error'})
        
        return installed_apps
    
    def extract_wifi_networks(self, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract WiFi network history from iOS device"""
        wifi_networks = []
        
        try:
            # Get WiFi networks from device settings backup
            result = subprocess.run(
                ['idevicebackup2', 'unback', '--system', 'WiFi'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'SSID' in line or 'BSSID' in line:
                        wifi_networks.append({'data': line.strip()})
        except Exception as e:
            wifi_networks.append({'error': str(e), 'source': 'error'})
        
        return wifi_networks
    
    def extract_system_logs(self, device_id: Optional[str] = None, lines: int = 1000) -> List[Dict[str, Any]]:
        """Extract system logs from iOS device"""
        system_logs = []
        
        try:
            # Get system logs using idevicesyslog
            result = subprocess.run(
                ['idevicesyslog', '-n', str(lines)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout:
                log_lines = result.stdout.split('\n')
                for line in log_lines:
                    if line.strip():
                        system_logs.append({'log': line.strip()})
        except Exception as e:
            system_logs.append({'error': str(e), 'source': 'error'})
        
        return system_logs
    
    def extract_whatsapp_artifacts(self, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract WhatsApp artifacts from iOS device (messages, media, databases)"""
        whatsapp_artifacts = []
        
        try:
            # Extract WhatsApp data from backup
            result = subprocess.run(
                ['idevicebackup2', 'unback', '--system', 'WhatsApp'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout:
                files = result.stdout.split('\n')
                for file in files:
                    if file.strip():
                        whatsapp_artifacts.append({
                            'source': 'WhatsApp',
                            'path': file.strip(),
                            'access': 'backup',
                            'type': 'artifact'
                        })
            
            # Also try to extract from app container
            result = subprocess.run(
                ['idevicebackup2', 'unback', '--app', 'net.whatsapp.WhatsApp'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout:
                files = result.stdout.split('\n')
                for file in files:
                    if file.strip():
                        whatsapp_artifacts.append({
                            'source': 'WhatsApp',
                            'path': file.strip(),
                            'access': 'app_container',
                            'type': 'artifact'
                        })
        except Exception as e:
            whatsapp_artifacts.append({'error': str(e), 'source': 'WhatsApp'})
        
        return whatsapp_artifacts
    
    def extract_instagram_artifacts(self, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract Instagram artifacts from iOS device (photos, videos, messages, databases)"""
        instagram_artifacts = []
        
        try:
            # Extract Instagram data from backup
            result = subprocess.run(
                ['idevicebackup2', 'unback', '--system', 'Instagram'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout:
                files = result.stdout.split('\n')
                for file in files:
                    if file.strip():
                        instagram_artifacts.append({
                            'source': 'Instagram',
                            'path': file.strip(),
                            'access': 'backup',
                            'type': 'artifact'
                        })
            
            # Also try to extract from app container
            result = subprocess.run(
                ['idevicebackup2', 'unback', '--app', 'com.instagram.android'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout:
                files = result.stdout.split('\n')
                for file in files:
                    if file.strip():
                        instagram_artifacts.append({
                            'source': 'Instagram',
                            'path': file.strip(),
                            'access': 'app_container',
                            'type': 'artifact'
                        })
        except Exception as e:
            instagram_artifacts.append({'error': str(e), 'source': 'Instagram'})
        
        return instagram_artifacts
    
    def extract_messaging_app_artifacts(self, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract messaging app artifacts from iOS (Telegram, Signal, Messenger)"""
        messaging_artifacts = []
        
        messaging_apps = {
            'Telegram': ['org.telegram.messenger', 'Telegram'],
            'Signal': ['org.signal.Signal', 'Signal'],
            'Facebook Messenger': ['com.facebook.Messenger', 'Messenger']
        }
        
        try:
            for app_name, identifiers in messaging_apps.items():
                for identifier in identifiers:
                    # Try system backup
                    result = subprocess.run(
                        ['idevicebackup2', 'unback', '--system', identifier],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode == 0 and result.stdout:
                        files = result.stdout.split('\n')
                        for file in files:
                            if file.strip():
                                messaging_artifacts.append({
                                    'source': app_name,
                                    'path': file.strip(),
                                    'access': 'backup',
                                    'type': 'artifact'
                                })
                    
                    # Try app container
                    result = subprocess.run(
                        ['idevicebackup2', 'unback', '--app', identifier],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode == 0 and result.stdout:
                        files = result.stdout.split('\n')
                        for file in files:
                            if file.strip():
                                messaging_artifacts.append({
                                    'source': app_name,
                                    'path': file.strip(),
                                    'access': 'app_container',
                                    'type': 'artifact'
                                })
        except Exception as e:
            messaging_artifacts.append({'error': str(e), 'source': 'Messaging Apps'})
        
        return messaging_artifacts
    
    def extract_media_files(self, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract media files from iOS device using idevicebackup2"""
        media_files = []
        
        try:
            # Extract photos and videos from device backup
            result = subprocess.run(
                ['idevicebackup2', 'unback', '--system', 'MediaLibrary'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout:
                files = result.stdout.split('\n')
                for file_path in files:
                    if file_path.strip():
                        file_name = file_path.split('/')[-1]
                        file_ext = file_name.split('.')[-1].lower()
                        
                        # Check if it's a media file
                        if file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'mp4', 'avi', 'mkv', 'mov', 'mp3', 'wav', 'aac', 'm4a']:
                            media_files.append({
                                'path': file_path.strip(),
                                'name': file_name,
                                'type': 'photo' if file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'] else 'video' if file_ext in ['mp4', 'avi', 'mkv', 'mov'] else 'audio',
                                'extension': file_ext,
                                'source': 'iOS Device',
                                'access': 'backup'
                            })
        except Exception as e:
            logger.warning(f"Error extracting media files from iOS: {e}")
        
        return media_files
    
    def extract_all_forensic_data(self, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Extract all forensic data from iOS device"""
        forensic_data = {
            'call_logs': self.extract_call_logs(device_id),
            'browser_history': self.extract_browser_history(device_id),
            'installed_apps': self.extract_installed_apps(device_id),
            'wifi_networks': self.extract_wifi_networks(device_id),
            'system_logs': self.extract_system_logs(device_id),
            'whatsapp_artifacts': self.extract_whatsapp_artifacts(device_id),
            'instagram_artifacts': self.extract_instagram_artifacts(device_id),
            'messaging_app_artifacts': self.extract_messaging_app_artifacts(device_id),
            'media_files': self.extract_media_files(device_id)
        }
        return forensic_data

    def status(self):
        return {"name": self.name, "status": "idle"}

    def abort(self):
        pass
