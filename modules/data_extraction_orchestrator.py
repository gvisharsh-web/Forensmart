from __future__ import annotations

"""
Data Extraction Orchestrator for ForenSmart
============================================

Central hub for one-click data extraction that coordinates all forensic analysis modules
based on consent levels and investigation requirements.

This module provides:
- Unified data extraction API
- Consent-level based module orchestration
- Progress tracking and error handling
- Result aggregation and reporting

Author: ForenSmart Development Team
"""

import os
import shutil
import streamlit as st
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import json
import logging
import sqlite3
from pathlib import Path

# Import ForenSmart modules
from .consent import ConsentManager, ConsentLevel
from .shared_utils import (
    ArtifactPathBuilder,
    ResultsRepository,
    parse_sms_dump,
    parse_calls_dump,
)
from modules.file_handler import file_handler
from modules.extraction_validator import ExtractionValidator
from modules.extraction_progress import ProgressManager
from modules.approval_sync import ApprovalSync
from modules.device_manager import DeviceManager
from modules.consent_portal import ConsentAuditTrail, ConsentPortalEnhancer  # NEW: Integrated consent portal

try:
    from adapters.android_adb import AndroidADB  # type: ignore
except Exception:  # pragma: no cover - adb optional
    AndroidADB = None  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractionModule:
    """Base class for all extraction modules"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.is_available = True

    def extract(self, device_id: str, **kwargs) -> Dict[str, Any]:
        """Extract data from device - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement extract method")

    def get_status(self) -> Dict[str, Any]:
        """Get module status and capabilities"""
        return {
            'name': self.name,
            'description': self.description,
            'available': self.is_available,
            'last_updated': getattr(self, '_last_updated', None)
        }


class DeviceInfoExtractor(ExtractionModule):
    """Extracts basic device information"""

    def __init__(self):
        super().__init__("Device Information", "Basic device identification and hardware specs")

    def extract(self, device_id: str, **kwargs) -> Dict[str, Any]:
        """Extract basic device information"""
        try:
            # Simulate device info extraction
            device_info = {
                'device_id': device_id,
                'model': f'Device-{device_id[-4:]}',
                'manufacturer': 'Android Device',
                'os_version': 'Android 13',
                'serial_number': f'SN{device_id[-8:]}',
                'imei': device_id if len(device_id) > 10 else f'IMEI{device_id}',
                'storage_capacity': '128GB',
                'ram': '8GB',
                'extracted_at': datetime.now().isoformat()
            }

            logger.info(f"Device info extracted for {device_id}")
            return {'status': 'success', 'data': device_info}

        except Exception as e:
            logger.error(f"Device info extraction failed: {e}")
            return {'status': 'error', 'error': str(e)}


class CommunicationExtractor(ExtractionModule):
    """Extracts communication data (SMS, calls, contacts)"""

    def __init__(self):
        super().__init__("Communications", "SMS, call logs, contacts, and messaging data")

    def extract(self, device_id: str, **kwargs) -> Dict[str, Any]:
        """Extract communication data with consent checks and resilient ADB-backed retrieval."""
        try:
            consent_manager = kwargs.get('consent_manager')
            case_id: Optional[str] = kwargs.get('case_id')
            orchestrator = kwargs.get('orchestrator')

            # Consent checks
            if consent_manager and case_id:
                messaging_consent = consent_manager.verify_messaging_consent(
                    case_id, device_id, require_pin=True)

                if messaging_consent.get('vault_required') and not messaging_consent.get('vault_verified'):
                    secure_message = (
                        "Provide the password pattern or PIN to unlock messaging data. "
                        "Authentication is verified securely without revealing credentials "
                        "to investigators. Confirm protection by validating your PIN, "
                        "pattern, or password."
                    )
                    return {
                        'status': 'vault_verification_required',
                        'message': secure_message,
                        'vault_entries': messaging_consent.get('vault_entries', []),
                        'auth_prompt': 'PIN/PATTERN/PASSWORD',
                        'consent_details': messaging_consent
                    }

                if not messaging_consent.get('access_granted', False):
                    denial_message = messaging_consent.get(
                        'message', 'Messaging access denied - consent verification required')
                    return {
                        'status': 'messaging_consent_denied',
                        'message': denial_message,
                        'consent_details': messaging_consent,
                        'required_action': messaging_consent.get('reason', 'verify_consent')
                    }

            sms: List[Dict[str, Any]] = []
            calls: List[Dict[str, Any]] = []
            method = 'simulated'
            errors: List[str] = []
            artifacts: Dict[str, str] = {}

            adb = orchestrator._get_adb() if orchestrator else None
            if orchestrator and adb:
                device_check = orchestrator._ensure_device(device_id)
                if not device_check.get('ok'):
                    errors.append(device_check.get('message', 'Unknown ADB connectivity issue.'))
                else:
                    db_dir = ArtifactPathBuilder.resolve(case_id, 'android', 'dbs', ensure_dir=True)
                    dump_dir = ArtifactPathBuilder.resolve(case_id, 'android', 'provider_dumps', ensure_dir=True)

                    try:
                        pulled = adb.pull_databases(case_id or 'unknown', db_dir)
                        if pulled:
                            method = 'adb_dbs'
                            sms_path = pulled.get('sms_db')
                            if sms_path:
                                sms = self._parse_sms_sqlite(sms_path)
                                artifacts['sms_database'] = sms_path
                            call_path = pulled.get('calllog_db')
                            if call_path:
                                calls = self._parse_calllog_sqlite(call_path)
                                artifacts['calllog_database'] = call_path
                    except Exception as exc:
                        errors.append(f'ADB sqlite pull failed: {exc}')

                    try:
                        dumps = adb.dump_content_providers(case_id or 'unknown', dump_dir)
                        if dumps:
                            if method == 'simulated':
                                method = 'content_provider'
                            sms_dump = dumps.get('sms_dump')
                            if sms_dump and not sms:
                                sms = parse_sms_dump(sms_dump)
                            call_dump = dumps.get('calllog_dump')
                            if call_dump and not calls:
                                calls = parse_calls_dump(call_dump)
                                for row in calls:
                                    duration = row.get('duration')
                                    if isinstance(duration, str) and duration.isdigit():
                                        row['duration'] = int(duration)
                            artifacts.update({k: v for k, v in dumps.items() if v})
                    except Exception as exc:
                        errors.append(f'ADB provider dump failed: {exc}')
            elif orchestrator and not adb:
                errors.append('AndroidADB adapter unavailable in current environment.')
            else:
                errors.append('Orchestrator context missing for communications extraction.')

            if not sms and not calls:
                errors.append('No communications retrieved from device. Ensure SMS/Call permissions are granted.')

            if artifacts:
                for ext in ['txt', 'sqlite']:
                    if ext not in file_handler.extension_database:
                        file_handler.register_custom_format(ext, 'pandas', 'data')

            artifact_counts = {
                key: len(value) if isinstance(value, list) else 1
                for key, value in artifacts.items()
            }

            comms_data = {
                'sms_messages': sms,
                'call_logs': calls,
                'contacts': [],
                'artifacts': artifacts,
                'artifact_counts': artifact_counts,
                'extracted_at': datetime.now().isoformat(),
                'extraction_method': method,
                'errors': errors,
            }
            logger.info(
                "Communication data extraction finished for %s via %s", device_id, method)
            return {'status': 'success', 'data': comms_data}

        except Exception as exc:
            logger.error("Communication extraction failed: %s", exc, exc_info=True)
            return {'status': 'error', 'error': str(exc)}

    @staticmethod
    def _parse_sms_sqlite(path: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if not path or not os.path.exists(path):
            return messages
        try:
            with sqlite3.connect(path) as conn:
                conn.row_factory = sqlite3.Row
                query = (
                    "SELECT address, date, date_sent, body, type, read, status"
                    " FROM sms ORDER BY date DESC LIMIT 1000"
                )
                for row in conn.execute(query):
                    messages.append({
                        'address': row['address'],
                        'body': row['body'],
                        'timestamp': row['date'],
                        'timestamp_sent': row['date_sent'],
                        'type': row['type'],
                        'read': row['read'],
                        'status': row['status'],
                    })
        except Exception as exc:
            logger.warning('Failed to parse SMS database %s: %s', path, exc)
        return messages

    @staticmethod
    def _parse_calllog_sqlite(path: str) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        if not path or not os.path.exists(path):
            return calls
        try:
            with sqlite3.connect(path) as conn:
                conn.row_factory = sqlite3.Row
                query = (
                    "SELECT number, duration, type, date, name, phone_account_address"
                    " FROM calls ORDER BY date DESC LIMIT 1000"
                )
                for row in conn.execute(query):
                    calls.append({
                        'number': row['number'],
                        'duration': row['duration'],
                        'type': row['type'],
                        'timestamp': row['date'],
                        'name': row['name'],
                        'account': row['phone_account_address'],
                    })
        except Exception as exc:
            logger.warning('Failed to parse call log database %s: %s', path, exc)
        return calls


class LocationExtractor(ExtractionModule):
    """Extracts location and GPS data"""

    def __init__(self):
        super().__init__("Location Data", "GPS coordinates, WiFi networks, and location history")

    def extract(self, device_id: str, **kwargs) -> Dict[str, Any]:
        """Extract location data via dumpsys and register generated artifacts."""
        try:
            case_id: Optional[str] = kwargs.get('case_id')
            orchestrator = kwargs.get('orchestrator')

            gps_points: List[Dict[str, Any]] = []
            wifi_networks: List[Dict[str, Any]] = []
            cell_towers: List[Dict[str, Any]] = []
            artifacts: Dict[str, str] = {}
            method = 'simulated'
            errors: List[str] = []

            adb = orchestrator._get_adb() if orchestrator else None
            if orchestrator and adb:
                device_check = orchestrator._ensure_device(device_id)
                if not device_check.get('ok'):
                    errors.append(device_check.get('message', 'Unknown ADB connectivity issue.'))
                else:
                    loc_dir = ArtifactPathBuilder.resolve(case_id, 'android', 'location', ensure_dir=True)
                    try:
                        dumps = adb.extract_location_data(case_id or 'unknown', loc_dir)
                        if dumps:
                            method = 'adb_dumpsys'
                            artifacts.update(dumps)
                            gps_points = self._load_json(dumps.get('gps_coordinates'))
                            wifi_networks = self._load_json(dumps.get('wifi_networks'))
                            cell_towers = self._load_json(dumps.get('cell_towers'))
                    except Exception as exc:
                        errors.append(f'Location dumpsys failed: {exc}')
            elif orchestrator and not adb:
                errors.append('AndroidADB adapter unavailable in current environment.')
            else:
                errors.append('Orchestrator context missing for location extraction.')

            if not gps_points and not wifi_networks and not cell_towers:
                errors.append('No location artifacts retrieved from device. Enable GPS/location permissions and re-run.')

            if artifacts:
                for ext in ['json']:
                    if ext not in file_handler.extension_database:
                        file_handler.register_custom_format(ext, 'builtins', 'data')

            artifact_counts = {
                key: len(value) if isinstance(value, list) else 1
                for key, value in artifacts.items()
            }

            location_data = {
                'gps_coordinates': gps_points,
                'wifi_networks': wifi_networks,
                'cell_towers': cell_towers,
                'artifacts': artifacts,
                'artifact_counts': artifact_counts,
                'extracted_at': datetime.now().isoformat(),
                'extraction_method': method,
                'errors': errors,
            }
            logger.info(
                "Location data extraction finished for %s via %s", device_id, method)
            return {'status': 'success', 'data': location_data}

        except Exception as exc:
            logger.error("Location extraction failed: %s", exc, exc_info=True)
            return {'status': 'error', 'error': str(exc)}

    @staticmethod
    def _load_json(path: Optional[str]) -> List[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            return []
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
                if isinstance(payload, list):
                    return payload
        except Exception as exc:
            logger.warning('Failed to load location artifact %s: %s', path, exc)
        return []


class SecurityExtractor(ExtractionModule):
    """Extracts security-related data"""

    def __init__(self):
        super().__init__("Security Data",
              "Passwords, biometrics, encryption keys, and security logs")

    def extract(self, device_id: str, **kwargs) -> Dict[str, Any]:
        """Extract security data"""
        try:
            # Simulate security data extraction (this would be highly restricted)
            security_data = {
                'password_hashes': [],  # Would require FULL consent
                'biometric_data': [],   # Would require LEGAL consent
                'encryption_keys': [],  # Would require LEGAL consent
                'security_logs': [
                    {
                        'timestamp': datetime.now().isoformat(),
                        'event': 'Device unlocked',
                        'method': 'PIN'
                    }
                ],
                'app_permissions': [
                    {
                        'app_name': 'Test App',
                        'package': 'com.test.app',
                        'version': '1.0.0',
                        'installed_date': datetime.now().isoformat()
                    }
                ],
                'extracted_at': datetime.now().isoformat()
            }

            logger.info(f"Security data extracted for {device_id}")
            return {'status': 'success', 'data': security_data}

        except Exception as e:
            logger.error(f"Security extraction failed: {e}")
            return {'status': 'error', 'error': str(e)}


class MediaExtractor(ExtractionModule):
    """Extracts media files and content"""

    def __init__(self):
        super().__init__("Media Files", "Photos, videos, audio files, and multimedia content")

        self._extension_map = {
            'photos': {
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.heic', '.heif', '.raw', '.dng'
            },
            'videos': {
                '.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg'
            },
            'audio': {
                '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.opus', '.aiff'
            },
            'documents': {
                '.pdf', '.txt', '.csv', '.json', '.xml', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx'
            },
        }

    def extract(self, device_id: str, **kwargs) -> Dict[str, Any]:
        """Extract media files"""
        try:
            case_id: Optional[str] = kwargs.get('case_id')
            orchestrator = kwargs.get('orchestrator')
            source_roots = [Path(path) for path in kwargs.get('source_roots', [])]

            artifact_root = Path(ArtifactPathBuilder.resolve(case_id, 'media', ensure_dir=True))
            device_pull_root = artifact_root / 'device_pull'

            extraction_method = 'sample_data'
            errors: List[str] = []

            adb = orchestrator._get_adb() if orchestrator else None
            if orchestrator and adb:
                device_check = orchestrator._ensure_device(device_id)
                if not device_check.get('ok'):
                    return {'status': 'error', 'error': device_check.get('message', 'Unable to verify device connection.')}
                else:
                    remote_roots = kwargs.get('remote_media_roots') or [
                        '/sdcard/DCIM',
                        '/sdcard/DCIM/Camera',
                        '/sdcard/Pictures',
                        '/sdcard/Pictures/Screenshots',
                        '/sdcard/Movies',
                        '/sdcard/Download',
                        '/sdcard/Documents',
                        '/sdcard/Music',
                        '/sdcard/Recordings',
                        '/sdcard/WhatsApp/Media',
                    ]
                    pulled_any = False
                    pull_directory = getattr(adb, 'pull_directory', None)
                    for remote in remote_roots:
                        remote_name = Path(remote).name or 'media'
                        dest_dir = device_pull_root / remote_name
                        pulled = False
                        enforce_unlock = True
                        if callable(pull_directory):
                            try:
                                pulled = bool(pull_directory(remote, str(dest_dir), device_id=device_id))
                            except Exception as pull_exc:
                                errors.append(f"ADB pull failed for {remote}: {pull_exc}")
                        if pulled and dest_dir.exists():
                            source_roots.append(dest_dir)
                            pulled_any = True
                    if pulled_any:
                        extraction_method = 'device_pull'

            default_root = Path('phone_test_data')
            if default_root.exists():
                source_roots.append(default_root)

            collected = {
                'photos': [],
                'videos': [],
                'audio': [],
                'documents': [],
            }
            artifact_paths: Dict[str, List[str]] = {}

            def _category_for(path: Path) -> Optional[str]:
                ext = path.suffix.lower()
                for category, extensions in self._extension_map.items():
                    if ext in extensions:
                        return category
                return None

            def _unique_destination(base: Path) -> Path:
                if not base.exists():
                    return base
                stem = base.stem
                suffix = base.suffix
                counter = 1
                while True:
                    candidate = base.with_name(f"{stem}_{counter}{suffix}")
                    if not candidate.exists():
                        return candidate
                    counter += 1

            files_copied = 0
            for root in source_roots:
                if not root or not Path(root).exists():
                    continue
                for file_path in Path(root).rglob('*'):
                    if not file_path.is_file():
                        continue
                    category = _category_for(file_path)
                    if not category:
                        continue
                    target_dir = artifact_root / category
                    target_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = _unique_destination(target_dir / file_path.name)
                    try:
                        shutil.copy2(file_path, dest_path)
                    except Exception as copy_exc:
                        errors.append(f"Failed to copy {file_path} -> {dest_path}: {copy_exc}")
                        continue

                    stat = dest_path.stat()
                    entry = {
                        'filename': dest_path.name,
                        'path': str(dest_path),
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    }

                    if category == 'photos':
                        try:
                            from PIL import Image  # type: ignore
                            with Image.open(dest_path) as img:
                                width, height = img.size
                            entry['metadata'] = {'width': width, 'height': height}
                        except Exception:
                            pass

                    collected[category].append(entry)
                    artifact_paths.setdefault(category, []).append(str(dest_path))
                    files_copied += 1

            artifact_counts = {category: len(entries) for category, entries in collected.items()}
            artifact_counts['total'] = files_copied

            media_data = {
                'photos': collected['photos'],
                'videos': collected['videos'],
                'audio': collected['audio'],
                'documents': collected['documents'],
                'artifacts': artifact_paths,
                'artifact_counts': artifact_counts,
                'extracted_at': datetime.now().isoformat(),
                'extraction_method': extraction_method,
                'errors': errors,
            }

            if files_copied == 0:
                media_data['note'] = 'No media artifacts discovered in available sources.'

            logger.info("Media data prepared for case %s (files: %s)", case_id or 'unknown', files_copied)
            return {'status': 'success', 'data': media_data}

        except Exception as e:
            logger.error(f"Media extraction failed: {e}")
            return {'status': 'error', 'error': str(e)}


class SystemExtractor(ExtractionModule):
    """Extracts system-level data"""

    def __init__(self):
        super().__init__("System Data", "System logs, configuration, and diagnostic information")

    def extract(self, device_id: str, **kwargs) -> Dict[str, Any]:
        """Extract system data"""
        try:
            # Simulate system data extraction
            system_data = {
                'system_logs': [
                    {
                        'timestamp': datetime.now().isoformat(),
                        'level': 'INFO',
                        'message': 'Device booted successfully',
                        'source': 'system'
                    }
                ],
                'installed_apps': [
                    {
                        'name': 'Test App',
                        'package': 'com.test.app',
                        'version': '1.0.0',
                        'installed_date': datetime.now().isoformat()
                    }
                ],
                'network_config': {
                    'wifi_enabled': True,
                    'bluetooth_enabled': False,
                    'mobile_data_enabled': True,
                    'airplane_mode': False
                },
                'battery_info': {
                    'level': 85,
                    'status': 'charging',
                    'temperature': 32.5
                },
                'extracted_at': datetime.now().isoformat()
            }

            logger.info(f"System data extracted for {device_id}")
            return {'status': 'success', 'data': system_data}

        except Exception as e:
            logger.error(f"System extraction failed: {e}")
            return {'status': 'error', 'error': str(e)}


class ExtractionMonitor:
    """Tracks active extractions across cases."""

    PHONE_THRESHOLDS = {
        'cpu': 60,    # %
        'mem': 65,    # %
        'disk_io': 25,  # MB/s
        'temp': 45    # °C
    }

    def __init__(self):
        self.active_cases = {}
        self.system_stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': []
        }
        self.device_type = 'desktop'  # Updated by ADB detection

    def add_case(self, case_id: str, total_modules: int):
        self.active_cases[case_id] = {
            'start_time': datetime.now(),
            'progress': 0,
            'total': total_modules,
            'completed': 0,
            'current_module': None,
            'status': 'running'
        }

    def update_progress(self, case_id: str, progress: float, module: str):
        if case_id in self.active_cases:
            self.active_cases[case_id]['progress'] = progress
            self.active_cases[case_id]['current_module'] = module
            if progress >= 100:
                self.active_cases[case_id]['completed'] += 1
                self.active_cases[case_id]['status'] = 'completed' if \
                    self.active_cases[case_id]['completed'] >= self.active_cases[case_id]['total'] \
                    else 'running'

    def record_system_stats(self):
        """Capture system resource usage for forensic auditing."""
        try:
            import psutil
            self.system_stats['cpu_usage'].append(
                (datetime.now(), psutil.cpu_percent()))
            self.system_stats['memory_usage'].append(
                (datetime.now(), psutil.virtual_memory().percent))
            self.system_stats['disk_io'].append(
                (datetime.now(), psutil.disk_io_counters().read_bytes))
        except ImportError:
            pass

    def check_phone_limits(self) -> Dict[str, bool]:
        """Returns dict of exceeded thresholds for phones."""
        limits = {k: False for k in self.PHONE_THRESHOLDS}
        if self.device_type != 'phone':
            return limits

        try:
            import psutil
            # CPU check
            cpu = psutil.cpu_percent()
            limits['cpu'] = cpu > self.PHONE_THRESHOLDS['cpu']

            # Memory check
            mem = psutil.virtual_memory().percent
            limits['mem'] = mem > self.PHONE_THRESHOLDS['mem']

            # Disk check
            io = psutil.disk_io_counters().read_bytes / (1024 * 1024)
            limits['disk_io'] = io > self.PHONE_THRESHOLDS['disk_io']

            # Temp check (Android only)
            try:
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    temp = int(f.read()) / 1000
                    limits['temp'] = temp > self.PHONE_THRESHOLDS['temp']
            except Exception:
                pass

        except Exception:
            pass

        return limits


class DataExtractionOrchestrator:
    """
    Central orchestrator for one-click data extraction across all forensic modules.

    This class coordinates the extraction of data from various sources based on
    the consent level granted for a case.
    """

    def __init__(self, consent_manager: ConsentManager):
        """Initialize the orchestrator with a consent manager"""
        self.consent_manager = consent_manager

        # Initialize all extraction modules
        self.modules = {
            'device_info': DeviceInfoExtractor(),
            'communications': CommunicationExtractor(),
            'location': LocationExtractor(),
            'security': SecurityExtractor(),
            'media': MediaExtractor(),
            'system': SystemExtractor()
        }

        self.module_min_levels = {
            'device_info': ConsentLevel.BASIC,
            'communications': ConsentLevel.STANDARD,
            'location': ConsentLevel.STANDARD,
            'security': ConsentLevel.FULL,
            'media': ConsentLevel.FULL,
            'system': ConsentLevel.FULL
        }

        # Define which modules run at which consent levels
        self.consent_level_modules = {
            ConsentLevel.BASIC: ['device_info'],
            ConsentLevel.STANDARD: ['device_info', 'communications', 'location'],
            ConsentLevel.FULL: ['device_info', 'communications', 'location', 'security', 'media', 'system'],
            ConsentLevel.LEGAL: ['device_info', 'communications',
                'location', 'security', 'media', 'system']
        }

        self.monitor = ExtractionMonitor()
        self._adb_client: Optional[AndroidADB] = None
        self._adb_summary: Dict[str, Any] = {'available': False}
        self._pending_resumptions: Dict[str, Any] = {}
        self._resume_futures: Dict[str, Any] = {}
        self._resume_results: Dict[str, Dict[str, Any]] = {}
        self._background_progress: Dict[str, Dict[str, Any]] = {}
        self._event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._event_history: Dict[str, List[Dict[str, Any]]] = {}
        self._event_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._ensure_default_extensions()

        logger.info("Data Extraction Orchestrator initialized")

        summary = self._refresh_adb_summary()
        if summary.get('connected'):
            self.monitor.device_type = 'phone'

    def _ensure_default_extensions(self) -> None:
        defaults = {
            'txt': ('builtins', 'text', True),
        }
        for ext, (package, category, safe) in defaults.items():
            if ext not in getattr(file_handler, 'extension_database', {}):
                file_handler.register_custom_format(ext, package, category, safe=safe)

    def _get_adb(self) -> Optional[AndroidADB]:
        if self._adb_client is not None:
            return self._adb_client
        if AndroidADB is None:
            return None
        try:
            self._adb_client = AndroidADB()
            return self._adb_client
        except Exception as exc:
            logger.warning("Failed to initialize AndroidADB: %s", exc)
            self._adb_client = None
            return None

    def _refresh_adb_summary(self) -> Dict[str, Any]:
        adb = self._get_adb()
        if not adb:
            summary = {'available': False, 'installed': False, 'devices': []}
            self._adb_summary = summary
            return summary
        summary = adb.device_summary()
        summary['available'] = True
        self._adb_summary = summary
        return summary

    def _ensure_device(self, device_id: Optional[str]) -> Dict[str, Any]:
        summary = self._refresh_adb_summary()
        if not summary.get('available'):
            return {'ok': False, 'message': 'ADB not available. Install Android platform-tools.'}
        if not summary.get('connected'):
            return {'ok': False, 'message': 'No authorised Android device detected via ADB.'}
        if device_id and not any(d.get('serial') == device_id for d in summary.get('devices', [])):
            return {'ok': False, 'message': f'Device {device_id} not detected via ADB.'}
        return {'ok': True, 'summary': summary}

    def register_event_callback(self, callback: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        self._event_callback = callback

    def _dispatch_event(self, payload: Dict[str, Any]) -> None:
        if self._event_callback:
            try:
                self._event_callback(payload)
            except Exception as exc:
                logger.debug('Event callback failed: %s', exc)

    def get_events(self, case_id: str, start: int = 0) -> List[Dict[str, Any]]:
        with self._event_lock:
            history = list(self._event_history.get(case_id, []))
        if start <= 0:
            return history
        return history[start:]

    def get_event_count(self, case_id: str) -> int:
        with self._event_lock:
            return len(self._event_history.get(case_id, []))

    def clear_pending_resume(self, case_id: str) -> None:
        with self._state_lock:
            self._pending_resumptions.pop(case_id, None)

    def get_background_progress(self, case_id: str) -> Optional[Dict[str, Any]]:
        with self._state_lock:
            return self._background_progress.get(case_id)

    def pop_resume_result(self, case_id: str) -> Optional[Dict[str, Any]]:
        with self._state_lock:
            return self._resume_results.pop(case_id, None)

    def _register_artifact_extensions(self, artifacts: Optional[Dict[str, Any]]) -> None:
        if not artifacts or not isinstance(artifacts, dict):
            return

        mapping = {
            'sqlite': ('sqlite3', 'database', True),
            'db': ('sqlite3', 'database', True),
            'json': ('builtins', 'data', True),
            'txt': ('builtins', 'text', True),
            'csv': ('pandas', 'data', True),
        }

        for value in artifacts.values():
            paths: List[str] = []
            if isinstance(value, str):
                paths.append(value)
            elif isinstance(value, list):
                paths.extend([item for item in value if isinstance(item, str)])
            for path in paths:
                ext = os.path.splitext(path)[1].lstrip('.').lower()
                if not ext:
                    continue
                existing = file_handler.extension_database.get(ext)
                if existing:
                    continue
                package, category, safe = mapping.get(ext, ('pandas', 'artifact', True))
                file_handler.register_custom_format(ext, package, category, safe=safe)

    def _schedule_resume(
        self,
        case_id: str,
        device_id: str,
        modules_remaining: Optional[List[str]],
        progress_callback,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        with self._state_lock:
            existing_future = self._resume_futures.get(case_id)
            if existing_future and not existing_future.done():
                return
            if existing_future and existing_future.done():
                self._resume_futures.pop(case_id, None)

        event_callback = event_callback or self._event_callback

        started = datetime.now().isoformat()

        def _run_resume() -> Dict[str, Any]:
            if event_callback:
                self.register_event_callback(event_callback)
            with self._state_lock:
                self._background_progress[case_id] = {
                    'started': started,
                    'status': 'running',
                    'progress': 0.0,
                    'message': 'Resuming extraction…',
                }

            def bg_progress(progress: float, message: str) -> None:
                with self._state_lock:
                    self._background_progress[case_id] = {
                        'started': started,
                        'status': 'running',
                        'progress': progress,
                        'message': message,
                        'updated': datetime.now().isoformat(),
                    }

            try:
                result = self.extract_all_data(
                    case_id,
                    device_id,
                    bg_progress,
                    modules_remaining,
                )
                self._finalize_results(case_id, result)
                return result
            finally:
                with self._state_lock:
                    self._background_progress.pop(case_id, None)
                self.register_event_callback(event_callback)

        future = self._executor.submit(_run_resume)
        with self._state_lock:
            self._resume_futures[case_id] = future

        def _store_result(fut):
            try:
                result = fut.result()
            except Exception as exc:
                logger.error('Background resume failed for %s: %s', case_id, exc, exc_info=True)
                result = {
                    'case_id': case_id,
                    'status': 'failed',
                    'errors': [str(exc)],
                }
            with self._state_lock:
                self._resume_results[case_id] = result
                self._resume_futures.pop(case_id, None)
                self._pending_resumptions.pop(case_id, None)

        future.add_done_callback(_store_result)

        with self._state_lock:
            pending = self._pending_resumptions.get(case_id)
            if isinstance(pending, dict):
                pending['scheduled'] = True

    def get_resume_future(self, case_id: str):
        with self._state_lock:
            return self._resume_futures.get(case_id)

    def clear_resume_future(self, case_id: str) -> None:
        with self._state_lock:
            self._resume_futures.pop(case_id, None)

    def _record_event(
        self,
        results: Dict[str, Any],
        module: str,
        message: str,
        *,
        severity: str = 'error',
        attempt: Optional[int] = None,
        final: bool = False,
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            'module': module,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
        }
        if attempt is not None:
            entry['attempt'] = attempt
        if final:
            entry['final'] = True
        results.setdefault('ui_events', []).append(entry)

        bucket = None
        if severity in {'error', 'fatal'}:
            bucket = 'errors'
        elif severity == 'warning':
            bucket = 'warnings'
        if bucket:
            previous = results.setdefault(bucket, [])
            previous.append(entry)
        case_id = results.get('case_id')
        if case_id:
            with self._event_lock:
                history = self._event_history.setdefault(case_id, [])
                history.append(entry)
        self._dispatch_event(entry)
        return entry

    def _merge_module_runs(
        self,
        existing: Optional[List[Dict[str, Any]]],
        new: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for source in (existing or []):
            if isinstance(source, dict) and source.get('name'):
                merged[source['name']] = source
        for source in (new or []):
            if isinstance(source, dict) and source.get('name'):
                merged[source['name']] = source
        return list(merged.values())

    def _persist_results(self, case_id: str, new_results: Dict[str, Any]) -> None:
        print(f"DEBUG: _persist_results called for case_id={case_id}")
        try:
            existing = ResultsRepository.load(case_id) or {}
            merged = existing.copy()
            merged.setdefault('case_id', case_id)
            merged.setdefault('device_id', new_results.get('device_id'))

            merge_keys = {
                'module_logs': 'dict',
                'retry_attempts': 'dict',
                'ui_events': 'list',
                'warnings': 'list',
                'errors': 'list',
            }

            for key, value in new_results.items():
                if key in {'data', 'modules_run'}:
                    continue
                if key in merge_keys:
                    continue
                merged[key] = value

            # Merge retry attempts
            retry_attempts: Dict[str, Any] = {}
            if isinstance(existing.get('retry_attempts'), dict):
                retry_attempts.update(existing['retry_attempts'])
            if isinstance(new_results.get('retry_attempts'), dict):
                retry_attempts.update(new_results['retry_attempts'])
            if retry_attempts:
                merged['retry_attempts'] = retry_attempts

            # Merge module logs
            module_logs: Dict[str, List[Dict[str, Any]]] = {}
            if isinstance(existing.get('module_logs'), dict):
                module_logs.update(existing['module_logs'])
            if isinstance(new_results.get('module_logs'), dict):
                for key, entries in new_results['module_logs'].items():
                    if key in module_logs:
                        module_logs[key] = (module_logs[key] or []) + (entries or [])
                    else:
                        module_logs[key] = entries
            if module_logs:
                merged['module_logs'] = module_logs

            # Merge events and warnings
            for list_key in ('ui_events', 'warnings', 'errors'):
                existing_list = existing.get(list_key) if isinstance(existing.get(list_key), list) else []
                new_list = new_results.get(list_key) if isinstance(new_results.get(list_key), list) else []
                if existing_list or new_list:
                    merged[list_key] = (existing_list or []) + (new_list or [])

            merged_data: Dict[str, Any] = {}
            if isinstance(existing.get('data'), dict):
                merged_data.update(existing['data'])
            if isinstance(new_results.get('data'), dict):
                for module_key, payload in new_results['data'].items():
                    enriched = dict(payload or {})
                    existing_payload = merged_data.get(module_key) or {}
                    if isinstance(existing_payload, dict):
                        enriched = {**existing_payload, **enriched}
                    # Provide artifact counts for dashboard summaries
                    artifacts = enriched.get('artifacts') or {}
                    enriched['artifact_counts'] = {
                        key: len(value) if isinstance(value, list) else 1
                        for key, value in artifacts.items()
                    }
                    merged_data[module_key] = enriched
            if merged_data:
                merged['data'] = merged_data

            merged['modules_run'] = self._merge_module_runs(
                existing.get('modules_run'),
                new_results.get('modules_run'),
            )

            ResultsRepository.save(case_id, merged)
        except Exception as exc:
            logger.warning('Failed to persist extraction results for %s: %s', case_id, exc)
            print(f"DEBUG: _persist_results failed for case_id={case_id} with error: {exc}")

    def _finalize_results(self, case_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        self._persist_results(case_id, results)
        return results

    def resume_after_vault(self, case_id: str) -> Optional[Dict[str, Any]]:
        with self._state_lock:
            pending = self._pending_resumptions.get(case_id)
        if not pending:
            return None

        modules_remaining = pending.get('modules_remaining') or None
        device_id = pending['device_id']
        progress_callback = pending.get('progress_callback')

        self._schedule_resume(
            case_id,
            device_id,
            modules_remaining,
            progress_callback,
            event_callback=None,
        )
        return self.get_background_progress(case_id)

    def extract_all_data(
        self,
        case_id: str,
        device_id: str,
        progress_callback=None,
        modules_override: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """One-click extraction with retry mechanism and enhanced validation"""
        start_time = datetime.now()
        results = {
            'case_id': case_id,
            'device_id': device_id,
            'extraction_started': start_time.isoformat(),
            'status': 'in_progress',
            'modules_run': [],
            'data': {},
            'errors': [],
            'retry_attempts': {}
        }

        max_retries = 2  # Maximum retry attempts per module

        try:
            # Get consent session
            session = self.consent_manager.get_session(case_id)
            if not session:
                raise ValueError(f"No consent session for case {case_id}")

            # Validate extraction readiness with enhanced validator
            validation_result = ExtractionValidator.validate_extraction_ready(
                case_id=case_id,
                device_id=device_id,
                session=session,
                required_level=ConsentLevel.STANDARD
            )
            
            if not validation_result["ready"]:
                results['status'] = 'blocked'
                results['errors'].extend(validation_result["errors"])
                results['validation_checks'] = validation_result["checks"]
                logger.warning(f"Extraction blocked for {case_id}: {validation_result['errors']}")
                return results

            # Determine which modules to run
            modules_for_level = self.consent_level_modules.get(session.level, [])
            if modules_override:
                modules_to_run = [
                    module for module in modules_for_level if module in set(modules_override)
                ]
            else:
                modules_to_run = modules_for_level

            if not modules_to_run:
                results['status'] = 'skipped'
                results['message'] = (
                    'No modules selected for execution. Adjust consent level or retry selection.'
                )
                return results

            results['modules_requested'] = modules_to_run
            module_logs: Dict[str, List[Dict[str, Any]]] = {module: [] for module in modules_to_run}

            # Check approval status with ApprovalSync
            if not ApprovalSync.is_approved(case_id):
                message = 'Awaiting nominee approval for extraction'
                results['status'] = 'pending_approval'
                results['errors'].append(message)
                logger.info(f"Extraction pending approval for {case_id}")
                return results
            
            # Check device health with DeviceManager
            device_health = DeviceManager.get_device_health(device_id)
            if device_health.get("issues"):
                message = f"Device issues detected: {', '.join(device_health['issues'])}"
                results['status'] = 'blocked'
                results['errors'].append(message)
                logger.warning(f"Device health check failed for {device_id}: {message}")
                return results

            unlock_status = self.consent_manager.get_unlock_status(case_id)
            results['unlock_status'] = unlock_status

            if (session.level.value >= ConsentLevel.STANDARD.value
                    and unlock_status.get('status') != 'verified'):
                message = (
                    'Unlock verification required. Send OTP and verify before running '
                    'communications, location, or higher-sensitivity modules.'
                )
                results['status'] = 'blocked'
                results['errors'].append(message)
                results['alert'] = message
                results['required_verification'] = {
                    'case_id': case_id,
                    'needed_for': [
                        m for m in modules_to_run
                        if self.module_min_levels.get(m, ConsentLevel.BASIC).value
                        >= ConsentLevel.STANDARD.value
                    ]
                }
                try:
                    self.consent_manager.record_audit_event(case_id, 'extraction_blocked', {
                        'device_id': device_id,
                        'modules': modules_to_run,
                        'unlock_status': unlock_status.get('status')
                    })
                except Exception:
                    logger.warning('Failed to record extraction_blocked audit event for case %s', case_id)
                return results
            
            # Create progress tracker for real-time monitoring
            progress_tracker = ProgressManager.create_tracker(case_id, 'full_extraction')

            vault_pending = False
            pending_payload: Optional[Dict[str, Any]] = None

            for module_name in modules_to_run:
                retry_count = 0
                last_error = None

                while retry_count <= max_retries:
                    try:
                        completed_modules = len(results['modules_run'])
                        progress = (completed_modules / len(modules_to_run)) * 100
                        if progress_callback:
                            status = (
                                f"Retry {retry_count}" if retry_count > 0 else "Initial attempt"
                            )
                            progress_callback(
                                progress,
                                f"{status} - {module_name} (attempt {retry_count + 1})"
                            )

                        entry = {
                            'attempt': retry_count + 1,
                            'timestamp': datetime.now().isoformat(),
                            'event': 'start'
                        }
                        module_logs[module_name].append(entry)
                        self._record_event(
                            results,
                            module_name,
                            f"Starting attempt {retry_count + 1}",
                            severity='info',
                            attempt=retry_count + 1,
                        )

                        module = self.modules[module_name]
                        module_kwargs: Dict[str, Any] = {
                            'case_id': case_id,
                            'orchestrator': self,
                        }
                        if module_name == 'communications':
                            module_kwargs['consent_manager'] = self.consent_manager
                        module_result = module.extract(device_id, **module_kwargs)

                        if module_result['status'] == 'success':
                            module_logs[module_name].append({
                                'attempt': retry_count + 1,
                                'timestamp': datetime.now().isoformat(),
                                'event': 'success'
                            })
                            self._record_event(
                                results,
                                module_name,
                                'Module completed successfully',
                                severity='info',
                                attempt=retry_count + 1,
                                final=True,
                            )
                            results['data'][module_name] = module_result['data']
                            self._register_artifact_extensions(module_result['data'].get('artifacts'))
                            results['modules_run'].append({
                                'name': module_name,
                                'status': 'success',
                                'attempts': retry_count + 1,
                                'extracted_at': datetime.now().isoformat(),
                                'logs': module_logs[module_name]
                            })
                            break

                        elif module_result['status'] == 'vault_verification_required':
                            module_logs[module_name].append({
                                'attempt': retry_count + 1,
                                'timestamp': datetime.now().isoformat(),
                                'event': 'vault_prompt',
                                'details': module_result.get('message')
                            })
                            self._record_event(
                                results,
                                module_name,
                                module_result.get('message', 'Vault verification required'),
                                severity='warning',
                                attempt=retry_count + 1,
                            )
                            results['modules_run'].append({
                                'name': module_name,
                                'status': 'pending_vault',
                                'attempts': retry_count + 1,
                                'logs': module_logs[module_name]
                            })
                            pending_payload = {
                                'module': module_name,
                                'vault_entries': module_result.get('vault_entries', []),
                                'message': module_result.get('message', 'Vault verification required'),
                                'auth_prompt': module_result.get('auth_prompt', 'PIN'),
                                'consent_details': module_result.get('consent_details')
                            }
                            results['alert'] = module_result.get('message')
                            results['errors'].append({
                                'module': module_name,
                                'error': 'Vault verification required',
                                'attempts': retry_count + 1,
                                'timestamp': datetime.now().isoformat()
                            })
                            vault_pending = True
                            break

                        else:
                            raise ValueError(module_result.get(
                                'error', f"Module status: {module_result['status']}"))

                    except Exception as e:
                        module_logs[module_name].append({
                            'attempt': retry_count + 1,
                            'timestamp': datetime.now().isoformat(),
                            'event': 'error',
                            'details': str(e)
                        })
                        self._record_event(
                            results,
                            module_name,
                            str(e),
                            severity='error',
                            attempt=retry_count + 1,
                        )
                        results['errors'].append({
                            'module': module_name,
                            'error': str(e),
                            'attempt': retry_count + 1,
                            'timestamp': datetime.now().isoformat()
                        })
                        last_error = str(e)
                        retry_count += 1
                        results['retry_attempts'][module_name] = retry_count

                        if retry_count > max_retries:
                            results['errors'].append({
                                'module': module_name,
                                'error': last_error,
                                'attempts': retry_count,
                                'timestamp': datetime.now().isoformat()
                            })
                            self._record_event(
                                results,
                                module_name,
                                f"Module failed after {retry_count} attempts: {last_error}",
                                severity='fatal',
                                attempt=retry_count,
                                final=True,
                            )
                            results['modules_run'].append({
                                'name': module_name,
                                'status': 'error',
                                'error': last_error,
                                'attempts': retry_count,
                                'logs': module_logs[module_name]
                            })
                        else:
                            time.sleep(1 * retry_count)  # Exponential backoff

                if vault_pending:
                    break

                if (len(results['modules_run']) % 2) == 0:
                    self.monitor.record_system_stats()

            if vault_pending:
                results['module_logs'] = module_logs
                if pending_payload:
                    results['vault_verification_pending'] = pending_payload
                    with self._state_lock:
                        self._pending_resumptions[case_id] = {
                            'device_id': device_id,
                            'modules_remaining': [
                                m for m in modules_to_run
                                if m not in {entry['name'] for entry in results['modules_run']}
                            ],
                            'progress_callback': progress_callback,
                            'timestamp': datetime.now().isoformat(),
                        }
                results['status'] = 'vault_pending'
                return results

            self.clear_pending_resume(case_id)
            results['module_logs'] = module_logs

            # Persist intermediate state for UI readers
            self._persist_results(case_id, results)

            # Calculate completion stats
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            results['extraction_completed'] = end_time.isoformat()
            results['duration_seconds'] = duration
            results['total_modules'] = len(modules_to_run)
            results['successful_modules'] = len(
                [m for m in results['modules_run'] if m['status'] == 'success'])
            results['failed_modules'] = len(
                [m for m in results['modules_run'] if m['status'] == 'error'])

            # Set final status
            if results['failed_modules'] == 0:
                results['status'] = 'completed'
                logger.info(
                    f"Extraction completed successfully in {duration:.2f} seconds")
            elif results['successful_modules'] > 0:
                results['status'] = 'partial_success'
                logger.warning(
                    f"Extraction completed with {results['failed_modules']} failed modules")
            else:
                results['status'] = 'failed'
                logger.error("Extraction failed completely")

            if progress_callback:
                progress_callback(100.0, f"Extraction {results['status']}")

            # NEW: Record extraction in audit trail
            try:
                ConsentAuditTrail.record_approval(
                    case_id=case_id,
                    decision=f"extraction_{results['status']}",
                    nominee_name=session.nominee_name if session else "Unknown",
                    device_id=device_id,
                    purpose=f"Data extraction - {results['successful_modules']}/{results['total_modules']} modules successful"
                )
            except Exception as audit_error:
                logger.warning(f"Failed to record audit trail: {audit_error}")

            return self._finalize_results(case_id, results)

        except Exception as e:
            logger.error('Extraction failed: %s', e, exc_info=True)
            results['status'] = 'failed'
            results['errors'].append(str(e))
            return self._finalize_results(case_id, results)

    def run_extraction(
        self,
        case_id: str,
        device_id: str,
        progress_callback=None,
        modules_override: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper for extract_all_data."""
        return self.extract_all_data(
            case_id,
            device_id,
            progress_callback=progress_callback,
            modules_override=modules_override
        )

    def get_module_status(self) -> Dict[str, Any]:
        """Get status of all modules"""
        return {
            module_name: module.get_status()
            for module_name, module in self.modules.items()
        }

    def provide_data_or_hint(self, case_id: str, domain: str) -> Dict[str, Any]:
        """Return lightweight hints about data availability for UI modules."""
        session = self.consent_manager.get_session(case_id)
        if not session:
            return {'status': 'error', 'message': 'No consent session'}

        domain = domain.lower()
        results = ResultsRepository.load(case_id) or {}
        data = results.get('data', {}) if isinstance(results, dict) else {}

        if domain == 'comms':
            comms = data.get('communications', {})
            if comms.get('sms_messages') or comms.get('call_logs'):
                return {'status': 'ok'}
            return {'status': 'missing', 'message': 'No communication data found.'}

        if domain == 'location':
            loc = data.get('location', {})
            if loc.get('gps_coordinates') or loc.get('cell_towers'):
                return {'status': 'ok'}
            return {'status': 'missing', 'message': 'No location data found.'}

        return {'status': 'unknown', 'message': f'No hint available for domain {domain}.'}

    def validate_extraction_requirements(self, case_id: str, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Validate that extraction can proceed"""
        validation = {
            'can_proceed': False,
            'issues': [],
            'consent_level': None,
            'available_modules': []
        }

        try:
            # Check consent
            session = self.consent_manager.get_session(case_id)
            if not session:
                validation['issues'].append(
                    f"No consent session found for case {case_id}")
                return validation

            validation['consent_level'] = session.level.name

            # Check device connectivity (placeholder - would need actual device detection)
            # For now, assume device is available
            validation['device_available'] = bool(device_id)

            # Determine available modules
            modules_to_run = self.consent_level_modules.get(session.level, [])
            validation['available_modules'] = modules_to_run

            # Check module availability
            unavailable_modules = []
            for module_name in modules_to_run:
                module = self.modules.get(module_name)
                if not module or not module.is_available:
                    unavailable_modules.append(module_name)

            if unavailable_modules:
                validation['issues'].append(
                    f"Modules not available: {', '.join(unavailable_modules)}")

            # Overall validation
            validation['can_proceed'] = len(validation['issues']) == 0

        except Exception as e:
            validation['issues'].append(f"Validation error: {str(e)}")

        return validation

    def estimate_extraction_time(self, consent_level: ConsentLevel) -> float:
        """Estimate extraction time in seconds based on consent level"""
        base_times = {
            ConsentLevel.BASIC: 30,      # Quick device info
            ConsentLevel.STANDARD: 120,  # Communications + location
            ConsentLevel.FULL: 300,      # All modules
            ConsentLevel.LEGAL: 600      # Extended analysis
        }

        return base_times.get(consent_level, 60)

    def get_extraction_summary(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable summary of extraction results"""
        summary_lines = [
            f"📊 Extraction Summary for Case {results.get('case_id', 'Unknown')}",
            f"🔢 Consent Level: {results.get('consent_level', 'Unknown')}",
            f"📱 Device: {results.get('device_id', 'Unknown')}",
            f"⚡ Status: {results.get('status', 'Unknown')}",
            f"⏱️ Duration: {results.get('duration_seconds', 0):.1f} seconds",
            f"📦 Modules Run: {results.get('total_modules', 0)}",
            f"✅ Successful: {results.get('successful_modules', 0)}",
            f"❌ Failed: {results.get('failed_modules', 0)}"
        ]

        if results.get('errors'):
            summary_lines.append("🚨 Errors:")
            for error in results['errors']:
                summary_lines.append(
                    f"   • {error['module']}: {error['error']}")

        return "\n".join(summary_lines)

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """
        Returns structured metrics for dashboard consumption
        Formats:
        {
            'active_cases': List[CaseStatus],
            'system_stats': SystemMetrics,
            'device_status': DeviceInfo
        }
        """
        return {
            'active_cases': self._format_active_cases(),
            'system_stats': self.monitor.system_stats,
            'device_status': self._get_device_status()
        }

    def _format_active_cases(self) -> List[Dict]:
        """Format running cases for dashboard"""
        return [{
            'case_id': k,
            'progress': v['progress'],
            'completed': v['completed'],
            'total': v['total'],
            'status': v['status'],
            'duration': str(datetime.now() - v['start_time'])[:-7]
        } for k, v in self.monitor.active_cases.items()]

    def _get_device_status(self) -> Dict[str, Any]:
        """Get connected device info without UI components"""
        try:
            from adapters.android_adb import AndroidADB
            devices = AndroidADB().list_devices() or []
            return {
                'connected': devices,
                'rooted': [d for d in devices if d.get('rooted')]
            }
        except Exception:
            return {'connected': [], 'rooted': []}
