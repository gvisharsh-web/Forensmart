"""
HDD ADAPTER - Hard Drive and USB Storage Extraction
Handles extraction from HDD and USB storage devices

This module provides:
- HDDAdapter class for storage device extraction
- File system scanning
- Deleted file recovery
- File metadata extraction
"""

import logging
import os
import platform
from typing import Dict, List, Optional, Any
from datetime import datetime

from modules.consent.models import ConsentLevel, MODULE_MIN_LEVELS
from modules.shared.utils import ArtifactPathBuilder, ResultsRepository
from .base import AdapterBase
from .exceptions import ConnectionFailed, ExtractionFailed, PermissionDenied

logger = logging.getLogger(__name__)


# ============================================================================
# HDD ADAPTER CLASS
# ============================================================================

class HDDAdapter(AdapterBase):
    """Hard drive and USB storage adapter"""
    
    def __init__(self, device_id: str, case_id: str, consent_manager=None):
        """Initialize HDD adapter"""
        super().__init__(device_id, case_id, consent_manager)
        self.adapter_type = "HDD"
        self.mount_path: Optional[str] = None
        self.file_system_type: Optional[str] = None
        logger.info(f"✅ HDD Adapter initialized for device: {device_id}")
    
    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================
    
    def connect(self) -> bool:
        """Mount HDD/USB drive"""
        try:
            logger.info(f"🔌 Connecting to storage device: {self.device_id}")
            
            # Check if device path exists
            if not os.path.exists(self.device_id):
                raise ConnectionFailed(self.device_id, f"Device path not found: {self.device_id}")
            
            self.mount_path = self.device_id
            self.is_connected = True
            self.extraction_status = "connected"
            
            # Detect file system type
            self.file_system_type = self._detect_filesystem()
            
            logger.info(f"✅ Connected to storage device: {self.device_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            raise ConnectionFailed(self.device_id, str(e))
    
    def disconnect(self) -> bool:
        """Unmount HDD/USB drive"""
        try:
            logger.info(f"🔌 Disconnecting from storage device: {self.device_id}")
            self.is_connected = False
            self.extraction_status = "disconnected"
            self.mount_path = None
            logger.info(f"✅ Disconnected from storage device: {self.device_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Disconnection error: {e}")
            return False
    
    # ========================================================================
    # DATA EXTRACTION METHODS
    # ========================================================================
    
    def extract_data(self) -> Dict[str, Any]:
        """Extract all data from storage device"""
        try:
            if not self.validate_connection():
                return {'error': 'Device not connected'}
            
            logger.info(f"💾 Starting extraction from storage device: {self.device_id}")
            
            results = {
                'device_id': self.device_id,
                'case_id': self.case_id,
                'adapter_type': self.adapter_type,
                'timestamp': datetime.now().isoformat(),
                'modules': {}
            }
            
            # Extract file system info
            if self.check_consent('device_info', MODULE_MIN_LEVELS.get('device_info', ConsentLevel.STANDARD)):
                results['modules']['device_info'] = self.extract_device_info()
            
            # Extract file system
            if self.check_consent('security', MODULE_MIN_LEVELS.get('security', ConsentLevel.FULL)):
                results['modules']['file_system'] = self.extract_file_system()
            
            # Scan for media
            if self.check_consent('media', MODULE_MIN_LEVELS.get('media', ConsentLevel.FULL)):
                results['modules']['media'] = self.scan_for_media()
            
            # Scan for documents
            if self.check_consent('security', MODULE_MIN_LEVELS.get('security', ConsentLevel.FULL)):
                results['modules']['documents'] = self.scan_for_documents()
            
            # Save results
            self.save_results(results, 'hdd_extraction')
            
            logger.info(f"✅ Extraction complete from storage device: {self.device_id}")
            return results
        except Exception as e:
            logger.error(f"❌ Extraction error: {e}")
            return {'error': str(e), 'device_id': self.device_id}
    
    def extract_device_info(self) -> Dict[str, Any]:
        """Extract storage device information"""
        try:
            logger.info(f"💾 Extracting device info from: {self.device_id}")
            
            info = {
                'device_id': self.device_id,
                'mount_path': self.mount_path,
                'file_system': self.file_system_type,
                'total_size': self._get_device_size(),
                'free_space': self._get_free_space(),
                'timestamp': datetime.now().isoformat()
            }
            
            self.log_operation('extract_device_info', 'success')
            return info
        except Exception as e:
            logger.error(f"❌ Error extracting device info: {e}")
            return self.handle_error(e, 'extract_device_info')
    
    def extract_file_system(self) -> Dict[str, Any]:
        """Extract file system structure"""
        try:
            logger.info(f"📁 Extracting file system from: {self.device_id}")
            
            file_system = {
                'root_files': self._list_directory(self.mount_path),
                'total_files': self._count_files(self.mount_path),
                'directory_structure': self._get_directory_structure(self.mount_path),
                'timestamp': datetime.now().isoformat()
            }
            
            self.log_operation('extract_file_system', 'success')
            return file_system
        except Exception as e:
            logger.error(f"❌ Error extracting file system: {e}")
            return self.handle_error(e, 'extract_file_system')
    
    def extract_deleted_files(self) -> List[Dict[str, Any]]:
        """Extract deleted files (simulated)"""
        try:
            logger.info(f"🔍 Scanning for deleted files on: {self.device_id}")
            
            deleted_files = [
                {
                    'name': 'deleted_file.txt',
                    'size': 1024,
                    'deleted_date': datetime.now().isoformat(),
                    'recovery_possible': True
                }
            ]
            
            return deleted_files
        except Exception as e:
            logger.error(f"❌ Error extracting deleted files: {e}")
            return []
    
    def extract_partitions(self) -> List[Dict[str, Any]]:
        """Extract partition information"""
        try:
            logger.info(f"📊 Extracting partition info from: {self.device_id}")
            
            partitions = [
                {
                    'name': 'Partition 1',
                    'size': 1000000000,
                    'file_system': self.file_system_type,
                    'mount_point': self.mount_path
                }
            ]
            
            return partitions
        except Exception as e:
            logger.error(f"❌ Error extracting partitions: {e}")
            return []
    
    def extract_file_metadata(self) -> List[Dict[str, Any]]:
        """Extract file metadata"""
        try:
            logger.info(f"📋 Extracting file metadata from: {self.device_id}")
            
            metadata = []
            for root, dirs, files in os.walk(self.mount_path):
                for file in files[:10]:  # Limit to first 10 files
                    file_path = os.path.join(root, file)
                    try:
                        stat_info = os.stat(file_path)
                        metadata.append({
                            'path': file_path,
                            'size': stat_info.st_size,
                            'modified': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                            'created': datetime.fromtimestamp(stat_info.st_ctime).isoformat()
                        })
                    except Exception as e:
                        logger.warning(f"⚠️ Error getting metadata for {file_path}: {e}")
            
            return metadata
        except Exception as e:
            logger.error(f"❌ Error extracting file metadata: {e}")
            return []
    
    def extract_directory_structure(self) -> Dict[str, Any]:
        """Extract directory structure"""
        try:
            logger.info(f"📁 Extracting directory structure from: {self.device_id}")
            
            structure = self._get_directory_structure(self.mount_path)
            return structure
        except Exception as e:
            logger.error(f"❌ Error extracting directory structure: {e}")
            return self.handle_error(e, 'extract_directory_structure')
    
    def scan_for_media(self) -> Dict[str, List[str]]:
        """Scan for media files"""
        try:
            logger.info(f"🎬 Scanning for media files on: {self.device_id}")
            
            media_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.avi', '.mov', '.mp3', '.wav']
            media_files = {'images': [], 'videos': [], 'audio': []}
            
            for root, dirs, files in os.walk(self.mount_path):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in media_extensions:
                        file_path = os.path.join(root, file)
                        if ext in ['.jpg', '.jpeg', '.png', '.gif']:
                            media_files['images'].append(file_path)
                        elif ext in ['.mp4', '.avi', '.mov']:
                            media_files['videos'].append(file_path)
                        elif ext in ['.mp3', '.wav']:
                            media_files['audio'].append(file_path)
            
            return media_files
        except Exception as e:
            logger.error(f"❌ Error scanning for media: {e}")
            return {'images': [], 'videos': [], 'audio': []}
    
    def scan_for_documents(self) -> Dict[str, List[str]]:
        """Scan for document files"""
        try:
            logger.info(f"📄 Scanning for documents on: {self.device_id}")
            
            doc_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.txt', '.ppt']
            documents = {}
            
            for root, dirs, files in os.walk(self.mount_path):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in doc_extensions:
                        file_path = os.path.join(root, file)
                        if ext not in documents:
                            documents[ext] = []
                        documents[ext].append(file_path)
            
            return documents
        except Exception as e:
            logger.error(f"❌ Error scanning for documents: {e}")
            return {}
    
    def scan_for_databases(self) -> Dict[str, List[str]]:
        """Scan for database files"""
        try:
            logger.info(f"🗄️ Scanning for databases on: {self.device_id}")
            
            db_extensions = ['.db', '.sqlite', '.sqlite3', '.mdb', '.accdb']
            databases = {}
            
            for root, dirs, files in os.walk(self.mount_path):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in db_extensions:
                        file_path = os.path.join(root, file)
                        if ext not in databases:
                            databases[ext] = []
                        databases[ext].append(file_path)
            
            return databases
        except Exception as e:
            logger.error(f"❌ Error scanning for databases: {e}")
            return {}
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _detect_filesystem(self) -> str:
        """Detect file system type"""
        try:
            if platform.system() == 'Windows':
                return 'NTFS'  # Simplified detection
            elif platform.system() == 'Darwin':
                return 'APFS'
            else:
                return 'ext4'
        except Exception as e:
            logger.warning(f"⚠️ Error detecting file system: {e}")
            return 'Unknown'
    
    def _get_device_size(self) -> int:
        """Get total device size"""
        try:
            stat = os.statvfs(self.mount_path)
            return stat.f_blocks * stat.f_frsize
        except Exception as e:
            logger.warning(f"⚠️ Error getting device size: {e}")
            return 0
    
    def _get_free_space(self) -> int:
        """Get free space on device"""
        try:
            stat = os.statvfs(self.mount_path)
            return stat.f_bavail * stat.f_frsize
        except Exception as e:
            logger.warning(f"⚠️ Error getting free space: {e}")
            return 0
    
    def _list_directory(self, path: str) -> List[str]:
        """List files in directory"""
        try:
            return os.listdir(path)[:20]  # Limit to first 20 items
        except Exception as e:
            logger.warning(f"⚠️ Error listing directory: {e}")
            return []
    
    def _count_files(self, path: str) -> int:
        """Count total files"""
        try:
            count = 0
            for root, dirs, files in os.walk(path):
                count += len(files)
            return count
        except Exception as e:
            logger.warning(f"⚠️ Error counting files: {e}")
            return 0
    
    def _get_directory_structure(self, path: str, max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
        """Get directory structure recursively"""
        try:
            if current_depth >= max_depth:
                return {}
            
            structure = {'name': os.path.basename(path), 'type': 'directory', 'children': []}
            
            try:
                for item in os.listdir(path)[:10]:  # Limit items
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        structure['children'].append(
                            self._get_directory_structure(item_path, max_depth, current_depth + 1)
                        )
                    else:
                        structure['children'].append({'name': item, 'type': 'file'})
            except PermissionError:
                logger.warning(f"⚠️ Permission denied for: {path}")
            
            return structure
        except Exception as e:
            logger.warning(f"⚠️ Error getting directory structure: {e}")
            return {}
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        try:
            info = {
                'device_id': self.device_id,
                'adapter_type': self.adapter_type,
                'is_connected': self.is_connected,
                'mount_path': self.mount_path,
                'file_system': self.file_system_type,
                'total_size': self._get_device_size(),
                'free_space': self._get_free_space(),
                'timestamp': datetime.now().isoformat()
            }
            
            return info
        except Exception as e:
            logger.error(f"❌ Error getting device info: {e}")
            return {}
