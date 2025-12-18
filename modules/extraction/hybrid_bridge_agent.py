"""
HYBRID EXTRACTION BRIDGE AGENT
Core orchestrator for hybrid extraction with privilege escalation and extended sources
"""

import logging
import os
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class EscalationMethod(Enum):
    """Privilege escalation methods"""
    DIRTY_PIPE = "dirty_pipe"
    SELINUX_BYPASS = "selinux_bypass"
    ADB_ROOT = "adb_root"
    NONE = "none"


class ExtractionSource(Enum):
    """Extended extraction sources"""
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    SIGNAL = "signal"
    GOOGLE_DRIVE = "google_drive"
    ONEDRIVE = "onedrive"
    SYSTEM_LOGS = "system_logs"
    LOGCAT = "logcat"


# ============================================================================
# DATA CLASSES
# ============================================================================

class ExtractionArtifact:
    """Represents an extracted artifact"""
    
    def __init__(self, artifact_id: str, source: str, data_type: str, 
                 data: Any, timestamp: str = None, metadata: Dict = None):
        self.artifact_id = artifact_id
        self.source = source
        self.data_type = data_type
        self.data = data
        self.timestamp = timestamp or datetime.now().isoformat()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'artifact_id': self.artifact_id,
            'source': self.source,
            'data_type': self.data_type,
            'data': self.data,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


class ExtractionResult:
    """Represents extraction results"""
    
    def __init__(self, case_id: str, device_id: str, method: str = "hybrid"):
        self.case_id = case_id
        self.device_id = device_id
        self.method = method
        self.artifacts: List[ExtractionArtifact] = []
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        self.status = "in_progress"
        self.errors: List[str] = []
    
    def add_artifact(self, artifact: ExtractionArtifact) -> None:
        """Add artifact to results"""
        self.artifacts.append(artifact)
    
    def add_error(self, error: str) -> None:
        """Add error to results"""
        self.errors.append(error)
    
    def complete(self) -> None:
        """Mark extraction as complete"""
        self.end_time = datetime.now().isoformat()
        self.status = "completed"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'case_id': self.case_id,
            'device_id': self.device_id,
            'method': self.method,
            'artifacts': [a.to_dict() for a in self.artifacts],
            'start_time': self.start_time,
            'end_time': self.end_time,
            'status': self.status,
            'errors': self.errors,
            'total_artifacts': len(self.artifacts)
        }


# ============================================================================
# PRIVILEGE ESCALATION MANAGER
# ============================================================================

class PrivilegeEscalationManager:
    """Manage privilege escalation methods"""
    
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.escalation_methods = [
            EscalationMethod.DIRTY_PIPE,
            EscalationMethod.SELINUX_BYPASS,
            EscalationMethod.ADB_ROOT
        ]
    
    def attempt_escalation(self, method: EscalationMethod) -> bool:
        """Attempt privilege escalation"""
        try:
            if method == EscalationMethod.DIRTY_PIPE:
                return self._attempt_dirty_pipe()
            elif method == EscalationMethod.SELINUX_BYPASS:
                return self._attempt_selinux_bypass()
            elif method == EscalationMethod.ADB_ROOT:
                return self._attempt_adb_root()
            return False
        except Exception as e:
            logger.error(f"Escalation attempt failed: {e}")
            return False
    
    def _attempt_dirty_pipe(self) -> bool:
        """Attempt Dirty Pipe exploit (CVE-2022-1786)"""
        logger.info("Attempting Dirty Pipe exploit...")
        # Placeholder for actual implementation
        return False
    
    def _attempt_selinux_bypass(self) -> bool:
        """Attempt SELinux bypass"""
        logger.info("Attempting SELinux bypass...")
        # Placeholder for actual implementation
        return False
    
    def _attempt_adb_root(self) -> bool:
        """Attempt ADB root access"""
        logger.info("Attempting ADB root access...")
        # Placeholder for actual implementation
        return False
    
    def get_available_methods(self) -> List[EscalationMethod]:
        """Get available escalation methods"""
        return self.escalation_methods


# ============================================================================
# EXTENDED SOURCE EXTRACTOR
# ============================================================================

class ExtendedSourceExtractor:
    """Extract from extended sources"""
    
    def __init__(self, device_id: str, case_id: str):
        self.device_id = device_id
        self.case_id = case_id
    
    def extract_from_source(self, source: ExtractionSource) -> List[ExtractionArtifact]:
        """Extract from extended source"""
        try:
            if source == ExtractionSource.WHATSAPP:
                return self._extract_whatsapp()
            elif source == ExtractionSource.TELEGRAM:
                return self._extract_telegram()
            elif source == ExtractionSource.SIGNAL:
                return self._extract_signal()
            elif source == ExtractionSource.GOOGLE_DRIVE:
                return self._extract_google_drive()
            elif source == ExtractionSource.ONEDRIVE:
                return self._extract_onedrive()
            elif source == ExtractionSource.SYSTEM_LOGS:
                return self._extract_system_logs()
            elif source == ExtractionSource.LOGCAT:
                return self._extract_logcat()
            return []
        except Exception as e:
            logger.error(f"Error extracting from {source.value}: {e}")
            return []
    
    def _extract_whatsapp(self) -> List[ExtractionArtifact]:
        """Extract WhatsApp data from device"""
        logger.info("Extracting WhatsApp data...")
        artifacts = []
        try:
            import subprocess
            
            # WhatsApp database paths on Android
            whatsapp_paths = [
                "/data/data/com.whatsapp/databases/msgstore.db",
                "/data/data/com.whatsapp.w4b/databases/msgstore.db",
                "/data/data/com.whatsapp/files/",
            ]
            
            for path in whatsapp_paths:
                try:
                    # Try to pull WhatsApp data via ADB
                    result = subprocess.run(
                        ['adb', '-s', self.device_id, 'pull', path, '/tmp/whatsapp_extract/'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0:
                        artifact = ExtractionArtifact(
                            artifact_id=f"whatsapp_{len(artifacts)}",
                            source="WhatsApp",
                            data_type="messaging_app",
                            data={"path": path, "status": "extracted"},
                            metadata={"app": "WhatsApp", "extraction_method": "adb_pull"}
                        )
                        artifacts.append(artifact)
                        logger.info(f"✅ Extracted WhatsApp data from {path}")
                except Exception as e:
                    logger.debug(f"Could not extract from {path}: {e}")
            
            # If no database found, create sample artifact
            if not artifacts:
                artifact = ExtractionArtifact(
                    artifact_id="whatsapp_sample",
                    source="WhatsApp",
                    data_type="messaging_app",
                    data={"messages": 0, "contacts": 0, "status": "no_data"},
                    metadata={"app": "WhatsApp", "extraction_method": "adb_pull"}
                )
                artifacts.append(artifact)
        
        except Exception as e:
            logger.error(f"Error extracting WhatsApp: {e}")
        
        return artifacts
    
    def _extract_telegram(self) -> List[ExtractionArtifact]:
        """Extract Telegram data from device"""
        logger.info("Extracting Telegram data...")
        artifacts = []
        try:
            import subprocess
            
            # Telegram database paths on Android
            telegram_paths = [
                "/data/data/org.telegram.messenger/databases/",
                "/data/data/org.telegram.messenger.web/databases/",
            ]
            
            for path in telegram_paths:
                try:
                    result = subprocess.run(
                        ['adb', '-s', self.device_id, 'pull', path, '/tmp/telegram_extract/'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0:
                        artifact = ExtractionArtifact(
                            artifact_id=f"telegram_{len(artifacts)}",
                            source="Telegram",
                            data_type="messaging_app",
                            data={"path": path, "status": "extracted"},
                            metadata={"app": "Telegram", "extraction_method": "adb_pull"}
                        )
                        artifacts.append(artifact)
                        logger.info(f"✅ Extracted Telegram data from {path}")
                except Exception as e:
                    logger.debug(f"Could not extract from {path}: {e}")
            
            if not artifacts:
                artifact = ExtractionArtifact(
                    artifact_id="telegram_sample",
                    source="Telegram",
                    data_type="messaging_app",
                    data={"messages": 0, "chats": 0, "status": "no_data"},
                    metadata={"app": "Telegram", "extraction_method": "adb_pull"}
                )
                artifacts.append(artifact)
        
        except Exception as e:
            logger.error(f"Error extracting Telegram: {e}")
        
        return artifacts
    
    def _extract_signal(self) -> List[ExtractionArtifact]:
        """Extract Signal data from device"""
        logger.info("Extracting Signal data...")
        artifacts = []
        try:
            import subprocess
            
            # Signal database paths on Android
            signal_paths = [
                "/data/data/org.signal.android/databases/",
            ]
            
            for path in signal_paths:
                try:
                    result = subprocess.run(
                        ['adb', '-s', self.device_id, 'pull', path, '/tmp/signal_extract/'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0:
                        artifact = ExtractionArtifact(
                            artifact_id=f"signal_{len(artifacts)}",
                            source="Signal",
                            data_type="messaging_app",
                            data={"path": path, "status": "extracted"},
                            metadata={"app": "Signal", "extraction_method": "adb_pull"}
                        )
                        artifacts.append(artifact)
                        logger.info(f"✅ Extracted Signal data from {path}")
                except Exception as e:
                    logger.debug(f"Could not extract from {path}: {e}")
            
            if not artifacts:
                artifact = ExtractionArtifact(
                    artifact_id="signal_sample",
                    source="Signal",
                    data_type="messaging_app",
                    data={"messages": 0, "conversations": 0, "status": "no_data"},
                    metadata={"app": "Signal", "extraction_method": "adb_pull"}
                )
                artifacts.append(artifact)
        
        except Exception as e:
            logger.error(f"Error extracting Signal: {e}")
        
        return artifacts
    
    def _extract_google_drive(self) -> List[ExtractionArtifact]:
        """Extract Google Drive data from device"""
        logger.info("Extracting Google Drive data...")
        artifacts = []
        try:
            import subprocess
            
            # Google Drive cache paths on Android
            drive_paths = [
                "/data/data/com.google.android.apps.docs/cache/",
                "/data/data/com.google.android.apps.docs/files/",
            ]
            
            for path in drive_paths:
                try:
                    result = subprocess.run(
                        ['adb', '-s', self.device_id, 'pull', path, '/tmp/gdrive_extract/'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0:
                        artifact = ExtractionArtifact(
                            artifact_id=f"gdrive_{len(artifacts)}",
                            source="Google Drive",
                            data_type="cloud_storage",
                            data={"path": path, "status": "extracted"},
                            metadata={"service": "Google Drive", "extraction_method": "adb_pull"}
                        )
                        artifacts.append(artifact)
                        logger.info(f"✅ Extracted Google Drive data from {path}")
                except Exception as e:
                    logger.debug(f"Could not extract from {path}: {e}")
            
            if not artifacts:
                artifact = ExtractionArtifact(
                    artifact_id="gdrive_sample",
                    source="Google Drive",
                    data_type="cloud_storage",
                    data={"files": 0, "status": "no_data"},
                    metadata={"service": "Google Drive", "extraction_method": "adb_pull"}
                )
                artifacts.append(artifact)
        
        except Exception as e:
            logger.error(f"Error extracting Google Drive: {e}")
        
        return artifacts
    
    def _extract_onedrive(self) -> List[ExtractionArtifact]:
        """Extract OneDrive data from device"""
        logger.info("Extracting OneDrive data...")
        artifacts = []
        try:
            import subprocess
            
            # OneDrive cache paths on Android
            onedrive_paths = [
                "/data/data/com.microsoft.skydrive/cache/",
                "/data/data/com.microsoft.skydrive/files/",
            ]
            
            for path in onedrive_paths:
                try:
                    result = subprocess.run(
                        ['adb', '-s', self.device_id, 'pull', path, '/tmp/onedrive_extract/'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0:
                        artifact = ExtractionArtifact(
                            artifact_id=f"onedrive_{len(artifacts)}",
                            source="OneDrive",
                            data_type="cloud_storage",
                            data={"path": path, "status": "extracted"},
                            metadata={"service": "OneDrive", "extraction_method": "adb_pull"}
                        )
                        artifacts.append(artifact)
                        logger.info(f"✅ Extracted OneDrive data from {path}")
                except Exception as e:
                    logger.debug(f"Could not extract from {path}: {e}")
            
            if not artifacts:
                artifact = ExtractionArtifact(
                    artifact_id="onedrive_sample",
                    source="OneDrive",
                    data_type="cloud_storage",
                    data={"files": 0, "status": "no_data"},
                    metadata={"service": "OneDrive", "extraction_method": "adb_pull"}
                )
                artifacts.append(artifact)
        
        except Exception as e:
            logger.error(f"Error extracting OneDrive: {e}")
        
        return artifacts
    
    def _extract_system_logs(self) -> List[ExtractionArtifact]:
        """Extract system logs from device"""
        logger.info("Extracting system logs...")
        artifacts = []
        try:
            import subprocess
            
            # System log paths on Android
            log_paths = [
                "/data/anr/",  # ANR (Application Not Responding) logs
                "/data/tombstones/",  # Crash logs
                "/data/system/log/",  # System logs
            ]
            
            for path in log_paths:
                try:
                    result = subprocess.run(
                        ['adb', '-s', self.device_id, 'pull', path, '/tmp/syslogs_extract/'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0:
                        artifact = ExtractionArtifact(
                            artifact_id=f"syslogs_{len(artifacts)}",
                            source="System Logs",
                            data_type="system_logs",
                            data={"path": path, "status": "extracted"},
                            metadata={"log_type": "system", "extraction_method": "adb_pull"}
                        )
                        artifacts.append(artifact)
                        logger.info(f"✅ Extracted system logs from {path}")
                except Exception as e:
                    logger.debug(f"Could not extract from {path}: {e}")
            
            if not artifacts:
                artifact = ExtractionArtifact(
                    artifact_id="syslogs_sample",
                    source="System Logs",
                    data_type="system_logs",
                    data={"entries": 0, "status": "no_data"},
                    metadata={"log_type": "system", "extraction_method": "adb_pull"}
                )
                artifacts.append(artifact)
        
        except Exception as e:
            logger.error(f"Error extracting system logs: {e}")
        
        return artifacts
    
    def _extract_logcat(self) -> List[ExtractionArtifact]:
        """Extract logcat from device"""
        logger.info("Extracting logcat...")
        artifacts = []
        try:
            import subprocess
            
            # Get logcat output
            result = subprocess.run(
                ['adb', '-s', self.device_id, 'logcat', '-d', '-v', 'threadtime', '*:V'],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0 and result.stdout:
                # Parse logcat output
                logcat_lines = result.stdout.split('\n')
                
                artifact = ExtractionArtifact(
                    artifact_id="logcat_dump",
                    source="Logcat",
                    data_type="system_logs",
                    data={
                        "entries": len(logcat_lines),
                        "status": "extracted",
                        "sample_lines": logcat_lines[:10]  # First 10 lines as sample
                    },
                    metadata={"log_type": "logcat", "extraction_method": "adb_logcat"}
                )
                artifacts.append(artifact)
                logger.info(f"✅ Extracted {len(logcat_lines)} logcat entries")
            else:
                artifact = ExtractionArtifact(
                    artifact_id="logcat_sample",
                    source="Logcat",
                    data_type="system_logs",
                    data={"entries": 0, "status": "no_data"},
                    metadata={"log_type": "logcat", "extraction_method": "adb_logcat"}
                )
                artifacts.append(artifact)
        
        except Exception as e:
            logger.error(f"Error extracting logcat: {e}")
            artifact = ExtractionArtifact(
                artifact_id="logcat_error",
                source="Logcat",
                data_type="system_logs",
                data={"error": str(e), "status": "failed"},
                metadata={"log_type": "logcat", "extraction_method": "adb_logcat"}
            )
            artifacts.append(artifact)
        
        return artifacts


# ============================================================================
# DATA DEDUPLICATOR
# ============================================================================

class DataDeduplicator:
    """Remove duplicate artifacts"""
    
    @staticmethod
    def deduplicate(artifacts: List[ExtractionArtifact]) -> List[ExtractionArtifact]:
        """Remove duplicate artifacts"""
        seen = set()
        deduplicated = []
        
        for artifact in artifacts:
            # Create hash of artifact data
            artifact_hash = hashlib.md5(
                json.dumps(artifact.to_dict(), sort_keys=True).encode()
            ).hexdigest()
            
            if artifact_hash not in seen:
                seen.add(artifact_hash)
                deduplicated.append(artifact)
        
        logger.info(f"Deduplicated {len(artifacts)} artifacts to {len(deduplicated)}")
        return deduplicated


# ============================================================================
# EXTRACTION BRIDGE AGENT
# ============================================================================

class ExtractionBridgeAgent:
    """Core orchestrator for hybrid extraction"""
    
    def __init__(self, device_id: str, case_id: str):
        self.device_id = device_id
        self.case_id = case_id
        self.escalation_manager = PrivilegeEscalationManager(device_id)
        self.extended_extractor = ExtendedSourceExtractor(device_id, case_id)
        self.deduplicator = DataDeduplicator()
        self.result = ExtractionResult(case_id, device_id)
    
    def extract(self, enable_escalation: bool = False, 
                enable_extended_sources: bool = True,
                progress_callback: Optional[Callable[[str, int], None]] = None) -> ExtractionResult:
        """Execute hybrid extraction"""
        
        try:
            if progress_callback:
                progress_callback("Starting hybrid extraction...", 5)
            
            logger.info(f"Starting hybrid extraction for device {self.device_id}")
            
            # Attempt privilege escalation if enabled
            if enable_escalation:
                if progress_callback:
                    progress_callback("Attempting privilege escalation...", 15)
                
                for method in self.escalation_manager.get_available_methods():
                    if self.escalation_manager.attempt_escalation(method):
                        logger.info(f"✅ Escalation successful: {method.value}")
                        break
            
            # Extract from extended sources if enabled
            if enable_extended_sources:
                if progress_callback:
                    progress_callback("Extracting from extended sources...", 40)
                
                for source in ExtractionSource:
                    artifacts = self.extended_extractor.extract_from_source(source)
                    for artifact in artifacts:
                        self.result.add_artifact(artifact)
            
            # Deduplicate artifacts
            if progress_callback:
                progress_callback("Deduplicating artifacts...", 70)
            
            self.result.artifacts = self.deduplicator.deduplicate(self.result.artifacts)
            
            # Mark as complete
            if progress_callback:
                progress_callback("Extraction complete", 100)
            
            self.result.complete()
            logger.info(f"✅ Hybrid extraction completed: {len(self.result.artifacts)} artifacts")
            
            return self.result
        
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}", exc_info=True)
            self.result.add_error(str(e))
            self.result.complete()
            return self.result


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

_bridge_agents: Dict[str, ExtractionBridgeAgent] = {}


def get_bridge_agent(device_id: str, case_id: str) -> ExtractionBridgeAgent:
    """Get or create bridge agent for device"""
    
    agent_key = f"{case_id}:{device_id}"
    
    if agent_key not in _bridge_agents:
        _bridge_agents[agent_key] = ExtractionBridgeAgent(device_id, case_id)
        logger.info(f"✅ Created bridge agent for {device_id}")
    
    return _bridge_agents[agent_key]


def clear_bridge_agents() -> None:
    """Clear all bridge agents"""
    _bridge_agents.clear()
    logger.info("✅ Cleared all bridge agents")
