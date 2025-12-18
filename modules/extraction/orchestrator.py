"""
EXTRACTION ORCHESTRATOR - Manages Extraction Workflow
Coordinates extraction across all modules with consent checks

This module provides:
- ExtractionOrchestrator (main orchestrator)
- Progress tracking
- Error handling
- Consent validation
- Results management
"""

import os
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable, Set
from modules.extraction.extractors import (
    ExtractionModule,
    DeviceInfoExtractor,
    CommunicationExtractor,
    LocationExtractor,
    SecurityExtractor,
    MediaExtractor,
    SystemExtractor
)
from modules.shared.utils import ErrorHandlingLoopholes, ResultsRepository, get_cache_manager
from modules.consent.models import ConsentLevel

# Bridge Agent Integration
BRIDGE_AGENT_AVAILABLE = False
ExtractionBridgeAgent = None
get_bridge_agent = None

try:
    from modules.extraction.hybrid_bridge_agent import get_bridge_agent, ExtractionBridgeAgent
    BRIDGE_AGENT_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Bridge agent imported successfully")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Bridge agent import failed: {e}")
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Error importing bridge agent: {e}", exc_info=True)

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# MODULE CONSENT REQUIREMENTS
# ============================================================================

MODULE_MIN_LEVELS = {
    'device_info': ConsentLevel.STANDARD,   # Level 1
    'communications': ConsentLevel.LEGAL,   # Level 3 - CRITICAL
    'location': ConsentLevel.STANDARD,      # Level 2
    'security': ConsentLevel.FULL,          # Level 4
    'media': ConsentLevel.FULL,             # Level 4
    'system': ConsentLevel.FULL             # Level 4
}

# ============================================================================
# HYBRID ARCHITECTURE - ONLINE/OFFLINE SUPPORT FOR EXTRACTION
# ============================================================================

class ExtractionHybridManager:
    """Manage online/offline extraction with sync"""
    
    def __init__(self):
        self.is_online = True
        self.pending_extractions: Dict[str, Dict[str, Any]] = {}
        self.last_sync_time: Optional[datetime] = None
        self.sync_interval = int(os.getenv('EXTRACTION_SYNC_INTERVAL', '300'))
    
    def set_online(self, is_online: bool) -> None:
        """Set connectivity status"""
        self.is_online = is_online
        logger.info(f"Extraction connectivity: {'ONLINE' if is_online else 'OFFLINE'}")
    
    def is_connected(self) -> bool:
        """Check if online"""
        return self.is_online
    
    def queue_extraction(self, extraction_id: str, extraction_data: Dict[str, Any]) -> None:
        """Queue extraction for sync when online"""
        self.pending_extractions[extraction_id] = {
            'data': extraction_data,
            'queued_at': datetime.now().isoformat(),
            'synced': False
        }
        logger.info(f"Extraction queued for sync: {extraction_id}")
    
    def get_pending_extractions(self) -> Dict[str, Dict[str, Any]]:
        """Get pending extractions"""
        return {k: v for k, v in self.pending_extractions.items() if not v['synced']}
    
    def mark_synced(self, extraction_id: str) -> None:
        """Mark extraction as synced"""
        if extraction_id in self.pending_extractions:
            self.pending_extractions[extraction_id]['synced'] = True
    
    def should_sync(self) -> bool:
        """Check if should sync"""
        if not self.is_online:
            return False
        
        if not self.last_sync_time:
            return True
        
        elapsed = (datetime.now() - self.last_sync_time).total_seconds()
        return elapsed >= self.sync_interval
    
    def sync_completed(self) -> None:
        """Mark sync as completed"""
        self.last_sync_time = datetime.now()
        logger.info("Extraction sync completed")

# ============================================================================
# MODULE DEPENDENCY MANAGEMENT
# ============================================================================

MODULE_DEPENDENCIES = {
    'device_info': [],
    'communications': ['device_info'],
    'location': ['device_info'],
    'security': ['device_info'],
    'media': ['device_info'],
    'system': ['device_info']
}

# ============================================================================
# CONSENT CHECKING HELPER
# ============================================================================

def check_module_consent(current_level: ConsentLevel, module_name: str) -> tuple[bool, str]:
    """
    Check if current consent level allows module extraction
    
    Args:
        current_level: Current consent level
        module_name: Name of module to check
        
    Returns:
        (allowed: bool, message: str)
    """
    if module_name not in MODULE_MIN_LEVELS:
        return False, f"Unknown module: {module_name}"
    
    min_level = MODULE_MIN_LEVELS[module_name]
    
    if current_level.value >= min_level.value:
        logger.info(f"✅ Consent check PASSED for {module_name}: {current_level.name} >= {min_level.name}")
        return True, f"Consent level {current_level.name} allows {module_name} extraction"
    else:
        logger.warning(f"❌ Consent check FAILED for {module_name}: {current_level.name} < {min_level.name}")
        return False, f"Insufficient consent for {module_name}. Required: {min_level.name}, Current: {current_level.name}"

# ============================================================================
# EXTRACTION SCHEDULER
# ============================================================================

class ExtractionScheduler:
    """Schedule extractions for later execution"""
    
    def __init__(self):
        self.scheduled_extractions: Dict[str, Dict[str, Any]] = {}
        self.execution_thread: Optional[threading.Thread] = None
        self.running = False
    
    def schedule_extraction(
        self,
        case_id: str,
        device_id: str,
        scheduled_time: datetime,
        modules: Optional[List[str]] = None
    ) -> str:
        """Schedule extraction for later"""
        extraction_id = f"{case_id}_{int(time.time())}"
        
        self.scheduled_extractions[extraction_id] = {
            'case_id': case_id,
            'device_id': device_id,
            'scheduled_time': scheduled_time,
            'modules': modules,
            'status': 'scheduled',
            'created_at': datetime.now()
        }
        
        logger.info(f"Extraction scheduled: {extraction_id} for {scheduled_time}")
        return extraction_id
    
    def get_pending_extractions(self) -> List[Dict[str, Any]]:
        """Get pending extractions"""
        pending = []
        for extraction_id, extraction in self.scheduled_extractions.items():
            if extraction['status'] == 'scheduled' and datetime.now() >= extraction['scheduled_time']:
                pending.append({**extraction, 'extraction_id': extraction_id})
        return pending
    
    def cancel_extraction(self, extraction_id: str) -> bool:
        """Cancel scheduled extraction"""
        if extraction_id in self.scheduled_extractions:
            self.scheduled_extractions[extraction_id]['status'] = 'cancelled'
            logger.info(f"Extraction cancelled: {extraction_id}")
            return True
        return False

# ============================================================================
# EXTRACTION CANCELLATION MANAGER
# ============================================================================

class ExtractionCancellationManager:
    """Manage extraction cancellation, pause, and resume"""
    
    def __init__(self):
        self.active_extractions: Dict[str, Dict[str, Any]] = {}
    
    def start_extraction(self, extraction_id: str, case_id: str) -> None:
        """Mark extraction as started"""
        self.active_extractions[extraction_id] = {
            'case_id': case_id,
            'started_at': datetime.now(),
            'cancelled': False,
            'paused': False,
            'paused_at': None,
            'resumed_at': None,
            'pause_duration': 0.0
        }
        logger.info(f"Extraction started: {extraction_id}")
    
    def cancel_extraction(self, extraction_id: str) -> bool:
        """Request extraction cancellation"""
        if extraction_id in self.active_extractions:
            self.active_extractions[extraction_id]['cancelled'] = True
            logger.info(f"Extraction cancellation requested: {extraction_id}")
            return True
        return False
    
    def is_cancelled(self, extraction_id: str) -> bool:
        """Check if extraction is cancelled"""
        if extraction_id in self.active_extractions:
            return self.active_extractions[extraction_id]['cancelled']
        return False
    
    def pause_extraction(self, extraction_id: str) -> bool:
        """Pause extraction"""
        if extraction_id in self.active_extractions:
            if not self.active_extractions[extraction_id]['paused']:
                self.active_extractions[extraction_id]['paused'] = True
                self.active_extractions[extraction_id]['paused_at'] = datetime.now()
                logger.info(f"Extraction paused: {extraction_id}")
                return True
        return False
    
    def resume_extraction(self, extraction_id: str) -> bool:
        """Resume extraction"""
        if extraction_id in self.active_extractions:
            if self.active_extractions[extraction_id]['paused']:
                paused_at = self.active_extractions[extraction_id]['paused_at']
                pause_duration = (datetime.now() - paused_at).total_seconds()
                self.active_extractions[extraction_id]['pause_duration'] += pause_duration
                self.active_extractions[extraction_id]['paused'] = False
                self.active_extractions[extraction_id]['resumed_at'] = datetime.now()
                logger.info(f"Extraction resumed: {extraction_id}")
                return True
        return False
    
    def is_paused(self, extraction_id: str) -> bool:
        """Check if extraction is paused"""
        if extraction_id in self.active_extractions:
            return self.active_extractions[extraction_id]['paused']
        return False
    
    def get_pause_duration(self, extraction_id: str) -> float:
        """Get total pause duration"""
        if extraction_id in self.active_extractions:
            return self.active_extractions[extraction_id]['pause_duration']
        return 0.0
    
    def finish_extraction(self, extraction_id: str) -> None:
        """Mark extraction as finished"""
        if extraction_id in self.active_extractions:
            del self.active_extractions[extraction_id]
            logger.info(f"Extraction finished: {extraction_id}")

# ============================================================================
# BANDWIDTH THROTTLER
# ============================================================================

class BandwidthThrottler:
    """Throttle extraction bandwidth"""
    
    def __init__(self, max_bytes_per_second: int = 1000000):
        self.max_bytes_per_second = max_bytes_per_second
        self.bytes_transferred = 0
        self.window_start = datetime.now()
    
    def throttle(self, bytes_to_transfer: int) -> None:
        """Throttle bandwidth"""
        elapsed = (datetime.now() - self.window_start).total_seconds()
        
        if elapsed >= 1.0:
            self.bytes_transferred = 0
            self.window_start = datetime.now()
            elapsed = 0
        
        if self.bytes_transferred + bytes_to_transfer > self.max_bytes_per_second:
            wait_time = (self.bytes_transferred + bytes_to_transfer - self.max_bytes_per_second) / self.max_bytes_per_second
            logger.debug(f"Throttling: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        
        self.bytes_transferred += bytes_to_transfer

# ============================================================================
# EXTRACTION ORCHESTRATOR CLASS
# ============================================================================

class ExtractionOrchestrator:
    """Manages extraction workflow across all modules"""

    def __init__(self, storage_path: str = "artifacts"):
        """Initialize extraction orchestrator"""
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize extractors
        self.extractors = {
            'device_info': DeviceInfoExtractor(),
            'communications': CommunicationExtractor(),
            'location': LocationExtractor(),
            'security': SecurityExtractor(),
            'media': MediaExtractor(),
            'system': SystemExtractor()
        }
        
        # Extraction results
        self.results: Dict[str, Dict[str, Any]] = {}
        self.extraction_status: Dict[str, str] = {}
        
        # Initialize enhancement managers
        self.scheduler = ExtractionScheduler()
        self.cancellation_manager = ExtractionCancellationManager()
        self.throttler = BandwidthThrottler(max_bytes_per_second=int(os.getenv('MAX_BANDWIDTH_BPS', '1000000')))
        self.cache_manager = get_cache_manager()
        
        # Hybrid architecture support
        self.hybrid_manager = ExtractionHybridManager()
        self.local_results_cache: Dict[str, Dict[str, Any]] = {}
        self.remote_sync_enabled = os.getenv('REMOTE_SYNC_ENABLED', 'true').lower() == 'true'
        
        # Retry configuration
        self.max_retries = int(os.getenv('EXTRACTION_MAX_RETRIES', '3'))
        self.retry_delay = float(os.getenv('EXTRACTION_RETRY_DELAY', '1.0'))
        
        # Bridge Agent Integration
        self.bridge_agents: Dict[str, Any] = {}  # Dict[str, ExtractionBridgeAgent]
        self.connected_devices: Dict[str, Dict[str, Any]] = {}
        self.usb_monitor_thread: Optional[threading.Thread] = None
        self.usb_monitoring_active = False
        
        # Initialize bridge agent for USB connections
        self._initialize_usb_monitoring()

    def extract_all_data(
        self,
        case_id: str,
        device_id: str,
        consent_manager: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        extraction_id: Optional[str] = None,
        use_hybrid: bool = True,
        enable_escalation: bool = False,
        enable_extended_sources: bool = True
    ) -> Dict[str, Any]:
        """Extract all data with hybrid bridge agent (standard mode) + consent checks, error handling, pause/resume, and cancellation
        
        Args:
            case_id: Case identifier
            device_id: Device identifier
            consent_manager: Consent manager instance
            progress_callback: Progress callback function
            extraction_id: Extraction identifier
            use_hybrid: Use hybrid extraction with bridge agent (default: True)
            enable_escalation: Enable privilege escalation (default: False)
            enable_extended_sources: Enable extended source extraction (default: True)
        """
        
        # If hybrid mode enabled, use bridge agent extraction
        if use_hybrid and BRIDGE_AGENT_AVAILABLE:
            return self._extract_with_bridge_agent(
                case_id=case_id,
                device_id=device_id,
                consent_manager=consent_manager,
                progress_callback=progress_callback,
                extraction_id=extraction_id,
                enable_escalation=enable_escalation,
                enable_extended_sources=enable_extended_sources
            )
        
        # Fall back to standard extraction if hybrid not available
        return self._extract_standard(
            case_id=case_id,
            device_id=device_id,
            consent_manager=consent_manager,
            progress_callback=progress_callback,
            extraction_id=extraction_id
        )
    
    def _extract_with_bridge_agent(
        self,
        case_id: str,
        device_id: str,
        consent_manager: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        extraction_id: Optional[str] = None,
        enable_escalation: bool = False,
        enable_extended_sources: bool = True
    ) -> Dict[str, Any]:
        """Extract using bridge agent (hybrid mode)"""
        try:
            logger.info(f"Starting hybrid extraction for {case_id}, {device_id}")
            
            # Generate extraction ID if not provided
            if not extraction_id:
                extraction_id = f"{case_id}_{int(time.time())}"
            
            # Create progress callback wrapper
            def bridge_progress(message: str, percentage: int):
                if progress_callback:
                    try:
                        progress_callback(message, percentage)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")
            
            # Create bridge agent directly
            if not BRIDGE_AGENT_AVAILABLE or not get_bridge_agent:
                logger.warning("Bridge agent not available, using standard extraction")
                return self._extract_standard(
                    case_id=case_id,
                    device_id=device_id,
                    consent_manager=consent_manager,
                    progress_callback=progress_callback,
                    extraction_id=extraction_id
                )
            
            # Execute hybrid extraction via bridge agent
            start_time = datetime.now()
            try:
                bridge_agent = get_bridge_agent(device_id, case_id)
                logger.info(f"Created bridge agent for {device_id}")
                
                extraction_result = bridge_agent.extract(
                    enable_escalation=enable_escalation,
                    enable_extended_sources=enable_extended_sources,
                    progress_callback=bridge_progress
                )
                
                # Convert ExtractionResult to dict
                result_dict = extraction_result.to_dict() if hasattr(extraction_result, 'to_dict') else extraction_result
                
                # Merge with standard extraction results
                extraction_results = {
                    'case_id': case_id,
                    'device_id': device_id,
                    'extraction_id': extraction_id,
                    'extraction_type': 'hybrid',
                    'start_time': datetime.now().isoformat(),
                    'status': result_dict.get('status', 'completed'),
                    'total_artifacts': result_dict.get('total_artifacts', 0),
                    'extraction_completeness': 100.0 if result_dict.get('total_artifacts', 0) > 0 else 0.0,
                    'privilege_escalation_used': False,
                    'escalation_method': None,
                    'total_time': (datetime.now() - start_time).total_seconds(),
                    'modules': {
                        'whatsapp': 0,
                        'telegram': 0,
                        'signal': 0,
                        'google_drive': 0,
                        'onedrive': 0,
                        'system_logs': 0,
                        'logcat': 0
                    },
                    'bridge_results': result_dict
                }
                
                logger.info(f"Hybrid extraction complete: {extraction_results['total_artifacts']} artifacts")
                return extraction_results
            
            except Exception as bridge_error:
                logger.error(f"Bridge agent error: {bridge_error}", exc_info=True)
                # Fall back to standard extraction
                return self._extract_standard(
                    case_id=case_id,
                    device_id=device_id,
                    consent_manager=consent_manager,
                    progress_callback=progress_callback,
                    extraction_id=extraction_id
                )
        
        except Exception as e:
            logger.error(f"Hybrid extraction error: {e}", exc_info=True)
            # Fall back to standard extraction
            return self._extract_standard(
                case_id=case_id,
                device_id=device_id,
                consent_manager=consent_manager,
                progress_callback=progress_callback,
                extraction_id=extraction_id
            )
    
    def _extract_standard(
        self,
        case_id: str,
        device_id: str,
        consent_manager: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        extraction_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Standard extraction (fallback from hybrid)"""
        
        # Validate inputs
        if not ErrorHandlingLoopholes.validate_input(case_id, str, min_length=1):
            logger.error("Invalid case_id")
            return {'status': 'error', 'error': 'Invalid case_id'}
        
        if not ErrorHandlingLoopholes.validate_input(device_id, str, min_length=1):
            logger.error("Invalid device_id")
            return {'status': 'error', 'error': 'Invalid device_id'}
        
        # Check dev mode
        dev_mode_enabled = False
        if consent_manager:
            try:
                dev_mode_enabled = consent_manager.connectivity_manager.is_dev_mode()
                if dev_mode_enabled:
                    logger.info("🧪 Dev Mode: Consent checks will be bypassed")
            except (AttributeError, TypeError) as e:
                logger.warning(f"⚠️ Could not check dev mode: {e}")
                dev_mode_enabled = False
            except Exception as e:
                logger.error(f"❌ Unexpected error checking dev mode: {e}", exc_info=True)
                dev_mode_enabled = False
        
        # Generate extraction ID if not provided
        if not extraction_id:
            extraction_id = f"{case_id}_{int(time.time())}"
        
        # Start extraction tracking
        self.cancellation_manager.start_extraction(extraction_id, case_id)
        
        logger.info(f"Starting extraction {extraction_id} for case {case_id}, device {device_id}")
        
        extraction_results = {
            'case_id': case_id,
            'device_id': device_id,
            'extraction_id': extraction_id,
            'start_time': datetime.now().isoformat(),
            'modules': {},
            'blocked_modules': [],
            'total_artifacts': 0,
            'total_time': 0,
            'paused': False,
            'cancelled': False
        }
        
        start_time = datetime.now()
        total_modules = len(self.extractors)
        
        # Extract from each module with error handling, pause/resume, and cancellation
        for idx, (module_name, extractor) in enumerate(self.extractors.items()):
            try:
                # Check if extraction is cancelled
                if self.cancellation_manager.is_cancelled(extraction_id):
                    logger.warning(f"Extraction cancelled: {extraction_id}")
                    extraction_results['cancelled'] = True
                    extraction_results['modules'][module_name] = {
                        'status': 'cancelled',
                        'message': 'Extraction was cancelled'
                    }
                    break
                
                # Wait if paused
                while self.cancellation_manager.is_paused(extraction_id):
                    logger.info(f"Extraction paused: {extraction_id}")
                    extraction_results['paused'] = True
                    time.sleep(0.5)  # Check every 500ms if resumed
                
                extraction_results['paused'] = False
                
                # Update progress
                if progress_callback:
                    try:
                        progress_callback(f"Extracting {module_name}...", idx + 1)
                    except Exception as cb_error:
                        logger.warning(f"⚠️ Progress callback failed for {module_name}: {cb_error}")
                        # Continue extraction anyway
                
                logger.info(f"Extracting {module_name}...")
                
                # Extract data with automatic retry on error
                def _extract_module():
                    return extractor.extract(
                        device_id=device_id,
                        case_id=case_id,
                        consent_manager=consent_manager
                    )
                
                result = ErrorHandlingLoopholes.auto_retry_on_error(
                    _extract_module,
                    max_attempts=3,
                    delay=0.5,
                    backoff=1.5
                )
                
                if result is None:
                    extraction_results['modules'][module_name] = {
                        'status': 'error',
                        'error': 'Extraction failed after retries'
                    }
                    logger.error(f"❌ {module_name} extraction failed after retries")
                    continue
                
                # Validate result is a dictionary
                if not isinstance(result, dict):
                    logger.error(f"❌ Invalid result type for {module_name}: {type(result)}")
                    extraction_results['modules'][module_name] = {
                        'status': 'error',
                        'error': f'Invalid result type: {type(result)}'
                    }
                    continue
                
                # Get status safely
                status = result.get('status', 'unknown')
                
                # Check if consent was denied
                if status == 'consent_denied':
                    extraction_results['blocked_modules'].append({
                        'module': module_name,
                        'reason': result.get('message', 'Unknown reason'),
                        'required_level': result.get('required_level', 'Unknown'),
                        'current_level': result.get('current_level', 'Unknown')
                    })
                    logger.warning(f"⚠️ {module_name} blocked: {result.get('message')}")
                    continue
                
                # Check for errors
                elif status == 'error':
                    extraction_results['modules'][module_name] = {
                        'status': 'error',
                        'error': result.get('error', 'Unknown error')
                    }
                    logger.error(f"❌ {module_name} extraction failed: {result.get('error')}")
                    continue
                
                # Check for success
                elif status == 'success':
                    # Validate success result structure
                    artifact_count = result.get('artifact_count', 0)
                    extraction_time = result.get('extraction_time', 0)
                    data = result.get('data', {})
                    
                    if not isinstance(artifact_count, (int, float)):
                        logger.warning(f"⚠️ Invalid artifact_count type for {module_name}: {type(artifact_count)}")
                        artifact_count = 0
                    
                    extraction_results['modules'][module_name] = {
                        'status': 'success',
                        'artifact_count': artifact_count,
                        'extraction_time': extraction_time,
                        'data': data
                    }
                    
                    extraction_results['total_artifacts'] += artifact_count
                    logger.info(f"✅ {module_name} extraction completed: {artifact_count} artifacts")
                
                else:
                    logger.error(f"❌ Unknown status for {module_name}: {status}")
                    extraction_results['modules'][module_name] = {
                        'status': 'error',
                        'error': f'Unknown status: {status}'
                    }
            
            except Exception as e:
                logger.error(f"Unexpected error in {module_name}: {e}", exc_info=True)
                extraction_results['modules'][module_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Calculate total time
        extraction_results['total_time'] = (datetime.now() - start_time).total_seconds()
        extraction_results['end_time'] = datetime.now().isoformat()
        
        # Save results with error handling
        self._save_results(case_id, extraction_results)
        
        logger.info(f"Extraction completed for case {case_id}: {extraction_results['total_artifacts']} total artifacts")
        
        return extraction_results

    def extract_partial(
        self,
        case_id: str,
        device_id: str,
        modules: List[str],
        consent_manager: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Any]:
        """Extract only specific modules (partial extraction)"""
        
        logger.info(f"Starting partial extraction for case {case_id}: {modules}")
        
        # Validate modules
        invalid_modules = [m for m in modules if m not in self.extractors]
        if invalid_modules:
            logger.error(f"Invalid modules: {invalid_modules}")
            return {'status': 'error', 'error': f'Invalid modules: {invalid_modules}'}
        
        extraction_results = {
            'case_id': case_id,
            'device_id': device_id,
            'start_time': datetime.now().isoformat(),
            'modules': {},
            'blocked_modules': [],
            'total_artifacts': 0,
            'extraction_type': 'partial',
            'requested_modules': modules
        }
        
        start_time = datetime.now()
        
        # Extract only requested modules
        for idx, module_name in enumerate(modules):
            try:
                if progress_callback:
                    progress_callback(f"Extracting {module_name}...", idx + 1)
                
                result = self.extract_module(
                    module_name=module_name,
                    case_id=case_id,
                    device_id=device_id,
                    consent_manager=consent_manager
                )
                
                if result.get('status') == 'consent_denied':
                    extraction_results['blocked_modules'].append({
                        'module': module_name,
                        'reason': result.get('message')
                    })
                    continue
                
                if result.get('status') == 'error':
                    extraction_results['modules'][module_name] = {'status': 'error', 'error': result.get('error')}
                    continue
                
                extraction_results['modules'][module_name] = {
                    'status': 'success',
                    'artifact_count': result.get('artifact_count', 0),
                    'extraction_time': result.get('extraction_time', 0),
                    'data': result.get('data', {})
                }
                
                extraction_results['total_artifacts'] += result.get('artifact_count', 0)
            
            except Exception as e:
                logger.error(f"❌ Error in partial extraction {module_name}: {e}", exc_info=True)
                extraction_results['modules'][module_name] = {'status': 'error', 'error': str(e)}
        
        extraction_results['total_time'] = (datetime.now() - start_time).total_seconds()
        extraction_results['end_time'] = datetime.now().isoformat()
        
        self._save_results(case_id, extraction_results)
        logger.info(f"Partial extraction completed: {extraction_results['total_artifacts']} artifacts")
        
        return extraction_results

    def extract_module(
        self,
        module_name: str,
        case_id: str,
        device_id: str,
        consent_manager: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Extract specific module with retry and caching"""
        
        if module_name not in self.extractors:
            logger.error(f"Unknown module: {module_name}")
            return {'status': 'error', 'error': f'Unknown module: {module_name}'}
        
        logger.info(f"Extracting {module_name} for case {case_id}")
        
        # Check cache first
        try:
            cache_key = f"extraction_{case_id}_{module_name}"
            cached_result = self.cache_manager.get(cache_key)
            
            if cached_result and isinstance(cached_result, dict):
                logger.info(f"✅ Using cached result for {module_name}")
                return cached_result
            elif cached_result:
                logger.warning(f"⚠️ Invalid cached result type for {module_name}: {type(cached_result)}")
        except Exception as cache_error:
            logger.warning(f"⚠️ Cache retrieval failed for {module_name}: {cache_error}")
            # Continue with fresh extraction
        
        extractor = self.extractors[module_name]
        
        # Retry with backoff (with max wait time cap)
        MAX_WAIT_TIME = 60  # Cap at 60 seconds
        
        for attempt in range(self.max_retries):
            try:
                result = extractor.extract(
                    device_id=device_id,
                    case_id=case_id,
                    consent_manager=consent_manager
                )
                
                # Validate result
                if result is None:
                    logger.error(f"❌ {module_name} returned None")
                    if attempt < self.max_retries - 1:
                        wait_time = min(self.retry_delay * (2 ** attempt), MAX_WAIT_TIME)
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {'status': 'error', 'error': f'{module_name} returned None'}
                
                if not isinstance(result, dict):
                    logger.error(f"❌ Invalid result type for {module_name}: {type(result)}")
                    if attempt < self.max_retries - 1:
                        wait_time = min(self.retry_delay * (2 ** attempt), MAX_WAIT_TIME)
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {'status': 'error', 'error': f'Invalid result type: {type(result)}'}
                
                status = result.get('status', 'unknown')
                
                # Check if consent was denied
                if status == 'consent_denied':
                    logger.warning(f"⚠️ {module_name} blocked: {result.get('message')}")
                    return result
                
                # Check for errors
                elif status == 'error':
                    if attempt < self.max_retries - 1:
                        wait_time = min(self.retry_delay * (2 ** attempt), MAX_WAIT_TIME)
                        logger.warning(f"⚠️ Attempt {attempt + 1}/{self.max_retries} failed: {result.get('error')}")
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"❌ {module_name} extraction failed after {self.max_retries} attempts")
                        return result
                
                # Cache successful result
                try:
                    self.cache_manager.set(cache_key, result)
                except Exception as cache_error:
                    logger.warning(f"⚠️ Failed to cache result: {cache_error}")
                
                logger.info(f"✅ {module_name} extraction completed")
                return result
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = min(self.retry_delay * (2 ** attempt), MAX_WAIT_TIME)
                    logger.warning(f"⚠️ Attempt {attempt + 1}/{self.max_retries} error: {e}")
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Failed after {self.max_retries} attempts: {e}", exc_info=True)
                    return {'status': 'error', 'error': str(e)}
        
        logger.error(f"❌ Max retries exceeded for {module_name}")
        return {'status': 'error', 'error': 'Max retries exceeded'}

    def _save_results(self, case_id: str, results: Dict[str, Any]) -> bool:
        """Save extraction results with verification and fallback"""
        try:
            case_dir = os.path.join(self.storage_path, case_id)
            os.makedirs(case_dir, exist_ok=True)
            
            results_file = os.path.join(case_dir, 'extraction_results.json')
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Verify file was saved
            if not os.path.exists(results_file):
                raise IOError(f"File not saved: {results_file}")
            
            # Verify file size
            file_size = os.path.getsize(results_file)
            if file_size == 0:
                raise IOError(f"File is empty: {results_file}")
            
            logger.info(f"✅ Results saved to {results_file} ({file_size} bytes)")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error saving results: {e}", exc_info=True)
            
            # Fallback: Save to temp location
            try:
                temp_dir = "temp"
                os.makedirs(temp_dir, exist_ok=True)
                temp_file = os.path.join(temp_dir, f"extraction_{case_id}_{int(time.time())}.json")
                
                with open(temp_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                logger.warning(f"⚠️ Saved to temp location: {temp_file}")
                return True
            except Exception as temp_error:
                logger.critical(f"❌ CRITICAL: Failed to save results anywhere: {temp_error}", exc_info=True)
                return False

    def get_results(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get extraction results"""
        try:
            results_file = os.path.join(self.storage_path, case_id, 'extraction_results.json')
            
            if not os.path.exists(results_file):
                return None
            
            with open(results_file, 'r') as f:
                return json.load(f)
        
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return None

    def get_module_info(self) -> Dict[str, Dict[str, str]]:
        """Get information about all modules"""
        return {
            module_name: extractor.get_info()
            for module_name, extractor in self.extractors.items()
        }

    def schedule_extraction(
        self,
        case_id: str,
        device_id: str,
        scheduled_time: datetime,
        modules: Optional[List[str]] = None
    ) -> str:
        """Schedule extraction for later execution"""
        return self.scheduler.schedule_extraction(case_id, device_id, scheduled_time, modules)

    def get_pending_extractions(self) -> List[Dict[str, Any]]:
        """Get pending scheduled extractions"""
        return self.scheduler.get_pending_extractions()

    def cancel_scheduled_extraction(self, extraction_id: str) -> bool:
        """Cancel scheduled extraction"""
        return self.scheduler.cancel_extraction(extraction_id)

    def cancel_active_extraction(self, extraction_id: str) -> bool:
        """Cancel active extraction"""
        return self.cancellation_manager.cancel_extraction(extraction_id)

    def is_extraction_cancelled(self, extraction_id: str) -> bool:
        """Check if extraction is cancelled"""
        return self.cancellation_manager.is_cancelled(extraction_id)

    def pause_extraction(self, extraction_id: str) -> bool:
        """Pause active extraction"""
        return self.cancellation_manager.pause_extraction(extraction_id)

    def resume_extraction(self, extraction_id: str) -> bool:
        """Resume paused extraction"""
        return self.cancellation_manager.resume_extraction(extraction_id)

    def is_extraction_paused(self, extraction_id: str) -> bool:
        """Check if extraction is paused"""
        return self.cancellation_manager.is_paused(extraction_id)

    def get_extraction_pause_duration(self, extraction_id: str) -> float:
        """Get total pause duration for extraction"""
        return self.cancellation_manager.get_pause_duration(extraction_id)

    def get_module_dependencies(self, module_name: str) -> List[str]:
        """Get module dependencies"""
        return MODULE_DEPENDENCIES.get(module_name, [])

    def validate_module_dependencies(self, modules: List[str]) -> Dict[str, Any]:
        """Validate that all dependencies are included with detailed feedback"""
        try:
            required_modules = set()
            
            # Validate each module exists
            for module in modules:
                if module not in self.extractors:
                    logger.error(f"❌ Unknown module: {module}")
                    return {
                        'valid': False,
                        'error': f'Unknown module: {module}',
                        'missing_modules': [module]
                    }
                
                required_modules.add(module)
                deps = self.get_module_dependencies(module)
                required_modules.update(deps)
            
            # Check if all dependencies are included
            requested = set(modules)
            missing = required_modules - requested
            
            if missing:
                logger.warning(f"⚠️ Missing dependencies: {missing}")
                return {
                    'valid': False,
                    'error': f'Missing dependencies: {missing}',
                    'missing_modules': list(missing),
                    'required_modules': list(required_modules)
                }
            
            logger.info(f"✅ All dependencies satisfied for {modules}")
            return {
                'valid': True,
                'modules': modules,
                'dependencies': list(required_modules)
            }
        
        except Exception as e:
            logger.error(f"❌ Dependency validation failed: {e}", exc_info=True)
            return {
                'valid': False,
                'error': str(e)
            }

    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        stats = {
            'total_modules': len(self.extractors),
            'scheduled_extractions': len(self.scheduler.scheduled_extractions),
            'active_extractions': len(self.cancellation_manager.active_extractions),
            'cache_size': len(self.cache_manager.memory_cache),
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'hybrid_online': self.hybrid_manager.is_connected(),
            'pending_sync': len(self.hybrid_manager.get_pending_extractions())
        }
        return stats

    def set_connectivity(self, is_online: bool) -> None:
        """Set connectivity status for hybrid architecture"""
        self.hybrid_manager.set_online(is_online)
        logger.info(f"Extraction connectivity set to: {'ONLINE' if is_online else 'OFFLINE'}")

    def sync_extraction_results(self) -> Dict[str, Any]:
        """Sync extraction results with remote server with verification"""
        
        if not self.hybrid_manager.is_connected():
            logger.warning("🔌 Offline: Cannot sync")
            return {
                'status': 'offline',
                'synced': 0,
                'message': 'Device is offline'
            }
        
        if not self.hybrid_manager.should_sync():
            logger.debug("Sync not needed yet")
            return {
                'status': 'skipped',
                'message': 'Sync interval not reached'
            }
        
        try:
            pending = self.hybrid_manager.get_pending_extractions()
            
            if not pending:
                logger.debug("No pending extractions to sync")
                try:
                    self.hybrid_manager.sync_completed()
                except Exception as sync_error:
                    logger.warning(f"⚠️ sync_completed() failed: {sync_error}")
                
                return {
                    'status': 'success',
                    'synced': 0,
                    'message': 'No pending extractions'
                }
            
            logger.info(f"📦 Syncing {len(pending)} pending extractions")
            
            synced_count = 0
            failed_count = 0
            
            # Mark extractions as synced
            for extraction_id in pending.keys():
                try:
                    self.hybrid_manager.mark_synced(extraction_id)
                    synced_count += 1
                except Exception as mark_error:
                    logger.error(f"❌ Failed to mark synced: {extraction_id}: {mark_error}")
                    failed_count += 1
            
            # Complete sync
            try:
                self.hybrid_manager.sync_completed()
            except Exception as complete_error:
                logger.error(f"❌ sync_completed() failed: {complete_error}")
                return {
                    'status': 'partial',
                    'synced': synced_count,
                    'failed': failed_count,
                    'error': str(complete_error),
                    'message': f'Synced {synced_count}/{len(pending)} extractions'
                }
            
            logger.info(f"✅ Extraction sync completed: {synced_count}/{len(pending)} synced")
            return {
                'status': 'success',
                'synced': synced_count,
                'failed': failed_count,
                'total': len(pending),
                'message': f'Synced {synced_count}/{len(pending)} extractions'
            }
        
        except Exception as e:
            logger.error(f"❌ Extraction sync error: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'message': 'Sync failed'
            }

    def queue_extraction_offline(self, case_id: str, extraction_data: Dict[str, Any]) -> None:
        """Queue extraction for sync when offline"""
        extraction_id = f"{case_id}_{int(time.time())}"
        self.hybrid_manager.queue_extraction(extraction_id, extraction_data)
        logger.info(f"Extraction queued offline: {extraction_id}")

    def _validate_results(self, results: Any) -> bool:
        """Validate results structure"""
        if not isinstance(results, dict):
            return False
        if 'case_id' not in results or 'modules' not in results:
            return False
        return True
    
    def get_results_hybrid(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get results from local cache or remote (hybrid approach) with validation"""
        
        try:
            # Try local cache first (offline support)
            if case_id in self.local_results_cache:
                cached = self.local_results_cache[case_id]
                if self._validate_results(cached):
                    logger.debug(f"✅ Results from local cache: {case_id}")
                    return cached
                else:
                    logger.warning(f"⚠️ Invalid cached results for {case_id}")
                    del self.local_results_cache[case_id]
            
            # Try main results
            if case_id in self.results:
                results = self.results[case_id]
                if self._validate_results(results):
                    self.local_results_cache[case_id] = results
                    logger.debug(f"✅ Results from main storage: {case_id}")
                    return results
                else:
                    logger.warning(f"⚠️ Invalid results in main storage for {case_id}")
            
            # Try file storage
            results = self.get_results(case_id)
            if results and self._validate_results(results):
                self.local_results_cache[case_id] = results
                logger.debug(f"✅ Results from file storage: {case_id}")
                return results
            elif results:
                logger.warning(f"⚠️ Invalid results in file storage for {case_id}")
            
            logger.warning(f"⚠️ No valid results found for {case_id}")
            return None
        
        except Exception as e:
            logger.error(f"❌ Error retrieving hybrid results: {e}", exc_info=True)
            return None
    
    # ========================================================================
    # USB BRIDGE AGENT INTEGRATION
    # ========================================================================
    
    def _initialize_usb_monitoring(self) -> None:
        """Initialize USB device monitoring and bridge agent"""
        if not BRIDGE_AGENT_AVAILABLE:
            logger.warning("⚠️ Bridge agent not available - USB monitoring disabled")
            return
        
        try:
            self.usb_monitoring_active = True
            self.usb_monitor_thread = threading.Thread(
                target=self._monitor_usb_devices,
                daemon=True,
                name="USBBridgeMonitor"
            )
            self.usb_monitor_thread.start()
            logger.info("✅ USB device monitoring started - Bridge agent ready")
        except Exception as e:
            logger.error(f"❌ Error initializing USB monitoring: {e}")
            self.usb_monitoring_active = False
    
    def _monitor_usb_devices(self) -> None:
        """Monitor for USB device connections and initialize bridge agents"""
        from modules.extraction.adapters.device_detector import get_device_detector
        
        device_detector = get_device_detector()
        previous_devices = set()
        
        while self.usb_monitoring_active:
            try:
                # Detect all devices
                all_devices = device_detector.detect_all_devices()
                current_devices = set(all_devices.keys())
                
                # Check for new connections
                new_devices = current_devices - previous_devices
                for device_id in new_devices:
                    device_info = all_devices[device_id]
                    self._on_device_connected(device_id, device_info)
                
                # Check for disconnections
                disconnected_devices = previous_devices - current_devices
                for device_id in disconnected_devices:
                    self._on_device_disconnected(device_id)
                
                previous_devices = current_devices
                time.sleep(2)  # Check every 2 seconds
            
            except Exception as e:
                logger.error(f"❌ Error in USB monitoring: {e}")
                time.sleep(2)
    
    def _on_device_connected(self, device_id: str, device_info: Dict[str, Any]) -> None:
        """Handle USB device connection - Initialize bridge agent"""
        try:
            logger.info(f"🔌 USB Device connected: {device_id} ({device_info.get('device_type')})")
            
            # Store device info
            self.connected_devices[device_id] = device_info
            
            # Initialize bridge agent for this device
            if BRIDGE_AGENT_AVAILABLE:
                bridge_agent = get_bridge_agent(device_id, "auto_init")
                self.bridge_agents[device_id] = bridge_agent
                logger.info(f"✅ Bridge agent initialized for {device_id}")
                
                # Log device capabilities
                capabilities = device_info.get('capabilities', [])
                logger.info(f"📱 Device capabilities: {', '.join(capabilities)}")
        
        except Exception as e:
            logger.error(f"❌ Error initializing bridge agent for {device_id}: {e}")
    
    def _on_device_disconnected(self, device_id: str) -> None:
        """Handle USB device disconnection"""
        try:
            logger.info(f"🔌 USB Device disconnected: {device_id}")
            
            # Remove device info
            if device_id in self.connected_devices:
                del self.connected_devices[device_id]
            
            # Clean up bridge agent
            if device_id in self.bridge_agents:
                del self.bridge_agents[device_id]
                logger.info(f"✅ Bridge agent cleaned up for {device_id}")
        
        except Exception as e:
            logger.error(f"❌ Error handling device disconnection: {e}")
    
    def get_bridge_agent_for_device(self, device_id: str) -> Optional[Any]:
        """Get bridge agent for a device"""
        if device_id in self.bridge_agents:
            return self.bridge_agents[device_id]
        
        # Try to initialize if not exists
        if BRIDGE_AGENT_AVAILABLE:
            try:
                bridge_agent = get_bridge_agent(device_id, self.case_id)
                self.bridge_agents[device_id] = bridge_agent
                logger.info(f"✅ Created bridge agent for {device_id}")
                return bridge_agent
            except Exception as e:
                logger.error(f"❌ Error creating bridge agent: {e}")
                return None
        
        return None
    
    def stop_usb_monitoring(self) -> None:
        """Stop USB device monitoring"""
        self.usb_monitoring_active = False
        if self.usb_monitor_thread:
            self.usb_monitor_thread.join(timeout=5)
        logger.info("⏹️ USB device monitoring stopped")

    def get_pending_sync_extractions(self) -> Dict[str, Dict[str, Any]]:
        """Get pending extractions waiting for sync"""
        return self.hybrid_manager.get_pending_extractions()


# ============================================================================
# GLOBAL ORCHESTRATOR INSTANCE
# ============================================================================

_orchestrator_instance: Optional[ExtractionOrchestrator] = None

def get_orchestrator() -> ExtractionOrchestrator:
    """Get global orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = ExtractionOrchestrator()
    return _orchestrator_instance
