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

    def extract_all_data(
        self,
        case_id: str,
        device_id: str,
        consent_manager: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        extraction_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract all data with consent checks, error handling, pause/resume, and cancellation"""
        
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
            except:
                pass
        
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
                    progress_callback(f"Extracting {module_name}...", idx + 1)
                
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
                    logger.error(f"{module_name} extraction failed after retries")
                    continue
                
                # Check if consent was denied
                if result.get('status') == 'consent_denied':
                    extraction_results['blocked_modules'].append({
                        'module': module_name,
                        'reason': result.get('message'),
                        'required_level': result.get('required_level'),
                        'current_level': result.get('current_level')
                    })
                    logger.warning(f"{module_name} blocked: {result.get('message')}")
                    continue
                
                # Check for errors
                if result.get('status') == 'error':
                    extraction_results['modules'][module_name] = {
                        'status': 'error',
                        'error': result.get('error')
                    }
                    logger.error(f"{module_name} extraction failed: {result.get('error')}")
                    continue
                
                # Store successful extraction
                extraction_results['modules'][module_name] = {
                    'status': 'success',
                    'artifact_count': result.get('artifact_count', 0),
                    'extraction_time': result.get('extraction_time', 0),
                    'data': result.get('data', {})
                }
                
                extraction_results['total_artifacts'] += result.get('artifact_count', 0)
                
                logger.info(f"{module_name} extraction completed: {result.get('artifact_count', 0)} artifacts")
            
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
                logger.error(f"Error in partial extraction {module_name}: {e}")
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
        cache_key = f"extraction_{case_id}_{module_name}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            logger.info(f"Using cached result for {module_name}")
            return cached_result
        
        extractor = self.extractors[module_name]
        
        # Retry with backoff
        for attempt in range(self.max_retries):
            try:
                result = extractor.extract(
                    device_id=device_id,
                    case_id=case_id,
                    consent_manager=consent_manager
                )
                
                # Check if consent was denied
                if result.get('status') == 'consent_denied':
                    logger.warning(f"{module_name} blocked: {result.get('message')}")
                    return result
                
                # Check for errors
                if result.get('status') == 'error':
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(f"{module_name} failed, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"{module_name} extraction failed after {self.max_retries} attempts")
                        return result
                
                # Cache successful result
                self.cache_manager.set(cache_key, result)
                
                logger.info(f"{module_name} extraction completed")
                return result
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Unexpected error in {module_name}: {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Unexpected error in {module_name} after {self.max_retries} attempts: {e}")
                    return {'status': 'error', 'error': str(e)}
        
        return {'status': 'error', 'error': 'Max retries exceeded'}

    def _save_results(self, case_id: str, results: Dict[str, Any]):
        """Save extraction results"""
        try:
            case_dir = os.path.join(self.storage_path, case_id)
            os.makedirs(case_dir, exist_ok=True)
            
            results_file = os.path.join(case_dir, 'extraction_results.json')
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {results_file}")
        
        except Exception as e:
            logger.error(f"Error saving results: {e}")

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

    def validate_module_dependencies(self, modules: List[str]) -> bool:
        """Validate that all dependencies are included"""
        required_modules = set()
        for module in modules:
            required_modules.add(module)
            required_modules.update(self.get_module_dependencies(module))
        
        return required_modules.issubset(set(modules))

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

    def sync_extraction_results(self) -> bool:
        """Sync extraction results with remote server"""
        
        if not self.hybrid_manager.is_connected():
            logger.warning("Cannot sync: offline")
            return False
        
        if not self.hybrid_manager.should_sync():
            logger.debug("Sync not needed yet")
            return False
        
        try:
            pending = self.hybrid_manager.get_pending_extractions()
            
            if not pending:
                logger.debug("No pending extractions to sync")
                self.hybrid_manager.sync_completed()
                return True
            
            logger.info(f"Syncing {len(pending)} pending extractions")
            
            # In production, sync with remote server
            # For now, just mark as synced
            for extraction_id in pending.keys():
                self.hybrid_manager.mark_synced(extraction_id)
            
            self.hybrid_manager.sync_completed()
            logger.info("Extraction sync completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Extraction sync error: {e}")
            return False

    def queue_extraction_offline(self, case_id: str, extraction_data: Dict[str, Any]) -> None:
        """Queue extraction for sync when offline"""
        extraction_id = f"{case_id}_{int(time.time())}"
        self.hybrid_manager.queue_extraction(extraction_id, extraction_data)
        logger.info(f"Extraction queued offline: {extraction_id}")

    def get_results_hybrid(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get results from local cache or remote (hybrid approach)"""
        
        # Try local cache first (offline support)
        if case_id in self.local_results_cache:
            logger.debug(f"Results from local cache: {case_id}")
            return self.local_results_cache[case_id]
        
        # Try main results
        if case_id in self.results:
            results = self.results[case_id]
            # Cache locally
            self.local_results_cache[case_id] = results
            return results
        
        # Try file storage
        results = self.get_results(case_id)
        if results:
            # Cache locally
            self.local_results_cache[case_id] = results
            return results
        
        return None

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
