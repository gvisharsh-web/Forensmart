"""
HYBRID INTEGRATION MODULE
Integrates ExtractionBridgeAgent with existing ExtractionOrchestrator

This module provides:
- HybridExtractionAdapter: Adapter to integrate bridge agent with orchestrator
- WebAppBridgeHandler: Handles ADB commands for web app (no direct USB access)
- Seamless switching between standard and hybrid extraction
- Backward compatibility with existing extraction flow
"""

import logging
import time
import os
import json
import threading
import subprocess
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime

from modules.extraction.hybrid_bridge_agent import (
    ExtractionBridgeAgent,
    get_bridge_agent,
    ExtractionSource,
    EscalationMethod
)
from modules.shared.utils import ResultsRepository, ArtifactPathBuilder

logger = logging.getLogger(__name__)

# ============================================================================
# HYBRID EXTRACTION ADAPTER
# ============================================================================

class HybridExtractionAdapter:
    """Adapter to integrate bridge agent with existing orchestrator"""
    
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self.bridge_agents: Dict[str, ExtractionBridgeAgent] = {}
        self.hybrid_enabled = True
        
    def extract_all_data_hybrid(
        self,
        case_id: str,
        device_id: str,
        consent_manager: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        extraction_id: Optional[str] = None,
        enable_escalation: bool = False,
        enable_extended_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Execute hybrid extraction combining standard and bridge agent methods
        
        Args:
            case_id: Case identifier
            device_id: Device identifier
            consent_manager: Consent manager instance
            progress_callback: Progress callback function
            extraction_id: Extraction identifier
            enable_escalation: Enable privilege escalation
            enable_extended_sources: Enable extended source extraction
        
        Returns:
            Combined extraction results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting hybrid extraction for case {case_id}, device {device_id}")
            
            # Generate extraction ID if not provided
            if not extraction_id:
                extraction_id = f"{case_id}_{int(time.time())}"
            
            # Step 1: Run standard extraction
            self._update_progress(progress_callback, "Running standard extraction...", 5)
            standard_results = self.orchestrator.extract_all_data(
                case_id=case_id,
                device_id=device_id,
                consent_manager=consent_manager,
                progress_callback=self._create_sub_progress(progress_callback, 5, 45),
                extraction_id=extraction_id
            )
            
            # Step 2: Run bridge agent extraction
            self._update_progress(progress_callback, "Running hybrid bridge extraction...", 50)
            bridge_agent = get_bridge_agent(device_id, case_id)
            bridge_results = bridge_agent.execute_hybrid_extraction(
                enable_escalation=enable_escalation,
                enable_extended_sources=enable_extended_sources,
                progress_callback=self._create_sub_progress(progress_callback, 50, 95)
            )
            
            # Step 3: Merge results
            self._update_progress(progress_callback, "Merging extraction results...", 95)
            merged_results = self._merge_results(
                standard_results,
                bridge_results,
                case_id,
                device_id,
                extraction_id
            )
            
            # Step 4: Save merged results
            self._update_progress(progress_callback, "Saving hybrid extraction results...", 98)
            self._save_hybrid_results(case_id, merged_results)
            
            self._update_progress(progress_callback, "Hybrid extraction complete", 100)
            
            logger.info(f"Hybrid extraction completed: {merged_results['total_artifacts']} total artifacts")
            
            return merged_results
        
        except Exception as e:
            logger.error(f"Hybrid extraction error: {e}", exc_info=True)
            return {
                'status': 'error',
                'case_id': case_id,
                'device_id': device_id,
                'error': str(e),
                'extraction_type': 'hybrid',
                'duration_seconds': time.time() - start_time
            }
    
    def extract_with_escalation(
        self,
        case_id: str,
        device_id: str,
        consent_manager: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        escalation_methods: Optional[List[EscalationMethod]] = None
    ) -> Dict[str, Any]:
        """
        Extract with privilege escalation attempts
        
        Args:
            case_id: Case identifier
            device_id: Device identifier
            consent_manager: Consent manager instance
            progress_callback: Progress callback function
            escalation_methods: List of escalation methods to try
        
        Returns:
            Extraction results with escalation info
        """
        try:
            logger.info(f"Starting escalation-enabled extraction for case {case_id}")
            
            bridge_agent = get_bridge_agent(device_id, case_id)
            
            # Attempt escalation
            if escalation_methods:
                success, message = bridge_agent.privilege_manager.escalate_privileges(escalation_methods)
            else:
                success, message = bridge_agent.privilege_manager.escalate_privileges()
            
            logger.info(f"Escalation result: {success}, {message}")
            
            # Run extraction with escalation
            return self.extract_all_data_hybrid(
                case_id=case_id,
                device_id=device_id,
                consent_manager=consent_manager,
                progress_callback=progress_callback,
                enable_escalation=True,
                enable_extended_sources=True
            )
        
        except Exception as e:
            logger.error(f"Escalation extraction error: {e}", exc_info=True)
            return {
                'status': 'error',
                'case_id': case_id,
                'device_id': device_id,
                'error': str(e),
                'extraction_type': 'escalation'
            }
    
    def _merge_results(
        self,
        standard_results: Dict[str, Any],
        bridge_results: Dict[str, Any],
        case_id: str,
        device_id: str,
        extraction_id: str
    ) -> Dict[str, Any]:
        """Merge standard and bridge extraction results"""
        
        # Get artifact counts
        standard_artifacts = standard_results.get('total_artifacts', 0)
        bridge_artifacts = bridge_results.get('total_artifacts', 0)
        
        # Merge module results
        merged_modules = standard_results.get('modules', {}).copy()
        
        # Add bridge results as separate section
        bridge_modules = bridge_results.get('extraction_results', {})
        
        return {
            'status': 'success',
            'extraction_type': 'hybrid',
            'extraction_id': extraction_id,
            'case_id': case_id,
            'device_id': device_id,
            'timestamp': datetime.now().isoformat(),
            'standard_extraction': {
                'status': standard_results.get('status'),
                'artifacts': standard_artifacts,
                'modules': merged_modules,
                'blocked_modules': standard_results.get('blocked_modules', []),
                'duration_seconds': standard_results.get('total_time', 0)
            },
            'bridge_extraction': {
                'status': bridge_results.get('status'),
                'artifacts': bridge_artifacts,
                'completeness': bridge_results.get('extraction_completeness', 0),
                'escalation_used': bridge_results.get('privilege_escalation_used', False),
                'escalation_method': bridge_results.get('escalation_method'),
                'sources': bridge_modules,
                'duration_seconds': bridge_results.get('duration_seconds', 0)
            },
            'total_artifacts': standard_artifacts + bridge_artifacts,
            'extraction_completeness': bridge_results.get('extraction_completeness', 0),
            'privilege_escalation_used': bridge_results.get('privilege_escalation_used', False),
            'escalation_method': bridge_results.get('escalation_method'),
            'total_duration_seconds': (
                standard_results.get('total_time', 0) + 
                bridge_results.get('duration_seconds', 0)
            )
        }
    
    def _save_hybrid_results(self, case_id: str, results: Dict[str, Any]) -> bool:
        """Save hybrid extraction results"""
        try:
            # Save to artifact storage
            artifact_path = ArtifactPathBuilder.resolve(case_id, "extraction", ensure_dir=True)
            results_file = os.path.join(artifact_path, "hybrid_extraction_results.json")
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Hybrid extraction results saved to {results_file}")
            
            # Also save to results repository
            ResultsRepository.save(case_id, {'hybrid_extraction': results})
            
            return True
        except Exception as e:
            logger.error(f"Error saving hybrid results: {e}")
            return False
    
    def _create_sub_progress(
        self,
        callback: Optional[Callable[[str, int], None]],
        start_percent: int,
        end_percent: int
    ) -> Optional[Callable[[str, int], None]]:
        """Create a sub-progress callback for nested operations"""
        if not callback:
            return None
        
        def sub_progress(message: str, sub_percent: int) -> None:
            # Map sub_percent (0-100) to range (start_percent, end_percent)
            mapped_percent = start_percent + int((sub_percent / 100) * (end_percent - start_percent))
            callback(message, mapped_percent)
        
        return sub_progress
    
    def _update_progress(
        self,
        callback: Optional[Callable[[str, int], None]],
        message: str,
        percentage: int
    ) -> None:
        """Update progress via callback"""
        if callback:
            try:
                callback(message, percentage)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

# ============================================================================
# INTEGRATION HELPER FUNCTIONS
# ============================================================================

def create_hybrid_adapter(orchestrator: Any) -> HybridExtractionAdapter:
    """Create hybrid extraction adapter"""
    return HybridExtractionAdapter(orchestrator)

def get_extraction_completeness_report(
    case_id: str,
    standard_artifacts: int,
    bridge_artifacts: int,
    escalation_used: bool
) -> Dict[str, Any]:
    """Generate extraction completeness report"""
    total_artifacts = standard_artifacts + bridge_artifacts
    
    # Calculate completeness based on artifact count
    # Expected: 1000+ artifacts for complete extraction
    expected_count = 1000
    completeness = min(100.0, (total_artifacts / expected_count) * 100)
    
    return {
        'case_id': case_id,
        'total_artifacts': total_artifacts,
        'standard_artifacts': standard_artifacts,
        'bridge_artifacts': bridge_artifacts,
        'completeness_percentage': round(completeness, 2),
        'escalation_used': escalation_used,
        'extraction_quality': 'excellent' if completeness >= 80 else 'good' if completeness >= 60 else 'fair' if completeness >= 40 else 'poor',
        'timestamp': datetime.now().isoformat()
    }

def compare_extraction_methods(
    standard_results: Dict[str, Any],
    bridge_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare standard vs bridge extraction methods"""
    return {
        'method_comparison': {
            'standard': {
                'artifacts': standard_results.get('total_artifacts', 0),
                'modules_successful': len([m for m in standard_results.get('modules', {}).values() if m.get('status') == 'success']),
                'modules_blocked': len(standard_results.get('blocked_modules', [])),
                'duration_seconds': standard_results.get('total_time', 0)
            },
            'bridge': {
                'artifacts': bridge_results.get('total_artifacts', 0),
                'sources_extracted': len(bridge_results.get('extraction_results', {})),
                'escalation_used': bridge_results.get('privilege_escalation_used', False),
                'duration_seconds': bridge_results.get('duration_seconds', 0)
            }
        },
        'improvement': {
            'additional_artifacts': bridge_results.get('total_artifacts', 0),
            'completeness_gain': bridge_results.get('extraction_completeness', 0),
            'escalation_benefit': 'Yes' if bridge_results.get('privilege_escalation_used') else 'No'
        }
    }

# ============================================================================
# WEB APP BRIDGE HANDLER - ADB COMMAND EXECUTION FOR WEB APP
# ============================================================================

class WebAppBridgeHandler:
    """Handles ADB commands for web app (web app cannot access USB directly)"""
    
    def __init__(self):
        self.connected_devices: Dict[str, Dict[str, Any]] = {}
        self.extraction_queue: Dict[str, Dict[str, Any]] = {}
        self.extraction_results: Dict[str, Dict[str, Any]] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None
        
        logger.info("✅ WebAppBridgeHandler initialized")
    
    def start_adb_bridge(self) -> None:
        """Start ADB bridge for web app"""
        if self.monitoring_active:
            logger.warning("⚠️ ADB bridge already running")
            return
        
        self.monitoring_active = True
        
        # Start device monitoring
        self.monitor_thread = threading.Thread(
            target=self._monitor_adb_devices,
            daemon=True,
            name="ADBBridgeMonitor"
        )
        self.monitor_thread.start()
        
        # Start extraction processing
        self.process_thread = threading.Thread(
            target=self._process_adb_extractions,
            daemon=True,
            name="ADBBridgeProcessor"
        )
        self.process_thread.start()
        
        logger.info("🔌 ADB bridge started - Ready to handle web app requests")
    
    def stop_adb_bridge(self) -> None:
        """Stop ADB bridge"""
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        if self.process_thread:
            self.process_thread.join(timeout=5)
        
        logger.info("⏹️ ADB bridge stopped")
    
    def _monitor_adb_devices(self) -> None:
        """Monitor ADB devices connected via USB"""
        previous_devices = set()
        
        while self.monitoring_active:
            try:
                # Execute: adb devices
                result = subprocess.run(
                    ['adb', 'devices'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    # Parse device list
                    lines = result.stdout.strip().split('\n')[1:]
                    current_devices = set()
                    
                    for line in lines:
                        if line.strip() and '\t' in line:
                            device_id, status = line.split('\t')
                            device_id = device_id.strip()
                            status = status.strip()
                            
                            if status == 'device':
                                current_devices.add(device_id)
                                
                                # New device connected
                                if device_id not in previous_devices:
                                    self._on_adb_device_connected(device_id)
                    
                    # Check for disconnections
                    disconnected = previous_devices - current_devices
                    for device_id in disconnected:
                        self._on_adb_device_disconnected(device_id)
                    
                    previous_devices = current_devices
                
                time.sleep(2)
            
            except Exception as e:
                logger.error(f"❌ Error monitoring ADB devices: {e}")
                time.sleep(2)
    
    def _on_adb_device_connected(self, device_id: str) -> None:
        """Handle ADB device connection"""
        try:
            logger.info(f"🔌 ADB Device connected: {device_id}")
            
            # Get device info via ADB
            device_info = self._get_adb_device_info(device_id)
            self.connected_devices[device_id] = device_info
            
            # Initialize bridge agent
            bridge_agent = get_bridge_agent(device_id, "web_app")
            
            logger.info(f"✅ Bridge agent ready for web app: {device_id}")
        
        except Exception as e:
            logger.error(f"❌ Error on device connection: {e}")
    
    def _on_adb_device_disconnected(self, device_id: str) -> None:
        """Handle ADB device disconnection"""
        try:
            logger.info(f"🔌 ADB Device disconnected: {device_id}")
            
            if device_id in self.connected_devices:
                del self.connected_devices[device_id]
        
        except Exception as e:
            logger.error(f"❌ Error on device disconnection: {e}")
    
    def _get_adb_device_info(self, device_id: str) -> Dict[str, Any]:
        """Get device info via ADB commands"""
        try:
            # Get model
            result = subprocess.run(
                ['adb', '-s', device_id, 'shell', 'getprop', 'ro.product.model'],
                capture_output=True,
                text=True,
                timeout=5
            )
            model = result.stdout.strip() if result.returncode == 0 else 'Unknown'
            
            # Get Android version
            result = subprocess.run(
                ['adb', '-s', device_id, 'shell', 'getprop', 'ro.build.version.release'],
                capture_output=True,
                text=True,
                timeout=5
            )
            android_version = result.stdout.strip() if result.returncode == 0 else 'Unknown'
            
            return {
                'device_id': device_id,
                'model': model,
                'android_version': android_version,
                'status': 'connected',
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Error getting device info: {e}")
            return {'device_id': device_id, 'status': 'connected'}
    
    def queue_web_extraction(
        self,
        request_id: str,
        device_id: str,
        case_id: str,
        enable_escalation: bool = False,
        enable_extended_sources: bool = True
    ) -> str:
        """Queue extraction request from web app"""
        try:
            self.extraction_queue[request_id] = {
                'request_id': request_id,
                'device_id': device_id,
                'case_id': case_id,
                'enable_escalation': enable_escalation,
                'enable_extended_sources': enable_extended_sources,
                'queued_at': datetime.now().isoformat()
            }
            
            logger.info(f"📋 Web extraction queued: {request_id}")
            return request_id
        
        except Exception as e:
            logger.error(f"❌ Error queuing extraction: {e}")
            return None
    
    def _process_adb_extractions(self) -> None:
        """Process extraction requests from web app"""
        while self.monitoring_active:
            try:
                if not self.extraction_queue:
                    time.sleep(1)
                    continue
                
                request_id, request = next(iter(self.extraction_queue.items()))
                del self.extraction_queue[request_id]
                
                # Execute extraction via bridge agent
                self._execute_web_extraction(request_id, request)
                
            except Exception as e:
                logger.error(f"❌ Error processing extractions: {e}")
                time.sleep(1)
    
    def _execute_web_extraction(self, request_id: str, request: Dict[str, Any]) -> None:
        """Execute extraction for web app"""
        try:
            device_id = request['device_id']
            
            logger.info(f"🔄 Executing web extraction: {request_id} on {device_id}")
            
            # Get bridge agent
            bridge_agent = get_bridge_agent(device_id, "web_app")
            
            # Create progress callback
            def progress_callback(message: str, percentage: int):
                logger.info(f"📊 {request_id}: {percentage}% - {message}")
            
            # Execute hybrid extraction
            results = bridge_agent.execute_hybrid_extraction(
                enable_escalation=request['enable_escalation'],
                enable_extended_sources=request['enable_extended_sources'],
                progress_callback=progress_callback
            )
            
            # Store results
            self.extraction_results[request_id] = {
                'status': 'success',
                'request_id': request_id,
                'device_id': device_id,
                'case_id': request['case_id'],
                'results': results,
                'completed_at': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Web extraction complete: {request_id}")
        
        except Exception as e:
            logger.error(f"❌ Error executing web extraction: {e}", exc_info=True)
            self.extraction_results[request_id] = {
                'status': 'error',
                'error': str(e),
                'request_id': request_id,
                'completed_at': datetime.now().isoformat()
            }
    
    def get_web_extraction_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get extraction result for web app"""
        return self.extraction_results.get(request_id)
    
    def get_web_extraction_status(self, request_id: str) -> str:
        """Get extraction status for web app"""
        if request_id in self.extraction_queue:
            return "queued"
        elif request_id in self.extraction_results:
            return self.extraction_results[request_id].get('status', 'unknown')
        else:
            return "not_found"
    
    def get_connected_adb_devices(self) -> List[Dict[str, Any]]:
        """Get list of connected ADB devices"""
        return list(self.connected_devices.values())

# ============================================================================
# GLOBAL WEB APP BRIDGE HANDLER INSTANCE
# ============================================================================

_web_app_bridge_handler: Optional[WebAppBridgeHandler] = None

def get_web_app_bridge_handler() -> WebAppBridgeHandler:
    """Get or create global web app bridge handler"""
    global _web_app_bridge_handler
    if _web_app_bridge_handler is None:
        _web_app_bridge_handler = WebAppBridgeHandler()
        _web_app_bridge_handler.start_adb_bridge()
    return _web_app_bridge_handler
