"""
Modern Extraction UI for ForenSmart
===================================

Provides a modern, user-friendly extraction interface with real-time progress tracking,
artifact visualization, and multi-platform support (Android, iOS, HDD).

Features:
- Real-time progress bars with percentage display
- Live artifact feed during extraction
- Multi-stage extraction tracking
- Error handling and recovery
- One-click extraction workflows
"""

import streamlit as st
from typing import Optional, Dict, List, Any, Callable
import json
import os
import logging
from datetime import datetime
from pathlib import Path
import time
import threading

from modules.ui.progress_ui import (
    ProgressTracker,
    ProgressStatus,
    render_progress_bar,
    render_extraction_progress,
    render_live_artifact_feed,
    render_multi_stage_progress
)
from modules.shared.utils import ArtifactPathBuilder
from modules.extraction.orchestrator import DataExtractionOrchestrator
from modules.consent.models import ConsentManager
from modules.consent.models import ConsentLevel
from modules.consent.portal import ConsentAuditTrail

logger = logging.getLogger(__name__)


class ExtractionUIManager:
    """Manages extraction UI state and interactions."""
    
    def __init__(self):
        self.extraction_history: List[Dict[str, Any]] = []
        self.active_extractions: Dict[str, ProgressTracker] = {}
        
    def start_extraction(
        self,
        case_id: str,
        extraction_type: str,
        total_steps: int = 100
    ) -> ProgressTracker:
        """Start a new extraction and return tracker."""
        tracker = ProgressTracker(total_steps)
        tracker.start()
        key = f"{case_id}_{extraction_type}"
        self.active_extractions[key] = tracker
        return tracker
    
    def get_extraction_tracker(self, case_id: str, extraction_type: str) -> Optional[ProgressTracker]:
        """Get tracker for active extraction."""
        key = f"{case_id}_{extraction_type}"
        return self.active_extractions.get(key)
    
    def complete_extraction(self, case_id: str, extraction_type: str, artifacts_count: int):
        """Mark extraction as complete."""
        key = f"{case_id}_{extraction_type}"
        if key in self.active_extractions:
            tracker = self.active_extractions[key]
            tracker.complete()
            tracker.artifacts_count = artifacts_count
            
            # Add to history
            self.extraction_history.append({
                'case_id': case_id,
                'type': extraction_type,
                'timestamp': datetime.now().isoformat(),
                'artifacts': artifacts_count,
                'status': 'completed'
            })


def get_extraction_ui_manager() -> ExtractionUIManager:
    """Get or create extraction UI manager in session state."""
    if 'extraction_ui_manager' not in st.session_state:
        st.session_state['extraction_ui_manager'] = ExtractionUIManager()
    return st.session_state['extraction_ui_manager']


def render_extraction_tab(case_id: str) -> None:
    """
    Render the main extraction tab with modern UI.
    
    Args:
        case_id: Case ID for extraction
    """
    
    st.markdown("# 📱 Data Extraction")
    
    # Import required modules
    from modules.dashboard_merged import get_consent_manager
    from modules.approval.utils import get_approval_decision
    from modules.extraction.validator import ExtractionValidator
    from modules.approval.manager import ApprovalManager, ApprovalSync
    from modules.shared.device_manager import DeviceManager
    from modules.extraction.progress import ProgressManager
    from modules.consent.portal import ConsentPortalEnhancer
    from modules.extraction.orchestrator import DataExtractionOrchestrator, MODULE_MIN_LEVELS
    
    cm = get_consent_manager()
    session = cm.get_session(case_id)
    orchestrator = DataExtractionOrchestrator(cm)

    # ========================================================================
    # PHASE 3: CONSENT LEVEL DISPLAY AND MODULE REQUIREMENTS
    # ========================================================================
    
    # Display current consent level with status
    if session and session.level:
        # Get consent level info with fallback for older ConsentManager instances
        try:
            if hasattr(cm, 'get_consent_level_info'):
                consent_info = cm.get_consent_level_info(case_id)
            else:
                raise AttributeError("Method not found")
        except (AttributeError, Exception):
            # Fallback for older instances or errors
            consent_info = {
                'level': session.level.name if session.level else None,
                'level_value': session.level.value if session.level else None,
                'locked': getattr(session, '_consent_level_locked', False),
                'set_at': getattr(session, '_consent_level_set_at', None),
                'scope': 'Unknown'
            }
        
        # Show consent level card
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**[CONSENT] Level:** `{session.level.name}`")
            st.caption(f"Scope: {consent_info.get('scope', 'Unknown')}")
        with col2:
            if consent_info.get('locked'):
                st.markdown("[LOCKED] Locked")
            else:
                st.markdown("[UNLOCKED] Unlocked")
        with col3:
            if consent_info.get('set_at'):
                st.caption(f"Set: {consent_info['set_at'][:10]}")
        
        st.divider()
        
        # Show module requirements and status
        st.markdown("### [MODULES] Module Requirements")
        module_cols = st.columns(2)
        
        for idx, (module_name, min_level) in enumerate(MODULE_MIN_LEVELS.items()):
            col = module_cols[idx % 2]
            with col:
                allowed, message = orchestrator.check_module_consent(module_name, session.level)
                
                if allowed:
                    st.success(message)
                else:
                    st.error(message)
                    # Show upgrade button if blocked
                    if st.button(f"[INFO] Learn about {min_level.name} consent", key=f"btn_learn_{module_name}"):
                        st.info(f"{module_name} requires {min_level.name} consent level to extract data.")
        
        st.divider()

    # Ensure consent level is set to at least LEGAL
    if not session or session.level is None or session.level == ConsentLevel.NONE:
        logger.info(f"[CONSENT] Setting consent level to LEGAL for {case_id}")
        result = cm.set_consent_level_immutable(case_id, ConsentLevel.LEGAL, "Auto-set for extraction")
        logger.info(f"[CONSENT] Set result: {result}")
        session = cm.get_session(case_id)  # Refresh session
        logger.info(f"[CONSENT] After refresh, session.level = {session.level if session else 'NO SESSION'}")
        st.rerun()
    
    consent_ok = session and session.level and session.level.value >= ConsentLevel.LEGAL.value
    if not consent_ok:
        st.warning("[WARNING] Insufficient consent. Please obtain at least LEGAL consent from the 'Consent' tab before extraction.")
        st.info("[INFO] Attempting to set consent level to LEGAL...")
        cm.set_consent_level_immutable(case_id, ConsentLevel.LEGAL, "Auto-set for extraction")
        st.rerun()

    # Check both old and new approval methods with ApprovalSync
    unlock_status = cm.get_unlock_status(case_id) if session else {}
    unlock_verified = unlock_status.get('status') == 'verified'
    
    # Check our new approval file first (from merged dashboard)
    approval_file = Path('audit/approvals') / f"{case_id}_approval.json"
    
    if approval_file.exists():
        try:
            approval_data = json.loads(approval_file.read_text())
            if approval_data.get('decision') == 'approved':
                unlock_verified = True
                st.success("✅ **Nominee Approved** - Extraction is unlocked!")
            elif approval_data.get('decision') == 'denied':
                unlock_verified = False
                st.error("🔐 Nominee denied the unlock request. Generate a new approval link in the Consent tab.")
        except json.JSONDecodeError as e:
            logger.error(f"Approval file corrupted: {e}", exc_info=True)
            st.warning(f"Approval file corrupted: {e}")
        except PermissionError as e:
            logger.error(f"Permission denied reading approval: {e}", exc_info=True)
            st.warning(f"Permission denied: {e}")
        except Exception as e:
            logger.error(f"Could not read approval: {type(e).__name__}: {e}", exc_info=True)
            st.warning(f"Could not read approval: {e}")
    # Fallback to ApprovalSync for real-time approval status
    else:
        try:
            if ApprovalSync.is_approved(case_id):
                unlock_verified = True
                st.success("✅ **Nominee Approved** - Extraction is unlocked!")
            elif ApprovalSync.is_denied(case_id):
                unlock_verified = False
                st.error("🔐 Nominee denied the unlock request. Generate a new approval link in the Consent tab.")
            elif ApprovalSync.is_approval_expired(case_id):
                unlock_verified = False
                st.warning("⏳ Approval expired. Request new approval from the Consent tab.")
        except AttributeError as e:
            logger.error(f"ApprovalSync method not found: {e}")
            st.warning("Could not check approval status")
        except Exception as e:
            logger.error(f"Approval check failed: {e}", exc_info=True)
            st.warning(f"Could not check approval status: {e}")
    
    if consent_ok and not unlock_verified:
        status = unlock_status.get('status', 'pending')
        if status == 'denied':
            st.error("🔐 Nominee denied the unlock request. Generate a new approval link in the Consent tab.")
        else:
            st.info("⏳ Waiting for nominee approval. Share the approval link from the Consent tab.")

    # Check for device connection with DeviceManager
    device_id = cm.ensure_device_id(case_id)
    
    # Normalize device ID (handle dict vs string)
    if isinstance(device_id, dict):
        device_id = device_id.get('serial') or device_id.get('device_id') or str(device_id)
    
    device_ok = device_id and device_id != 'UNKNOWN_DEVICE'
    
    if device_ok:
        # Validate device health
        device_health = DeviceManager.get_device_health(device_id)
        if device_health.get("issues"):
            st.warning(f"⚠️ Device issues: {', '.join(device_health['issues'])}")
            device_ok = False
        if device_health.get("warnings"):
            for warning in device_health["warnings"]:
                st.warning(f"⚠️ {warning}")
    else:
        st.warning("⚠️ No device connected. Please connect a device and ensure it's recognized before starting extraction.")

    buttons_disabled = not (consent_ok and device_ok and unlock_verified)
    
    # Show approval delivery options with ConsentPortalEnhancer if not approved yet
    if consent_ok and not unlock_verified:
        st.divider()
        st.markdown("### 📤 Need Approval?")
        if st.button("Show Approval Delivery Options", key="btn_show_approval_options"):
            ConsentPortalEnhancer.render_delivery_ui(
                approval_link=f"https://forensmart-consent.streamlit.app?case={case_id}",
                nominee_phone=session.nominee_phone if session else "",
                nominee_email="",
                nominee_name="",
                case_id=case_id
            )
        st.divider()
    
    # Extraction type selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Android")
        if st.button("🚀 Start Android Extraction", key="btn_android_extract", disabled=buttons_disabled):
            st.session_state['extraction_type'] = 'android'
            st.session_state['start_extraction'] = True
    
    with col2:
        st.markdown("### iOS")
        if st.button("🚀 Start iOS Extraction", key="btn_ios_extract", disabled=buttons_disabled):
            st.session_state['extraction_type'] = 'ios'
            st.session_state['start_extraction'] = True
    
    with col3:
        st.markdown("### HDD")
        if st.button("🚀 Start HDD Extraction", key="btn_hdd_extract", disabled=buttons_disabled):
            st.session_state['extraction_type'] = 'hdd'
            st.session_state['start_extraction'] = True
    
    st.divider()
    
    # Show active extraction if one is running
    if st.session_state.get('start_extraction'):
        extraction_type = st.session_state.get('extraction_type', 'android')
        manager = get_extraction_ui_manager()
        
        # Validate extraction readiness BEFORE starting
        validation_result = ExtractionValidator.validate_extraction_ready(
            case_id=case_id,
            device_id=device_id,
            session=session,
            required_level=ConsentLevel.STANDARD
        )
        
        if not validation_result["ready"]:
            st.error("❌ **Extraction Cannot Start**")
            st.error("**Errors:**")
            for error in validation_result["errors"]:
                st.write(f"- {error}")
            if validation_result["warnings"]:
                st.warning("**Warnings:**")
                for warning in validation_result["warnings"]:
                    st.write(f"- {warning}")
            st.session_state['start_extraction'] = False
            st.stop()
        
        # Get or create tracker with progress manager
        progress_tracker = ProgressManager.create_tracker(case_id, extraction_type)
        tracker = manager.get_extraction_tracker(case_id, extraction_type)
        if not tracker:
            tracker = manager.start_extraction(case_id, extraction_type)
        
        # Initialize thread state if not exists
        if 'extraction_thread' not in st.session_state:
            st.session_state.extraction_thread = None
            st.session_state.extraction_completed = False
        # Initialize extraction_in_progress if not exists (but don't reset if tracker is running)
        if 'extraction_in_progress' not in st.session_state:
            st.session_state.extraction_in_progress = False
        # If tracker is running, ensure extraction_in_progress is True
        if tracker.status == ProgressStatus.RUNNING:
            st.session_state.extraction_in_progress = True
        
        # Progress bar placeholders
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Check if we need to start the extraction
        if tracker.status != ProgressStatus.RUNNING and not st.session_state.get('extraction_in_progress'):
            logger.info(f"✅ Extraction conditions met for case {case_id}")
            logger.info(f"   - Consent OK: {consent_ok}")
            logger.info(f"   - Device OK: {device_ok}")
            logger.info(f"   - Unlock verified: {unlock_verified}")

            tracker.start()
            progress_tracker.start_module("initialization")
            st.session_state['extraction_in_progress'] = True
            logger.info(f"🚀 EXTRACTION STARTING FOR CASE {case_id}")

            # Show starting message
            with status_placeholder.container():
                st.info("🚀 Starting extraction...")
        
        # Execute extraction if in progress
        logger.info(f"DEBUG: extraction_in_progress={st.session_state.get('extraction_in_progress')}, tracker.status={tracker.status}, RUNNING={ProgressStatus.RUNNING}")
        if st.session_state.get('extraction_in_progress') and tracker.status == ProgressStatus.RUNNING:
            logger.info(f"🚀 Running extraction for case {case_id}")

            def progress_callback(progress, message, artifacts=0):
                """Update progress in real-time."""
                tracker.update(int(progress), message, artifacts)
                logger.info(f"Progress: {progress}% - {message}")

            try:
                # Initialize results
                device_id_for_extraction = session.device_id if session else "unknown_device"
                logger.info(f"Device ID for extraction: {device_id_for_extraction}")
                results = {
                    'status': 'in_progress',
                    'start_time': datetime.now().isoformat(),
                    'case_id': case_id,
                    'device_id': device_id_for_extraction,
                    'extraction_type': extraction_type,
                    'artifacts': {}
                }

                # Get consent manager and orchestrator
                consent_manager = get_consent_manager()
                orchestrator = DataExtractionOrchestrator(consent_manager)

                # Start extraction
                progress_callback(0, "Starting extraction...")

                try:
                    extraction_results = orchestrator.extract_all_data(
                        case_id=case_id,
                        device_id=device_id_for_extraction,
                        progress_callback=progress_callback
                    )

                    # Update results with extraction results
                    results.update(extraction_results)
                    results['status'] = 'completed'
                    results['end_time'] = datetime.now().isoformat()

                    # Count artifacts from all modules
                    total_artifacts = 0
                    artifact_details = {}

                    if 'data' in extraction_results:
                        for module_name, module_data in extraction_results['data'].items():
                            module_artifacts = 0

                            # Check for direct artifacts
                            if 'artifacts' in module_data and isinstance(module_data['artifacts'], dict):
                                for artifact_type, artifact_list in module_data['artifacts'].items():
                                    if isinstance(artifact_list, list):
                                        count = len(artifact_list)
                                        module_artifacts += count
                                        # Store artifact details
                                        if module_name not in artifact_details:
                                            artifact_details[module_name] = {}
                                        artifact_details[module_name][artifact_type] = count

                            # Check for artifact counts in module data
                            if 'artifact_counts' in module_data and isinstance(module_data['artifact_counts'], dict):
                                for artifact_type, count in module_data['artifact_counts'].items():
                                    if isinstance(count, int) and count > 0:
                                        module_artifacts += count
                                        # Store artifact details
                                        if module_name not in artifact_details:
                                            artifact_details[module_name] = {}
                                        artifact_details[module_name][f"{artifact_type}_count"] = count

                            total_artifacts += module_artifacts

                    # Store the detailed artifact counts in the results
                    results['artifact_details'] = artifact_details
                    results['total_artifacts'] = total_artifacts

                    # Update tracker with final count
                    progress_callback(100, "Extraction completed", total_artifacts)
                    tracker.complete()

                    # Store results in session
                    st.session_state['extraction_results'] = results
                    st.session_state['extraction_completed'] = True
                    st.session_state['extraction_in_progress'] = False
                    
                    # Save results to file for persistence
                    try:
                        results_dir = Path('artifacts') / case_id / 'extraction'
                        results_dir.mkdir(parents=True, exist_ok=True)
                        results_file = results_dir / 'extraction_results.json'
                        results_file.write_text(json.dumps(results, indent=2))
                        logger.info(f"Extraction results saved to {results_file}")
                    except Exception as save_e:
                        logger.error(f"Failed to save extraction results: {save_e}", exc_info=True)

                except Exception as e:
                    logger.error(f"Extraction failed: {e}", exc_info=True)
                    results['status'] = 'failed'
                    results['error'] = str(e)
                    
                    # ========================================================================
                    # PHASE 3: DISPLAY MODULE-LEVEL CONSENT ERRORS
                    # ========================================================================
                    # Check if error is consent-related
                    error_str = str(e).lower()
                    if 'consent' in error_str or 'messaging_consent_denied' in error_str:
                        results['error_type'] = 'consent_denied'
                        results['blocked_modules'] = []
                        
                        # Identify which modules are blocked
                        for module_name, min_level in MODULE_MIN_LEVELS.items():
                            if module_name in error_str or 'messaging' in error_str:
                                allowed, message = orchestrator.check_module_consent(module_name, session.level)
                                if not allowed:
                                    results['blocked_modules'].append({
                                        'module': module_name,
                                        'required': min_level.name,
                                        'current': session.level.name,
                                        'message': message
                                    })
                    
                    st.session_state['extraction_results'] = results
                    st.session_state['extraction_completed'] = True
                    st.session_state['extraction_in_progress'] = False
                    tracker.error(str(e))
                    progress_callback(0, f"Extraction failed: {e}")

            except Exception as e:
                logger.error(f"Extraction setup failed: {e}", exc_info=True)
                st.session_state['extraction_in_progress'] = False
                tracker.error(str(e))

        # Display progress
        if tracker.status == ProgressStatus.RUNNING:
            st.info(f"⏳ Extraction in progress: {tracker.get_percentage()}%")
        elif tracker.status == ProgressStatus.COMPLETED:
            st.success(f"✅ Extraction completed! {tracker.artifacts_count} artifacts extracted")
            manager.complete_extraction(case_id, extraction_type, tracker.artifacts_count)
            st.session_state['start_extraction'] = False
            
            # Display extraction results
            if st.session_state.get('extraction_results'):
                results = st.session_state['extraction_results']
                st.divider()
                st.markdown("### 📊 Extraction Results Summary")
                
                # Show artifact breakdown by module
                if results.get('artifact_details'):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Artifacts", results.get('total_artifacts', 0))
                    with col2:
                        st.metric("Modules Completed", len(results.get('artifact_details', {})))
                    with col3:
                        st.metric("Status", results.get('status', 'unknown').upper())
                    
                    st.divider()
                    st.markdown("#### 📦 Artifacts by Module")
                    
                    for module_name, artifacts in results['artifact_details'].items():
                        with st.expander(f"📁 {module_name.upper()}"):
                            total_module_artifacts = sum(v for v in artifacts.values() if isinstance(v, int))
                            st.metric(f"{module_name} Artifacts", total_module_artifacts)
                            
                            # Show breakdown
                            for artifact_type, count in artifacts.items():
                                if isinstance(count, int):
                                    st.write(f"- **{artifact_type}**: {count} items")
        elif tracker.status == ProgressStatus.ERROR:
            st.error(f"❌ Extraction failed: {tracker.message}")
            st.session_state['start_extraction'] = False
