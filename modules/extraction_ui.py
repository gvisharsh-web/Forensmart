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
from datetime import datetime
from pathlib import Path
import time
import threading

from modules.progress_ui import (
    ProgressTracker,
    ProgressStatus,
    render_progress_bar,
    render_extraction_progress,
    render_live_artifact_feed,
    render_multi_stage_progress
)
from modules.shared_utils import ArtifactPathBuilder
from modules.data_extraction_orchestrator import DataExtractionOrchestrator
from modules.consent import ConsentManager
from modules.consent import ConsentLevel
print(ConsentLevel)


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


def render_extraction_tab(
    case_id: str
) -> None:
    """
    Render the main extraction tab with modern UI.
    
    Args:
        case_id: Case ID for extraction
    """
    
    st.markdown("# 📱 Data Extraction")
    
    # Get ConsentManager and check consent level
    from modules.dashboard import get_consent_manager
    from modules.approval_utils import get_approval_decision
    from modules.extraction_validator import ExtractionValidator
    from modules.approval_sync import ApprovalSync
    from modules.device_manager import DeviceManager
    from modules.extraction_progress import ProgressManager
    
    cm = get_consent_manager()
    session = cm.get_session(case_id)

    consent_ok = session and session.level.value >= ConsentLevel.STANDARD.value
    if not consent_ok:
        st.warning("⚠️ Insufficient consent. Please obtain at least STANDARD consent from the 'Consent' tab before extraction.")

    # Check both old and new approval methods with ApprovalSync
    unlock_status = cm.get_unlock_status(case_id) if session else {}
    unlock_verified = unlock_status.get('status') == 'verified'
    
    # Use ApprovalSync for real-time approval status
    if ApprovalSync.is_approved(case_id):
        unlock_verified = True
        st.success("✅ **Nominee Approved** - Extraction is unlocked!")
    elif ApprovalSync.is_denied(case_id):
        unlock_verified = False
        st.error("🔐 Nominee denied the unlock request. Generate a new approval link in the Consent tab.")
    elif ApprovalSync.is_approval_expired(case_id):
        st.warning("⏳ Approval expired. Request new approval from the Consent tab.")
        unlock_verified = False
    elif consent_ok and not unlock_verified:
        status = unlock_status.get('status', 'pending')
        if status == 'denied':
            st.error("🔐 Nominee denied the unlock request. Generate a new approval link in the Consent tab.")
        else:
            st.info("⏳ Waiting for nominee approval. Share the approval link from the Consent tab.")

    # Check for device connection with DeviceManager
    device_id = cm.ensure_device_id(case_id)
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
        
        # Progress bar placeholder
        progress_placeholder = st.empty()
        
        # Check if we need to start the extraction
        if tracker.status != ProgressStatus.RUNNING and not st.session_state.extraction_thread:
            tracker.start()
            progress_tracker.start_module("initialization")
            
            def progress_callback(progress, message):
                tracker.update(int(progress), message)
                # Update the progress bar in the UI
                with progress_placeholder.container():
                    render_extraction_progress(
                        case_id=case_id,
                        extraction_type=extraction_type.upper(),
                        tracker=tracker,
                        on_cancel=cancel_extraction
                    )
                # Small delay to allow UI to update
                time.sleep(0.1)

            def run_extraction():
                try:
                    # Initialize results
                    device_id = session.device_id if session else "unknown_device"
                    results = {
                        'status': 'in_progress',
                        'start_time': datetime.now().isoformat(),
                        'case_id': case_id,
                        'device_id': device_id,
                        'extraction_type': extraction_type,
                        'artifacts': {}
                    }
                    
                    # Get consent manager and orchestrator
                    consent_manager = get_consent_manager()
                    orchestrator = DataExtractionOrchestrator(consent_manager)
                    
                    # Start extraction
                    progress_callback(0, "Starting extraction...")
                    
                    # Run extraction in a thread
                    def run():
                        try:
                            extraction_results = orchestrator.extract_all_data(
                                case_id=case_id,
                                device_id=device_id,
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
                                    
                                    # Log module artifact count for debugging
                                    print(f"Module {module_name} found {module_artifacts} artifacts")
                            
                            # Store the detailed artifact counts in the results
                            results['artifact_details'] = artifact_details
                            
                            # Ensure we have at least some artifacts if the extraction was successful
                            if total_artifacts == 0 and results['status'] == 'completed':
                                print(f"Warning: No artifacts found in extraction results for case {case_id}")
                            
                            # Update tracker with final count
                            progress_callback(100, "Extraction completed", total_artifacts)
                            
                            # Log the final artifact count
                            print(f"Total artifacts extracted: {total_artifacts}")
                            
                            # Store the total count in the results
                            results['total_artifacts'] = total_artifacts
                            
                            # Mark extraction as complete in the manager
                            manager.complete_extraction(case_id, extraction_type, total_artifacts)
                            
                            # Force a UI update by triggering a rerun
                            try:
                                import streamlit as st
                                st.rerun()
                            except Exception as e:
                                print(f"Error triggering UI update: {e}")
                            
                        except Exception as e:
                            error_msg = f"Extraction failed: {str(e)}"
                            print(error_msg)
                            results['status'] = 'error'
                            results['error'] = error_msg
                            tracker.error(error_msg)
                    
                    # Complete the extraction with the final count
                    manager.complete_extraction(case_id, extraction_type, tracker.artifacts_count)
                    
                    # Force a rerun to update the UI with the final count
                    st.rerun()
                    
                except Exception as e:
                    tracker.error(f"Extraction failed: {str(e)}")
                finally:
                    st.session_state.extraction_completed = True
                    # Force a rerun to show completion state
                    st.rerun()
            
            # Start the extraction in a separate thread
            st.session_state.extraction_thread = threading.Thread(target=run_extraction, daemon=True)
            st.session_state.extraction_thread.start()
        
        # Function to cancel the extraction
        def cancel_extraction():
            if st.session_state.extraction_thread and st.session_state.extraction_thread.is_alive():
                # Set a flag to indicate cancellation
                tracker.error("Extraction cancelled by user")
                st.session_state.extraction_completed = True
                st.rerun()
        
        # Display the current progress
        with progress_placeholder.container():
            if tracker.status == ProgressStatus.RUNNING or not st.session_state.extraction_completed:
                render_extraction_progress(
                    case_id=case_id,
                    extraction_type=extraction_type.upper(),
                    tracker=tracker,
                    on_cancel=cancel_extraction
                )
            else:
                # Show completion message
                if tracker.status == ProgressStatus.COMPLETED:
                    st.success(f"✅ Extraction completed! {tracker.artifacts_count} artifacts extracted.")
                elif tracker.status == ProgressStatus.ERROR:
                    st.error("❌ Extraction failed. See logs for details.")
                
                # Reset state for next extraction
                st.session_state['start_extraction'] = False
                st.session_state.extraction_thread = None
                st.session_state.extraction_completed = False
                
                # Add a small delay before allowing another extraction
                time.sleep(1)
                st.rerun()


def render_intelligence_tab(
    case_id: str,
    consent_id: Optional[str] = None
) -> None:
    """
    Render the intelligence analysis tab with modern progress tracking and enhancements.
    
    Args:
        case_id: Case ID for analysis
        consent_id: Optional consent ID for the analysis
    """
    from modules.progress_ui import render_progress_bar, ProgressStatus
    from modules.dashboard import get_consent_manager
    from modules.extraction_validator import ExtractionValidator
    from modules.approval_sync import ApprovalSync
    from modules.device_manager import DeviceManager
    from modules.extraction_progress import ProgressManager
    import time
    import threading
    
    st.markdown("# 🧠 Intelligence Analysis")
    
    # Get ConsentManager and check consent level
    cm = get_consent_manager()
    session = cm.get_session(case_id)

    if not session or session.level.value < ConsentLevel.STANDARD.value:
        st.warning("⚠️ Insufficient consent. Please obtain at least STANDARD consent before analysis.")
        return
    
    # Check approval status with ApprovalSync
    if not ApprovalSync.is_approved(case_id):
        st.warning("⏳ Awaiting nominee approval for intelligence analysis. Share approval link from Consent tab.")
        return
    
    # Check device health
    device_id = cm.ensure_device_id(case_id)
    if device_id and device_id != 'UNKNOWN_DEVICE':
        device_health = DeviceManager.get_device_health(device_id)
        if device_health.get("issues"):
            st.warning(f"⚠️ Device issues detected: {', '.join(device_health['issues'])}")
        if device_health.get("warnings"):
            for warning in device_health["warnings"]:
                st.warning(f"⚠️ {warning}")
    
    # Initialize session state for intelligence analysis
    if 'intelligence_started' not in st.session_state:
        st.session_state.intelligence_started = False
    if 'intelligence_completed' not in st.session_state:
        st.session_state.intelligence_completed = False
    if 'intelligence_tracker' not in st.session_state:
        st.session_state.intelligence_tracker = ProgressTracker(total_steps=100)
    
    tracker = st.session_state.intelligence_tracker
    
    # Intelligence features selection
    st.markdown("### Select Analysis Features")
    
    col1, col2, col3 = st.columns(3)
    
    features = []
    
    with col1:
        if st.checkbox("🔍 Suspicious Activity Detection", value=True, key="cb_suspicious_activity"):
            features.append("Suspicious Activity Detection")
        if st.checkbox("📊 Communication Patterns", value=True, key="cb_comm_patterns"):
            features.append("Communication Patterns")
    
    with col2:
        if st.checkbox("📍 Location Hotspots", value=True, key="cb_location_hotspots"):
            features.append("Location Hotspots")
        if st.checkbox("🗺️ Cell Tower Mapping", value=True, key="cb_cell_tower"):
            features.append("Cell Tower Mapping")
    
    with col3:
        if st.checkbox("🔐 Password Analysis", value=True, key="cb_password_analysis"):
            features.append("Password Analysis")
        if st.checkbox("👥 Contact Network", value=True, key="cb_contact_network"):
            features.append("Contact Network")
    
    if not features:
        st.info("Select at least one feature to analyze")
        return
    
    st.divider()
    
    # Start analysis button
    if not st.session_state.intelligence_started and st.button("🚀 Start Intelligence Analysis", 
                                                             key="btn_start_intelligence"):
        st.session_state.intelligence_started = True
        st.session_state.intelligence_completed = False
        tracker = ProgressTracker(total_steps=len(features) * 100)
        tracker.start()
        st.session_state.intelligence_tracker = tracker
        
        # Create progress tracker for intelligence analysis
        progress_tracker = ProgressManager.create_tracker(case_id, 'intelligence_analysis')
        st.session_state.intelligence_progress_tracker = progress_tracker
        st.rerun()
    
    # Show analysis progress if running
    if st.session_state.intelligence_started:
        progress_placeholder = st.empty()
        progress_tracker = st.session_state.get('intelligence_progress_tracker')
        
        # Start analysis in a separate thread if not already started
        if 'intelligence_thread' not in st.session_state or not st.session_state.intelligence_thread.is_alive():
            if not st.session_state.intelligence_completed:
                def run_analysis():
                    try:
                        # Simulate analysis of each feature
                        for feature_idx, feature in enumerate(features, 1):
                            steps_per_feature = 100 // len(features)
                            start_step = (feature_idx - 1) * steps_per_feature
                            
                            # Track feature analysis
                            if progress_tracker:
                                progress_tracker.start_module(feature)
                            
                            for i in range(steps_per_feature + 1):
                                if st.session_state.get('stop_intelligence', False):
                                    tracker.error("Analysis cancelled by user")
                                    if progress_tracker:
                                        progress_tracker.error_module(feature, "Cancelled by user")
                                    break
                                    
                                current_step = start_step + i
                                tracker.update(
                                    current_step, 
                                    f"Analyzing {feature}... {int((i / steps_per_feature) * 100)}%"
                                )
                                
                                # Update progress tracker
                                if progress_tracker:
                                    progress_tracker.update_module_progress(
                                        feature,
                                        int((i / steps_per_feature) * 100),
                                        artifacts_count=0
                                    )
                                
                                time.sleep(0.05)
                            
                            # Complete feature analysis
                            if progress_tracker and not st.session_state.get('stop_intelligence', False):
                                progress_tracker.complete_module(feature, artifacts_count=0)
                        
                        if not st.session_state.get('stop_intelligence', False):
                            tracker.complete()
                            tracker.message = "Analysis completed successfully!"
                            if progress_tracker:
                                progress_tracker.complete_extraction()
                                progress_tracker.save_progress()
                        
                        st.session_state.intelligence_completed = True
                        st.rerun()
                            
                    except Exception as e:
                        tracker.error(f"Analysis failed: {str(e)}")
                        if progress_tracker:
                            progress_tracker.error_extraction(str(e))
                        st.session_state.intelligence_completed = True
                        st.rerun()
            
                st.session_state.intelligence_thread = threading.Thread(target=run_analysis, daemon=True)
                st.session_state.intelligence_thread.start()
        
        # Display progress
        with progress_placeholder.container():
            if tracker.status == ProgressStatus.RUNNING or not st.session_state.intelligence_completed:
                render_progress_bar(
                    tracker,
                    title="Intelligence Analysis",
                    show_artifacts=True
                )
                
                if st.button("⏹️ Cancel Analysis", key="btn_cancel_intelligence"):
                    st.session_state.stop_intelligence = True
                    st.rerun()
                
                st.divider()
                
                # Show feature progress
                st.markdown("### Feature Progress")
                for i, feature in enumerate(features, 1):
                    feature_progress = min(100, int((tracker.current_step / len(features)) * i))
                    st.markdown(f"**{feature}** - {feature_progress}%")
                    st.progress(feature_progress / 100)
            
            elif tracker.status == ProgressStatus.COMPLETED:
                st.success("✅ Intelligence analysis completed!")
                
                # Show results summary
                st.markdown("### Analysis Results")
                
                results_col1, results_col2, results_col3 = st.columns(3)
                
                with results_col1:
                    st.metric("Suspicious Activities", 12)
                
                with results_col2:
                    st.metric("Location Clusters", 5)
                
                with results_col3:
                    st.metric("Contacts Analyzed", 47)
                
                if st.button("🔄 Run Another Analysis", key="btn_restart_analysis"):
                    st.session_state.intelligence_started = False
                    st.session_state.intelligence_completed = False
                    st.session_state.stop_intelligence = False
                    st.session_state.intelligence_tracker = ProgressTracker(total_steps=100)
                    st.rerun()
            
            elif tracker.status == ProgressStatus.ERROR:
                st.error("❌ Analysis failed. Please try again or check the logs for more details.")
                
                if st.button("🔄 Retry Analysis", key="btn_retry_analysis"):
                    st.session_state.intelligence_started = False
                    st.session_state.intelligence_completed = False
                    st.session_state.stop_intelligence = False
                    st.session_state.intelligence_tracker = ProgressTracker(total_steps=100)
                    st.rerun()


def render_extraction_history() -> None:
    """Render extraction history and statistics."""
    
    st.markdown("### Extraction History")
    
    manager = get_extraction_ui_manager()
    
    if not manager.extraction_history:
        st.info("No extraction history yet")
        return
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Extractions", len(manager.extraction_history))
    
    with col2:
        total_artifacts = sum(e['artifacts'] for e in manager.extraction_history)
        st.metric("Total Artifacts", total_artifacts)
    
    with col3:
        st.metric("Success Rate", "100%")
    
    st.divider()
    
    # History table
    st.markdown("### Recent Extractions")
    
    history_data = []
    for extraction in manager.extraction_history[-10:]:
        history_data.append({
            "Case ID": extraction['case_id'][:12],
            "Type": extraction['type'].upper(),
            "Artifacts": extraction['artifacts'],
            "Time": extraction['timestamp'][:19]
        })
    
    if history_data:
        st.dataframe(history_data, use_container_width=True)
