"""
Modern Progress Bar UI Component for ForenSmart
===============================================

Provides a sleek, modern progress bar UI with animations and status tracking
for extraction and intelligence module operations.

Features:
- Animated progress bars with percentage display
- Multi-stage progress tracking
- Status indicators (pending, running, completed, error)
- Real-time artifact counting
- Modern dark theme styling
"""

import streamlit as st
from typing import Optional, Dict, List, Any, Callable
import time
from enum import Enum


class ProgressStatus(Enum):
    """Status states for progress tracking."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class ProgressTracker:
    """Manages progress state for extraction/intelligence operations."""
    
    def __init__(self, total_steps: int = 100):
        self.total_steps = total_steps
        self.current_step = 0
        self.status = ProgressStatus.PENDING
        self.message = ""
        self.artifacts_count = 0
        self.errors: List[str] = []
        self._last_update_time = 0  # Timestamp of last update
        self._update_threshold = 0.2  # Minimum seconds between updates (5 FPS)
        
    def should_update(self) -> bool:
        """Check if enough time has passed since last update for smooth animation."""
        import time
        current_time = time.time()
        return (current_time - self._last_update_time) >= self._update_threshold

    def update(self, step: int, message: str = "", artifacts: int = 0) -> bool:
        """
        Update progress state.
        
        Args:
            step: Current step (0 to total_steps)
            message: Status message to display
            artifacts: Optional artifact count update
            
        Returns:
            bool: True if the update was applied, False if throttled
        """
        if not self.should_update() and step < self.total_steps:
            return False
            
        self.current_step = min(step, self.total_steps)
        self.message = message
        if artifacts > 0:
            self.artifacts_count = artifacts
            
        import time
        self._last_update_time = time.time()
        return True
            
    def start(self):
        """Mark operation as started."""
        self.status = ProgressStatus.RUNNING
        self.current_step = 0
        self._last_update_time = 0  # Reset update timer
        
    def complete(self):
        """Mark operation as completed."""
        self.status = ProgressStatus.COMPLETED
        self.current_step = self.total_steps
        
    def error(self, error_msg: str):
        """Mark operation as errored."""
        self.status = ProgressStatus.ERROR
        self.errors.append(error_msg)
        
    def get_percentage(self) -> int:
        """Get completion percentage."""
        if self.total_steps == 0:
            return 0
        return int((self.current_step / self.total_steps) * 100)


def render_progress_bar(
    tracker: ProgressTracker,
    title: str = "Processing",
    show_artifacts: bool = True,
    show_details: bool = True
) -> None:
    """
    Render a modern progress bar UI.
    
    Args:
        tracker: ProgressTracker instance
        title: Title to display above progress bar
        show_artifacts: Whether to show artifact count
        show_details: Whether to show detailed status message
    """
    
    # Title
    st.markdown(f"### {title}")
    
    # Progress bar container with custom styling
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Custom progress bar with percentage
        percentage = tracker.get_percentage()
        
        # Create progress bar HTML
        progress_html = f"""
        <div style="
            width: 100%;
            height: 12px;
            background-color: #1a1a2e;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
            margin: 10px 0;
        ">
            <div style="
                width: {percentage}%;
                height: 100%;
                background: linear-gradient(90deg, #00d4ff, #0099ff);
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
                transition: width 0.3s ease;
            "></div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
    
    with col2:
        # Percentage display
        status_color = {
            ProgressStatus.PENDING: "#888888",
            ProgressStatus.RUNNING: "#00d4ff",
            ProgressStatus.COMPLETED: "#00ff88",
            ProgressStatus.ERROR: "#ff4444"
        }.get(tracker.status, "#888888")
        
        st.markdown(
            f"<div style='text-align: center; color: {status_color}; font-weight: bold; font-size: 18px;'>"
            f"{percentage}%</div>",
            unsafe_allow_html=True
        )
    
    # Status indicator and message
    if show_details:
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            # Status icon
            status_icon = {
                ProgressStatus.PENDING: "⏳",
                ProgressStatus.RUNNING: "⚙️",
                ProgressStatus.COMPLETED: "✅",
                ProgressStatus.ERROR: "❌"
            }.get(tracker.status, "⏳")
            st.markdown(f"<div style='font-size: 24px;'>{status_icon}</div>", unsafe_allow_html=True)
        
        with col2:
            # Status message
            if tracker.message:
                st.markdown(f"**{tracker.message}**")
            elif tracker.status == ProgressStatus.RUNNING:
                st.markdown("**Processing...**")
            elif tracker.status == ProgressStatus.COMPLETED:
                st.markdown("**Completed successfully**")
            elif tracker.status == ProgressStatus.ERROR:
                st.markdown("**Error occurred**")
        
        with col3:
            # Artifacts count
            if show_artifacts and tracker.artifacts_count > 0:
                st.markdown(
                    f"<div style='text-align: center; color: #00d4ff;'>"
                    f"<div style='font-size: 20px; font-weight: bold;'>{tracker.artifacts_count}</div>"
                    f"<div style='font-size: 12px;'>artifacts</div></div>",
                    unsafe_allow_html=True
                )
    
    # Error display
    if tracker.status == ProgressStatus.ERROR and tracker.errors:
        st.error("\n".join(tracker.errors))


def render_multi_stage_progress(
    stages: List[Dict[str, Any]],
    title: str = "Multi-Stage Processing"
) -> None:
    """
    Render multi-stage progress tracking.
    
    Args:
        stages: List of stage dicts with keys:
                - name: str (stage name)
                - status: ProgressStatus
                - percentage: int (0-100)
                - message: str (optional)
        title: Title to display
    """
    
    st.markdown(f"### {title}")
    
    for i, stage in enumerate(stages, 1):
        with st.container():
            # Stage header
            status_icon = {
                ProgressStatus.PENDING: "⏳",
                ProgressStatus.RUNNING: "⚙️",
                ProgressStatus.COMPLETED: "✅",
                ProgressStatus.ERROR: "❌"
            }.get(stage.get("status", ProgressStatus.PENDING), "⏳")
            
            col1, col2 = st.columns([0.5, 9.5])
            with col1:
                st.markdown(f"<div style='font-size: 20px;'>{status_icon}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**{i}. {stage['name']}**")
            
            # Progress bar
            percentage = stage.get("percentage", 0)
            progress_html = f"""
            <div style="
                width: 100%;
                height: 8px;
                background-color: #1a1a2e;
                border-radius: 5px;
                overflow: hidden;
                margin: 5px 0 10px 0;
            ">
                <div style="
                    width: {percentage}%;
                    height: 100%;
                    background: linear-gradient(90deg, #00d4ff, #0099ff);
                    border-radius: 5px;
                    transition: width 0.3s ease;
                "></div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
            
            # Message
            if stage.get("message"):
                st.caption(stage["message"])
            
            st.divider()


def render_extraction_progress(
    case_id: str,
    extraction_type: str,
    tracker: ProgressTracker,
    on_cancel: Optional[Callable] = None
) -> None:
    """
    Render extraction progress UI with controls.
    
    Args:
        case_id: Case ID being extracted
        extraction_type: Type of extraction (Android, iOS, HDD, etc.)
        tracker: ProgressTracker instance
        on_cancel: Optional callback for cancel button
    """
    
    st.markdown(f"## {extraction_type} Extraction")
    
    # Case info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Case ID", case_id[:12] + "..." if len(case_id) > 12 else case_id)
    with col2:
        st.metric("Type", extraction_type)
    with col3:
        st.metric("Artifacts", tracker.artifacts_count)
    
    st.divider()
    
    # Progress bar
    render_progress_bar(tracker, "Extraction Progress", show_artifacts=True)
    
    st.divider()
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if tracker.status == ProgressStatus.RUNNING:
            if st.button("⏸️ Pause", key=f"pause_{case_id}"):
                st.info("Pause functionality coming soon")
    
    with col2:
        if tracker.status == ProgressStatus.RUNNING and on_cancel:
            if st.button("🛑 Cancel", key=f"cancel_{case_id}"):
                on_cancel()
                st.warning("Extraction cancelled")
    
    with col3:
        if tracker.status == ProgressStatus.COMPLETED:
            if st.button("📊 View Results", key=f"results_{case_id}"):
                st.success("Results view coming soon")


def render_intelligence_progress(
    case_id: str,
    features: List[str],
    tracker: ProgressTracker
) -> None:
    """
    Render intelligence module progress UI.
    
    Args:
        case_id: Case ID
        features: List of intelligence features being analyzed
        tracker: ProgressTracker instance
    """
    
    st.markdown("## Intelligence Analysis")
    
    # Case info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Case ID", case_id[:12] + "..." if len(case_id) > 12 else case_id)
    with col2:
        st.metric("Features", len(features))
    
    st.divider()
    
    # Overall progress
    render_progress_bar(tracker, "Overall Analysis Progress", show_artifacts=False)
    
    st.divider()
    
    # Feature-level progress
    st.markdown("### Analyzing Features")
    
    for i, feature in enumerate(features, 1):
        # Simulate feature progress (in real implementation, track actual progress)
        feature_percentage = min(100, int((tracker.get_percentage() / len(features)) * (i + 1)))
        
        col1, col2 = st.columns([0.5, 9.5])
        with col1:
            if feature_percentage >= 100:
                st.markdown("✅")
            elif feature_percentage > 0:
                st.markdown("⚙️")
            else:
                st.markdown("⏳")
        
        with col2:
            st.markdown(f"**{feature}** - {feature_percentage}%")
            
            # Mini progress bar
            progress_html = f"""
            <div style="
                width: 100%;
                height: 4px;
                background-color: #1a1a2e;
                border-radius: 2px;
                overflow: hidden;
                margin: 2px 0;
            ">
                <div style="
                    width: {feature_percentage}%;
                    height: 100%;
                    background: linear-gradient(90deg, #00d4ff, #0099ff);
                    border-radius: 2px;
                "></div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)


def render_live_artifact_feed(
    artifacts: List[Dict[str, Any]],
    max_items: int = 10
) -> None:
    """
    Render a live feed of extracted artifacts.
    
    Args:
        artifacts: List of artifact dicts with keys:
                  - name: str
                  - type: str
                  - timestamp: str
                  - size: str (optional)
        max_items: Maximum items to display
    """
    
    st.markdown("### Live Artifact Feed")
    
    if not artifacts:
        st.info("No artifacts extracted yet")
        return
    
    # Show most recent artifacts
    recent = artifacts[-max_items:][::-1]
    
    for artifact in recent:
        with st.container():
            col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
            
            with col1:
                # Type icon
                type_icon = {
                    "message": "💬",
                    "call": "📞",
                    "location": "📍",
                    "image": "🖼️",
                    "video": "🎥",
                    "file": "📄",
                    "contact": "👤"
                }.get(artifact.get("type", "file"), "📦")
                st.markdown(f"<div style='font-size: 20px;'>{type_icon}</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**{artifact['name']}**")
            
            with col3:
                st.caption(artifact.get("timestamp", ""))
            
            with col4:
                if artifact.get("size"):
                    st.caption(artifact["size"])
            
            st.divider()
