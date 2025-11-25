"""Real-time extraction progress tracking and monitoring."""
from __future__ import annotations

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModuleProgress:
    """Progress for a single extraction module."""
    module_name: str
    status: str  # pending, running, completed, error
    progress_percent: int
    artifacts_count: int
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error_message: Optional[str] = None


class ExtractionProgressTracker:
    """Track extraction progress in real-time."""

    def __init__(self, case_id: str, extraction_type: str):
        self.case_id = case_id
        self.extraction_type = extraction_type
        self.start_time = datetime.now()
        self.modules: Dict[str, ModuleProgress] = {}
        self.total_artifacts = 0
        self.status = "pending"  # pending, running, completed, error
        self.error_message = None

    def start_module(self, module_name: str) -> None:
        """Mark module as started."""
        self.modules[module_name] = ModuleProgress(
            module_name=module_name,
            status="running",
            progress_percent=0,
            artifacts_count=0,
            start_time=datetime.now().isoformat()
        )
        self.status = "running"
        logger.info(f"Started extraction module: {module_name}")

    def update_module_progress(
        self,
        module_name: str,
        progress_percent: int,
        artifacts_count: int = 0
    ) -> None:
        """Update module progress."""
        if module_name not in self.modules:
            self.start_module(module_name)
        
        module = self.modules[module_name]
        module.progress_percent = min(100, max(0, progress_percent))
        module.artifacts_count = artifacts_count
        self.total_artifacts += artifacts_count

    def complete_module(self, module_name: str, artifacts_count: int = 0) -> None:
        """Mark module as completed."""
        if module_name not in self.modules:
            self.start_module(module_name)
        
        module = self.modules[module_name]
        module.status = "completed"
        module.progress_percent = 100
        module.artifacts_count = artifacts_count
        module.end_time = datetime.now().isoformat()
        self.total_artifacts += artifacts_count
        logger.info(f"Completed extraction module: {module_name} ({artifacts_count} artifacts)")

    def error_module(self, module_name: str, error_message: str) -> None:
        """Mark module as errored."""
        if module_name not in self.modules:
            self.start_module(module_name)
        
        module = self.modules[module_name]
        module.status = "error"
        module.error_message = error_message
        module.end_time = datetime.now().isoformat()
        logger.error(f"Error in extraction module {module_name}: {error_message}")

    def get_overall_progress(self) -> int:
        """Get overall progress percentage."""
        if not self.modules:
            return 0
        
        total_progress = sum(m.progress_percent for m in self.modules.values())
        return int(total_progress / len(self.modules))

    def get_status_summary(self) -> Dict[str, Any]:
        """Get extraction status summary."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "case_id": self.case_id,
            "extraction_type": self.extraction_type,
            "status": self.status,
            "overall_progress": self.get_overall_progress(),
            "total_artifacts": self.total_artifacts,
            "modules_completed": sum(1 for m in self.modules.values() if m.status == "completed"),
            "modules_running": sum(1 for m in self.modules.values() if m.status == "running"),
            "modules_error": sum(1 for m in self.modules.values() if m.status == "error"),
            "elapsed_seconds": int(elapsed),
            "modules": {
                name: asdict(module)
                for name, module in self.modules.items()
            }
        }

    def complete_extraction(self) -> None:
        """Mark extraction as completed."""
        self.status = "completed"
        logger.info(
            f"Extraction completed: {self.case_id} - "
            f"{self.total_artifacts} artifacts extracted"
        )

    def error_extraction(self, error_message: str) -> None:
        """Mark extraction as errored."""
        self.status = "error"
        self.error_message = error_message
        logger.error(f"Extraction failed: {error_message}")

    def save_progress(self) -> bool:
        """Save progress to file."""
        try:
            progress_dir = Path("reports") / self.case_id
            progress_dir.mkdir(parents=True, exist_ok=True)
            
            progress_file = progress_dir / f"extraction_progress_{self.extraction_type}.json"
            summary = self.get_status_summary()
            
            progress_file.write_text(json.dumps(summary, indent=2))
            logger.info(f"Saved extraction progress to {progress_file}")
            return True
        except PermissionError as e:
            logger.error(f"Permission denied saving progress to {progress_file}: {e}", exc_info=True)
            return False
        except IOError as e:
            logger.error(f"IO error saving progress: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Failed to save extraction progress: {type(e).__name__}: {e}", exc_info=True)
            return False

    @staticmethod
    def load_progress(case_id: str, extraction_type: str) -> Optional[Dict[str, Any]]:
        """Load saved extraction progress."""
        try:
            progress_file = Path("reports") / case_id / f"extraction_progress_{extraction_type}.json"
            if not progress_file.exists():
                logger.debug(f"Progress file not found: {progress_file}")
                return None
            
            content = progress_file.read_text()
            return json.loads(content)
        
        except FileNotFoundError:
            logger.debug(f"Progress file not found for {case_id}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Progress file corrupted for {case_id}: {e}", exc_info=True)
            return None
        except PermissionError as e:
            logger.error(f"Permission denied reading progress: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Failed to load extraction progress: {type(e).__name__}: {e}", exc_info=True)
            return None


class ProgressManager:
    """Manage multiple extraction progress trackers."""

    _trackers: Dict[str, ExtractionProgressTracker] = {}

    @staticmethod
    def create_tracker(case_id: str, extraction_type: str) -> ExtractionProgressTracker:
        """Create new progress tracker."""
        key = f"{case_id}_{extraction_type}"
        tracker = ExtractionProgressTracker(case_id, extraction_type)
        ProgressManager._trackers[key] = tracker
        return tracker

    @staticmethod
    def get_tracker(case_id: str, extraction_type: str) -> Optional[ExtractionProgressTracker]:
        """Get existing progress tracker."""
        key = f"{case_id}_{extraction_type}"
        return ProgressManager._trackers.get(key)

    @staticmethod
    def get_all_trackers(case_id: str) -> List[ExtractionProgressTracker]:
        """Get all trackers for a case."""
        return [
            t for k, t in ProgressManager._trackers.items()
            if k.startswith(f"{case_id}_")
        ]

    @staticmethod
    def remove_tracker(case_id: str, extraction_type: str) -> None:
        """Remove progress tracker."""
        key = f"{case_id}_{extraction_type}"
        ProgressManager._trackers.pop(key, None)


__all__ = ["ExtractionProgressTracker", "ProgressManager", "ModuleProgress"]
