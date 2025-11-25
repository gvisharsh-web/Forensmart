"""Utilities for managing approval decisions and consent records."""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def get_approvals_file() -> Path:
    """Get path to shared approvals file - prioritizes project directory for offline access."""
    # Primary: Project directory (for offline/local access)
    project_approvals_dir = Path(__file__).resolve().parent.parent / 'audit' / 'approvals'
    try:
        project_approvals_dir.mkdir(parents=True, exist_ok=True)
        return project_approvals_dir / 'approvals.json'
    except Exception as e:
        logger.warning(f"Failed to use project directory: {e}")
        pass
    
    # Fallback: User home directory (for cloud/online deployments)
    approvals_dir = Path.home() / '.forensmart'
    try:
        approvals_dir.mkdir(parents=True, exist_ok=True)
        return approvals_dir / 'approvals.json'
    except Exception:
        pass
    
    # Fallback paths
    fallback_paths = [
        Path('/tmp/forensmart_approvals.json'),  # Linux/Mac temp
        Path('C:\\ProgramData\\ForenSmart\\approvals.json'),  # Windows shared
    ]
    
    for path in fallback_paths:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        except Exception:
            continue
    
    # Last resort: current directory
    return Path('.forensmart_approvals.json')


def get_approval_decision(case_id: str) -> Optional[str]:
    """Get approval decision for a case."""
    try:
        approvals_file = get_approvals_file()
        if not approvals_file.exists():
            return None
        
        data = json.loads(approvals_file.read_text(encoding="utf-8"))
        if case_id in data:
            return data[case_id].get('decision')
        return None
    except Exception as e:
        logger.error(f"Failed to get approval decision: {e}")
        return None


def save_approval_decision(
    case_id: str,
    decision: str,
    nominee_name: Optional[str] = None,
    message: Optional[str] = None
) -> bool:
    """Save approval decision to file."""
    try:
        approvals_file = get_approvals_file()
        approvals = {}
        
        if approvals_file.exists():
            try:
                approvals = json.loads(approvals_file.read_text(encoding="utf-8"))
            except Exception:
                approvals = {}
        
        approvals[case_id] = {
            'decision': decision,
            'timestamp': datetime.now().isoformat(),
            'nominee_name': nominee_name or '',
            'message': message or '',
        }
        
        approvals_file.write_text(json.dumps(approvals, indent=2), encoding="utf-8")
        logger.info(f"Approval decision saved for {case_id}: {decision}")
        return True
    except Exception as e:
        logger.error(f"Failed to save approval decision: {e}")
        return False


def get_approval_data(case_id: str) -> Optional[Dict[str, Any]]:
    """Get full approval data for a case."""
    try:
        approvals_file = get_approvals_file()
        if not approvals_file.exists():
            return None
        
        data = json.loads(approvals_file.read_text(encoding="utf-8"))
        if case_id in data:
            return data[case_id]
        return None
    except Exception as e:
        logger.error(f"Failed to get approval data: {e}")
        return None


def clear_approval(case_id: str) -> bool:
    """Clear approval for a case."""
    try:
        approvals_file = get_approvals_file()
        if not approvals_file.exists():
            return True
        
        data = json.loads(approvals_file.read_text(encoding="utf-8"))
        if case_id in data:
            del data[case_id]
            approvals_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.info(f"Approval cleared for {case_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to clear approval: {e}")
        return False


__all__ = [
    'get_approvals_file',
    'get_approval_decision',
    'save_approval_decision',
    'get_approval_data',
    'clear_approval',
]
