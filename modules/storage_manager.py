"""
ForenSmart Storage Manager
==========================

Comprehensive storage management with:
- Artifact deletion and cleanup
- Case deletion with full cleanup
- Storage analytics and monitoring
- Disk space management
- Deletion history and audit trail
- Safe deletion with verification

Author: ForenSmart Development Team
"""

import os
import shutil
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import threading

logger = logging.getLogger(__name__)

def _handle_rmtree_error(func, path: str, exc_info: tuple) -> None:
    """
    Error handler for shutil.rmtree.

    This handler will:
    1. Log the specific error.
    2. If it's a FileNotFoundError on Windows, attempt to fix trailing space issues.
    3. If it's a PermissionError, attempt to make the file writable and retry.
    """
    exc_type, exc_value, _ = exc_info
    logger.warning(f"Deletion error in rmtree: {exc_type.__name__} on path '{path}': {exc_value}")

    # Handle trailing space issue on Windows for directories
    if exc_type is FileNotFoundError and os.path.isdir(path) and os.name == 'nt':
        new_path = path.rstrip('. \t')
        if new_path != path and not os.path.exists(new_path):
            try:
                os.rename(path, new_path)
                logger.info(f"Renamed '{path}' to '{new_path}' to fix trailing space issue.")
                func(new_path)
                return
            except Exception as e:
                logger.error(f"rmtree handler failed to fix trailing space for '{path}': {e}")

    # Handle permission errors by trying to make the file writable
    if exc_type is PermissionError:
        try:
            # Make the file/dir writable and try again
            os.chmod(path, 0o777)
            func(path) # Retry the original function (e.g., os.remove)
            logger.info(f"Successfully deleted '{path}' after fixing permissions.")
        except Exception as e:
            logger.error(f"rmtree handler failed to fix permissions for '{path}': {e}")


class StorageAnalytics:
    """Analyze storage usage and provide statistics."""
    
    @staticmethod
    def get_directory_size(path: str) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception as e:
            logger.error(f"Error calculating directory size: {e}")
        return total_size
    
    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Convert bytes to human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    @staticmethod
    def get_case_storage_info(case_id: str) -> Dict[str, Any]:
        """Get storage information for a specific case."""
        artifacts_dir = os.path.join('artifacts', case_id)
        reports_dir = os.path.join('reports', case_id)
        consent_dir = os.path.join('consent_records', case_id)
        
        artifacts_size = StorageAnalytics.get_directory_size(artifacts_dir) if os.path.exists(artifacts_dir) else 0
        reports_size = StorageAnalytics.get_directory_size(reports_dir) if os.path.exists(reports_dir) else 0
        consent_size = StorageAnalytics.get_directory_size(consent_dir) if os.path.exists(consent_dir) else 0
        
        total_size = artifacts_size + reports_size + consent_size
        
        return {
            'case_id': case_id,
            'artifacts_size': artifacts_size,
            'artifacts_size_formatted': StorageAnalytics.format_size(artifacts_size),
            'reports_size': reports_size,
            'reports_size_formatted': StorageAnalytics.format_size(reports_size),
            'consent_size': consent_size,
            'consent_size_formatted': StorageAnalytics.format_size(consent_size),
            'total_size': total_size,
            'total_size_formatted': StorageAnalytics.format_size(total_size),
            'artifacts_exist': os.path.exists(artifacts_dir),
            'reports_exist': os.path.exists(reports_dir),
            'consent_exist': os.path.exists(consent_dir),
        }
    
    @staticmethod
    def get_total_storage_info() -> Dict[str, Any]:
        """Get total storage usage across all cases."""
        artifacts_total = StorageAnalytics.get_directory_size('artifacts') if os.path.exists('artifacts') else 0
        reports_total = StorageAnalytics.get_directory_size('reports') if os.path.exists('reports') else 0
        consent_total = StorageAnalytics.get_directory_size('consent_records') if os.path.exists('consent_records') else 0
        
        total_size = artifacts_total + reports_total + consent_total
        
        # Count cases
        case_count = 0
        if os.path.exists('artifacts'):
            case_count = len([d for d in os.listdir('artifacts') if os.path.isdir(os.path.join('artifacts', d))])
        
        return {
            'artifacts_total': artifacts_total,
            'artifacts_total_formatted': StorageAnalytics.format_size(artifacts_total),
            'reports_total': reports_total,
            'reports_total_formatted': StorageAnalytics.format_size(reports_total),
            'consent_total': consent_total,
            'consent_total_formatted': StorageAnalytics.format_size(consent_total),
            'total_size': total_size,
            'total_size_formatted': StorageAnalytics.format_size(total_size),
            'case_count': case_count,
        }
    
    @staticmethod
    def list_cases_by_size() -> List[Dict[str, Any]]:
        """List all cases sorted by storage usage."""
        cases = []
        
        if os.path.exists('artifacts'):
            for case_dir in os.listdir('artifacts'):
                case_path = os.path.join('artifacts', case_dir)
                if os.path.isdir(case_path):
                    info = StorageAnalytics.get_case_storage_info(case_dir)
                    cases.append(info)
        
        # Sort by total size (descending)
        cases.sort(key=lambda x: x['total_size'], reverse=True)
        return cases


class DeletionAudit:
    """Track deletion history and audit trail."""
    
    AUDIT_FILE = 'audit/deletion_history.json'
    
    @staticmethod
    def _ensure_audit_dir():
        """Ensure audit directory exists."""
        os.makedirs(os.path.dirname(DeletionAudit.AUDIT_FILE), exist_ok=True)
    
    @staticmethod
    def _load_history() -> List[Dict[str, Any]]:
        """Load deletion history."""
        DeletionAudit._ensure_audit_dir()
        if os.path.exists(DeletionAudit.AUDIT_FILE):
            try:
                with open(DeletionAudit.AUDIT_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading deletion history: {e}")
        return []
    
    @staticmethod
    def _save_history(history: List[Dict[str, Any]]):
        """Save deletion history."""
        DeletionAudit._ensure_audit_dir()
        try:
            with open(DeletionAudit.AUDIT_FILE, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving deletion history: {e}")
    
    @staticmethod
    def log_deletion(case_id: str, deleted_items: Dict[str, Any], reason: str = "User deletion"):
        """Log a deletion event."""
        history = DeletionAudit._load_history()
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'case_id': case_id,
            'deleted_items': deleted_items,
            'reason': reason,
            'status': 'completed'
        }
        
        history.append(entry)
        DeletionAudit._save_history(history)
        logger.info(f"Deletion logged for case {case_id}")
    
    @staticmethod
    def get_deletion_history(case_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get deletion history, optionally filtered by case."""
        history = DeletionAudit._load_history()
        
        if case_id:
            history = [h for h in history if h.get('case_id') == case_id]
        
        return history


class StorageManager:
    """Comprehensive storage management."""
    
    @staticmethod
    def delete_artifact_directory(case_id: str, artifact_type: str = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Delete specific artifact directory for a case.
        
        Args:
            case_id: Case ID
            artifact_type: Type of artifacts (android, ios, hdd, etc.) or None for all
        
        Returns:
            Tuple of (success, message, deleted_info)
        """
        try:
            if artifact_type:
                artifact_path = os.path.join('artifacts', case_id, artifact_type)
            else:
                artifact_path = os.path.join('artifacts', case_id)
            
            if not os.path.exists(artifact_path):
                return True, f"Artifact path not found: {artifact_path}", {}
            
            # Get size before deletion
            size_before = StorageAnalytics.get_directory_size(artifact_path)
            
            # Delete directory
            shutil.rmtree(artifact_path, onerror=_handle_rmtree_error)
            
            deleted_info = {
                'path': artifact_path,
                'size_freed': size_before,
                'size_freed_formatted': StorageAnalytics.format_size(size_before),
                'type': artifact_type or 'all'
            }
            
            logger.info(f"Deleted {artifact_type or 'all'} artifacts for case {case_id}")
            return True, f"Successfully deleted {artifact_type or 'all'} artifacts", deleted_info
            
        except Exception as e:
            logger.error(f"Error deleting artifacts: {e}")
            return False, f"Error deleting artifacts: {str(e)}", {}
    
    @staticmethod
    def delete_case_reports(case_id: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Delete all reports for a case."""
        try:
            reports_path = os.path.join('reports', case_id)
            
            if not os.path.exists(reports_path):
                return False, "Reports not found", {}
            
            size_before = StorageAnalytics.get_directory_size(reports_path)
            shutil.rmtree(reports_path, onerror=_handle_rmtree_error)
            
            deleted_info = {
                'path': reports_path,
                'size_freed': size_before,
                'size_freed_formatted': StorageAnalytics.format_size(size_before)
            }
            
            logger.info(f"Deleted reports for case {case_id}")
            return True, "Reports deleted successfully", deleted_info
            
        except Exception as e:
            logger.error(f"Error deleting reports: {e}")
            return False, f"Error deleting reports: {str(e)}", {}
    
    @staticmethod
    def delete_case_consent_data(case_id: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Delete consent data for a case."""
        try:
            consent_path = os.path.join('consent_records', case_id)
            
            if not os.path.exists(consent_path):
                return False, "Consent data not found", {}
            
            size_before = StorageAnalytics.get_directory_size(consent_path)
            shutil.rmtree(consent_path, onerror=_handle_rmtree_error)
            
            deleted_info = {
                'path': consent_path,
                'size_freed': size_before,
                'size_freed_formatted': StorageAnalytics.format_size(size_before)
            }
            
            logger.info(f"Deleted consent data for case {case_id}")
            return True, "Consent data deleted successfully", deleted_info
            
        except Exception as e:
            logger.error(f"Error deleting consent data: {e}")
            return False, f"Error deleting consent data: {str(e)}", {}
    
    @staticmethod
    def delete_case_snapshots(case_id: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Delete case snapshots."""
        try:
            snapshots_dir = 'case_snapshots'
            deleted_count = 0
            total_size = 0
            
            if os.path.exists(snapshots_dir):
                for filename in os.listdir(snapshots_dir):
                    if case_id in filename:
                        filepath = os.path.join(snapshots_dir, filename)
                        if os.path.isfile(filepath):
                            total_size += os.path.getsize(filepath)
                            os.remove(filepath)
                            deleted_count += 1
            
            deleted_info = {
                'files_deleted': deleted_count,
                'size_freed': total_size,
                'size_freed_formatted': StorageAnalytics.format_size(total_size)
            }
            
            logger.info(f"Deleted {deleted_count} snapshots for case {case_id}")
            return True, f"Deleted {deleted_count} snapshots", deleted_info
            
        except Exception as e:
            logger.error(f"Error deleting snapshots: {e}")
            return False, f"Error deleting snapshots: {str(e)}", {}
    
    @staticmethod
    def delete_entire_case(case_id: str, consent_manager=None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Delete entire case including all artifacts, reports, consent data, and snapshots.
        
        Args:
            case_id: Case ID to delete
            consent_manager: Optional ConsentManager instance to clean up sessions
        
        Returns:
            Tuple of (success, message, deleted_info)
        """
        try:
            overall_success = True
            deleted_info = {
                'case_id': case_id,
                'artifacts': {},
                'reports': {},
                'consent_data': {},
                'snapshots': {},
                'session_deleted': False,
                'total_size_freed': 0
            }
            
            # Delete artifacts
            success, msg, info = StorageManager.delete_artifact_directory(case_id)
            if success:
                deleted_info['artifacts'] = info
                deleted_info['total_size_freed'] += info.get('size_freed', 0)
            else:
                overall_success = False
            
            # Delete reports
            success, msg, info = StorageManager.delete_case_reports(case_id)
            if success:
                deleted_info['reports'] = info
                deleted_info['total_size_freed'] += info.get('size_freed', 0)
            else:
                overall_success = False
            
            # Delete consent data
            success, msg, info = StorageManager.delete_case_consent_data(case_id)
            if success:
                deleted_info['consent_data'] = info
                deleted_info['total_size_freed'] += info.get('size_freed', 0)
            else:
                overall_success = False
            
            # Delete snapshots
            success, msg, info = StorageManager.delete_case_snapshots(case_id)
            if success:
                deleted_info['snapshots'] = info
                deleted_info['total_size_freed'] += info.get('size_freed', 0)
            else:
                overall_success = False
            
            # Delete consent session if manager provided
            if consent_manager:
                try:
                    if case_id in consent_manager.sessions:
                        del consent_manager.sessions[case_id]
                        deleted_info['session_deleted'] = True
                except Exception as e:
                    logger.warning(f"Could not delete consent session: {e}")
                    overall_success = False
            
            # Log deletion
            deleted_info['total_size_freed_formatted'] = StorageAnalytics.format_size(
                deleted_info['total_size_freed']
            )
            DeletionAudit.log_deletion(case_id, deleted_info)
            
            if overall_success:
                message = f"Case '{case_id}' deleted successfully. Freed {deleted_info['total_size_freed_formatted']}"
                logger.info(message)
            else:
                message = f"Case '{case_id}' partially deleted. Please check logs."
                logger.warning(message)

            return overall_success, message, deleted_info
            
        except Exception as e:
            logger.error(f"Error deleting case: {e}")
            return False, f"Error deleting case: {str(e)}", {}
    
    @staticmethod
    def cleanup_orphaned_artifacts(dry_run: bool = True) -> Tuple[int, int, Dict[str, Any]]:
        """
        Clean up orphaned artifacts (artifacts without corresponding consent records).
        
        Args:
            dry_run: If True, only report what would be deleted
        
        Returns:
            Tuple of (files_deleted, size_freed, details)
        """
        files_deleted = 0
        size_freed = 0
        details = {'orphaned_cases': []}
        
        try:
            if not os.path.exists('artifacts'):
                return 0, 0, details
            
            # Get list of cases with consent records
            consent_cases = set()
            if os.path.exists('consent_records'):
                consent_cases = set(os.listdir('consent_records'))
            
            # Check each artifact case
            for case_dir in os.listdir('artifacts'):
                case_path = os.path.join('artifacts', case_dir)
                
                if not os.path.isdir(case_path):
                    continue
                
                if case_dir not in consent_cases:
                    # This is an orphaned case
                    size = StorageAnalytics.get_directory_size(case_path)
                    
                    details['orphaned_cases'].append({
                        'case_id': case_dir,
                        'size': size,
                        'size_formatted': StorageAnalytics.format_size(size)
                    })
                    
                    if not dry_run:
                        try:
                            shutil.rmtree(case_path, onerror=_handle_rmtree_error)
                            files_deleted += 1
                            size_freed += size
                        except Exception as e:
                            logger.error(f"Error deleting orphaned case {case_dir}: {e}")
        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        return files_deleted, size_freed, details
    
    @staticmethod
    def get_deletion_candidates(min_age_days: int = 30) -> List[Dict[str, Any]]:
        """
        Get cases that are candidates for deletion based on age.
        
        Args:
            min_age_days: Minimum age in days to consider for deletion
        
        Returns:
            List of candidate cases with info
        """
        candidates = []
        
        try:
            if not os.path.exists('consent_records'):
                return candidates
            
            current_time = datetime.now().timestamp()
            min_age_seconds = min_age_days * 24 * 60 * 60
            
            for case_dir in os.listdir('consent_records'):
                case_path = os.path.join('consent_records', case_dir)
                
                if not os.path.isdir(case_path):
                    continue
                
                # Get modification time
                mod_time = os.path.getmtime(case_path)
                age_seconds = current_time - mod_time
                age_days = age_seconds / (24 * 60 * 60)
                
                if age_seconds >= min_age_seconds:
                    storage_info = StorageAnalytics.get_case_storage_info(case_dir)
                    candidates.append({
                        'case_id': case_dir,
                        'age_days': int(age_days),
                        'storage_info': storage_info,
                        'last_modified': datetime.fromtimestamp(mod_time).isoformat()
                    })
            
            # Sort by age (oldest first)
            candidates.sort(key=lambda x: x['age_days'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting deletion candidates: {e}")
        
        return candidates
