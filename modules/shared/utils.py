"""Shared helper utilities for ForenSmart modules."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


class ArtifactPathBuilder:
    """Centralise artifact path resolution."""

    BASE_DIR = "artifacts"

    @classmethod
    def resolve(
        cls,
        case_id: Optional[str],
        *segments: str,
        ensure_dir: bool = False,
        ensure_parent: bool = False,
    ) -> str:
        safe_case = (case_id or "default_case").strip() or "default_case"
        path = os.path.join(cls.BASE_DIR, safe_case, *segments)
        if ensure_dir:
            os.makedirs(path, exist_ok=True)
        elif ensure_parent:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


class ResultsRepository:
    """Handle loading and saving extraction results."""

    BASE_DIR = "reports"

    @classmethod
    def _path(cls, case_id: Optional[str]) -> str:
        safe_case = (case_id or "default_case").strip() or "default_case"
        return os.path.join(cls.BASE_DIR, safe_case, "results.json")

    @classmethod
    def load(cls, case_id: Optional[str]) -> Optional[Dict[str, Any]]:
        path = cls._path(case_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None

    @classmethod
    def save(cls, case_id: Optional[str], data: Dict[str, Any]) -> None:
        path = cls._path(case_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, default=str)


class MediaManifest:
    """Build media manifests for viewer consumption."""

    @staticmethod
    def build(case_id: str, results: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []

        manifest: List[Dict[str, Any]] = []
        media_section = (results.get("data") or {}).get("media", {})
        media_map = {
            "photos": "photo",
            "videos": "video",
            "audio": "audio",
        }
        for collection, kind in media_map.items():
            items = media_section.get(collection) or []
            for item in items:
                if not isinstance(item, dict):
                    continue
                manifest.append(
                    {
                        "case_id": case_id,
                        "type": kind,
                        "path": item.get("path"),
                        "metadata": item,
                    }
                )
        return manifest


class ConsentVaultHelper:
    """Thin wrapper to interact with ConsentManager's privacy vault."""

    @staticmethod
    def store_entry(consent_manager: Any, case_id: str, device_id: str, secret: str,
                    auth_type: str = "PIN", consent_level: str = "STANDARD") -> Optional[str]:
        vault = getattr(consent_manager, "privacy_vault", None)
        if not vault:
            return None
        try:
            return vault.store_pin_pattern(case_id, device_id, secret, auth_type, consent_level)
        except Exception:
            return None

    @staticmethod
    def verify_entry(consent_manager: Any, vault_id: str, attempt: str, case_id: str) -> bool:
        vault = getattr(consent_manager, "privacy_vault", None)
        if not vault:
            return False
        try:
            return bool(vault.verify_pin_pattern(vault_id, attempt, case_id))
        except Exception:
            return False


class ArtifactIndex:
    """Provide lightweight artifact listings for a case."""

    @staticmethod
    def list_files(case_id: str, *subdirs: str, max_depth: int = 3) -> List[str]:
        root = ArtifactPathBuilder.resolve(case_id, *subdirs)
        if not os.path.isdir(root):
            return []
        results: List[str] = []
        for current_root, dirs, files in os.walk(root):
            depth = current_root.replace(root, '').count(os.sep)
            if depth >= max_depth:
                dirs[:] = []
            for name in files:
                results.append(os.path.join(current_root, name))
        return sorted(results)


class ConsentAuditReader:
    """Access consent audit logs from disk."""

    BASE_DIR = os.path.join("audit", "consent_records")

    @classmethod
    def list_events(
        cls, case_id: Optional[str], limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not case_id:
            return []
        path = os.path.join(cls.BASE_DIR, case_id, "access_log.json")
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                events: List[Dict[str, Any]] = json.load(handle)
        except Exception:
            return []

        if limit is not None and limit >= 0:
            return events[-limit:]
        return events


class ProgressLogFormatter:
    """Format module log dictionaries for UI tables."""

    @staticmethod
    def from_module_logs(module_logs: Optional[Dict[str, Iterable[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        if not module_logs:
            return []

        rows: List[Dict[str, Any]] = []
        for module_name, entries in module_logs.items():
            for entry in entries or []:
                if not isinstance(entry, dict):
                    continue
                rows.append(
                    {
                        "module": module_name,
                        "attempt": entry.get("attempt"),
                        "event": entry.get("event"),
                        "timestamp": entry.get("timestamp"),
                        "details": entry.get("details"),
                    }
                )

        def _sort_key(row: Dict[str, Any]):
            ts = row.get("timestamp")
            if isinstance(ts, str):
                try:
                    return datetime.fromisoformat(ts)
                except ValueError:
                    return ts
            return ts or ""

        rows.sort(key=_sort_key)
        return rows


class ExtractionValidator:
    """Determine potential issues in extraction results."""

    @staticmethod
    def scan(results: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
        issues = {"warnings": [], "errors": []}
        if not results:
            issues["warnings"].append("No extraction results available.")
            return issues

        status = results.get("status")
        if status not in {"completed", "partial_success"}:
            issues["warnings"].append(f"Extraction status is {status}.")

        modules_run = results.get("modules_run") or []
        if not modules_run:
            issues["warnings"].append("No modules reported in results.")

        for module in modules_run:
            if module.get("status") == "error":
                issues["errors"].append(
                    f"Module {module.get('name')} failed: {module.get('error', 'unknown error')}"
                )

        for key in ("communications", "location", "media"):
            if key not in (results.get("data") or {}):
                issues["warnings"].append(f"Missing {key} section in results data.")

        return issues


class ResultsDiff:
    """Compare two extraction results dictionaries."""

    @staticmethod
    def compare(old: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if old is None:
            return {"status": "new", "changed_modules": []}
        if new is None:
            return {"status": "removed", "changed_modules": []}

        old_modules = {m.get("name"): m for m in old.get("modules_run", [])}
        new_modules = {m.get("name"): m for m in new.get("modules_run", [])}

        changed = []
        for name, new_entry in new_modules.items():
            old_entry = old_modules.get(name)
            if not old_entry:
                changed.append({"module": name, "change": "added"})
                continue
            if old_entry.get("status") != new_entry.get("status") or old_entry.get("attempts") != new_entry.get("attempts"):
                changed.append({"module": name, "change": "updated"})

        for name in old_modules:
            if name not in new_modules:
                changed.append({"module": name, "change": "removed"})

        return {"status": "diff", "changed_modules": changed}


class JobTracker:
    """Minimal in-memory tracking for long running jobs."""

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(self, job_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self._jobs[job_id] = {
                "status": "running",
                "metadata": metadata or {},
                "progress": 0,
                "history": [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "event": "started",
                        "details": metadata or {},
                    }
                ],
            }

    def update(self, job_id: str, progress: int, message: str) -> None:
        with self._lock:
            job = self._jobs.setdefault(job_id, {})
            job.setdefault("history", []).append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "event": "progress",
                    "details": message,
                }
            )
            job["progress"] = progress
            job.setdefault("status", "running")

    def complete(self, job_id: str, status: str = "completed", details: Optional[str] = None) -> None:
        with self._lock:
            job = self._jobs.setdefault(job_id, {})
            job["status"] = status
            job.setdefault("history", []).append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "event": "completed",
                    "details": details,
                }
            )

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return json.loads(json.dumps(self._jobs))


class AsyncJobRegistry(JobTracker):
    """High-level helper for registering background extraction jobs."""

    def schedule(self, job_id: str, label: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.start(job_id, metadata={**(metadata or {}), 'label': label})
        return self.get(job_id) or {}

    def mark(self, job_id: str, status: str, details: Optional[str] = None) -> None:
        if status == 'completed':
            self.complete(job_id, status='completed', details=details)
        else:
            self.complete(job_id, status='failed', details=details)


def adb_root_access_message(summary: Dict[str, Any], feature: str) -> str:
    """Generate contextual messaging when ADB-root-only data is unavailable."""
    if not summary.get('installed'):
        return 'ADB is not installed or not accessible in PATH; install platform-tools to enable device extraction.'
    devices = summary.get('devices') or []
    if not devices:
        return 'No Android devices detected via ADB. Connect a device and accept the USB debugging prompt.'
    default = summary.get('default_device')
    status = default.get('status') if isinstance(default, dict) else None
    if status != 'device':
        return 'Connected device is not authorised. Unlock the phone and accept the RSA fingerprint prompt.'
    return (
        f'{feature} requires elevated access to the device data partition. '
        'Ensure the device is rooted or use consent-backed content provider dumps as a fallback.'
    )


def parse_sms_dump(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            for line in handle:
                if 'body=' not in line:
                    continue
                rows.append(
                    {
                        'message': _extract_between(line, 'body=', ','),
                        'address': _extract_between(line, 'address=', ','),
                        'timestamp': _extract_between(line, 'date=', ','),
                    }
                )
    except Exception:
        return []
    return rows


def parse_calls_dump(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            for line in handle:
                if 'number=' not in line:
                    continue
                rows.append(
                    {
                        'number': _extract_between(line, 'number=', ','),
                        'duration': _extract_between(line, 'duration=', ','),
                        'type': _extract_between(line, 'type=', ','),
                        'timestamp': _extract_between(line, 'date=', ','),
                    }
                )
    except Exception:
        return []
    return rows


def _extract_between(text: str, start: str, end: str) -> str:
    try:
        i = text.index(start) + len(start)
        j = text.index(end, i)
        return text[i:j]
    except ValueError:
        return ''


def format_system_checks(summary: Dict[str, Any]) -> Dict[str, List[str]]:
    warnings: List[str] = []
    info: List[str] = []
    if not summary.get('installed'):
        warnings.append('ADB not installed; install Android platform-tools and ensure `adb` is on PATH.')
        return {'warnings': warnings, 'info': info}

    devices = summary.get('devices') or []
    if not devices:
        warnings.append('No Android devices detected via ADB.')
    else:
        for device in devices:
            serial = device.get('serial', 'unknown')
            status = device.get('status', 'unknown')
            if status == 'device':
                info.append(f"Device {serial} ready via ADB.")
            elif status == 'unauthorized':
                warnings.append(f"Device {serial} is unauthorized. Accept the RSA prompt on the handset.")
            else:
                warnings.append(f"Device {serial} status: {status}.")

    return {'warnings': warnings, 'info': info}


def case_selection_options(sessions: Dict[str, Any]) -> List[str]:
    return sorted(sessions.keys())


def persist_case_snapshot(consent_manager: Any, case_id: Optional[str]) -> Tuple[bool, str]:
    if not case_id:
        return False, 'No case selected.'
    if consent_manager.persist_session(case_id):
        return True, 'Consent snapshot saved.'
    return False, 'No consent session found to save.'


def render_consent_status(session: Optional[Any]) -> Dict[str, str]:
    if not session:
        return {'level': 'N/A', 'message': 'No active consent session.'}
    level = getattr(session, 'level', None)
    level_name = level.name if level else 'Unknown'
    unlock = getattr(session, 'metadata', {}).get('unlock_status') if hasattr(session, 'metadata') else None
    unlock_status = (unlock or {}).get('status', 'unverified')
    return {
        'level': level_name,
        'unlock_status': unlock_status,
        'message': f"Consent level {level_name} • Unlock {unlock_status}",
    }


def consent_otp_controls(consent_manager: Any, case_id: str, phone_number: str) -> Tuple[bool, str]:
    if not consent_manager:
        return False, 'Consent manager unavailable.'
    try:
        consent_manager.send_verification_sms(case_id, phone_number)
    except Exception as exc:
        return False, f'Failed to send OTP: {exc}'
    return True, 'OTP sent successfully.'


def render_vault_entries(entries: List[Dict[str, Any]]) -> List[str]:
    rendered = []
    for entry in entries:
        rendered.append(
            f"Vault {entry.get('vault_id')} ({entry.get('auth_type', 'PIN')}) added {entry.get('created_at')}"
        )
    return rendered


def summarize_results_diff(diff: Dict[str, Any]) -> str:
    if not diff:
        return 'No change detected.'
    status = diff.get('status', 'unknown')
    modules = diff.get('changed_modules', [])
    lines = [f'Result diff status: {status}']
    for module in modules:
        lines.append(f"• {module.get('module')}: {module.get('change')}")
    return '\n'.join(lines)


def build_report_sections(results: Dict[str, Any], selected: List[str]) -> Dict[str, Any]:
    sections: Dict[str, Any] = {}
    data = results.get('data', {}) if isinstance(results, dict) else {}
    mapping = {
        'Communications': 'communications',
        'Location': 'location',
        'Media': 'media',
        'System': 'system',
        'Security': 'security',
        'Errors': 'errors',
    }
    for label in selected:
        key = mapping.get(label)
        if key and key in data:
            sections[label] = data[key]
    if 'Summary' in selected:
        sections['Summary'] = {
            'status': results.get('status'),
            'modules_run': results.get('modules_run'),
            'duration_seconds': results.get('duration_seconds'),
        }
    return sections


def capture_diagnostics_snapshot(orchestrator: Any, consent_manager: Any, case_id: str) -> Dict[str, Any]:
    snapshot = {
        'case_id': case_id,
        'created_at': datetime.now().isoformat(),
        'consent': {},
        'system': {},
    }
    if consent_manager:
        session = consent_manager.get_session(case_id)
        snapshot['consent']['status'] = render_consent_status(session)
    if orchestrator:
        try:
            snapshot['system']['modules'] = orchestrator.get_module_status()
        except Exception:
            snapshot['system']['modules'] = {}
    return snapshot
