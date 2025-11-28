"""
CLOUD ANALYSIS REPORT - Module-Specific Report

Generates detailed reports for cloud analysis:
- Cloud accounts
- Sync status
- Cloud storage
- Account activity
- Security analysis

Features:
- Comprehensive docstrings
- Error handling with custom exceptions
- Structured logging
- Performance optimization
"""

import logging
import json
from typing import Dict, Any
from datetime import datetime
from functools import lru_cache

logger = logging.getLogger(__name__)

class ModuleReportException(Exception):
    """Base exception for module report errors"""
    pass

class ReportGenerationError(ModuleReportException):
    """Raised when report generation fails"""
    pass

class StructuredLogger:
    """Structured logging with JSON context"""
    
    @staticmethod
    def log_with_context(level: str, message: str, **context) -> None:
        """Log with context information"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message,
                'context': context
            }
            log_level = getattr(logging, level.upper(), logging.INFO)
            logger.log(log_level, json.dumps(log_entry))
        except Exception as e:
            logger.error(f"Error in structured logging: {str(e)}")

class CloudAnalysisReport:
    """
    Generate detailed cloud analysis reports.
    
    Creates comprehensive reports from cloud analysis module including
    cloud accounts, sync status, storage, activity, and security.
    """
    
    def __init__(self, case_id: str = ""):
        """Initialize Cloud Analysis Report"""
        self.case_id = case_id
        logger.debug(f"CloudAnalysisReport initialized for case: {case_id}")
    
    def generate(self, cloud_data: Dict[str, Any]) -> str:
        """
        Generate cloud analysis report.
        
        Creates detailed report including cloud accounts, sync status,
        storage analysis, account activity, and security settings.
        
        Args:
            cloud_data (Dict[str, Any]): Cloud analysis data
            
        Returns:
            str: Formatted cloud analysis report
            
        Raises:
            ReportGenerationError: If report generation fails
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "DEBUG",
                "Generating cloud analysis report",
                case_id=self.case_id
            )
            
            accounts = cloud_data.get('accounts', [])
            
            report = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     CLOUD ANALYSIS REPORT                                     ║
║                          Case ID: {self.case_id:<50} ║
╚═══════════════════════════════════════════════════════════════════════════════╝

EXECUTIVE SUMMARY
─────────────────────────────────────────────────────────────────────────────────
Total Cloud Accounts:     {len(accounts)}
Report Generated:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CLOUD ACCOUNTS
─────────────────────────────────────────────────────────────────────────────────
"""
            
            for account in accounts:
                report += f"""
Account: {account.get('provider', 'Unknown')}
  Email: {account.get('email', 'N/A')}
  Status: {account.get('status', 'N/A')}
  Sync Enabled: {account.get('sync_enabled', 'N/A')}
  Last Sync: {account.get('last_sync', 'N/A')}
  Storage Used: {self._format_size(account.get('storage_used', 0))}
  Storage Total: {self._format_size(account.get('storage_total', 0))}
  Files: {account.get('file_count', 0):,}
  Folders: {account.get('folder_count', 0):,}
"""
            
            report += f"""
SYNC STATUS
─────────────────────────────────────────────────────────────────────────────────
Sync Enabled Accounts:    {len([a for a in accounts if a.get('sync_enabled')])}
Sync Disabled Accounts:   {len([a for a in accounts if not a.get('sync_enabled')])}
Last Sync Time:           {cloud_data.get('last_sync_time', 'N/A')}

STORAGE ANALYSIS
─────────────────────────────────────────────────────────────────────────────────
Total Cloud Storage:      {self._format_size(sum(a.get('storage_total', 0) for a in accounts))}
Total Used Storage:       {self._format_size(sum(a.get('storage_used', 0) for a in accounts))}
Total Free Storage:       {self._format_size(sum(a.get('storage_total', 0) - a.get('storage_used', 0) for a in accounts))}

ACCOUNT ACTIVITY
─────────────────────────────────────────────────────────────────────────────────
Recent Uploads:           {cloud_data.get('recent_uploads', 0)}
Recent Downloads:         {cloud_data.get('recent_downloads', 0)}
Recent Modifications:     {cloud_data.get('recent_modifications', 0)}
Shared Items:             {cloud_data.get('shared_items', 0)}

SECURITY ANALYSIS
─────────────────────────────────────────────────────────────────────────────────
Two-Factor Auth:          {cloud_data.get('two_factor_auth', 'N/A')}
Last Password Change:     {cloud_data.get('last_password_change', 'N/A')}
Active Sessions:          {cloud_data.get('active_sessions', 0)}
Suspicious Activities:    {cloud_data.get('suspicious_activities', 0)}

═══════════════════════════════════════════════════════════════════════════════
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            structured_logger.log_with_context(
                "DEBUG",
                "Cloud analysis report generated successfully",
                case_id=self.case_id,
                report_length=len(report)
            )
            
            return report
        
        except Exception as e:
            error_msg = f"Error generating cloud analysis report: {str(e)}"
            logger.error(error_msg)
            raise ReportGenerationError(error_msg) from e
    
    @staticmethod
    @lru_cache(maxsize=256)
    def _format_size(bytes_size: int) -> str:
        """Format bytes to human-readable size (cached)"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.2f} PB"
