"""
DEVICE INFORMATION REPORT - Module-Specific Report

Generates detailed reports for device information:
- Device specifications
- System information
- Storage analysis
- Application inventory
- Security settings

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

class DeviceInformationReport:
    """
    Generate detailed device information reports.
    
    Creates comprehensive reports from device information module including
    specifications, system info, storage, applications, and security settings.
    """
    
    def __init__(self, case_id: str = ""):
        """Initialize Device Information Report"""
        self.case_id = case_id
        logger.debug(f"DeviceInformationReport initialized for case: {case_id}")
    
    def generate(self, device_data: Dict[str, Any]) -> str:
        """
        Generate device information report.
        
        Creates detailed report including device specifications, system
        information, storage analysis, applications, and security settings.
        
        Args:
            device_data (Dict[str, Any]): Device information data
            
        Returns:
            str: Formatted device information report
            
        Raises:
            ReportGenerationError: If report generation fails
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "DEBUG",
                "Generating device information report",
                case_id=self.case_id
            )
            
            report = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                   DEVICE INFORMATION REPORT                                   ║
║                          Case ID: {self.case_id:<50} ║
╚═══════════════════════════════════════════════════════════════════════════════╝

DEVICE SPECIFICATIONS
─────────────────────────────────────────────────────────────────────────────────
Manufacturer:             {device_data.get('manufacturer', 'N/A')}
Model:                    {device_data.get('model', 'N/A')}
Device Type:              {device_data.get('device_type', 'N/A')}
Serial Number:            {device_data.get('serial_number', 'N/A')}
IMEI:                     {device_data.get('imei', 'N/A')}
IMSI:                     {device_data.get('imsi', 'N/A')}

SYSTEM INFORMATION
─────────────────────────────────────────────────────────────────────────────────
Operating System:         {device_data.get('os_name', 'N/A')}
OS Version:               {device_data.get('os_version', 'N/A')}
Build Number:             {device_data.get('build_number', 'N/A')}
Security Patch:           {device_data.get('security_patch', 'N/A')}
Bootloader:               {device_data.get('bootloader', 'N/A')}
Kernel Version:           {device_data.get('kernel_version', 'N/A')}

HARDWARE INFORMATION
─────────────────────────────────────────────────────────────────────────────────
Processor:                {device_data.get('processor', 'N/A')}
RAM:                      {self._format_size(device_data.get('ram', 0))}
Storage:                  {self._format_size(device_data.get('storage', 0))}
Display:                  {device_data.get('display', 'N/A')}
Battery:                  {device_data.get('battery', 'N/A')} mAh

STORAGE ANALYSIS
─────────────────────────────────────────────────────────────────────────────────
Total Storage:            {self._format_size(device_data.get('total_storage', 0))}
Used Storage:             {self._format_size(device_data.get('used_storage', 0))}
Free Storage:             {self._format_size(device_data.get('free_storage', 0))}
Storage Usage:            {self._calculate_percentage(device_data.get('used_storage', 0), device_data.get('total_storage', 1))}%

APPLICATION INVENTORY
─────────────────────────────────────────────────────────────────────────────────
Total Applications:       {device_data.get('total_apps', 0)}
System Applications:      {device_data.get('system_apps', 0)}
Third-party Apps:         {device_data.get('third_party_apps', 0)}
Suspicious Apps:          {device_data.get('suspicious_apps', 0)}

SECURITY SETTINGS
─────────────────────────────────────────────────────────────────────────────────
Lock Screen:              {device_data.get('lock_screen', 'N/A')}
Encryption:               {device_data.get('encryption', 'N/A')}
Developer Mode:           {device_data.get('developer_mode', 'N/A')}
USB Debugging:            {device_data.get('usb_debugging', 'N/A')}
Unknown Sources:          {device_data.get('unknown_sources', 'N/A')}

NETWORK INFORMATION
─────────────────────────────────────────────────────────────────────────────────
WiFi MAC Address:         {device_data.get('wifi_mac', 'N/A')}
Bluetooth MAC:            {device_data.get('bluetooth_mac', 'N/A')}
Mobile Network:           {device_data.get('mobile_network', 'N/A')}
SIM Card:                 {device_data.get('sim_card', 'N/A')}

═══════════════════════════════════════════════════════════════════════════════
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            structured_logger.log_with_context(
                "DEBUG",
                "Device information report generated successfully",
                case_id=self.case_id,
                report_length=len(report)
            )
            
            return report
        
        except Exception as e:
            error_msg = f"Error generating device information report: {str(e)}"
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
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _calculate_percentage(used: int, total: int) -> float:
        """Calculate percentage (cached)"""
        if total == 0:
            return 0.0
        return (used / total) * 100
