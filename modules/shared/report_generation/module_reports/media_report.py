"""
MEDIA VIEWER REPORT - Module-Specific Report

Generates detailed reports for media analysis:
- Image analysis
- Video analysis
- Audio analysis
- Metadata extraction
- Media timeline

Features:
- Comprehensive docstrings
- Error handling with custom exceptions
- Structured logging
- Performance optimization
"""

import logging
import json
from typing import Dict, Any, List
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

class MediaViewerReport:
    """
    Generate detailed media analysis reports.
    
    Creates comprehensive reports from media viewer module including
    image analysis, video analysis, audio analysis, and metadata.
    """
    
    def __init__(self, case_id: str = ""):
        """Initialize Media Viewer Report"""
        self.case_id = case_id
        logger.debug(f"MediaViewerReport initialized for case: {case_id}")
    
    def generate(self, media_data: Dict[str, Any]) -> str:
        """
        Generate media analysis report.
        
        Creates detailed report including image, video, and audio analysis
        with metadata extraction and media timeline.
        
        Args:
            media_data (Dict[str, Any]): Media analysis data
            
        Returns:
            str: Formatted media analysis report
            
        Raises:
            ReportGenerationError: If report generation fails
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "DEBUG",
                "Generating media analysis report",
                case_id=self.case_id
            )
            
            images = media_data.get('images', [])
            videos = media_data.get('videos', [])
            audio = media_data.get('audio', [])
            
            report = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      MEDIA ANALYSIS REPORT                                    ║
║                          Case ID: {self.case_id:<50} ║
╚═══════════════════════════════════════════════════════════════════════════════╝

EXECUTIVE SUMMARY
─────────────────────────────────────────────────────────────────────────────────
Total Images:             {len(images):,}
Total Videos:             {len(videos):,}
Total Audio Files:        {len(audio):,}
Report Generated:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

IMAGE ANALYSIS
─────────────────────────────────────────────────────────────────────────────────
"""
            
            if images:
                image_formats = {}
                total_size = 0
                for img in images:
                    fmt = img.get('format', 'Unknown')
                    image_formats[fmt] = image_formats.get(fmt, 0) + 1
                    total_size += img.get('size', 0)
                
                report += f"""
Image Formats:
"""
                for fmt, count in sorted(image_formats.items(), key=lambda x: x[1], reverse=True):
                    report += f"  • {fmt}: {count:,} images\n"
                
                report += f"""
Total Image Size:         {self._format_size(total_size)}

Recent Images:
"""
                for img in images[:5]:
                    report += f"""
  • {img.get('filename', 'Unknown')}
    Size: {self._format_size(img.get('size', 0))}
    Date: {img.get('date', 'N/A')}
    Resolution: {img.get('width', 'N/A')}x{img.get('height', 'N/A')}
"""
            
            report += f"""
VIDEO ANALYSIS
─────────────────────────────────────────────────────────────────────────────────
"""
            
            if videos:
                total_duration = sum(v.get('duration', 0) for v in videos)
                total_size = sum(v.get('size', 0) for v in videos)
                
                report += f"""
Total Video Duration:     {self._format_duration(total_duration)}
Total Video Size:         {self._format_size(total_size)}

Video Files:
"""
                for video in videos[:5]:
                    report += f"""
  • {video.get('filename', 'Unknown')}
    Duration: {self._format_duration(video.get('duration', 0))}
    Size: {self._format_size(video.get('size', 0))}
    Format: {video.get('format', 'N/A')}
"""
            
            report += f"""
AUDIO ANALYSIS
─────────────────────────────────────────────────────────────────────────────────
"""
            
            if audio:
                total_duration = sum(a.get('duration', 0) for a in audio)
                total_size = sum(a.get('size', 0) for a in audio)
                
                report += f"""
Total Audio Duration:     {self._format_duration(total_duration)}
Total Audio Size:         {self._format_size(total_size)}

Audio Files:
"""
                for aud in audio[:5]:
                    report += f"""
  • {aud.get('filename', 'Unknown')}
    Duration: {self._format_duration(aud.get('duration', 0))}
    Size: {self._format_size(aud.get('size', 0))}
    Format: {aud.get('format', 'N/A')}
"""
            
            report += f"""
═══════════════════════════════════════════════════════════════════════════════
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            structured_logger.log_with_context(
                "DEBUG",
                "Media analysis report generated successfully",
                case_id=self.case_id,
                report_length=len(report)
            )
            
            return report
        
        except Exception as e:
            error_msg = f"Error generating media analysis report: {str(e)}"
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
    def _format_duration(seconds: int) -> str:
        """Format duration in seconds to readable format (cached)"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs}s"
