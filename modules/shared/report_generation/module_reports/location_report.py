"""
LOCATION INTELLIGENCE REPORT - Module-Specific Report

Generates detailed reports for location intelligence analysis:
- GPS coordinates
- Location timeline
- Geofencing analysis
- Movement patterns
- Visited locations

Features:
- Comprehensive docstrings
- Error handling with custom exceptions
- Structured logging
- Performance optimization
"""

import logging
import json
from typing import Dict, Any, List, Tuple
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

class LocationIntelligenceReport:
    """
    Generate detailed location intelligence reports.
    
    Creates comprehensive reports from location intelligence module
    including GPS data, timelines, geofencing, and movement patterns.
    """
    
    def __init__(self, case_id: str = ""):
        """
        Initialize Location Intelligence Report.
        
        Args:
            case_id (str): Case ID for logging
        """
        self.case_id = case_id
        logger.debug(f"LocationIntelligenceReport initialized for case: {case_id}")
    
    def generate(self, location_data: Dict[str, Any]) -> str:
        """
        Generate location intelligence report.
        
        Creates detailed report including GPS coordinates, location timeline,
        geofencing analysis, movement patterns, and visited locations.
        
        Args:
            location_data (Dict[str, Any]): Location intelligence data
            
        Returns:
            str: Formatted location intelligence report
            
        Raises:
            ReportGenerationError: If report generation fails
            
        Example:
            >>> report_gen = LocationIntelligenceReport("CASE-001")
            >>> location_data = {
            ...     'gps_points': [...],
            ...     'locations': [...],
            ...     'timeline': [...]
            ... }
            >>> report = report_gen.generate(location_data)
            >>> len(report) > 1000
            True
        """
        try:
            structured_logger = StructuredLogger()
            structured_logger.log_with_context(
                "DEBUG",
                "Generating location intelligence report",
                case_id=self.case_id
            )
            
            # Extract data
            gps_points = location_data.get('gps_points', [])
            locations = location_data.get('locations', [])
            timeline = location_data.get('timeline', [])
            
            # Generate report
            report = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                  LOCATION INTELLIGENCE REPORT                                 ║
║                          Case ID: {self.case_id:<50} ║
╚═══════════════════════════════════════════════════════════════════════════════╝

EXECUTIVE SUMMARY
─────────────────────────────────────────────────────────────────────────────────
Total GPS Points:         {len(gps_points):,}
Unique Locations:         {len(locations):,}
Timeline Entries:         {len(timeline):,}
Report Generated:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

GPS ANALYSIS
─────────────────────────────────────────────────────────────────────────────────
"""
            
            if gps_points:
                # Calculate coverage area
                lats = [p.get('latitude', 0) for p in gps_points]
                lons = [p.get('longitude', 0) for p in gps_points]
                
                min_lat, max_lat = min(lats), max(lats)
                min_lon, max_lon = min(lons), max(lons)
                
                report += f"""
Latitude Range:           {min_lat:.6f}° to {max_lat:.6f}°
Longitude Range:          {min_lon:.6f}° to {max_lon:.6f}°
Coverage Area:            {self._calculate_area(min_lat, max_lat, min_lon, max_lon):.2f} km²

GPS Point Accuracy:
"""
                accuracy_levels = {}
                for point in gps_points:
                    acc = point.get('accuracy', 'Unknown')
                    accuracy_levels[acc] = accuracy_levels.get(acc, 0) + 1
                
                for acc, count in sorted(accuracy_levels.items(), key=lambda x: x[1], reverse=True):
                    report += f"  • {acc}: {count:,} points\n"
            
            report += f"""
LOCATION ANALYSIS
─────────────────────────────────────────────────────────────────────────────────
"""
            
            if locations:
                report += f"""
Visited Locations:
"""
                for i, location in enumerate(locations[:15], 1):
                    visit_count = location.get('visit_count', 1)
                    duration = location.get('duration', 0)
                    report += f"""
  {i}. {location.get('name', 'Unknown Location')}
     Coordinates: {location.get('latitude', 'N/A')}, {location.get('longitude', 'N/A')}
     Visits: {visit_count}
     Total Duration: {self._format_duration(duration)}
     First Visit: {location.get('first_visit', 'N/A')}
     Last Visit: {location.get('last_visit', 'N/A')}
"""
            
            report += f"""
MOVEMENT PATTERNS
─────────────────────────────────────────────────────────────────────────────────
"""
            
            if timeline:
                report += f"""
Timeline Analysis:
"""
                for i, entry in enumerate(timeline[:10], 1):
                    report += f"""
  {i}. {entry.get('timestamp', 'N/A')}
     Location: {entry.get('location', 'Unknown')}
     Coordinates: {entry.get('latitude', 'N/A')}, {entry.get('longitude', 'N/A')}
     Activity: {entry.get('activity', 'N/A')}
"""
            
            report += f"""
GEOFENCING ANALYSIS
─────────────────────────────────────────────────────────────────────────────────
Home Location:            {location_data.get('home_location', 'Not determined')}
Work Location:            {location_data.get('work_location', 'Not determined')}
Frequent Locations:       {location_data.get('frequent_locations_count', 0)}
Anomalous Locations:      {location_data.get('anomalous_locations_count', 0)}

═══════════════════════════════════════════════════════════════════════════════
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            structured_logger.log_with_context(
                "DEBUG",
                "Location intelligence report generated successfully",
                case_id=self.case_id,
                report_length=len(report)
            )
            
            return report
        
        except Exception as e:
            error_msg = f"Error generating location intelligence report: {str(e)}"
            logger.error(error_msg)
            raise ReportGenerationError(error_msg) from e
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _format_duration(seconds: int) -> str:
        """Format duration in seconds to readable format (cached)"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs}s"
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _calculate_area(min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> float:
        """Calculate coverage area in km² (cached)"""
        # Simplified calculation
        lat_diff = max_lat - min_lat
        lon_diff = max_lon - min_lon
        # Approximate km per degree at equator
        km_per_degree_lat = 111.32
        km_per_degree_lon = 111.32 * 0.9  # Approximate
        return (lat_diff * km_per_degree_lat) * (lon_diff * km_per_degree_lon)
