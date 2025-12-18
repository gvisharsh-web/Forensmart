"""
VALIDATORS MODULE - Input Validation for Forensmart
Provides validation functions for common data types and formats

This module provides:
- File path validation
- Coordinate validation
- Timestamp validation
- Device ID validation
- Media extension validation
"""

import os
import logging
from datetime import datetime
from typing import Any, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# FILE PATH VALIDATION
# ============================================================================

def validate_file_path(path: Any) -> bool:
    """
    Validate file path
    
    Args:
        path: Path to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if not isinstance(path, str):
            logger.warning(f"⚠️ Invalid path type: {type(path).__name__}")
            return False
        
        if not path or len(path) == 0:
            logger.warning("⚠️ Empty path")
            return False
        
        if len(path) > 260:  # Windows MAX_PATH
            logger.warning(f"⚠️ Path too long: {len(path)} characters")
            return False
        
        # Check for invalid characters
        invalid_chars = ['<', '>', '|', '\0']
        for char in invalid_chars:
            if char in path:
                logger.warning(f"⚠️ Invalid character in path: {char}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"❌ Error validating path: {e}")
        return False


# ============================================================================
# COORDINATE VALIDATION
# ============================================================================

def validate_coordinates(latitude: Any, longitude: Any) -> bool:
    """
    Validate GPS coordinates
    
    Args:
        latitude: Latitude value
        longitude: Longitude value
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Convert to float
        try:
            lat = float(latitude)
            lon = float(longitude)
        except (ValueError, TypeError) as e:
            logger.warning(f"⚠️ Invalid coordinate type: {type(latitude).__name__}, {type(longitude).__name__}")
            return False
        
        # Check ranges
        if not (-90 <= lat <= 90):
            logger.warning(f"⚠️ Latitude out of range: {lat} (must be -90 to 90)")
            return False
        
        if not (-180 <= lon <= 180):
            logger.warning(f"⚠️ Longitude out of range: {lon} (must be -180 to 180)")
            return False
        
        # Check for NaN or Infinity
        if lat != lat or lon != lon:  # NaN check
            logger.warning("⚠️ Coordinates contain NaN")
            return False
        
        if lat == float('inf') or lon == float('inf'):
            logger.warning("⚠️ Coordinates contain Infinity")
            return False
        
        return True
    except Exception as e:
        logger.error(f"❌ Error validating coordinates: {e}")
        return False


# ============================================================================
# TIMESTAMP VALIDATION
# ============================================================================

def validate_timestamp(timestamp: Any) -> bool:
    """
    Validate ISO format timestamp
    
    Args:
        timestamp: Timestamp to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if not isinstance(timestamp, str):
            logger.warning(f"⚠️ Invalid timestamp type: {type(timestamp).__name__}")
            return False
        
        if not timestamp or len(timestamp) == 0:
            logger.warning("⚠️ Empty timestamp")
            return False
        
        # Try to parse as ISO format
        try:
            datetime.fromisoformat(timestamp)
            return True
        except ValueError:
            logger.warning(f"⚠️ Invalid timestamp format: {timestamp}")
            return False
    except Exception as e:
        logger.error(f"❌ Error validating timestamp: {e}")
        return False


# ============================================================================
# DEVICE ID VALIDATION
# ============================================================================

def validate_device_id(device_id: Any) -> bool:
    """
    Validate device ID
    
    Args:
        device_id: Device ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if not isinstance(device_id, str):
            logger.warning(f"⚠️ Invalid device ID type: {type(device_id).__name__}")
            return False
        
        if not device_id or len(device_id) == 0:
            logger.warning("⚠️ Empty device ID")
            return False
        
        if len(device_id) > 100:
            logger.warning(f"⚠️ Device ID too long: {len(device_id)} characters")
            return False
        
        # Device IDs should be alphanumeric with some special chars
        valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_:.')
        for char in device_id:
            if char not in valid_chars:
                logger.warning(f"⚠️ Invalid character in device ID: {char}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"❌ Error validating device ID: {e}")
        return False


# ============================================================================
# MEDIA EXTENSION VALIDATION
# ============================================================================

def validate_media_extension(filename: Any) -> bool:
    """
    Validate media file extension
    
    Args:
        filename: Filename to validate
        
    Returns:
        True if valid media extension, False otherwise
    """
    try:
        if not isinstance(filename, str):
            logger.warning(f"⚠️ Invalid filename type: {type(filename).__name__}")
            return False
        
        if not filename or len(filename) == 0:
            logger.warning("⚠️ Empty filename")
            return False
        
        # Get extension
        if '.' not in filename:
            logger.warning(f"⚠️ No extension in filename: {filename}")
            return False
        
        ext = filename.split('.')[-1].lower()
        
        # Valid media extensions
        valid_extensions = {
            # Images
            'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff', 'ico', 'svg',
            # Videos
            'mp4', 'avi', 'mkv', 'mov', 'flv', 'wmv', 'webm', 'm4v', '3gp',
            # Audio
            'mp3', 'wav', 'aac', 'm4a', 'flac', 'ogg', 'wma', 'opus', 'aiff'
        }
        
        if ext not in valid_extensions:
            logger.debug(f"⚠️ Unknown media extension: {ext}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"❌ Error validating media extension: {e}")
        return False


# ============================================================================
# BATCH VALIDATION
# ============================================================================

def validate_media_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate a complete media file
    
    Args:
        file_path: Path to media file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Validate path
        if not validate_file_path(file_path):
            return False, "Invalid file path"
        
        # Validate extension
        if not validate_media_extension(file_path):
            return False, "Invalid media extension"
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ File does not exist: {file_path}")
            return False, "File does not exist"
        
        # Check if file is readable
        if not os.access(file_path, os.R_OK):
            logger.warning(f"⚠️ File is not readable: {file_path}")
            return False, "File is not readable"
        
        return True, ""
    except Exception as e:
        logger.error(f"❌ Error validating media file: {e}")
        return False, str(e)


# ============================================================================
# LOCATION VALIDATION
# ============================================================================

def validate_location(location: Any) -> Tuple[bool, str]:
    """
    Validate a location dictionary
    
    Args:
        location: Location dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not isinstance(location, dict):
            return False, "Location must be a dictionary"
        
        # Check required fields
        required_fields = ['latitude', 'longitude']
        for field in required_fields:
            if field not in location:
                return False, f"Missing required field: {field}"
        
        # Validate coordinates
        if not validate_coordinates(location['latitude'], location['longitude']):
            return False, "Invalid coordinates"
        
        # Validate optional timestamp
        if 'timestamp' in location:
            if not validate_timestamp(location['timestamp']):
                return False, "Invalid timestamp"
        
        return True, ""
    except Exception as e:
        logger.error(f"❌ Error validating location: {e}")
        return False, str(e)


# ============================================================================
# EXTRACTION DATA VALIDATION
# ============================================================================

def validate_extraction_data(data: Any) -> Tuple[bool, str]:
    """
    Validate extraction data structure
    
    Args:
        data: Extraction data to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not isinstance(data, dict):
            return False, "Data must be a dictionary"
        
        # Check for required keys
        if 'device_id' in data:
            if not validate_device_id(data['device_id']):
                return False, "Invalid device ID"
        
        # Validate media files if present
        if 'media_files' in data:
            if not isinstance(data['media_files'], list):
                return False, "Media files must be a list"
            
            for media in data['media_files']:
                if not isinstance(media, dict):
                    return False, "Each media file must be a dictionary"
                
                if 'path' not in media:
                    return False, "Media file missing path"
                
                if not validate_file_path(media['path']):
                    return False, f"Invalid media path: {media['path']}"
        
        return True, ""
    except Exception as e:
        logger.error(f"❌ Error validating extraction data: {e}")
        return False, str(e)


# ============================================================================
# SUMMARY
# ============================================================================

"""
Available Validation Functions:
- validate_file_path(path) -> bool
- validate_coordinates(lat, lon) -> bool
- validate_timestamp(timestamp) -> bool
- validate_device_id(device_id) -> bool
- validate_media_extension(filename) -> bool
- validate_media_file(file_path) -> (bool, str)
- validate_location(location) -> (bool, str)
- validate_extraction_data(data) -> (bool, str)

All functions log warnings/errors appropriately.
"""
