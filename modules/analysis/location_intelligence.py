"""
LOCATION INTELLIGENCE - Location Analysis Module
Analyzes location data and identifies movement patterns, anomalies, and risks
Integrated with fraud database for location-based threat detection

This module provides:
- Timeline visualization
- Geofencing detection
- Frequent locations
- Travel patterns
- Anomaly detection
- Heatmap generation
- Distance analysis
- Risk assessment
- Comparison with baseline
- Reporting & export
- Database integration
"""

import logging
import math
import re
import urllib.parse
import requests
import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict, Counter

from modules.analysis.models import DatabaseManager, updater
from modules.shared.utils import ArtifactPathBuilder, ResultsRepository

logger = logging.getLogger(__name__)

# ============================================================================
# OFFLINE QUEUE MANAGER
# ============================================================================

class OfflineQueueManager:
    """Manage offline operations and sync"""
    
    def __init__(self, queue_file: str = "offline_gps_queue.json"):
        """Initialize offline queue manager"""
        self.queue_file = queue_file
        self.queue = self._load_queue()
        self.is_online = self._check_connectivity()
    
    def _load_queue(self) -> List[Dict[str, Any]]:
        """Load queue from file"""
        try:
            if os.path.exists(self.queue_file):
                with open(self.queue_file, 'r') as f:
                    queue = json.load(f)
                    logger.info(f"✅ Loaded {len(queue)} pending operations from queue")
                    return queue
            return []
        except Exception as e:
            logger.warning(f"⚠️ Could not load queue: {e}")
            return []
    
    def _save_queue(self):
        """Save queue to file"""
        try:
            with open(self.queue_file, 'w') as f:
                json.dump(self.queue, f, indent=2)
            logger.debug(f"💾 Queue saved: {len(self.queue)} operations")
        except Exception as e:
            logger.error(f"❌ Could not save queue: {e}")
    
    def _check_connectivity(self) -> bool:
        """Check if online by pinging Google"""
        try:
            response = requests.head(
                "https://www.google.com",
                timeout=3
            )
            return response.status_code == 200
        except:
            return False
    
    def is_connected(self) -> bool:
        """Check current connectivity status"""
        self.is_online = self._check_connectivity()
        return self.is_online
    
    def add_to_queue(self, operation: Dict[str, Any]) -> bool:
        """Add operation to offline queue"""
        try:
            operation["queued_at"] = datetime.utcnow().isoformat()
            operation["status"] = "pending"
            self.queue.append(operation)
            self._save_queue()
            logger.info(f"📋 Operation queued: {operation.get('type', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"❌ Could not queue operation: {e}")
            return False
    
    def get_pending_operations(self) -> List[Dict[str, Any]]:
        """Get all pending operations"""
        return [op for op in self.queue if op.get("status") == "pending"]
    
    def mark_synced(self, operation_index: int):
        """Mark operation as synced"""
        try:
            if 0 <= operation_index < len(self.queue):
                self.queue[operation_index]["status"] = "synced"
                self.queue[operation_index]["synced_at"] = datetime.utcnow().isoformat()
                self._save_queue()
                logger.info(f"✅ Operation marked as synced")
        except Exception as e:
            logger.error(f"❌ Could not mark operation as synced: {e}")
    
    def clear_synced(self):
        """Remove synced operations from queue"""
        try:
            original_count = len(self.queue)
            self.queue = [op for op in self.queue if op.get("status") != "synced"]
            removed = original_count - len(self.queue)
            self._save_queue()
            logger.info(f"🗑️ Removed {removed} synced operations from queue")
        except Exception as e:
            logger.error(f"❌ Could not clear synced operations: {e}")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        pending = len(self.get_pending_operations())
        synced = len([op for op in self.queue if op.get("status") == "synced"])
        return {
            "total_operations": len(self.queue),
            "pending": pending,
            "synced": synced,
            "is_online": self.is_online
        }

# ============================================================================
# MAP DISPLAY MANAGER
# ============================================================================

class MapDisplayManager:
    """Manage map display with Google Maps and Folium fallback"""
    
    def __init__(self):
        """Initialize map manager"""
        self.google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY', '')
        self.api_call_count = 0
        self.api_quota_limit = 25000  # Monthly free tier
        self.fallback_mode = False
        
        try:
            import folium
            self.folium_available = True
        except ImportError:
            self.folium_available = False
            logger.warning("⚠️ Folium not available - install with: pip install folium")
        
        try:
            import googlemaps
            self.googlemaps_available = True if self.google_maps_api_key else False
        except ImportError:
            self.googlemaps_available = False
            logger.warning("⚠️ Google Maps not available - install with: pip install googlemaps")
    
    def check_api_quota(self) -> bool:
        """Check if API quota exceeded"""
        if self.api_call_count >= self.api_quota_limit:
            logger.warning(f"⚠️ API quota exceeded: {self.api_call_count}/{self.api_quota_limit}")
            self.fallback_mode = True
            return False
        return True
    
    def increment_api_call(self):
        """Increment API call counter"""
        self.api_call_count += 1
        if self.api_call_count % 1000 == 0:
            logger.info(f"📊 API calls: {self.api_call_count}/{self.api_quota_limit}")
    
    def get_google_maps_embed_url(self, latitude: float, longitude: float, 
                                  zoom: int = 15) -> str:
        """Get Google Maps embed URL"""
        return f"https://www.google.com/maps/embed/v1/place?q={latitude},{longitude}&key={self.google_maps_api_key}&zoom={zoom}"
    
    def create_folium_map(self, latitude: float, longitude: float, 
                         zoom: int = 15, location_name: str = "Location") -> Any:
        """Create Folium map"""
        try:
            import folium
            
            # Create map centered on coordinates
            map_obj = folium.Map(
                location=[latitude, longitude],
                zoom_start=zoom,
                tiles='OpenStreetMap'
            )
            
            # Add marker
            folium.Marker(
                location=[latitude, longitude],
                popup=location_name,
                tooltip=location_name,
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(map_obj)
            
            logger.info(f"✅ Folium map created: {latitude}, {longitude}")
            return map_obj
        except Exception as e:
            logger.error(f"Error creating Folium map: {e}")
            return None
    
    def get_map_display_info(self, latitude: float, longitude: float, 
                            location_name: str = "Location",
                            map_type: str = "auto") -> Dict[str, Any]:
        """Get map display information with fallback logic"""
        try:
            # Check quota
            if not self.check_api_quota():
                self.fallback_mode = True
                map_type = "folium"
            
            # Determine which map to use
            if map_type == "auto":
                if self.googlemaps_available and not self.fallback_mode:
                    map_type = "google_maps"
                elif self.folium_available:
                    map_type = "folium"
                else:
                    map_type = "coordinates_only"
            
            result = {
                "latitude": latitude,
                "longitude": longitude,
                "location_name": location_name,
                "map_type": map_type,
                "fallback_active": self.fallback_mode,
                "api_calls": self.api_call_count,
                "api_quota": self.api_quota_limit
            }
            
            # Add map-specific data
            if map_type == "google_maps":
                if self.googlemaps_available and self.check_api_quota():
                    result["embed_url"] = self.get_google_maps_embed_url(latitude, longitude)
                    self.increment_api_call()
                    result["status"] = "success"
                    logger.info(f"✅ Google Maps embed URL generated")
                else:
                    result["status"] = "fallback"
                    result["map_type"] = "folium"
            
            elif map_type == "folium":
                if self.folium_available:
                    result["map_object"] = self.create_folium_map(latitude, longitude, location_name=location_name)
                    result["status"] = "success"
                    logger.info(f"✅ Folium map created")
                else:
                    result["status"] = "error"
                    result["message"] = "Folium not available"
            
            else:  # coordinates_only
                result["status"] = "success"
                result["message"] = "Coordinates only (no map display)"
            
            return result
        except Exception as e:
            logger.error(f"Error getting map display info: {e}")
            return {
                "status": "error",
                "message": str(e),
                "latitude": latitude,
                "longitude": longitude
            }
    
    def get_available_map_types(self) -> List[str]:
        """Get available map types"""
        types = []
        
        if self.googlemaps_available and not self.fallback_mode:
            types.append("google_maps")
        
        if self.folium_available:
            types.append("folium")
        
        types.append("coordinates_only")
        
        return types

# ============================================================================
# LOCATION INTELLIGENCE CLASS
# ============================================================================

class LocationIntelligence:
    """Location Intelligence - Analyze locations and movement patterns"""
    
    def __init__(self):
        """Initialize location analyzer with database connection"""
        self.db = DatabaseManager()
        self.updater = updater
        self.earth_radius_km = 6371  # Earth radius in kilometers
        self.map_manager = MapDisplayManager()  # Initialize map manager
        self.offline_queue = OfflineQueueManager()  # Initialize offline queue
    
    # ========================================================================
    # URL EXPANSION (For shortened URLs)
    # ========================================================================
    
    def expand_shortened_url(self, short_url: str, timeout: int = 5) -> Optional[str]:
        """Expand shortened URL to get full URL"""
        try:
            # List of shortened URL services
            shortened_services = ['goo.gl', 'bit.ly', 'tinyurl.com', 'ow.ly', 'maps.app.goo.gl']
            
            if not any(service in short_url for service in shortened_services):
                return short_url  # Not a shortened URL
            
            logger.info(f"🔗 Expanding shortened URL: {short_url}")
            
            # Use requests to follow redirects
            response = requests.head(
                short_url,
                allow_redirects=True,
                timeout=timeout,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            
            expanded_url = response.url
            logger.info(f"✅ Expanded URL: {expanded_url}")
            return expanded_url
        except requests.exceptions.Timeout:
            logger.warning(f"⚠️ URL expansion timeout: {short_url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ URL expansion failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error expanding URL: {e}")
            return None
    
    # ========================================================================
    # GPS LINK PARSER
    # ========================================================================
    
    def parse_google_maps_link(self, url: str) -> Optional[Dict[str, Any]]:
        """Parse Google Maps link and extract coordinates"""
        try:
            # Check if it's a shortened URL
            if 'goo.gl/maps' in url or 'maps.app.goo.gl' in url or any(
                service in url for service in ['bit.ly', 'tinyurl.com', 'ow.ly']
            ):
                logger.info(f"🔗 Detected shortened URL: {url}")
                expanded = self.expand_shortened_url(url)
                
                if expanded:
                    logger.info(f"✅ Expanded to: {expanded}")
                    url = expanded  # Use expanded URL
                else:
                    logger.warning(f"⚠️ Could not expand shortened URL: {url}")
                    return {"error": "Could not expand shortened URL", "url": url}
            
            # Pattern 1: https://maps.google.com/?q=40.7128,-74.0060
            pattern1 = r'maps\.google\.com.*[?&]q=(-?\d+\.?\d*),(-?\d+\.?\d*)'
            match = re.search(pattern1, url)
            if match:
                lat, lon = float(match.group(1)), float(match.group(2))
                logger.info(f"✅ Parsed Google Maps link: {lat}, {lon}")
                return {"latitude": lat, "longitude": lon, "source": "google_maps"}
            
            # Pattern 2: https://www.google.com/maps/place/40.7128,-74.0060
            pattern2 = r'google\.com/maps/place/(-?\d+\.?\d*),(-?\d+\.?\d*)'
            match = re.search(pattern2, url)
            if match:
                lat, lon = float(match.group(1)), float(match.group(2))
                logger.info(f"✅ Parsed Google Maps place link: {lat}, {lon}")
                return {"latitude": lat, "longitude": lon, "source": "google_maps_place"}
            
            # Pattern 3: https://www.google.com/maps/@40.7128,-74.0060,15z
            pattern3 = r'google\.com/maps/@(-?\d+\.?\d*),(-?\d+\.?\d*)'
            match = re.search(pattern3, url)
            if match:
                lat, lon = float(match.group(1)), float(match.group(2))
                logger.info(f"✅ Parsed Google Maps @ link: {lat}, {lon}")
                return {"latitude": lat, "longitude": lon, "source": "google_maps_at"}
            
            return None
        except Exception as e:
            logger.error(f"Error parsing Google Maps link: {e}")
            return None
    
    def parse_whatsapp_location_link(self, url: str) -> Optional[Dict[str, Any]]:
        """Parse WhatsApp location link and extract coordinates"""
        try:
            # WhatsApp location links are typically Google Maps links
            # Pattern: https://maps.google.com/?q=40.7128,-74.0060
            result = self.parse_google_maps_link(url)
            if result:
                result["source"] = "whatsapp_location"
                logger.info(f"✅ Parsed WhatsApp location link: {result}")
                return result
            
            return None
        except Exception as e:
            logger.error(f"Error parsing WhatsApp location link: {e}")
            return None
    
    def parse_gps_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Parse generic GPS URL and extract coordinates"""
        try:
            # Pattern: geo:40.7128,-74.0060
            pattern = r'geo:(-?\d+\.?\d*),(-?\d+\.?\d*)'
            match = re.search(pattern, url)
            if match:
                lat, lon = float(match.group(1)), float(match.group(2))
                logger.info(f"✅ Parsed GPS URL: {lat}, {lon}")
                return {"latitude": lat, "longitude": lon, "source": "geo_url"}
            
            return None
        except Exception as e:
            logger.error(f"Error parsing GPS URL: {e}")
            return None
    
    def parse_location_link(self, link: str) -> Optional[Dict[str, Any]]:
        """Parse any location link (Google Maps, WhatsApp, GPS URL, etc.)"""
        try:
            if not link:
                return None
            
            link = link.strip()
            
            # Try Google Maps
            result = self.parse_google_maps_link(link)
            if result:
                return result
            
            # Try WhatsApp
            result = self.parse_whatsapp_location_link(link)
            if result:
                return result
            
            # Try GPS URL
            result = self.parse_gps_url(link)
            if result:
                return result
            
            logger.warning(f"Could not parse location link: {link}")
            return None
        except Exception as e:
            logger.error(f"Error parsing location link: {e}")
            return None
    
    # ========================================================================
    # COORDINATE INPUT
    # ========================================================================
    
    def validate_coordinates(self, latitude: float, longitude: float) -> bool:
        """Validate latitude and longitude"""
        try:
            lat = float(latitude)
            lon = float(longitude)
            
            # Latitude: -90 to 90
            if not (-90 <= lat <= 90):
                logger.warning(f"Invalid latitude: {lat}")
                return False
            
            # Longitude: -180 to 180
            if not (-180 <= lon <= 180):
                logger.warning(f"Invalid longitude: {lon}")
                return False
            
            return True
        except (ValueError, TypeError):
            logger.warning(f"Invalid coordinate format: {latitude}, {longitude}")
            return False
    
    def add_location_from_coordinates(self, latitude: float, longitude: float, 
                                     name: str = None, location_type: str = "USER_INPUT") -> Dict[str, Any]:
        """Add location from coordinate input"""
        try:
            if not self.validate_coordinates(latitude, longitude):
                return {"status": "error", "message": "Invalid coordinates"}
            
            lat = float(latitude)
            lon = float(longitude)
            
            location = {
                "latitude": lat,
                "longitude": lon,
                "name": name or f"Location ({lat}, {lon})",
                "type": location_type,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "user_input"
            }
            
            logger.info(f"✅ Location added from coordinates: {lat}, {lon}")
            return {"status": "success", "location": location}
        except Exception as e:
            logger.error(f"Error adding location from coordinates: {e}")
            return {"status": "error", "message": str(e)}
    
    def add_location_from_link(self, link: str, name: str = None, case_id: str = None,
                              added_by: str = None, notes: str = None) -> Dict[str, Any]:
        """Add location from GPS link and track in database (with offline support)"""
        try:
            parsed = self.parse_location_link(link)
            
            if not parsed:
                return {"status": "error", "message": "Could not parse location link"}
            
            if "error" in parsed:
                return {"status": "error", "message": parsed["error"]}
            
            lat = parsed.get("latitude")
            lon = parsed.get("longitude")
            source = parsed.get("source", "unknown")
            
            location = {
                "latitude": lat,
                "longitude": lon,
                "name": name or f"Location from {source}",
                "type": "GPS_LINK",
                "timestamp": datetime.utcnow().isoformat(),
                "source": source,
                "original_link": link
            }
            
            # Track in database if case_id provided
            if case_id:
                # Check if online
                if self.offline_queue.is_connected():
                    # Online - add directly to database
                    try:
                        gps_log = self.db.add_gps_link(
                            case_id=case_id,
                            link=link,
                            source=source,
                            latitude=lat,
                            longitude=lon,
                            location_name=name,
                            added_by=added_by,
                            notes=notes
                        )
                        location["db_id"] = gps_log.id
                        location["sync_status"] = "synced"
                        logger.info(f"✅ GPS link tracked in database: ID={gps_log.id}")
                    except Exception as db_error:
                        logger.warning(f"⚠️ Could not track in database: {db_error}")
                        # Queue for later sync
                        self._queue_gps_link_operation(case_id, link, source, lat, lon, name, added_by, notes)
                        location["sync_status"] = "queued"
                else:
                    # Offline - queue for later sync
                    logger.info(f"📡 Offline detected - queuing GPS link operation")
                    self._queue_gps_link_operation(case_id, link, source, lat, lon, name, added_by, notes)
                    location["sync_status"] = "queued"
            
            logger.info(f"✅ Location added from link: {lat}, {lon} (source: {source})")
            return {"status": "success", "location": location}
        except Exception as e:
            logger.error(f"Error adding location from link: {e}")
            return {"status": "error", "message": str(e)}
    
    def add_locations_from_csv(self, csv_data: str) -> Dict[str, Any]:
        """Add multiple locations from CSV format"""
        try:
            locations = []
            errors = []
            
            lines = csv_data.strip().split('\n')
            
            for i, line in enumerate(lines, start=1):
                try:
                    parts = line.split(',')
                    if len(parts) < 2:
                        errors.append(f"Line {i}: Invalid format (need at least latitude,longitude)")
                        continue
                    
                    lat = float(parts[0].strip())
                    lon = float(parts[1].strip())
                    name = parts[2].strip() if len(parts) > 2 else f"Location {i}"
                    
                    if not self.validate_coordinates(lat, lon):
                        errors.append(f"Line {i}: Invalid coordinates ({lat}, {lon})")
                        continue
                    
                    location = {
                        "latitude": lat,
                        "longitude": lon,
                        "name": name,
                        "type": "CSV_INPUT",
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": "csv_input"
                    }
                    locations.append(location)
                except (ValueError, IndexError) as e:
                    errors.append(f"Line {i}: {str(e)}")
            
            logger.info(f"✅ Added {len(locations)} locations from CSV")
            return {
                "status": "success" if locations else "error",
                "locations": locations,
                "errors": errors,
                "total_added": len(locations),
                "total_errors": len(errors)
            }
        except Exception as e:
            logger.error(f"Error adding locations from CSV: {e}")
            return {"status": "error", "message": str(e)}
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def haversine_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in kilometers"""
        try:
            lat1_rad = math.radians(lat1)
            lon1_rad = math.radians(lon1)
            lat2_rad = math.radians(lat2)
            lon2_rad = math.radians(lon2)
            
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            
            a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            
            return self.earth_radius_km * c
        except Exception as e:
            logger.error(f"Distance calculation error: {e}")
            return 0.0
    
    def is_within_radius(self, lat1: float, lon1: float, 
                        lat2: float, lon2: float, radius_km: float) -> bool:
        """Check if two locations are within radius"""
        distance = self.haversine_distance(lat1, lon1, lat2, lon2)
        return distance <= radius_km
    
    # ========================================================================
    # TIMELINE VISUALIZATION
    # ========================================================================
    
    def build_timeline(self, locations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build chronological location timeline"""
        try:
            if not locations:
                return {"timeline": [], "total_locations": 0}
            
            # Sort by timestamp
            sorted_locations = sorted(locations, key=lambda x: x.get("timestamp", ""))
            
            timeline = []
            for i, location in enumerate(sorted_locations):
                timestamp = location.get("timestamp", "")
                latitude = location.get("latitude", 0)
                longitude = location.get("longitude", 0)
                location_name = location.get("name", "Unknown")
                
                # Calculate duration if not last location
                duration = None
                if i < len(sorted_locations) - 1:
                    try:
                        current_time = datetime.fromisoformat(timestamp)
                        next_time = datetime.fromisoformat(sorted_locations[i+1].get("timestamp", ""))
                        duration = (next_time - current_time).total_seconds() / 60  # minutes
                    except:
                        pass
                
                timeline.append({
                    "index": i + 1,
                    "timestamp": timestamp,
                    "location": location_name,
                    "coordinates": [latitude, longitude],
                    "duration_minutes": duration
                })
            
            logger.info(f"✅ Timeline built: {len(timeline)} locations")
            return {
                "timeline": timeline,
                "total_locations": len(timeline)
            }
        except Exception as e:
            logger.error(f"Timeline building error: {e}")
            return {"timeline": [], "total_locations": 0}
    
    # ========================================================================
    # GEOFENCING DETECTION
    # ========================================================================
    
    def detect_geofence_violations(self, locations: List[Dict[str, Any]], 
                                   geofences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect geofence violations"""
        try:
            violations = []
            
            for location in locations:
                lat = location.get("latitude", 0)
                lon = location.get("longitude", 0)
                timestamp = location.get("timestamp", "")
                
                for geofence in geofences:
                    geofence_lat = geofence.get("latitude", 0)
                    geofence_lon = geofence.get("longitude", 0)
                    radius = geofence.get("radius_km", 5)
                    geofence_name = geofence.get("name", "Unknown")
                    
                    if self.is_within_radius(lat, lon, geofence_lat, geofence_lon, radius):
                        violations.append({
                            "geofence": geofence_name,
                            "coordinates": [lat, lon],
                            "timestamp": timestamp,
                            "radius_km": radius
                        })
            
            logger.info(f"✅ Geofence violations detected: {len(violations)}")
            return {
                "violations_detected": len(violations) > 0,
                "violations": violations,
                "total_violations": len(violations)
            }
        except Exception as e:
            logger.error(f"Geofence detection error: {e}")
            return {"violations_detected": False, "violations": [], "total_violations": 0}
    
    # ========================================================================
    # FREQUENT LOCATIONS
    # ========================================================================
    
    def identify_frequent_locations(self, locations: List[Dict[str, Any]], 
                                   cluster_radius_km: float = 1.0) -> Dict[str, Any]:
        """Identify frequently visited locations"""
        try:
            if not locations:
                return {"frequent_locations": [], "total_unique": 0}
            
            # Cluster locations
            clusters = []
            used = set()
            
            for i, loc1 in enumerate(locations):
                if i in used:
                    continue
                
                cluster = [loc1]
                used.add(i)
                
                for j, loc2 in enumerate(locations[i+1:], start=i+1):
                    if j in used:
                        continue
                    
                    if self.is_within_radius(
                        loc1.get("latitude", 0), loc1.get("longitude", 0),
                        loc2.get("latitude", 0), loc2.get("longitude", 0),
                        cluster_radius_km
                    ):
                        cluster.append(loc2)
                        used.add(j)
                
                clusters.append(cluster)
            
            # Rank clusters
            frequent_locations = []
            for cluster in sorted(clusters, key=len, reverse=True):
                avg_lat = sum(loc.get("latitude", 0) for loc in cluster) / len(cluster)
                avg_lon = sum(loc.get("longitude", 0) for loc in cluster) / len(cluster)
                
                frequent_locations.append({
                    "rank": len(frequent_locations) + 1,
                    "coordinates": [avg_lat, avg_lon],
                    "visits": len(cluster),
                    "percentage": round((len(cluster) / len(locations)) * 100, 2),
                    "type": cluster[0].get("type", "UNKNOWN")
                })
            
            logger.info(f"✅ Frequent locations identified: {len(frequent_locations)}")
            return {
                "frequent_locations": frequent_locations,
                "total_unique": len(frequent_locations)
            }
        except Exception as e:
            logger.error(f"Frequent locations error: {e}")
            return {"frequent_locations": [], "total_unique": 0}
            
            sorted_locations = sorted(locations, key=lambda x: x.get("timestamp", ""))
            
            for i in range(len(sorted_locations) - 1):
                from_loc = sorted_locations[i].get("name", "Unknown")
                to_loc = sorted_locations[i+1].get("name", "Unknown")
                
                try:
                    from_time = datetime.fromisoformat(sorted_locations[i].get("timestamp", ""))
                    to_time = datetime.fromisoformat(sorted_locations[i+1].get("timestamp", ""))
                    travel_time = (to_time - from_time).total_seconds() / 60  # minutes
                except:
                    travel_time = 0
                
                key = f"{from_loc} → {to_loc}"
                patterns[key]["count"] += 1
                patterns[key]["times"].append(travel_time)
            
            # Format patterns
            travel_patterns = []
            for pattern, data in sorted(patterns.items(), key=lambda x: x[1]["count"], reverse=True):
                avg_time = sum(data["times"]) / len(data["times"]) if data["times"] else 0
                travel_patterns.append({
                    "route": pattern,
                    "frequency": data["count"],
                    "average_time_minutes": round(avg_time, 2)
                })
            
            logger.info(f"✅ Travel patterns analyzed: {len(travel_patterns)}")
            return {"travel_patterns": travel_patterns}
        except Exception as e:
            logger.error(f"Travel pattern analysis error: {e}")
            return {"travel_patterns": []}
    
    # ========================================================================
    # ANOMALY DETECTION
    # ========================================================================
    
    def detect_anomalies(self, locations: List[Dict[str, Any]], 
                        baseline_locations: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Detect anomalous locations and patterns"""
        try:
            anomalies = []
            
            if not baseline_locations:
                baseline_locations = locations[:len(locations)//2]
            
            # Calculate baseline statistics
            baseline_lats = [loc.get("latitude", 0) for loc in baseline_locations]
            baseline_lons = [loc.get("longitude", 0) for loc in baseline_locations]
            
            avg_lat = sum(baseline_lats) / len(baseline_lats) if baseline_lats else 0
            avg_lon = sum(baseline_lons) / len(baseline_lons) if baseline_lons else 0
            
            # Check for anomalies
            for location in locations:
                lat = location.get("latitude", 0)
                lon = location.get("longitude", 0)
                distance = self.haversine_distance(lat, lon, avg_lat, avg_lon)
                
                # Anomaly if distance > 2 standard deviations
                if distance > 50:  # 50km threshold
                    anomalies.append({
                        "location": location.get("name", "Unknown"),
                        "coordinates": [lat, lon],
                        "distance_from_baseline_km": round(distance, 2),
                        "timestamp": location.get("timestamp", ""),
                        "risk": "HIGH" if distance > 100 else "MEDIUM"
                    })
            
            logger.info(f"✅ Anomalies detected: {len(anomalies)}")
            return {
                "anomalies_detected": len(anomalies) > 0,
                "anomalies": anomalies,
                "total_anomalies": len(anomalies)
            }
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return {"anomalies_detected": False, "anomalies": [], "total_anomalies": 0}
    
    # ========================================================================
    # DISTANCE ANALYSIS
    # ========================================================================
    
    def analyze_distances(self, locations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distances traveled"""
        try:
            if len(locations) < 2:
                return {"total_distance_km": 0, "analysis": {}}
            
            sorted_locations = sorted(locations, key=lambda x: x.get("timestamp", ""))
            total_distance = 0
            distances = []
            
            for i in range(len(sorted_locations) - 1):
                lat1 = sorted_locations[i].get("latitude", 0)
                lon1 = sorted_locations[i].get("longitude", 0)
                lat2 = sorted_locations[i+1].get("latitude", 0)
                lon2 = sorted_locations[i+1].get("longitude", 0)
                
                distance = self.haversine_distance(lat1, lon1, lat2, lon2)
                distances.append(distance)
                total_distance += distance
            
            avg_distance = sum(distances) / len(distances) if distances else 0
            max_distance = max(distances) if distances else 0
            
            return {
                "total_distance_km": round(total_distance, 2),
                "average_distance_per_trip_km": round(avg_distance, 2),
                "max_distance_single_trip_km": round(max_distance, 2),
                "total_trips": len(distances)
            }
        except Exception as e:
            logger.error(f"Distance analysis error: {e}")
            return {"total_distance_km": 0, "analysis": {}}
    
    # ========================================================================
    # RISK ASSESSMENT
    # ========================================================================
    
    def assess_location_risk(self, locations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess location-based risks"""
        try:
            high_risk_locations = []
            
            for location in locations:
                lat = str(location.get("latitude", 0))
                lon = str(location.get("longitude", 0))
                
                # Check database for suspicious locations
                suspicious_loc = self.db.get_location(lat, lon)
                
                if suspicious_loc:
                    high_risk_locations.append({
                        "location": suspicious_loc.location_name,
                        "coordinates": [lat, lon],
                        "type": suspicious_loc.location_type,
                        "risk_level": suspicious_loc.risk_level,
                        "known_fraudsters": suspicious_loc.known_fraudsters,
                        "known_harassers": suspicious_loc.known_harassers
                    })
            
            overall_risk = "HIGH" if high_risk_locations else "LOW"
            
            logger.info(f"✅ Location risk assessed: {len(high_risk_locations)} high-risk locations")
            return {
                "overall_risk": overall_risk,
                "high_risk_locations": high_risk_locations,
                "total_high_risk": len(high_risk_locations)
            }
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return {"overall_risk": "UNKNOWN", "high_risk_locations": [], "total_high_risk": 0}
    
    # ========================================================================
    # COMBINED ANALYSIS
    # ========================================================================
    
    def analyze_locations(self, locations: List[Dict[str, Any]], 
                         geofences: List[Dict[str, Any]] = None,
                         case_id: str = None, consent_manager: Any = None) -> Dict[str, Any]:
        """Comprehensive location analysis with consent verification"""
        
        # Check consent if available
        if consent_manager and case_id:
            try:
                from modules.consent.models import ConsentLevel, MODULE_MIN_LEVELS
                
                session = consent_manager.get_session(case_id)
                if session:
                    min_level = MODULE_MIN_LEVELS.get('location', ConsentLevel.STANDARD)
                    
                    if session.level < min_level:
                        logger.warning(f"Location analysis blocked: {session.level.name} < {min_level.name}")
                        return {
                            'status': 'consent_denied',
                            'message': f'Location analysis requires {min_level.name} consent',
                            'required_level': min_level.name,
                            'current_level': session.level.name,
                            'case_id': case_id
                        }
            except Exception as e:
                logger.error(f"Error checking consent: {e}")
        
        try:
            logger.info(f"🔍 Starting location analysis for {len(locations)} locations")
            
            # Build timeline
            timeline = self.build_timeline(locations)
            
            # Identify frequent locations
            frequent = self.identify_frequent_locations(locations)
            
            # Analyze travel patterns
            patterns = self.analyze_travel_patterns(locations)
            
            # Detect anomalies
            anomalies = self.detect_anomalies(locations)
            
            # Analyze distances
            distances = self.analyze_distances(locations)
            
            # Assess risks
            risks = self.assess_location_risk(locations)
            
            # Detect geofence violations
            geofence_violations = {"violations": []}
            if geofences:
                geofence_violations = self.detect_geofence_violations(locations, geofences)
            
            # Calculate overall risk
            overall_risk_score = 0.0
            if anomalies["anomalies_detected"]:
                overall_risk_score += 0.3
            if risks["overall_risk"] == "HIGH":
                overall_risk_score += 0.4
            if geofence_violations.get("violations_detected", False):
                overall_risk_score += 0.3
            
            result = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_locations": len(locations),
                "timeline": timeline,
                "frequent_locations": frequent,
                "travel_patterns": patterns,
                "anomalies": anomalies,
                "distances": distances,
                "risk_assessment": risks,
                "geofence_violations": geofence_violations,
                "overall_risk_score": round(min(overall_risk_score, 1.0), 2),
                "classification": "CRITICAL" if overall_risk_score > 0.8 else "HIGH" if overall_risk_score > 0.6 else "MEDIUM" if overall_risk_score > 0.4 else "LOW"
            }
            
            logger.info(f"✅ Location analysis complete: Risk={result['classification']}")
            return result
        except Exception as e:
            logger.error(f"Location analysis error: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # ========================================================================
    # OFFLINE OPERATIONS
    # ========================================================================
    
    def _queue_gps_link_operation(self, case_id: str, link: str, source: str,
                                 latitude: float, longitude: float, location_name: str = None,
                                 added_by: str = None, notes: str = None) -> bool:
        """Queue GPS link operation for offline sync"""
        operation = {
            "type": "add_gps_link",
            "case_id": case_id,
            "link": link,
            "source": source,
            "latitude": latitude,
            "longitude": longitude,
            "location_name": location_name,
            "added_by": added_by,
            "notes": notes
        }
        return self.offline_queue.add_to_queue(operation)
    
    def sync_pending_operations(self) -> Dict[str, Any]:
        """Sync all pending offline operations to database"""
        try:
            if not self.offline_queue.is_connected():
                return {
                    "status": "offline",
                    "message": "No internet connection",
                    "pending": len(self.offline_queue.get_pending_operations())
                }
            
            pending_ops = self.offline_queue.get_pending_operations()
            synced_count = 0
            failed_count = 0
            errors = []
            
            logger.info(f"🔄 Starting sync of {len(pending_ops)} pending operations")
            
            for idx, operation in enumerate(pending_ops):
                try:
                    if operation.get("type") == "add_gps_link":
                        gps_log = self.db.add_gps_link(
                            case_id=operation.get("case_id"),
                            link=operation.get("link"),
                            source=operation.get("source"),
                            latitude=operation.get("latitude"),
                            longitude=operation.get("longitude"),
                            location_name=operation.get("location_name"),
                            added_by=operation.get("added_by"),
                            notes=operation.get("notes")
                        )
                        self.offline_queue.mark_synced(idx)
                        synced_count += 1
                        logger.info(f"✅ Synced operation {idx+1}/{len(pending_ops)}")
                except Exception as e:
                    failed_count += 1
                    error_msg = f"Operation {idx+1}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(f"❌ {error_msg}")
            
            # Clean up synced operations
            self.offline_queue.clear_synced()
            
            result = {
                "status": "success",
                "synced": synced_count,
                "failed": failed_count,
                "total": len(pending_ops),
                "errors": errors if errors else None
            }
            
            logger.info(f"✅ Sync complete: {synced_count} synced, {failed_count} failed")
            return result
        except Exception as e:
            logger.error(f"Sync error: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_offline_status(self) -> Dict[str, Any]:
        """Get offline/online status and queue information"""
        stats = self.offline_queue.get_queue_stats()
        return {
            "is_online": self.offline_queue.is_connected(),
            "queue_stats": stats,
            "pending_operations": len(stats["pending"]),
            "synced_operations": len(stats["synced"]),
            "total_operations": stats["total_operations"]
        }
    
    # ========================================================================
    # ARTIFACT ROUTING
    # ========================================================================
    
    def save_analysis_results(self, case_id: str, results: Dict[str, Any]) -> bool:
        """Save location analysis results to artifact storage"""
        try:
            # Resolve artifact path
            artifact_path = ArtifactPathBuilder.resolve(
                case_id, 
                "analysis", 
                ensure_dir=True
            )
            
            # Save location analysis
            location_file = os.path.join(artifact_path, "location_analysis.json")
            
            with open(location_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"✅ Location analysis saved to {location_file}")
            
            # Also save to results repository
            ResultsRepository.save(case_id, {"location_analysis": results})
            
            return True
        except Exception as e:
            logger.error(f"❌ Error saving location analysis: {e}")
            return False
    
    def save_gps_links(self, case_id: str, gps_links: List[Dict[str, Any]]) -> bool:
        """Save GPS links to artifact storage"""
        try:
            # Resolve artifact path
            artifact_path = ArtifactPathBuilder.resolve(
                case_id, 
                "analysis", 
                ensure_dir=True
            )
            
            # Save GPS links
            gps_file = os.path.join(artifact_path, "gps_links.json")
            
            with open(gps_file, 'w') as f:
                json.dump(gps_links, f, indent=2, default=str)
            
            logger.info(f"✅ GPS links saved to {gps_file}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error saving GPS links: {e}")
            return False
    
    def load_analysis_results(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Load location analysis results from artifact storage"""
        try:
            artifact_path = ArtifactPathBuilder.resolve(case_id, "analysis")
            location_file = os.path.join(artifact_path, "location_analysis.json")
            
            if os.path.exists(location_file):
                with open(location_file, 'r') as f:
                    results = json.load(f)
                
                logger.info(f"✅ Location analysis loaded from {location_file}")
                return results
            
            return None
        except Exception as e:
            logger.error(f"❌ Error loading location analysis: {e}")
            return None
    
    def load_gps_links(self, case_id: str) -> Optional[List[Dict[str, Any]]]:
        """Load GPS links from artifact storage"""
        try:
            artifact_path = ArtifactPathBuilder.resolve(case_id, "analysis")
            gps_file = os.path.join(artifact_path, "gps_links.json")
            
            if os.path.exists(gps_file):
                with open(gps_file, 'r') as f:
                    links = json.load(f)
                
                logger.info(f"✅ GPS links loaded from {gps_file}")
                return links
            
            return None
        except Exception as e:
            logger.error(f"❌ Error loading GPS links: {e}")
            return None


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Example usage
    analyzer = LocationIntelligence()
    
    # Test locations
    test_locations = [
        {
            "name": "Home",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "timestamp": "2025-11-25T08:00:00",
            "type": "RESIDENCE"
        },
        {
            "name": "Work",
            "latitude": 40.7489,
            "longitude": -73.9680,
            "timestamp": "2025-11-25T09:00:00",
            "type": "WORKPLACE"
        },
        {
            "name": "Coffee Shop",
            "latitude": 40.7505,
            "longitude": -73.9972,
            "timestamp": "2025-11-25T12:30:00",
            "type": "CAFE"
        },
        {
            "name": "Home",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "timestamp": "2025-11-25T18:00:00",
            "type": "RESIDENCE"
        }
    ]
    
    result = analyzer.analyze_locations(test_locations)
    
    print(f"\n{'='*60}")
    print(f"Location Analysis Results")
    print(f"{'='*60}")
    print(f"Total Locations: {result.get('total_locations')}")
    print(f"Classification: {result.get('classification')}")
    print(f"Risk Score: {result.get('overall_risk_score')}")
    print(f"Frequent Locations: {len(result.get('frequent_locations', {}).get('frequent_locations', []))}")
    print(f"Anomalies: {result.get('anomalies', {}).get('total_anomalies', 0)}")
    print(f"Travel Patterns: {len(result.get('travel_patterns', {}).get('travel_patterns', []))}")
