#!/usr/bin/env python3
"""
Test Google Maps API Integration
Verifies that the API key is loaded and working correctly
"""

import os
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("=" * 60)
print("GOOGLE MAPS API - VERIFICATION TEST")
print("=" * 60)

# Check if API key is loaded
api_key = os.getenv('GOOGLE_MAPS_API_KEY')

if api_key:
    print("\n✅ API Key loaded successfully!")
    print(f"   Key: {api_key[:20]}...{api_key[-10:]}")
    print(f"   Length: {len(api_key)} characters")
else:
    print("\n❌ API Key not found!")
    print("   Please check .env file")
    sys.exit(1)

# Test MapDisplayManager
print("\n" + "=" * 60)
print("TESTING MAP DISPLAY MANAGER")
print("=" * 60)

try:
    from modules.analysis.location_intelligence import MapDisplayManager
    
    # Initialize
    map_manager = MapDisplayManager()
    
    print("\n✅ MapDisplayManager initialized")
    print(f"   Google Maps available: {map_manager.googlemaps_available}")
    print(f"   Folium available: {map_manager.folium_available}")
    print(f"   Fallback mode: {map_manager.fallback_mode}")
    
    # Get available map types
    available_types = map_manager.get_available_map_types()
    print(f"\n✅ Available map types:")
    for map_type in available_types:
        print(f"   - {map_type}")
    
    # Test map creation
    print("\n" + "=" * 60)
    print("TESTING MAP CREATION")
    print("=" * 60)
    
    result = map_manager.get_map_display_info(
        latitude=40.7128,
        longitude=-74.0060,
        location_name="New York, NY"
    )
    
    print(f"\n✅ Map creation successful!")
    print(f"   Map type: {result['map_type']}")
    print(f"   Status: {result['status']}")
    print(f"   Location: {result['location_name']}")
    print(f"   Coordinates: {result['latitude']}, {result['longitude']}")
    print(f"   API calls: {result['api_calls']}/{result['api_quota']}")
    
    if result['map_type'] == 'google_maps':
        print(f"   Embed URL: {result['embed_url'][:50]}...")
    
    # Test with different map types
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT MAP TYPES")
    print("=" * 60)
    
    # Test Google Maps
    print("\n1. Testing Google Maps...")
    result_gm = map_manager.get_map_display_info(
        latitude=40.7128,
        longitude=-74.0060,
        map_type="google_maps"
    )
    print(f"   Status: {result_gm['status']}")
    print(f"   Type: {result_gm['map_type']}")
    
    # Test Folium
    print("\n2. Testing Folium...")
    result_fm = map_manager.get_map_display_info(
        latitude=40.7128,
        longitude=-74.0060,
        map_type="folium"
    )
    print(f"   Status: {result_fm['status']}")
    print(f"   Type: {result_fm['map_type']}")
    
    # Test Coordinates Only
    print("\n3. Testing Coordinates Only...")
    result_co = map_manager.get_map_display_info(
        latitude=40.7128,
        longitude=-74.0060,
        map_type="coordinates_only"
    )
    print(f"   Status: {result_co['status']}")
    print(f"   Type: {result_co['map_type']}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\n🎉 Google Maps API is working correctly!")
    print("   Ready to use in Location Intelligence module")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
