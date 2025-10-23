# tests/test_polygon_contains.py
from shapely.geometry import Polygon, Point
# This tests the coordinate convention used elsewhere: shapely Point(x=lon, y=lat)
def test_polygon_contains_point():
    # polygon defined as list of (lon,lat) pairs (GeoJSON-style)
    coords = [(77.5940, 12.9710), (77.5960, 12.9710), (77.5960, 12.9730), (77.5940, 12.9730)]
    poly = Polygon(coords)
    # create a point that's inside (lon,lat)
    p = Point(77.5950, 12.9720)
    assert poly.contains(p)
    # outside point
    p2 = Point(77.6000, 12.9800)
    assert not poly.contains(p2)