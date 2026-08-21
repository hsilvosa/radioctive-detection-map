import json
import os
from pathlib import Path
from radiation_map.inference.predictor import SpatialRadiationPredictor

def build_regional_layer():
    predictor = SpatialRadiationPredictor(model_path="checkpoints/best_spatial_harmonic_net.pt")

    def get_center(coords):
        flat = []
        def extract_pts(c):
            if isinstance(c[0], (int, float)):
                flat.append((c[1], c[0])) # lat, lon
            else:
                for item in c:
                    extract_pts(item)
        extract_pts(coords)
        if not flat:
            return 0.0, 0.0
        avg_lat = sum(p[0] for p in flat) / len(flat)
        avg_lon = sum(p[1] for p in flat) / len(flat)
        return avg_lat, avg_lon

    def simplify_ring(pts, min_dist=0.25):
        if len(pts) <= 4:
            return pts
        res = [pts[0]]
        for p in pts[1:-1]:
            dx = p[0] - res[-1][0]
            dy = p[1] - res[-1][1]
            if (dx*dx + dy*dy) >= (min_dist * min_dist):
                res.append([round(p[0], 3), round(p[1], 3)])
        res.append(pts[-1])
        return res if len(res) >= 4 else pts

    def simplify_geom(coords, geom_type, min_dist=0.25):
        if geom_type == 'Polygon':
            return [simplify_ring(ring, min_dist) for ring in coords if len(ring) >= 4]
        elif geom_type == 'MultiPolygon':
            return [[simplify_ring(ring, min_dist) for ring in poly if len(ring) >= 4] for poly in coords]
        return coords

    all_features = []

    # 1. Add Japan Prefectures
    jp_path = Path("src/radiation_map/web/static/japan_prefectures.geojson")
    if jp_path.exists():
        with open(jp_path, "r", encoding="utf-8") as f:
            jp_data = json.load(f)
        for feat in jp_data["features"]:
            name = feat["properties"].get("nam_ja") or feat["properties"].get("nam") or feat["properties"].get("name", "Prefecture")
            geom = feat["geometry"]
            lat, lon = get_center(geom["coordinates"])
            pred = predictor.predict_point(lat, lon)
            all_features.append({
                "type": "Feature",
                "properties": {
                    "name": name,
                    "parent": "Japan",
                    "type": "Prefecture",
                    "radiation_usv": pred["radiation_usv"],
                    "ci_lower_usv": pred["ci_lower_usv"],
                    "ci_upper_usv": pred["ci_upper_usv"],
                    "center_lat": round(lat, 4),
                    "center_lon": round(lon, 4)
                },
                "geometry": {
                    "type": geom["type"],
                    "coordinates": simplify_geom(geom["coordinates"], geom["type"], min_dist=0.08)
                }
            })
        print(f"Added {len(jp_data['features'])} Japan prefectures")

    # 2. Add Admin-1 regions (US States, Canada provinces, Brazil, China, Australia, etc.)
    admin1_path = Path("src/radiation_map/web/static/admin1_regions.geojson")
    if admin1_path.exists():
        with open(admin1_path, "r", encoding="utf-8") as f:
            admin1_data = json.load(f)
        for feat in admin1_data["features"]:
            props = feat["properties"]
            admin_country = props.get("admin") or props.get("adm0_name") or "Country"
            if admin_country in ["Japan"]:
                continue
            name = props.get("name") or props.get("name_en") or "Region"
            geom = feat["geometry"]
            lat, lon = get_center(geom["coordinates"])
            pred = predictor.predict_point(lat, lon)
            all_features.append({
                "type": "Feature",
                "properties": {
                    "name": name,
                    "parent": admin_country,
                    "type": "State / Province",
                    "radiation_usv": pred["radiation_usv"],
                    "ci_lower_usv": pred["ci_lower_usv"],
                    "ci_upper_usv": pred["ci_upper_usv"],
                    "center_lat": round(lat, 4),
                    "center_lon": round(lon, 4)
                },
                "geometry": {
                    "type": geom["type"],
                    "coordinates": simplify_geom(geom["coordinates"], geom["type"], min_dist=0.25)
                }
            })
        print(f"Added subnational features. Total now: {len(all_features)}")

    # 3. Add remaining world countries
    world_path = Path("src/radiation_map/web/static/countries.geojson")
    if world_path.exists():
        with open(world_path, "r", encoding="utf-8") as f:
            world_data = json.load(f)
        added_countries = 0
        subdivided_countries = {"United States of America", "United States", "Japan", "Canada", "Australia", "Brazil", "China", "Russia", "India", "Indonesia", "South Africa"}
        for feat in world_data["features"]:
            props = feat["properties"]
            c_name = props.get("name", "Country")
            if c_name in subdivided_countries:
                continue
            geom = feat["geometry"]
            lat, lon = get_center(geom["coordinates"])
            pred = predictor.predict_point(lat, lon)
            all_features.append({
                "type": "Feature",
                "properties": {
                    "name": c_name,
                    "parent": c_name,
                    "type": "Country",
                    "radiation_usv": pred["radiation_usv"],
                    "ci_lower_usv": pred["ci_lower_usv"],
                    "ci_upper_usv": pred["ci_upper_usv"],
                    "center_lat": round(lat, 4),
                    "center_lon": round(lon, 4)
                },
                "geometry": {
                    "type": geom["type"],
                    "coordinates": simplify_geom(geom["coordinates"], geom["type"], min_dist=0.3)
                }
            })
            added_countries += 1
        print(f"Added {added_countries} country features. Total regions: {len(all_features)}")

    out_path = Path("src/radiation_map/web/static/regional_radiation.geojson")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": all_features}, f, separators=(',', ':'))

    print(f"Successfully generated {out_path} ({out_path.stat().st_size / 1024:.1f} KB, {len(all_features)} regions)")

if __name__ == "__main__":
    build_regional_layer()
