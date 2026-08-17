import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is on path for Config
sys.path.append('/home/ws/')
from utils.recursive_config import Config


# Configs and Paths
config = Config()
graph_path = config.get_subpath("scene_graph")
ending = config["pre_scanned_graphs"]["high_res"]
GRAPH_DIR = os.path.join(graph_path, ending)


def load_drawer_centroids(drawers_folder: Path) -> dict:
    """
    Load centroids from numeric-named JSON files in drawers_folder.
    Expects: {"centroid": [x, y, z], ...}
    """
    drawer_data = {}

    for json_file in sorted(Path(drawers_folder).glob("*.json"), key=lambda p: int(p.stem)):
        drawer_id = json_file.stem
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read {json_file.name}: {e}")
            continue

        centroid = data.get('centroid', None)
        if isinstance(centroid, list) and len(centroid) == 3:
            drawer_data[drawer_id] = {
                'centroid': np.array(centroid, dtype=float),
                'full_data': data
            }
        else:
            print(f"Warning: Unknown centroid format in {json_file.name}, skipping")

    return drawer_data


def create_text_label_geometry(text: str, position, font_size=0.05):
    """
    Create a text label as a 3D point cloud arranged like text.
    Labels lie in the XZ plane (constant Y) so they face along the Y direction,
    readable when looking from +Y towards −Y with Z up.
    """
    # Rasterize text to image
    img_size = (200, 100)
    img = Image.new('RGB', img_size, color='white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_pos = ((img_size[0] - text_w) // 2, (img_size[1] - text_h) // 2)
    draw.text(text_pos, text, fill='black', font=font)

    img_array = np.array(img)
    text_pixels = np.where(img_array[:, :, 0] < 128)  # black pixels

    if len(text_pixels[0]) == 0:
        return None

    # Build centered, scaled point set in 2D
    points_2d = np.column_stack((text_pixels[1], -text_pixels[0])).astype(float)  # flip y for up
    points_2d -= points_2d.mean(axis=0)
    std = points_2d.std()
    points_2d *= (font_size / (std + 1e-8))

    # Map to 3D in XZ plane (constant Y)
    points_3d = np.zeros((len(points_2d), 3), dtype=float)
    points_3d[:, 0] = -points_2d[:, 0]  # mirror X so it reads correctly from +Y
    points_3d[:, 2] = points_2d[:, 1]   # Z up
    points_3d[:, 1] = 0.0               # constant Y

    # Offset slightly toward +Y and above the centroid
    offset = np.array([0.0, 0.05, 0.12], dtype=float)
    points_3d += np.array(position, dtype=float) + offset

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.paint_uniform_color([0.0, 0.0, 0.0])  # black
    return pcd


def create_visualization_geometries(drawer_data: dict):
    """
    Create Open3D geometries for centroids and labels.
    - Smaller spheres for centroids
    - Labels face Y direction (readable from +Y viewing toward −Y)
    """
    geometries = []

    # Coordinate frame: X(red), Y(green), Z(blue)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    geometries.append(coord_frame)

    np.random.seed(42)
    colors = np.random.rand(len(drawer_data), 3)

    for idx, (drawer_id, data) in enumerate(sorted(drawer_data.items(), key=lambda x: int(x[0]))):
        centroid = data['centroid']

        # Smaller spheres (half-ish)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
        sphere.translate(centroid)
        sphere.paint_uniform_color(colors[idx])
        geometries.append(sphere)

        # Slightly larger, faint sphere for visibility
        outer = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
        outer.translate(centroid)
        outer.paint_uniform_color(colors[idx] * 0.6)
        geometries.append(outer)

        # 3D text label with the drawer ID
        label = create_text_label_geometry(drawer_id, centroid, font_size=0.04)
        if label is not None:
            geometries.append(label)

        # Console info
        print(f"Drawer {drawer_id}: Centroid = [{centroid[0]:+.3f}, {centroid[1]:+.3f}, {centroid[2]:+.3f}]")

    return geometries


def save_drawer_map_html(drawer_data: dict, output_file: str):
    """
    Save an HTML table with all centroid coordinates.
    """
    html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Drawer Centroid Map</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
h1 { color: #333; }
table { border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
th, td { border: 1px solid #ddd; padding: 10px 12px; text-align: left; }
th { background: #4CAF50; color: #fff; }
tr:nth-child(even) { background: #fafafa; }
.coord { font-family: monospace; color: #1976D2; }
.drawer-id { font-weight: bold; color: #2E7D32; }
.info { margin: 16px 0; padding: 12px; background: #e3f2fd; border-left: 4px solid #2196F3; }
</style>
</head>
<body>
<h1>Drawer Centroid Position Map</h1>
<div class="info">
Coordinate system and preferred view: look from +Y toward −Y (Z up). X left(−), Y forward(−), Z up(+).
</div>
<table>
<tr><th>Drawer ID</th><th>X (m)</th><th>Y (m)</th><th>Z (m)</th><th>Distance (m)</th></tr>
"""
    for drawer_id, data in sorted(drawer_data.items(), key=lambda x: int(x[0])):
        c = data['centroid']
        dist = float(np.linalg.norm(c))
        html += f"<tr><td class='drawer-id'>{drawer_id}</td><td class='coord'>{c[0]:+.3f}</td><td class='coord'>{c[1]:+.3f}</td><td class='coord'>{c[2]:+.3f}</td><td class='coord'>{dist:.3f}</td></tr>\n"

    html += """
</table>
</body>
</html>
"""
    with open(output_file, "w") as f:
        f.write(html)
    print(f"✓ HTML map saved to: {output_file}")


def save_coordinates_txt(drawer_data: dict, output_file: str):
    """
    Save a plain text list of centroids.
    """
    with open(output_file, "w") as f:
        f.write("DRAWER CENTROID COORDINATES\n")
        f.write("View: from +Y toward −Y (Z up). X left(−), Y forward(−), Z up(+)\n\n")
        f.write(f"{'ID':<6} | {'X (m)':>9} | {'Y (m)':>9} | {'Z (m)':>9} | {'Dist':>8}\n")
        f.write("-" * 50 + "\n")
        for drawer_id, data in sorted(drawer_data.items(), key=lambda x: int(x[0])):
            c = data['centroid']
            dist = float(np.linalg.norm(c))
            f.write(f"{drawer_id:<6} | {c[0]:+9.3f} | {c[1]:+9.3f} | {c[2]:+9.3f} | {dist:8.3f}\n")
    print(f"✓ Text coordinates saved to: {output_file}")


def visualize_drawers():
    """
    Entry point: find latest scene_graph date folder, load drawers, render and save outputs.
    """
    print("=" * 80)
    print("DRAWER CENTROID POSITION FINDER")
    print("=" * 80)

    latest_folder = Path(GRAPH_DIR)
    drawers_folder = latest_folder / "drawers"
    if not drawers_folder.exists():
        raise FileNotFoundError(f"Drawers folder not found: {drawers_folder}")

    print(f"Loading drawer data from: {drawers_folder}")
    drawer_data = load_drawer_centroids(drawers_folder)
    if not drawer_data:
        print("No drawer data found.")
        return

    print(f"Found {len(drawer_data)} drawers\n")

    geometries = create_visualization_geometries(drawer_data)

    # Outputs in the latest folder
    html_output = latest_folder / "drawer_map.html"
    txt_output = latest_folder / "drawer_coordinates.txt"
    save_drawer_map_html(drawer_data, str(html_output))
    save_coordinates_txt(drawer_data, str(txt_output))

    # Open3D visualization with default camera set to look along −Y, Z up
    print("\nOpening 3D visualization (default view: along −Y, Z up)...")
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Drawer Centroids - Labeled by ID", width=1280, height=720)

        for g in geometries:
            vis.add_geometry(g)

        ctr = vis.get_view_control()

        # Center camera on centroid mean
        all_centroids = np.array([v['centroid'] for v in drawer_data.values()])
        center = all_centroids.mean(axis=0) if len(all_centroids) else np.array([0.0, 0.0, 0.0])

        # Set orientation: look from +Y toward −Y, Z up
        ctr.set_lookat(center.tolist())
        ctr.set_front([0.0, -1.0, 0.0])  # −Y
        ctr.set_up([0.0, 0.0, 1.0])      # Z up
        ctr.set_zoom(0.6)

        vis.run()
        vis.destroy_window()
        print("✓ 3D visualization closed")
    except Exception as e:
        print(f"Warning: Could not open 3D visualization: {e}")
        print("HTML and text outputs were still generated.")

    # Open HTML in host browser
    print(f'Opening HTML map in browser...')
    try:
        os.system(f'"$BROWSER" "{html_output}"')
    except Exception:
        print(f"Open manually: {html_output}")


if __name__ == "__main__":
    try:
        visualize_drawers()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)