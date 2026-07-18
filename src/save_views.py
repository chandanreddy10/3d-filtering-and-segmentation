"""
Code to load Point cloud, filter noise, save View (and its metadata)
The Camera intrinsic, extrinsic is also stored.

Run:
    python save_views.py

To capture a View click S.
"""

from pathlib import Path
import yaml
import os
import open3d as o3d
import json

from utils.preprocess import preprocess_point_cloud
from views import camera_sampling

# Paths
ROOT_DIR = Path(__file__).parents[1]
CONFIG_FILE = ROOT_DIR / "config.yaml"

with open(CONFIG_FILE) as f:
    CONFIG = yaml.safe_load(f)


DATA_FOLDER = ROOT_DIR / CONFIG["DATA_FOLDER"]
OUTPUT_FOLDER = ROOT_DIR / CONFIG["OUTPUT_FOLDER"]

CAMERA_FOLDER = ROOT_DIR / CONFIG["CAMERA_PARAMS_JSON"]
VIEWS_FOLDER = ROOT_DIR / CONFIG["VIEWS_FOLDER"]

# Create folders
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
CAMERA_FOLDER.mkdir(parents=True, exist_ok=True)
VIEWS_FOLDER.mkdir(parents=True, exist_ok=True)

files = sorted(os.listdir(DATA_FOLDER))

for i, file in enumerate(files):
    print(i, file)

INDEX = 0
ply_file = files[INDEX]
ply_path = DATA_FOLDER / ply_file


pcd_name = Path(ply_file).stem

# Create sample folders
pcd_views_folder = VIEWS_FOLDER / pcd_name
pcd_camera_folder = CAMERA_FOLDER


pcd_views_folder.mkdir(parents=True, exist_ok=True)
pcd_camera_folder.mkdir(parents=True, exist_ok=True)

camera_json_path = pcd_camera_folder / "camera.json"

print(f"\nProcessing: {pcd_name}")
# Load / Generate cameras
file_exists = camera_json_path.exists()
views_len = False

if file_exists:

    with open(camera_json_path, "r") as f:
        views = json.load(f)

    if len(views) > 0:
        views_len = True

if views_len and file_exists:
    print("Camera parameters found. loading.......")
    with open(camera_json_path, "r") as f:
        views = json.load(f)
else:
    print("Camera parameters not found. Generating...")

    # Load point cloud
    pcd = o3d.io.read_point_cloud(str(ply_path))

    print(f"Loaded points: {len(pcd.points):,}")

    # Preprocess
    pcd = preprocess_point_cloud(pcd, voxel_size=0.005)

    # Interactive camera selection
    views = camera_sampling.inspect_camera_view(pcd)

    # Save cameras
    with open(camera_json_path, "w") as f:
        json.dump(views, f, indent=4)

    print(f"Saved camera parameters: {camera_json_path}")

# Load point cloud for rendering
pcd = o3d.io.read_point_cloud(str(ply_path))
pcd = preprocess_point_cloud(pcd, voxel_size=0.005)

for i, camera in enumerate(views):

    image_path = pcd_views_folder / f"view_{i:03d}.png"

    # Skip existing images
    if image_path.exists():
        print(f"Skipping existing: {image_path}")
        continue

    camera_sampling.render_camera_view_to_png(pcd, camera, output_path=str(image_path))

    print(f"Saved view: {image_path}")

print("\nFinished!")
