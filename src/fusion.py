from pathlib import Path
import yaml
import os
import open3d as o3d
import json
import numpy as np 

from utils.preprocess import preprocess_point_cloud
from views.camera_sampling import visualize_with_saved_camera
from postprocessing.fusion import capture_depth_from_pcd, create_labeled_point_cloud_from_depth_and_masks, visualize_labeled_point_cloud

# Paths
ROOT_DIR = Path(__file__).parents[1]
CONFIG_FILE = ROOT_DIR / "config.yaml"

with open(CONFIG_FILE) as f:
    CONFIG = yaml.safe_load(f)


DATA_FOLDER = ROOT_DIR / CONFIG["DATA_FOLDER"]
OUTPUT_FOLDER = ROOT_DIR / CONFIG["OUTPUT_FOLDER"]

CAMERA_FOLDER = ROOT_DIR / CONFIG["CAMERA_PARAMS_JSON"]
RENDERS_FOLDER = ROOT_DIR / CONFIG["VIEWS_FOLDER"]
MASKS_FOLDER = ROOT_DIR / CONFIG["SEGMENT_MASKS_FOLDER"]

files = sorted(os.listdir(DATA_FOLDER))

for i, file in enumerate(files):
    print(i, file)

INDEX = 0
ply_file = files[INDEX]
ply_path = DATA_FOLDER / ply_file

pcd_name = Path(ply_file).stem

pcd_views_folder = RENDERS_FOLDER / pcd_name
pcd_masks_folder = MASKS_FOLDER / pcd_name
pcd_camera_folder = CAMERA_FOLDER

camera_json_path = pcd_camera_folder / "camera.json"

with open(camera_json_path, "r") as file:
    camera_config = json.load(file)

print(f"\nProcessing: {pcd_name}")
pcd = o3d.io.read_point_cloud(str(ply_path))

print(f"Loaded points: {len(pcd.points):,}")

# Preprocess
pcd = preprocess_point_cloud(pcd, voxel_size=0.005)

for i, camera in enumerate(camera_config):

    image_path = pcd_views_folder / f"view_{i:03d}.png"
    masks_path = pcd_masks_folder / f"view_{i:03d}.npy"

    mask = np.load(masks_path)
    mask = mask.astype(bool)

    visualize_with_saved_camera(pcd, camera)

    depth_arr = capture_depth_from_pcd(pcd, camera)
    z = depth_arr / 1000.0
    valid_mask = (z > 0) & (z <= 3.0)

    camera_intrinsic = camera["intrinsic"]
    K = np.array(
    camera_intrinsic["intrinsic_matrix"],
    dtype=np.float64
    )

    width = camera_intrinsic["width"]
    height = camera_intrinsic["height"]

    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        K[0,0],  # fx
        K[1,1],  # fy
        K[0,2],  # cx
        K[1,2]   # cy
    )

    points, labels = create_labeled_point_cloud_from_depth_and_masks(
    depth_image=depth_arr,
    intrinsic=intrinsic_o3d.intrinsic_matrix,
    masks=mask,                     # shape (num_objects, H, W), bool
    extrinsic=camera["extrinsic"],
    depth_scale=1000.0,
    depth_trunc=1000.0,
    )

    # points[i] corresponds to labels[i]
    visualize_labeled_point_cloud(points, labels)
    
    break