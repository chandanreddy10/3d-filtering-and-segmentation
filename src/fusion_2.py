from pathlib import Path
import yaml
import os
import open3d as o3d
import json
import numpy as np

from utils.preprocess import preprocess_point_cloud, iterative_plate_removal
from views.camera_sampling import visualize_with_saved_camera
from postprocessing.fusion import (
    capture_depth_from_pcd,
    create_labeled_point_cloud_from_depth_and_masks,
    visualize_labeled_point_cloud,
)

from scipy.spatial import cKDTree

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


def visualise_single_view(pcd, camera_config, index=0):
    """
    Function to Visualise Single View.

    """
    camera = camera_config[index]
    i = index
    image_path = pcd_views_folder / f"view_{i:03d}.png"
    masks_path = pcd_masks_folder / f"view_{i:03d}.npy"

    mask = np.load(masks_path)
    mask = mask.astype(bool)

    visualize_with_saved_camera(pcd, camera)

    depth_arr = capture_depth_from_pcd(pcd, camera)
    z = depth_arr / 1000.0
    valid_mask = (z > 0) & (z <= 3.0)

    camera_intrinsic = camera["intrinsic"]
    K = np.array(camera_intrinsic["intrinsic_matrix"], dtype=np.float64)

    width = camera_intrinsic["width"]
    height = camera_intrinsic["height"]

    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2]  # fx  # fy  # cx  # cy
    )

    points, labels = create_labeled_point_cloud_from_depth_and_masks(
        depth_image=depth_arr,
        intrinsic=intrinsic_o3d.intrinsic_matrix,
        masks=mask,  # shape (num_objects, H, W), bool
        extrinsic=camera["extrinsic"],
        depth_scale=1000.0,
        depth_trunc=1000.0,
    )

    print(points[0], labels[0])
    # points[i] corresponds to labels[i]
    # pcd = visualize_labeled_point_cloud(points, labels)

    return points, labels


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

DIST_THRESHOLD = 0.2 #(in meters)

def remove_existing_points(new_points, fused_points, threshold):
    """
    Remove points which already exist in fused cloud.
    """

    if len(fused_points) == 0:
        return np.ones(len(new_points), dtype=bool)

    tree = cKDTree(fused_points)

    distances, _ = tree.query(new_points, k=1)

    # keep only points farther than threshold
    keep = distances > threshold

    return keep


def assign_labels_from_edges(new_points, fused_points, fused_labels, max_distance=0.05):
    """
    Assign each new point the label of the nearest existing fused point.

    Parameters
    ----------
    new_points : (N,3)
    fused_points : (M,3)
    fused_labels : (M,)
    max_distance : float

    Returns
    -------
    assigned_labels : (N,)
        -1 means no nearby labelled point.
    """

    if len(fused_points) == 0:
        return np.full(len(new_points), -1, dtype=int)

    tree = cKDTree(fused_points)

    distances, indices = tree.query(new_points, k=1)

    assigned = np.full(len(new_points), -1, dtype=int)

    valid = distances < max_distance

    assigned[valid] = fused_labels[indices[valid]]

    return assigned


def incremental_view_fusion(views, camera_config, threshold=0.1):

    fused_points = np.empty((0, 3))
    fused_labels = np.empty((0,), dtype=int)

    for view_id in views:

        print(f"\nProcessing view {view_id}")

        points, labels = visualise_single_view(pcd, camera_config, view_id)

        if len(points) == 0:
            continue

        if len(fused_points) == 0:

            fused_points = points.copy()
            fused_labels = labels.copy()

            print("Initialized fusion")

            continue
        keep_mask = remove_existing_points(points, fused_points, threshold)

        new_points = points[keep_mask]
        new_labels = labels[keep_mask]

        print("Original:", len(points), " New:", len(new_points))

        if len(new_points) == 0:
            continue
        new_labels = assign_labels_from_edges(
            new_points, fused_points, fused_labels, max_distance=0.05
        )
        fused_points = np.vstack([fused_points, new_points])

        fused_labels = np.concatenate([fused_labels, new_labels])

        print("Total fused:", len(fused_points))

    return fused_points, fused_labels

NUM_VIEWS = [12, 17, 19, 10, 7, 5]
fused_points, fused_labels = incremental_view_fusion(
    NUM_VIEWS, camera_config, threshold=0.2
)
print("Final points:", len(fused_points))
visualize_labeled_point_cloud(fused_points, fused_labels, visualise=True)
