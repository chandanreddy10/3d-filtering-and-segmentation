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

    camera = camera_config[index]
    i = index
    image_path = pcd_views_folder / f"view_{i:03d}.png"
    masks_path = pcd_masks_folder / f"view_{i:03d}.npy"

    mask = np.load(masks_path)
    mask = mask.astype(bool)

    visualize_with_saved_camera(pcd, camera)

    depth_arr = capture_depth_from_pcd(pcd, camera)
    depth_m = depth_arr.astype(np.float32) / 1000.0

    # Mask invalid pixels (0 means no depth)
    depth_display = np.ma.masked_equal(depth_m, 0)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_display, cmap="viridis")
    plt.colorbar(label="Depth (m)")
    plt.title("Depth Map")
    plt.xlabel("Pixel")
    plt.ylabel("Pixel")
    plt.show()
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
    pcd = visualize_labeled_point_cloud(points, labels, visualise=True)

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
# visualise_single_view(pcd, camera_config,17)
print(f"Loaded points: {len(pcd.points):,}")

# Preprocess
pcd = preprocess_point_cloud(pcd, voxel_size=0.005)
import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment


def bbox_overlap_ratio(box1, box2):
    """Intersection volume over the SMALLER box's volume.
    Much more robust than IoU when views only see partial slices
    of an object (common with single-view point clouds)."""
    min1, max1 = box1["min_bound"], box1["max_bound"]
    min2, max2 = box2["min_bound"], box2["max_bound"]

    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(inter_max - inter_min, 0)
    inter_vol = np.prod(inter_dims)

    vol1 = np.prod(np.maximum(max1 - min1, 1e-9))
    vol2 = np.prod(np.maximum(max2 - min2, 1e-9))

    smaller_vol = min(vol1, vol2)
    return inter_vol / smaller_vol if smaller_vol > 0 else 0.0


def centroid_dist_normalized(box1, box2):
    """Centroid distance normalized by average object size, as a
    secondary signal for borderline cases."""
    c1 = (box1["min_bound"] + box1["max_bound"]) / 2
    c2 = (box2["min_bound"] + box2["max_bound"]) / 2
    size1 = np.linalg.norm(box1["max_bound"] - box1["min_bound"])
    size2 = np.linalg.norm(box2["max_bound"] - box2["min_bound"])
    avg_size = max((size1 + size2) / 2, 1e-6)
    return np.linalg.norm(c1 - c2) / avg_size


merged_points = []
merged_labels = []

objects = {}       # obj_id -> {"min_bound", "max_bound"}
next_id = 0

OVERLAP_THRESHOLD = 0.05     # tune this (0.3-0.5 typical for overlap-ratio metric)
CENTROID_THRESHOLD = 0.6     # fallback: normalized centroid distance

N_VIEWS = 4

for index in range(N_VIEWS):
    print("Iterating", index)
    points, labels = visualise_single_view(pcd, camera_config, index)

    valid = labels != -1
    points_valid = points[valid]
    labels_valid = labels[valid]

    # build bboxes for current view
    current_boxes = {}
    for label in np.unique(labels_valid):
        cluster = points_valid[labels_valid == label]
        current_boxes[label] = {
            "min_bound": cluster.min(axis=0),
            "max_bound": cluster.max(axis=0),
        }

    label_mapping = {}

    if index == 0 or len(objects) == 0:
        # first view (or nothing accumulated yet): every cluster is a new object
        for label, bbox in current_boxes.items():
            objects[next_id] = bbox
            label_mapping[label] = next_id
            next_id += 1

    else:
        new_labels = list(current_boxes.keys())
        old_ids = list(objects.keys())

        n_new = len(new_labels)
        n_old = len(old_ids)

        # cost matrix: lower cost = better match
        cost = np.ones((n_new, n_old))

        for i, nl in enumerate(new_labels):
            for j, oid in enumerate(old_ids):
                overlap = bbox_overlap_ratio(current_boxes[nl], objects[oid])
                cost[i, j] = 1.0 - overlap

        # Hungarian algorithm: optimal one-to-one assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_new = set()

        for i, j in zip(row_ind, col_ind):
            nl = new_labels[i]
            oid = old_ids[j]
            overlap = 1.0 - cost[i, j]

            match_ok = overlap > OVERLAP_THRESHOLD

            # fallback check for borderline / partial-view cases
            if not match_ok:
                cdist = centroid_dist_normalized(current_boxes[nl], objects[oid])
                match_ok = cdist < CENTROID_THRESHOLD

            if match_ok:
                label_mapping[nl] = oid
                objects[oid]["min_bound"] = np.minimum(
                    objects[oid]["min_bound"], current_boxes[nl]["min_bound"]
                )
                objects[oid]["max_bound"] = np.maximum(
                    objects[oid]["max_bound"], current_boxes[nl]["max_bound"]
                )
                assigned_new.add(nl)

        # any new-view clusters that weren't matched become new objects
        for nl in new_labels:
            if nl not in assigned_new:
                objects[next_id] = current_boxes[nl]
                label_mapping[nl] = next_id
                next_id += 1

    # assign object IDs to points for this view
    for label, obj_id in label_mapping.items():
        mask = labels_valid == label
        merged_points.append(points_valid[mask])
        merged_labels.append(np.full(mask.sum(), obj_id))

print("Objects:", len(objects))

merged_points = np.vstack(merged_points)
merged_labels = np.concatenate(merged_labels)

print(merged_points.shape)
print(merged_labels.shape)

pcd_final = o3d.geometry.PointCloud()
pcd_final.points = o3d.utility.Vector3dVector(merged_points)

# one random color per unique object id (consistent across all its points)
colors = np.zeros((len(merged_points), 3))
rng = np.random.default_rng(42)
unique_ids = np.unique(merged_labels)
color_map = {oid: rng.random(3) for oid in unique_ids}

for oid in unique_ids:
    colors[merged_labels == oid] = color_map[oid]

pcd_final.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd_final])

##TODO
## Resolve the converging 