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

merged = o3d.geometry.PointCloud()
for index in range(20):
    points, labels = visualise_single_view(pcd, camera_config, index)
    valid = labels != -1

    points_valid = points[valid]
    labels_valid = labels[valid]

    bboxes = {}

    for label in np.unique(labels_valid):
        cluster = points_valid[labels_valid == label]

        bboxes[label] = {
            "x_range": (cluster[:, 0].min(), cluster[:, 0].max()),
            "y_range": (cluster[:, 1].min(), cluster[:, 1].max()),
            "z_range": (cluster[:, 2].min(), cluster[:, 2].max()),
        }
    print(len(bboxes))
    # merged +=pcd_view
    break
o3d.visualization.draw_geometries(
    [merged], window_name=ply_path.name, width=1280, height=720
)

visualize_labeled_point_cloud(points, labels)

merged_points = []
merged_labels = []

objects = {}
next_id = 0
IOU_THRESHOLD = 0.1


def bbox_iou_3d(box1, box2):

    min1, max1 = box1["min_bound"], box1["max_bound"]
    min2, max2 = box2["min_bound"], box2["max_bound"]

    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)

    inter_dims = np.maximum(inter_max - inter_min, 0)
    inter_vol = np.prod(inter_dims)

    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)

    union = vol1 + vol2 - inter_vol

    return inter_vol / union if union > 0 else 0


merged_points = []
merged_labels = []

objects = {}
next_id = 0

IOU_THRESHOLD = 0.1


for index in range(4):
    print("Iterating", index)
    points, labels = visualise_single_view(pcd, camera_config, index)

    valid = labels != -1

    points_valid = points[valid]
    labels_valid = labels[valid]

    current_boxes = {}

    # create bounding boxes for current view
    for label in np.unique(labels_valid):

        cluster = points_valid[labels_valid == label]

        current_boxes[label] = {
            "min_bound": cluster.min(axis=0),
            "max_bound": cluster.max(axis=0),
        }

    label_mapping = {}

    # first view
    if index == 0:

        for label, bbox in current_boxes.items():

            objects[next_id] = bbox

            label_mapping[label] = next_id

            next_id += 1

    else:

        for label, new_bbox in current_boxes.items():

            best_iou = 0
            best_id = None

            for obj_id, old_bbox in objects.items():

                iou = bbox_iou_3d(new_bbox, old_bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_id = obj_id

            if best_iou > IOU_THRESHOLD:

                # assign to existing object
                label_mapping[label] = best_id

                # expand bbox
                objects[best_id]["min_bound"] = np.minimum(
                    objects[best_id]["min_bound"], new_bbox["min_bound"]
                )

                objects[best_id]["max_bound"] = np.maximum(
                    objects[best_id]["max_bound"], new_bbox["max_bound"]
                )

            else:

                # create new object
                objects[next_id] = new_bbox

                label_mapping[label] = next_id

                next_id += 1

    # assign object IDs to points
    for label, obj_id in label_mapping.items():

        mask = labels_valid == label

        merged_points.append(points_valid[mask])

        merged_labels.append(np.full(mask.sum(), obj_id))


print("Objects:", len(objects))


# concatenate final cloud
merged_points = np.vstack(merged_points)
merged_labels = np.concatenate(merged_labels)


print(merged_points.shape)
print(merged_labels.shape)
pcd_final = o3d.geometry.PointCloud()

pcd_final.points = o3d.utility.Vector3dVector(merged_points)

# random color per object
colors = np.zeros((len(merged_points), 3))

for obj_id in np.unique(merged_labels):
    colors[merged_labels == obj_id] = np.random.rand(3)

pcd_final.colors = o3d.utility.Vector3dVector(colors)


o3d.visualization.draw_geometries([pcd_final])

##TODO: 
#The IOU is Noisy. Consider removing outliers.
#The IOU is sensitive to views and number of views.