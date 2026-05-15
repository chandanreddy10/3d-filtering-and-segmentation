from pathlib import Path
import yaml
import open3d as o3d
import numpy as np
import os
from collections import deque

from utils import preprocess

ROOT_DIR = Path(__file__).parents[1]
CONFIG_FILE = ROOT_DIR / "config.yaml"

with open(CONFIG_FILE, "r") as file:
    CONFIG = yaml.safe_load(file)

DATA_FOLDER = ROOT_DIR / CONFIG["DATA_FOLDER"]

point_cloud_files = os.listdir(DATA_FOLDER)
pcd_path = Path(DATA_FOLDER) / point_cloud_files[2]

pcd = o3d.io.read_point_cloud(str(pcd_path))

def region_growing_segmentation(
    pcd,
    k=30,
    distance_thresh=0.02,
    normal_thresh=0.8,
    min_cluster_size=50,
    visualize=True
):
    if isinstance(pcd, o3d.geometry.PointCloud):
        points = np.asarray(pcd.points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
        )
        normals = np.asarray(pcd.normals)
    else:
        raise ValueError("Input must be Open3D PointCloud")

    N = len(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    labels = -np.ones(N, dtype=int)
    cluster_id = 0

    for i in range(N):
        if labels[i] != -1:
            continue

        queue = deque([i])
        labels[i] = cluster_id
        cluster_size = 0

        while queue:
            idx = queue.popleft()
            _, neighbors, _ = pcd_tree.search_knn_vector_3d(points[idx], k)

            for j in neighbors:
                if labels[j] != -1:
                    continue

                if np.linalg.norm(points[idx] - points[j]) > distance_thresh:
                    continue

                if np.dot(normals[idx], normals[j]) < normal_thresh:
                    continue

                labels[j] = cluster_id
                queue.append(j)
                cluster_size += 1

        if cluster_size < min_cluster_size:
            labels[labels == cluster_id] = -1
        else:
            cluster_id += 1

    if visualize:
        colors = np.zeros((N, 3))
        for label in set(labels):
            if label == -1:
                color = np.array([0, 0, 0])
            else:
                np.random.seed(label + 1)
                color = np.random.rand(3)
            colors[labels == label] = color

        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])

    return labels


pcd = pcd.voxel_down_sample(0.004)

remaining, planes = preprocess.iterative_plate_removal(pcd, num_planes=3)

o3d.visualization.draw_geometries([remaining])

points = np.asarray(remaining.points)
z_min = np.percentile(points[:, 2], 1)
z_max = np.percentile(points[:, 2], 99)

last = preprocess.cylinder_filter(remaining)

last, _ = last.remove_statistical_outlier(20, 1.2)

o3d.visualization.draw_geometries([last])

output_path = Path(pcd_path).with_suffix("").as_posix() + "_processed.ply"

o3d.io.write_point_cloud(output_path, last)

region_growing_segmentation(last)