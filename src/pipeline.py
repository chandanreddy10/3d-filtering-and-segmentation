from pathlib import Path
import yaml
import os
import open3d as o3d

from preprocessing import preprocess
from utils import render_utils

ROOT_DIR = Path(__file__).parents[1]

CONFIG_FILE = ROOT_DIR / "config.yaml"

with open(CONFIG_FILE) as f:
    CONFIG = yaml.safe_load(f)

DATA_FOLDER = ROOT_DIR / CONFIG["DATA_FOLDER"]
OUTPUT_FOLDER = ROOT_DIR / CONFIG["OUTPUT_FOLDER"]
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

files = sorted(os.listdir(DATA_FOLDER))

for i, file in enumerate(files):
    print(i, file)

INDEX = 0

ply_path = DATA_FOLDER / files[INDEX]
pcd = o3d.io.read_point_cloud(ply_path)

# #Downsample
# pcd = preprocess.voxel_downsample(pcd, voxel_size=0.00001)
# #Noise Removal
# pcd = preprocess.statistical_outlier_removal(pcd)
# render_utils.render_point_cloud(pcd, Path(OUTPUT_FOLDER / "test_2.png"))
# mesh = o3d.io.read_triangle_mesh("scene.ply")


# mesh.compute_vertex_normals()


aligned_pcd = render_utils.align_point_cloud_to_top(pcd, distance_threshold=0.005)

render_utils.render_point_cloud(aligned_pcd, Path(OUTPUT_FOLDER / "test_2.png"))
