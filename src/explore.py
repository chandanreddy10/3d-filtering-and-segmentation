from pathlib import Path
import yaml
import open3d as o3d
import os

ROOT_DIR = Path(__file__).parents[1]
CONFIG_FILE = ROOT_DIR / "config.yaml"

with open(CONFIG_FILE, "r") as file:
    CONFIG = yaml.safe_load(file)

DATA_FOLDER = ROOT_DIR / CONFIG["DATA_FOLDER"]

# Load point cloud
point_cloud_files = os.listdir(DATA_FOLDER)
pcd_path = Path(DATA_FOLDER) / point_cloud_files[2]

# pcd = o3d.io.read_point_cloud(str(pcd_path))
# pcd = pcd.voxel_down_sample(voxel_size=0.001)

# # Visualize
# o3d.visualization.draw_geometries(
#     [pcd],
#     window_name="Point Cloud Viewer",
#     width=1280,
#     height=720
# )

from pathlib import Path

with open(pcd_path, "rb") as f:
    while True:
        line = f.readline().decode("ascii", errors="ignore")
        print(line.strip())

        if line.strip() == "end_header":
            break