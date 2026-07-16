import open3d as o3d
import numpy as np

from pathlib import Path
import yaml
import os
import json
 
import sys 

ROOT_DIR = Path(__file__).parents[1]
sys.path.append(str(ROOT_DIR))

from utils.preprocess import preprocess_point_cloud

ROOT_DIR_CONFIG= Path(__file__).parents[2]

CONFIG_FILE = ROOT_DIR_CONFIG / "config.yaml"

with open(CONFIG_FILE) as f:
    CONFIG = yaml.safe_load(f)

DATA_FOLDER = ROOT_DIR_CONFIG / CONFIG["DATA_FOLDER"]

files = sorted(os.listdir(DATA_FOLDER))

for i, file in enumerate(files):
    print(i, file)

INDEX = 1

ply_path = DATA_FOLDER / files[INDEX]

def inspect_camera_view(ply_path):

    pcd = o3d.io.read_point_cloud(ply_path)
    pcd = preprocess_point_cloud(pcd, voxel_size=0.005)
    camera_views_list = []
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.7, 0.7, 0.7])


    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window(
        window_name="Rotate to desired view",
        width=1280,
        height=720
    )

    vis.add_geometry(pcd)


    def save_camera(vis):
        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()

        camera_views_list.append({
            "intrinsic": {
                "width": params.intrinsic.width,
                "height": params.intrinsic.height,
                "intrinsic_matrix": params.intrinsic.intrinsic_matrix.tolist()
            },
            "extrinsic": params.extrinsic.tolist()
        })

        print(f"Camera {len(camera_views_list)} saved!")

        return False


    vis.register_key_callback(ord("S"), save_camera)

    print("Instructions:")
    print(" - Rotate with mouse")
    print(" - Zoom with scroll")
    print(" - Pan with Shift + mouse")
    print(" - Press S to save the current camera")

    vis.run()
    vis.destroy_window()

    with open("camera.json", "w") as f:
        json.dump(camera_views_list, f, indent=4)

    print(f"Saved {len(camera_views_list)} camera views to camera.json")

inspect_camera_view(ply_path)