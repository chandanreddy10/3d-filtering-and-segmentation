from pathlib import Path
import yaml
import os

from utils.ply_utils import visualize_point_cloud, inspect_ply_file

ROOT_DIR = Path(__file__).parents[1]

CONFIG_FILE = ROOT_DIR / "config.yaml"

with open(CONFIG_FILE) as f:
    CONFIG = yaml.safe_load(f)

DATA_FOLDER = ROOT_DIR / CONFIG["DATA_FOLDER"]

files = sorted(os.listdir(DATA_FOLDER))

for i, file in enumerate(files):
    print(i, file)

INDEX = 1

ply_path = DATA_FOLDER / files[INDEX]

inspect_ply_file(ply_path)
visualize_point_cloud(ply_path, voxel_size=0.001)

