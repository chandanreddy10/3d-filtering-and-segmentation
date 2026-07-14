from pathlib import Path
import yaml
import os

from utils.ply_utils import visualize, get_file_metadata

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

get_file_metadata(ply_path)
visualize(ply_path, voxel_size=0.001)

