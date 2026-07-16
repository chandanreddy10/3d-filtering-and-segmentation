from pathlib import Path
import yaml
import json
import os 

from segmentation_2d import segment_view

# Paths
ROOT_DIR = Path(__file__).parents[1]
CONFIG_FILE = ROOT_DIR / "config.yaml"

with open(CONFIG_FILE) as f:
    CONFIG = yaml.safe_load(f)

RENDERS_FOLDER = ROOT_DIR / CONFIG["VIEWS_FOLDER"]
SAVE_DIR = ROOT_DIR / "results"

sub_folder = "reconstruction_1"
images_path = RENDERS_FOLDER / sub_folder 

images = []
for image in os.listdir(images_path):
    images.append(Path(images_path / image))
results = segment_view.segment_images(images, save_dir="results")
print("done")

