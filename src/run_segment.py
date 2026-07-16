from pathlib import Path
import yaml
import json
import os 
from PIL import Image

from segmentation_2d import segment_view

# Paths
ROOT_DIR = Path(__file__).parents[1]
CONFIG_FILE = ROOT_DIR / "config.yaml"

with open(CONFIG_FILE) as f:
    CONFIG = yaml.safe_load(f)

RENDERS_FOLDER = ROOT_DIR / CONFIG["VIEWS_FOLDER"]
SEGMENT_MASKS_FOLDER = ROOT_DIR / CONFIG["SEGMENT_MASKS_FOLDER"]

sub_folder = "reconstruction_1"
SAVE_DIR = ROOT_DIR / "results" / sub_folder

SAVE_DIR.mkdir(parents=True, exist_ok=True)
SEGMENT_MASKS_FOLDER.mkdir(exist_ok = True)

images_path = RENDERS_FOLDER / sub_folder 
images = []
for image in os.listdir(images_path):
    images.append(Path(images_path / image))
result_dict = segment_view.segment_images(images, save_dir=SEGMENT_MASKS_FOLDER)

for (image_path, masks), img_path in zip(result_dict.items(), os.listdir(images_path)):

    image_path = Path(images_path / img_path)
    image = Image.open(image_path).convert("RGB")
    save_path = Path(SAVE_DIR / image_path.name)
    image_overlay = segment_view.overlay_masks(image, masks, save_path )

print("Saved Masks and overlaz images")
