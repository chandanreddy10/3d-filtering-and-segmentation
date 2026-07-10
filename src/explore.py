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
pcd = pcd.voxel_down_sample(0.001)
points = np.asarray(pcd.points)  # shape (N, 3)
xy = points[:, :2]   # (x, y)
xz = points[:, [0, 2]]   # keep height for coloring
yz = points[:, [1, 2]]

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].scatter(points[:,0], points[:,1], s=1)
axs[0].set_title("Top View (XY)")

axs[1].scatter(points[:,0], points[:,2], s=1)
axs[1].set_title("Front View (XZ)")

axs[2].scatter(points[:,1], points[:,2], s=1)
axs[2].set_title("Side View (YZ)")

for ax in axs:
    ax.axis("equal")

plt.show()