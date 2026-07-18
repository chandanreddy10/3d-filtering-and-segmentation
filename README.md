### The Repo is Work in Progress.

# Multi-View 3D Object Segmentation Pipeline

This project performs multi-view 3D object segmentation by combining:

1. **View generation** from input point clouds
2. **2D segmentation** using SAM3 on rendered views
3. **2D-to-3D projection and multi-view fusion**

The pipeline:

```
Point Cloud (.ply)
        |
        v
views.py
        |
        v
Rendered RGB Views + Camera Parameters
        |
        v
segment.py (SAM3)
        |
        v
2D Object Masks
        |
        v
fusion.py
        |
        v
Segmented 3D Point Cloud
```

---

# Installation

### Clone repository and install dependencies
# Pipeline Execution

The pipeline must be executed in the following order:

---

# Step 1: Generate Views

Run:

```bash
python save_views.py
```

This will:

- Load the input `.ply` point cloud
- Sample camera viewpoints
- Render RGB images
- Save camera intrinsic/extrinsic parameters


Output:

```
renders/
└── file_name/
    ├── view_000.png
    ├── view_001.png
    ├── view_002.png
    └── ...

camera_config/
└── camera.json
```

---
Run:

```bash
python run_segment.py
```

This will:

- Load rendered images
- Run SAM3 segmentation
- Save object masks


Output:

```
segment_masks/

└── file_name/
    ├── view_000.npy
    ├── view_001.npy
    └── ...
```

Each mask file contains:

```
(num_objects, height, width)
```

where:

- Each channel represents one object
- Boolean values indicate object pixels

---

# Step 3: 3D Fusion

Run:

```bash
python fusion.py
```

This will:

- Load camera parameters
- Load rendered depth maps
- Project 2D masks into 3D
- Fuse information from multiple views
- Remove duplicate points
- Propagate object identities

# GPU Requirement

SAM3 segmentation requires a CUDA-enabled GPU.

Recommended:

- NVIDIA GPU
- >= 8GB VRAM

Check GPU:

```bash
nvidia-smi
```

---
# Notes

The fusion stage uses:

- Depth-based unprojection
- Spatial duplicate removal
- Multi-view label propagation
- Nearest labelled point matching

The goal is to retain only unique 3D information while accumulating object labels across all camera views.
