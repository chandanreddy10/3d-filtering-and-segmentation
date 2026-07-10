from pathlib import Path
import open3d as o3d
import numpy as np


def get_ply_header(ply_path):
    """
    Reads PLY header and extracts metadata.
    """

    header = []

    with open(ply_path, "rb") as f:
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header.append(line)

            if line == "end_header":
                break

    return header


def inspect_ply_file(ply_path):
    """
    Prints information available in a PLY file.
    """

    print("=" * 60)
    print(f"PLY FILE: {ply_path.name}")
    print("=" * 60)

    header = get_ply_header(ply_path)

    properties = []

    for line in header:

        if line.startswith("property"):
            properties.append(line)

        if line.startswith("element"):
            print(line)

    print("\nAvailable properties:")
    for prop in properties:
        print("  ", prop)

    # Check attributes
    property_names = [p.split()[-1] for p in properties]

    print("\nDetected attributes:")

    checks = {
        "XYZ coordinates": all(x in property_names for x in ["x", "y", "z"]),
        "Normals": all(x in property_names for x in ["nx", "ny", "nz"]),
        "RGB colors": all(x in property_names for x in ["red", "green", "blue"]),
        "Intensity": "intensity" in property_names,
        "Labels": "label" in property_names,
    }

    for key, value in checks.items():
        print(f"{key:<20}: {value}")

    # Faces
    has_faces = any(line.startswith("element face") for line in header)

    print(f"{'Mesh faces':<20}: {has_faces}")

    print("=" * 60)


def visualize_point_cloud(ply_path, voxel_size=None):
    """
    Visualizes a PLY point cloud.
    """

    pcd = o3d.io.read_point_cloud(str(ply_path))

    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size)

    print(pcd)

    print("\nPoint cloud properties:")
    print("Colors :", pcd.has_colors())
    print("Normals:", pcd.has_normals())
    print("Points :", len(pcd.points))

    if not pcd.has_colors():
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    o3d.visualization.draw_geometries(
        [pcd], window_name=ply_path.name, width=1280, height=720
    )
