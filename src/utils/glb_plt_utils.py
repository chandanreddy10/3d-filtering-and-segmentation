from pathlib import Path
import open3d as o3d
import numpy as np

from pygltflib import GLTF2


def get_file_metadata(file_path):

    ext = Path(file_path).suffix.lower()

    if ext == ".ply":

        header = []

        with open(file_path, "rb") as f:
            while True:
                line = f.readline().decode("ascii", errors="ignore").strip()
                header.append(line)

                if line == "end_header":
                    break

        return {
            "type": "ply",
            "header": header
        }

    elif ext == ".glb":

        gltf = GLTF2().load(str(file_path))

        return {
            "type": "glb",
            "asset": gltf.asset,
            "scenes": len(gltf.scenes),
            "nodes": len(gltf.nodes),
            "meshes": len(gltf.meshes),
            "materials": len(gltf.materials),
            "textures": len(gltf.textures),
            "images": len(gltf.images),
            "cameras": len(gltf.cameras)
        }

    else:
        raise ValueError(f"Unsupported format: {ext}")



def inspect_file(file_path):

    file_path = Path(file_path)

    print("=" * 60)
    print(f"FILE: {file_path.name}")
    print("=" * 60)

    metadata = get_file_metadata(file_path)

    if metadata["type"] == "ply":

        header = metadata["header"]

        properties = []

        for line in header:

            if line.startswith("element"):
                print(line)

            if line.startswith("property"):
                properties.append(line)

        print("\nAvailable properties:")

        for prop in properties:
            print(" ", prop)

        names = [
            p.split()[-1]
            for p in properties
        ]

        print("\nDetected attributes:")

        checks = {
            "XYZ coordinates":
                all(x in names for x in ["x", "y", "z"]),

            "Normals":
                all(x in names for x in ["nx", "ny", "nz"]),

            "RGB colors":
                all(x in names for x in ["red", "green", "blue"]),

            "Intensity":
                "intensity" in names,

            "Labels":
                "label" in names
        }

        for key, value in checks.items():
            print(f"{key:<20}: {value}")

        print(
            f"{'Mesh faces':<20}:",
            any(
                x.startswith("element face")
                for x in header
            )
        )

    else:

        print("\nGLB metadata:")

        for key, value in metadata.items():

            if key != "type":
                print(f"{key:<20}: {value}")

    print("=" * 60)



def load_geometry(file_path):

    ext = Path(file_path).suffix.lower()

    if ext == ".ply":

        return o3d.io.read_point_cloud(
            str(file_path)
        )

    elif ext == ".glb":

        mesh = o3d.io.read_triangle_mesh(
            str(file_path)
        )

        mesh.compute_vertex_normals()

        return mesh

    else:
        raise ValueError(f"Unsupported format: {ext}")



def mesh_to_point_cloud(mesh):

    points = np.asarray(mesh.vertices)

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)

    if mesh.has_vertex_colors():
        pcd.colors = mesh.vertex_colors

    return pcd



def visualize(file_path, voxel_size=None):

    file_path = Path(file_path)

    geometry = load_geometry(file_path)

    print(geometry)

    if isinstance(
        geometry,
        o3d.geometry.PointCloud
    ):

        if voxel_size:
            geometry = geometry.voxel_down_sample(
                voxel_size
            )

        print("\nPoint cloud properties:")
        print("Colors:", geometry.has_colors())
        print("Normals:", geometry.has_normals())
        print("Points:", len(geometry.points))

        if not geometry.has_colors():
            geometry.paint_uniform_color(
                [0.7, 0.7, 0.7]
            )

    elif isinstance(
        geometry,
        o3d.geometry.TriangleMesh
    ):

        print("\nMesh properties:")
        print("Vertices:", len(geometry.vertices))
        print("Triangles:", len(geometry.triangles))

        if not geometry.has_vertex_colors():
            geometry.paint_uniform_color(
                [0.7, 0.7, 0.7]
            )

    o3d.visualization.draw_geometries(
        [geometry],
        window_name=file_path.name,
        width=1280,
        height=720
    )


if __name__ == "__main__":

    path = "model.glb"

    inspect_file(path)

    visualize(path)