import open3d as o3d

from pathlib import Path
import yaml
import os
import json 

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
import open3d as o3d
import numpy as np


def visualize_with_camera_pose(
    ply_path,
    camera_position=None,
    rot_x=0,
    rot_y=0,
    rot_z=0
):
    """
    Move and rotate Open3D camera.

    camera_position:
        [x, y, z] world coordinates

    rot_x, rot_y, rot_z:
        camera rotations in degrees
    """

    pcd = o3d.io.read_point_cloud(ply_path)

    if not pcd.has_colors():
        pcd.paint_uniform_color([0.7, 0.7, 0.7])


    vis = o3d.visualization.Visualizer()

    vis.create_window(
        window_name="Camera pose",
        width=1280,
        height=720
    )

    vis.add_geometry(pcd)


    vis.poll_events()
    vis.update_renderer()


    ctr = vis.get_view_control()

    params = ctr.convert_to_pinhole_camera_parameters()


    if camera_position is None:
        camera_position = np.array([0, 0, 5])
    else:
        camera_position = np.array(camera_position)


    center = np.asarray(
        pcd.get_center()
    )


    rx = np.deg2rad(rot_x)
    ry = np.deg2rad(rot_y)
    rz = np.deg2rad(rot_z)


    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])


    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])


    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])


    R = Rz @ Ry @ Rx


    extrinsic = np.eye(4)

    extrinsic[:3, :3] = R


    # world -> camera translation
    extrinsic[:3, 3] = -R @ camera_position


    params.extrinsic = extrinsic


    ctr.convert_from_pinhole_camera_parameters(
        params,
        allow_arbitrary=True
    )


    vis.poll_events()
    vis.update_renderer()


    print("Camera position:")
    print(camera_position)

    print("\nExtrinsic:")
    print(params.extrinsic)


    vis.run()

    vis.destroy_window()


# Example:
# visualize_with_camera_pose(
#     ply_path,
#     camera_position=[-0.5,0.2,-3],
#     rot_x=0,
#     rot_y=0,
#     rot_z=0
# )
import open3d as o3d
import numpy as np


def visualize_camera_on_radius(
    ply_path,
    radius=1.0,
    direction=[0, 0, 1]
):

    pcd = o3d.io.read_point_cloud(ply_path)

    if not pcd.has_colors():
        pcd.paint_uniform_color([0.7, 0.7, 0.7])


    centroid = np.asarray(
        pcd.get_center()
    )

    direction = np.asarray(direction)
    direction = direction / np.linalg.norm(direction)


    camera_position = centroid + radius * direction


    print("Centroid:")
    print(centroid)

    print("\nCamera position:")
    print(camera_position)


    vis = o3d.visualization.Visualizer()

    vis.create_window(
        window_name="Radius camera",
        width=1280,
        height=720
    )

    vis.add_geometry(pcd)


    vis.poll_events()
    vis.update_renderer()


    ctr = vis.get_view_control()

    params = ctr.convert_to_pinhole_camera_parameters()


    camera_direction = centroid - camera_position
    camera_direction /= np.linalg.norm(camera_direction)


    up = np.array([0, 1, 0])


    right = np.cross(
        camera_direction,
        up
    )

    right /= np.linalg.norm(right)


    up = np.cross(
        right,
        camera_direction
    )


    R = np.vstack([
        right,
        up,
        -camera_direction
    ])


    extrinsic = np.eye(4)

    extrinsic[:3, :3] = R

    extrinsic[:3, 3] = -R @ camera_position


    params.extrinsic = extrinsic


    ctr.convert_from_pinhole_camera_parameters(
        params,
        allow_arbitrary=True
    )


    vis.run()

    vis.destroy_window()

import open3d as o3d
import numpy as np


def visualize_camera_control(
    ply_path,
    radius=1.0,
    direction=[0, 0, 1],
    position_offset=[0, 0, 0],
    rot_x=0,
    rot_y=0,
    rot_z=0
):
    """
    Camera control around point cloud.

    radius:
        Distance from centroid.

    direction:
        Initial camera direction from centroid.

    position_offset:
        Additional camera translation.

    rot_x, rot_y, rot_z:
        Camera rotation in degrees.
    """

    pcd = o3d.io.read_point_cloud(ply_path)

    if not pcd.has_colors():
        pcd.paint_uniform_color([0.7, 0.7, 0.7])


    centroid = np.asarray(
        pcd.get_center()
    )


    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)


    camera_position = (
        centroid
        + radius * direction
        + np.asarray(position_offset)
    )


    look_at = centroid


    forward = look_at - camera_position
    forward /= np.linalg.norm(forward)


    up = np.array([0, 1, 0])


    right = np.cross(forward, up)
    right /= np.linalg.norm(right)


    up = np.cross(right, forward)


    R_base = np.vstack([
        right,
        up,
        -forward
    ])


    rx = np.deg2rad(rot_x)
    ry = np.deg2rad(rot_y)
    rz = np.deg2rad(rot_z)


    Rx = np.array([
        [1,0,0],
        [0,np.cos(rx),-np.sin(rx)],
        [0,np.sin(rx), np.cos(rx)]
    ])


    Ry = np.array([
        [np.cos(ry),0,np.sin(ry)],
        [0,1,0],
        [-np.sin(ry),0,np.cos(ry)]
    ])


    Rz = np.array([
        [np.cos(rz),-np.sin(rz),0],
        [np.sin(rz), np.cos(rz),0],
        [0,0,1]
    ])


    R = Rz @ Ry @ Rx @ R_base


    extrinsic = np.eye(4)

    extrinsic[:3,:3] = R

    extrinsic[:3,3] = -R @ camera_position


    vis = o3d.visualization.Visualizer()

    vis.create_window(
        window_name="Camera control",
        width=1280,
        height=720
    )

    vis.add_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()


    ctr = vis.get_view_control()

    params = ctr.convert_to_pinhole_camera_parameters()

    params.extrinsic = extrinsic

    ctr.convert_from_pinhole_camera_parameters(
        params,
        allow_arbitrary=True
    )


    print("Centroid:", centroid)
    print("Camera position:", camera_position)
    print("Extrinsic:")
    print(extrinsic)


    vis.run()

    vis.destroy_window()

import open3d as o3d
import numpy as np


def inspect_camera_view(ply_path):

    pcd = o3d.io.read_point_cloud(ply_path)
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
# import open3d as o3d


# def visualize_with_saved_camera(
#     ply_path,
#     camera_path="camera.json"
# ):

#     pcd = o3d.io.read_point_cloud(ply_path)

#     if not pcd.has_colors():
#         pcd.paint_uniform_color(
#             [0.7,0.7,0.7]
#         )


#     vis = o3d.visualization.Visualizer()

#     vis.create_window(
#         window_name="Saved camera view",
#         width=1280,
#         height=720
#     )

#     vis.add_geometry(pcd)


#     vis.poll_events()
#     vis.update_renderer()


#     ctr = vis.get_view_control()


#     params = o3d.io.read_pinhole_camera_parameters(
#         camera_path
#     )


#     ctr.convert_from_pinhole_camera_parameters(
#         params,
#         allow_arbitrary=True
#     )


#     vis.run()

#     vis.destroy_window()



# # visualize_with_saved_camera(
# #     ply_path,
# #     "camera.json"
# # )
# import open3d as o3d
# import numpy as np


# def create_orbit_views(
#     ply_path,
#     camera_json,
#     num_steps=8,
#     radius_scale=1.0
# ):
#     """
#     Creates circular camera views around the point cloud.

#     ply_path:
#         Point cloud file

#     camera_json:
#         Saved Open3D camera parameters

#     num_steps:
#         Number of orbit views

#     radius_scale:
#         Scale camera distance from centroid
#     """

#     pcd = o3d.io.read_point_cloud(ply_path)

#     if not pcd.has_colors():
#         pcd.paint_uniform_color(
#             [0.7, 0.7, 0.7]
#         )


#     centroid = np.asarray(
#         pcd.get_center()
#     )


#     params = o3d.io.read_pinhole_camera_parameters(
#         camera_json
#     )


#     extrinsic = params.extrinsic


#     R = extrinsic[:3, :3]
#     t = extrinsic[:3, 3]


#     camera_position = -R.T @ t


#     print("Original camera position:")
#     print(camera_position)


#     orbit_vector = camera_position - centroid

#     radius = np.linalg.norm(orbit_vector)

#     radius *= radius_scale


#     orbit_axis = np.array([1,0,0])


#     views = []


#     for i in range(num_steps):

#         angle = (
#             2*np.pi*i/num_steps
#         )


#         rot = o3d.geometry.get_rotation_matrix_from_axis_angle(
#             orbit_axis * angle
#         )


#         new_position = (
#             centroid
#             + rot @ orbit_vector
#         )


#         new_position = (
#             centroid
#             + radius * (
#                 new_position-centroid
#             ) / np.linalg.norm(
#                 new_position-centroid
#             )
#         )


#         forward = centroid - new_position

#         forward /= np.linalg.norm(forward)


#         up = np.array([0,1,0])


#         right = np.cross(
#             forward,
#             up
#         )

#         right /= np.linalg.norm(right)


#         up = np.cross(
#             right,
#             forward
#         )


#         R_new = np.vstack([
#             right,
#             up,
#             -forward
#         ])


#         extrinsic = np.eye(4)

#         extrinsic[:3,:3] = R_new

#         extrinsic[:3,3] = (
#             -R_new @ new_position
#         )


#         new_params = o3d.camera.PinholeCameraParameters()

#         new_params.intrinsic = params.intrinsic

#         new_params.extrinsic = extrinsic


#         views.append(new_params)


#     return views
# import open3d as o3d
# import os


# def save_orbit_views(
#     ply_path,
#     views,
#     output_dir="views"
# ):

#     os.makedirs(output_dir, exist_ok=True)

#     pcd = o3d.io.read_point_cloud(ply_path)

#     vis = o3d.visualization.Visualizer()

#     vis.create_window(
#         window_name="Orbit rendering",
#         width=1280,
#         height=720
#     )

#     vis.add_geometry(pcd)

#     vis.poll_events()
#     vis.update_renderer()

#     ctr = vis.get_view_control()


#     for i, params in enumerate(views):

#         print(f"Rendering view {i+1}/{len(views)}")

#         ctr.convert_from_pinhole_camera_parameters(
#             params,
#             allow_arbitrary=True
#         )

#         vis.poll_events()
#         vis.update_renderer()


#         filename = os.path.join(
#             output_dir,
#             f"view_{i:03d}.png"
#         )


#         vis.capture_screen_image(
#             filename,
#             do_render=True
#         )

#         print("Saved:", filename)


#     vis.destroy_window()
# views = create_orbit_views(
#     ply_path,
#     "camera.json",
#     num_steps=12
# )

# save_orbit_views(
#     ply_path,
#     views
# )

# #Next steps, Save Views based on Num_Steps. Then Sam3, then Reprojection.