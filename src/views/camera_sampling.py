"""
Script provides functions to inspect, and load a camera view.

"""

import open3d as o3d
import json
import numpy as np


def inspect_camera_view(pcd: o3d.geometry.PointCloud) -> list:
    """
    Function to Inspect a camera view
    Args:
    pcd : a point cloud

    Returns
    [list] : List of camera views

    """
    camera_views_list = []
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window(window_name="Rotate to desired view", width=1280, height=720)

    vis.add_geometry(pcd)

    def save_camera(vis):
        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()

        camera_views_list.append(
            {
                "intrinsic": {
                    "width": params.intrinsic.width,
                    "height": params.intrinsic.height,
                    "intrinsic_matrix": params.intrinsic.intrinsic_matrix.tolist(),
                },
                "extrinsic": params.extrinsic.tolist(),
            }
        )

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

    return camera_views_list


def load_camera_from_config(camera_config):
    """
    Convert one camera dictionary into Open3D PinholeCameraParameters
    """

    intrinsic_data = camera_config["intrinsic"]

    intrinsic = o3d.camera.PinholeCameraIntrinsic()

    intrinsic.set_intrinsics(
        width=intrinsic_data["width"],
        height=intrinsic_data["height"],
        fx=intrinsic_data["intrinsic_matrix"][0][0],
        fy=intrinsic_data["intrinsic_matrix"][1][1],
        cx=intrinsic_data["intrinsic_matrix"][0][2],
        cy=intrinsic_data["intrinsic_matrix"][1][2],
    )

    params = o3d.camera.PinholeCameraParameters()

    params.intrinsic = intrinsic
    params.extrinsic = np.array(camera_config["extrinsic"])

    return params


def visualize_with_saved_camera(pcd: o3d.geometry.PointCloud, config: dict):

    if not pcd.has_colors():
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    vis = o3d.visualization.Visualizer()

    vis.create_window(window_name="Saved camera view", width=1280, height=720)

    vis.add_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()

    ctr = vis.get_view_control()
    params = load_camera_from_config(config)

    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    vis.run()

    vis.destroy_window()


def render_camera_view_to_png(
    pcd: o3d.geometry.PointCloud,
    camera_config: dict,
    output_path: str,
    width: int = 1280,
    height: int = 720,
):
    """
    Render point cloud from a saved camera view and save PNG.
    """

    if not pcd.has_colors():
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    vis = o3d.visualization.Visualizer()

    vis.create_window(visible=False, width=width, height=height)

    vis.add_geometry(pcd)

    # Need one render update before applying camera
    vis.poll_events()
    vis.update_renderer()

    ctr = vis.get_view_control()

    params = load_camera_from_config(camera_config)

    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    # Render
    vis.poll_events()
    vis.update_renderer()

    # Save screenshot
    vis.capture_screen_image(output_path, do_render=True)

    vis.destroy_window()

    return output_path
