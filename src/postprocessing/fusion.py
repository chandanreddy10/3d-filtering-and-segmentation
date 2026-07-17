import open3d as o3d
import numpy as np


def capture_depth_from_pcd(pcd: o3d.geometry.PointCloud, camera: dict) -> np.ndarray:
    """
    The function takes a point cloud and returns the depth array.


    """
    width = camera["intrinsic"]["width"]
    height = camera["intrinsic"]["height"]

    K = np.array(camera["intrinsic"]["intrinsic_matrix"], dtype=np.float64)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    extrinsic = np.array(camera["extrinsic"], dtype=np.float64)

    vis = o3d.visualization.Visualizer()

    vis.create_window(width=width, height=height, visible=False)

    vis.add_geometry(pcd)

    # make points visible
    render_option = vis.get_render_option()
    render_option.point_size = 3.0

    ctr = vis.get_view_control()

    cam_params = ctr.convert_to_pinhole_camera_parameters()

    cam_params.intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

    cam_params.extrinsic = extrinsic

    ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

    vis.poll_events()
    vis.update_renderer()

    # depth in meters (main method)
    depth_float = vis.capture_depth_float_buffer(do_render=True)

    depth_float = np.asarray(depth_float)

    vis.destroy_window()

    # convert meters -> millimeters uint16
    depth_uint16 = (depth_float * 1000).astype(np.uint16)

    return depth_uint16
