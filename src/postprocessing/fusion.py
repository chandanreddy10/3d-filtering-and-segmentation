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

def create_labeled_point_cloud_from_depth_and_masks(
    depth_image: np.ndarray,
    intrinsic: np.ndarray,
    masks: np.ndarray,
    extrinsic: np.ndarray = None,
    depth_scale: float = 1000.0,
    depth_trunc: float = 1000.0,
    stride: int = 1,
    project_valid_depth_only: bool = True,
    background_label: int = -1,
):
    """
    Unprojects a depth image to 3D points and assigns each point an object
    label based on a set of 2D masks.

    """
    if extrinsic is None:
        extrinsic = np.eye(4)

    num_objects, H_mask, W_mask = masks.shape
    assert masks.shape[1:] == depth_image.shape, (
        f"masks spatial shape {masks.shape[1:]} must match "
        f"depth_image shape {depth_image.shape}"
    )

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    depth = depth_image[::stride, ::stride].astype(np.float64)
    masks_s = masks[:, ::stride, ::stride]  # (num_objects, H, W) strided
    H, W = depth.shape

    u, v = np.meshgrid(np.arange(W), np.arange(H))

    z = depth / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_cam = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Build a flat per-pixel label array: shape (H*W,)
    masks_flat = masks_s.reshape(num_objects, -1).astype(bool)  # (num_objects, H*W)
    labels = np.full(masks_flat.shape[1], background_label, dtype=np.int64)
    # Assign in reverse order so that object 0 has highest priority
    # if masks overlap (lowest index "wins").
    for obj_idx in range(num_objects - 1, -1, -1):
        labels[masks_flat[obj_idx]] = obj_idx

    if project_valid_depth_only:
        valid = (points_cam[:, 2] > 0) & (points_cam[:, 2] <= depth_trunc)
        points_cam = points_cam[valid]
        labels = labels[valid]
    else:
        invalid = (points_cam[:, 2] <= 0) | (points_cam[:, 2] > depth_trunc)
        points_cam[invalid] = 0.0
        labels[invalid] = background_label

    cam_to_world = np.linalg.inv(extrinsic)
    R = cam_to_world[:3, :3]
    t = cam_to_world[:3, 3]

    points_world = points_cam @ R.T + t

    return points_world, labels


def visualize_labeled_point_cloud(
    points: np.ndarray,
    labels: np.ndarray,
    background_label: int = -1,
    background_color=(0.6, 0.6, 0.6),
    seed: int = 42,
    point_size: float = 3.0,
):
    """
    Visualizes a point cloud colored by per-point object labels.
    """
    unique_labels = np.unique(labels)
    obj_labels = unique_labels[unique_labels != background_label]

    rng = np.random.default_rng(seed)
    color_map = {lbl: rng.uniform(0.15, 1.0, size=3) for lbl in obj_labels}
    color_map[background_label] = np.array(background_color)

    colors = np.array([color_map[lbl] for lbl in labels])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([0.05, 0.05, 0.05])
    vis.run()
    vis.destroy_window()

    return pcd