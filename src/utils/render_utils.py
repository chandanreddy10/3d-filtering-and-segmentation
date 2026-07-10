import open3d as o3d
import numpy as np


def render_point_cloud(
    pcd,
    output_path,
    resolution=(1024, 768),
    camera_position=None,
    camera_target=None,
    up_vector=np.array([0, 1, 0]),
    fov=60.0,
    distance_multiplier=2.0,
    point_size=3.0,
    background_color=np.array([1, 1, 1, 1]),
):
    """
    Render point cloud using Open3D EGL offscreen renderer.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Input point cloud.

    output_path : str
        Path to save rendered image.

    resolution : tuple
        (width, height)

    camera_position : np.ndarray
        Camera location [x,y,z].
        If None, automatically generated.

    camera_target : np.ndarray
        Point camera looks at.
        If None, uses point cloud center.

    up_vector : np.ndarray
        Camera up direction.

    fov : float
        Camera field of view in degrees.

    distance_multiplier : float
        Distance from object when camera_position
        is automatically calculated.

    point_size : float
        Size of rendered points.

    background_color : np.ndarray
        RGBA background.

    Returns
    -------
    image : open3d.geometry.Image
    """

    width, height = resolution

    bbox = pcd.get_axis_aligned_bounding_box()

    center = bbox.get_center()
    extent = np.linalg.norm(bbox.get_extent())

    if camera_target is None:
        camera_target = center

    if camera_position is None:
        camera_position = center + np.array([0, 0, extent * distance_multiplier])

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    material = o3d.visualization.rendering.MaterialRecord()

    material.shader = "defaultlit"

    material.point_size = point_size

    renderer.scene.add_geometry("pointcloud", pcd, material)

    # Background
    renderer.scene.set_background(background_color)

    # Camera
    renderer.setup_camera(fov, camera_target, camera_position, up_vector)

    # Render
    image = renderer.render_to_image()

    o3d.io.write_image(output_path, image)

    return image
