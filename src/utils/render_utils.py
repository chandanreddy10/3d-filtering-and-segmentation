import open3d as o3d
import numpy as np


def rotation_matrix_from_vectors(source, target):
    """
    Compute rotation matrix that rotates source vector to target vector.

    Parameters
    ----------
    source : np.ndarray
        Original vector.
    target : np.ndarray
        Desired vector.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)

    v = np.cross(source, target)
    c = np.dot(source, target)

    s = np.linalg.norm(v)

    # Already aligned
    if s < 1e-8:
        return np.eye(3)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))

    return R


def align_point_cloud_to_top(
    pcd,
    distance_threshold=0.005,
    voxel_size=None,
    center=True
):
    """
    Align point cloud so dominant plane becomes XY plane.
    """

    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(
            voxel_size
        )


    plane_model, _ = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=2000
    )


    normal = np.array(
        plane_model[:3]
    )

    normal /= np.linalg.norm(normal)


    if normal[2] < 0:
        normal = -normal


    R = rotation_matrix_from_vectors(
        normal,
        np.array([0,0,1])
    )


    pcd.rotate(
        R,
        center=pcd.get_center()
    )


    if center:
        pcd.translate(
            -pcd.get_center()
        )


    return pcd

def render_point_cloud(
    pcd,
    output_path,
    resolution=(1024, 768),
    camera_position=None,
    camera_target=None,
    up_vector=np.array([0, 0, 1]),
    fov=45.0,
    distance_multiplier=0.75,
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
        If None, automatically generated above the point cloud.
    camera_target : np.ndarray
        Point camera looks at.
        If None, uses point cloud center.
    up_vector : np.ndarray
        Camera up direction.
    fov : float
        Field of view in degrees.
    distance_multiplier : float
        Camera distance relative to point cloud size.
    point_size : float
        Size of rendered points.
    background_color : np.ndarray
        RGBA background color.

    Returns
    -------
    image : open3d.geometry.Image
        Rendered image.
    """
    width, height = resolution

    # Bounding box
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = np.linalg.norm(bbox.get_extent())

    # Default camera target
    if camera_target is None:
        camera_target = center

    # Default camera position (top view)
    if camera_position is None:
        camera_position = center + np.array([0, 0, extent * distance_multiplier])

    # Renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    # Material
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.point_size = point_size

    renderer.scene.add_geometry("point_cloud", pcd, material)

    # Background
    renderer.scene.set_background(background_color)

    # Camera
    renderer.setup_camera(fov, camera_target, camera_position, up_vector)

    # Render
    image = renderer.render_to_image()

    # Save
    o3d.io.write_image(output_path, image)

    return image