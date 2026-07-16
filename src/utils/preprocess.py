import numpy as np
import open3d as o3d

def voxel_downsample(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.02,
) -> o3d.geometry.PointCloud:
    """
    Downsample a point cloud using voxel grid filtering.
    """
    print(f"Original points: {len(pcd.points):,}")

    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    print(f"After voxel downsampling: {len(pcd.points):,}")

    return pcd


def remove_outliers(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    """
    Remove statistical outliers from a point cloud.
    """
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )

    print(f"After outlier removal: {len(pcd.points):,}")

    return pcd


def estimate_normals(
    pcd: o3d.geometry.PointCloud,
    radius: float = 0.1,
    max_nn: int = 30,
) -> o3d.geometry.PointCloud:
    """
    Estimate surface normals.
    """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn,
        )
    )

    return pcd

def cylinder_filter(
    pcd: o3d.geometry.PointCloud,
    radius: float = 0.85,
    z_min: float = None,
    z_max: float = None,
) -> o3d.geometry.PointCloud:
    points = np.asarray(pcd.points)

    center = np.mean(points[:, :2], axis=0)
    r = np.linalg.norm(points[:, :2] - center, axis=1)

    mask = r < radius

    if z_min is not None:
        mask &= points[:, 2] > z_min
    if z_max is not None:
        mask &= points[:, 2] < z_max

    return pcd.select_by_index(np.where(mask)[0])


def iterative_plate_removal(
    pcd: o3d.geometry.PointCloud, num_planes: int = 2, thickness: float = 0.02
) -> tuple[o3d.geometry.PointCloud, list[o3d.geometry.PointCloud]]:
    remaining = pcd
    plates = []

    for _ in range(num_planes):

        if len(remaining.points) < 5000:
            break

        plane_model, _ = remaining.segment_plane(
            distance_threshold=thickness, ransac_n=3, num_iterations=1000
        )

        a, b, c, d = plane_model
        points = np.asarray(remaining.points)

        distances = np.abs(points @ np.array([a, b, c]) + d)

        inliers = np.where(distances < thickness)[0]

        if len(inliers) < 5000:
            break

        plate = remaining.select_by_index(inliers)
        remaining = remaining.select_by_index(inliers, invert=True)

        plates.append(plate)

    return remaining, plates
