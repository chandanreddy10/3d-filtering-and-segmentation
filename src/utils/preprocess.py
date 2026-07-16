"""
Preprocessing Utilities.
Available :
1.Downsampling
2.SOR
3.ROR
4.Plate removal to remove dominant plane
5.Cylinder filter 


"""



import numpy as np
import open3d as o3d


def voxel_downsample(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.02
):
    """
    Downsample point cloud using voxel grid.
    """

    print(f"Before voxel downsampling: {len(pcd.points):,}")

    pcd = pcd.voxel_down_sample(
        voxel_size=voxel_size
    )

    print(f"After voxel downsampling: {len(pcd.points):,}")

    return pcd



def statistical_outlier_removal(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0
):
    """
    Remove points whose average distance from neighbors
    is larger than the global mean + std_ratio * std.
    """

    print(f"Before statistical removal: {len(pcd.points):,}")

    pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    print(f"After statistical removal: {len(pcd.points):,}")

    return pcd



def radius_outlier_removal(
    pcd: o3d.geometry.PointCloud,
    nb_points: int = 16,
    radius: float = 0.05
):
    """
    Remove points that do not have enough neighbors
    within a fixed radius.
    """

    print(f"Before radius removal: {len(pcd.points):,}")

    pcd, ind = pcd.remove_radius_outlier(
        nb_points=nb_points,
        radius=radius
    )

    print(f"After radius removal: {len(pcd.points):,}")

    return pcd



def preprocess_point_cloud(
    pcd,
    voxel_size=0.001,
    statistical=True,
    radius=False
):
    """
    Complete preprocessing pipeline.
    """

    # 1. Voxel downsampling
    pcd = voxel_downsample(
        pcd,
        voxel_size
    )

    # 2. Statistical outlier removal
    if statistical:
        pcd = statistical_outlier_removal(
            pcd,
            nb_neighbors=30,
            std_ratio=2.0
        )

    # 3. Radius outlier removal
    if radius:
        pcd = radius_outlier_removal(
            pcd,
            nb_points=16,
            radius=0.05
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
