import open3d as o3d


def voxel_downsample(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.001,
) -> o3d.geometry.PointCloud:
    """
    Downsample a point cloud using voxel grid filtering.
    """

    return pcd.voxel_down_sample(voxel_size)

import open3d as o3d


def statistical_outlier_removal(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
):
    """
    Remove statistical outliers.
    """

    filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )

    return filtered