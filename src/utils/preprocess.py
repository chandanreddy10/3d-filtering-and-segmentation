import numpy as np
import open3d as o3d


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
