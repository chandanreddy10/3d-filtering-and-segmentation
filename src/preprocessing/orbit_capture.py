import argparse
import os
import numpy as np
import open3d as o3d


def compute_centroid_and_radius(points: np.ndarray, up_axis: int = 1):
    """
    The Function Computes the centroid and the horizontal bounding radius (X) of a point cloud
    """
    centroid = points.mean(axis=0)

    horiz_axes = [i for i in range(3) if i != up_axis]
    horiz = points[:, horiz_axes] - centroid[horiz_axes]
    radius_X = np.max(np.linalg.norm(horiz, axis=1))

    return centroid, radius_X, up_axis


def camera_position_on_circle(center, R, theta, phi=0.0, up_axis=1):
    """
    Position a camera on a circle/shell of radius R around `center`.

    theta : azimuth angle (radians), sweeps all the way around the object.
    phi   : elevation angle (radians), 0 = level with the object's center,
            positive = looking down from above.
    """
    horiz_axes = [i for i in range(3) if i != up_axis]

    pos = np.array(center, dtype=float)
    pos[horiz_axes[0]] += R * np.cos(phi) * np.cos(theta)
    pos[horiz_axes[1]] += R * np.cos(phi) * np.sin(theta)
    pos[up_axis] += R * np.sin(phi)

    return pos


def look_at_matrix(eye, target, up):
    """
    Build a 4x4 world-to-camera (view) matrix from eye/target/up vectors.
    Not strictly needed for Open3D (which has its own camera controls),
    but included in case you want to export views to another renderer
    (OpenGL / Three.js / a custom pipeline).
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)

    view = np.eye(4)
    view[0, :3] = s
    view[1, :3] = u
    view[2, :3] = -f
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def load_point_cloud(ply_path: str) -> o3d.geometry.PointCloud:
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Could not find file: {ply_path}")

    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        raise ValueError(
            f"Loaded point cloud from {ply_path} but it has 0 points. "
            "Check the file is a valid .ply point cloud."
        )
    return pcd


def run_interactive(pcd, Y, up_axis, theta0, phi0):
    points = np.asarray(pcd.points)
    center, X, up_axis = compute_centroid_and_radius(points, up_axis)
    R = X + Y

    print(f"Centroid: {center}")
    print(f"Bounding radius X = {X:.4f}")
    print(f"Orbit radius R = X + Y = {R:.4f}")

    up_vec = np.zeros(3)
    up_vec[up_axis] = 1.0

    eye = camera_position_on_circle(center, R, theta0, phi0, up_axis)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Orbit Viewer", width=1280, height=720)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    # Open3D's ViewControl works with front/lookat/up/zoom rather than a raw
    # eye position, so we derive "front" (direction the camera looks) from
    # our computed eye position.
    front = center - eye
    front /= np.linalg.norm(front)

    ctr.set_lookat(center)
    ctr.set_front(front)          # direction FROM camera TO scene
    ctr.set_up(up_vec)
    ctr.set_zoom(0.7)             # tweak to taste; smaller = more zoomed in

    print("Window opened. Use the mouse to orbit/pan/zoom further.")
    print("Press 'q' or close the window to exit.")

    vis.run()
    vis.destroy_window()


def run_capture(pcd, Y, up_axis, phi, frames, out_dir):
    points = np.asarray(pcd.points)
    center, X, up_axis = compute_centroid_and_radius(points, up_axis)
    R = X + Y

    print(f"Centroid: {center}")
    print(f"Bounding radius X = {X:.4f}")
    print(f"Orbit radius R = X + Y = {R:.4f}")

    os.makedirs(out_dir, exist_ok=True)

    up_vec = np.zeros(3)
    up_vec[up_axis] = 1.0

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Capture", width=1280, height=720, visible=False)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_up(up_vec)
    ctr.set_zoom(0.7)

    thetas = np.linspace(0, 2 * np.pi, frames, endpoint=False)

    for i, theta in enumerate(thetas):
        eye = camera_position_on_circle(center, R, theta, phi, up_axis)
        front = center - eye
        front /= np.linalg.norm(front)

        ctr.set_front(front)
        vis.poll_events()
        vis.update_renderer()

        out_path = os.path.join(out_dir, f"frame_{i:03d}.png")
        vis.capture_screen_image(out_path, do_render=True)
        print(f"Saved {out_path}  (theta={np.degrees(theta):.1f} deg)")

    vis.destroy_window()
    print(f"\nDone. {frames} frames written to: {out_dir}")
    print("Combine into a video with e.g.:")
    print(f"  ffmpeg -framerate 24 -i {out_dir}/frame_%03d.png -c:v libx264 -pix_fmt yuv420p orbit.mp4")


# --------------------------------------------------------------------------
# 4. CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Orbit a camera around a point cloud from a .ply file.")
    parser.add_argument("ply_path", help="Path to the input .ply point cloud file.")
    parser.add_argument("--mode", choices=["interactive", "capture"], default="interactive",
                         help="interactive: open a live window. capture: save a sequence of PNG frames.")
    parser.add_argument("--y", type=float, default=1.0,
                         help="Standoff distance Y. Camera orbits at radius R = X + Y.")
    parser.add_argument("--up-axis", type=int, choices=[0, 1, 2], default=1,
                         help="Which axis is 'up': 0=X, 1=Y, 2=Z. Default 1 (Y-up).")
    parser.add_argument("--theta0", type=float, default=0.0,
                         help="[interactive] Starting azimuth angle in degrees.")
    parser.add_argument("--phi0", type=float, default=15.0,
                         help="[interactive] Starting elevation angle in degrees (0=level, 90=top-down).")
    parser.add_argument("--phi", type=float, default=15.0,
                         help="[capture] Elevation angle in degrees used for the whole sweep.")
    parser.add_argument("--frames", type=int, default=60,
                         help="[capture] Number of frames in the full 360-degree sweep.")
    parser.add_argument("--out-dir", default="./orbit_frames",
                         help="[capture] Directory to save PNG frames into.")

    args = parser.parse_args()

    pcd = load_point_cloud(args.ply_path)

    # Optional: estimate normals if missing, so shading looks reasonable.
    if not pcd.has_normals():
        pcd.estimate_normals()

    if args.mode == "interactive":
        run_interactive(
            pcd,
            Y=args.y,
            up_axis=args.up_axis,
            theta0=np.radians(args.theta0),
            phi0=np.radians(args.phi0),
        )
    else:
        run_capture(
            pcd,
            Y=args.y,
            up_axis=args.up_axis,
            phi=np.radians(args.phi),
            frames=args.frames,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()