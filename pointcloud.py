import numpy as np
import open3d as o3d

def depth_to_pointcloud(image, depth, fx=500, fy=500, cx=None, cy=None):
    """
    Converts an RGB image and its depth map into a 3D point cloud.
    Very simplistic pinhole camera model approach for demonstration.

    Args:
        image: (H,W,3) numpy array in RGB
        depth: (H,W) numpy array in [0,1] (or meters if you have real scale)
        fx, fy: Focal lengths
        cx, cy: Optical center (defaults to image center)
    Returns:
        open3d.geometry.PointCloud
    """
    h, w, _ = image.shape
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    # Flatten
    zs = depth.flatten()
    xs = (np.arange(w) - cx) / fx
    xs = np.tile(xs, h)
    ys = (np.arange(h) - cy) / fy
    ys = np.repeat(ys, w)

    # Multiply by depth
    xs = xs * zs
    ys = ys * zs

    points = np.stack((xs, ys, zs), axis=-1)  # (H*W, 3)

    # Colors
    colors = image.reshape(-1, 3).astype(np.float32) / 255.0

    # Filter out points with depth == 0 (or near zero)
    valid_mask = zs > 1e-6
    points = points[valid_mask]
    colors = colors[valid_mask]

    # Construct open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd
