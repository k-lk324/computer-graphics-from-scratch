import numpy as np
from typing import Tuple


def perspective_project(
    pts: np.ndarray, focal: float, R: np.ndarray, t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D world points to 2D image-plane points and depths.
    Input:
      pts:   (3xN) array of world points
      focal: scalar focal length (in world units)
      R:     (3x3) camera rotation (world -> camera)
      t:     (3,) camera translation (so p_cam = R.p_world + t)
    Output:
      pts_2d: (2xN) array of projected points on camera plane (in same units)
      depth:  (N,) array of Z_c for each point (camera-space depth)
    """
    cam3 = R @ pts + t.reshape(3, 1)
    Xc = cam3[0, :]
    Yc = cam3[1, :]
    Zc = cam3[2, :]

    # Project to image plane
    x_proj = (focal * Xc) / Zc
    y_proj = (focal * Yc) / Zc

    pts_2d = np.vstack([x_proj, y_proj])
    depth = Zc.copy()

    return pts_2d, depth


def lookat(
    eye: np.ndarray, up: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a camera rotation (3x3) and translation (3x1) so that
    any world point p_world maps to p_cam = R.p_world + t.
    The camera "looks at" target from eye, with given up.
    Convention (right-handed): camera's +Z points backwards, so
    z_cam = normalize(eye - target).
    Then x_cam = normalize(up x z_cam), y_cam = z_cam x x_cam.
    R = [x_cam; y_cam; z_cam] (as rows), t = -R.eye.
    Input:
      eye:    (3,) camera position in world
      up:     (3,) "up" direction (not necessarily unit)
      target: (3,) point camera looks at
    Output:
      R: (3x3) rotation
      t: (3,) translation
    """
    eye = eye.reshape(-1)
    up = up.reshape(-1)
    target = target.reshape(-1)

    z_cam = eye.astype(np.float64) - target.astype(np.float64)
    z_cam /= np.linalg.norm(z_cam)

    up_vec = up.astype(np.float64).reshape(-1)
    x_cam = np.cross(up_vec, z_cam)
    x_cam /= np.linalg.norm(x_cam)

    y_cam = np.cross(z_cam, x_cam)

    R = np.vstack([x_cam, y_cam, z_cam])
    t = -R @ eye.reshape(3)

    return R, t.reshape(3)



def rasterize(
    pts_2d: np.ndarray, plane_w: float, plane_h: float, res_w: int, res_h: int
) -> np.ndarray:
    """
    Map 2D points on the camera plane (size = plane_w x plane_h,
    centered at (0,0)) to pixel indices in an image of size res_w x res_h.
    Input:
        pts_2d: (2xN) array of (x', y') in plane coordinates
        plane_w, plane_h: floats, width and height of camera plane
        res_w, res_h: ints, image resolution
    Output:
        pix: (Nx2) integer array of pixel indices (u, v).
    """
    x = pts_2d[0, :]
    y = pts_2d[1, :]

    # Normalize from plane coords to [0..1], then scale to [0..res_w-1] etc.
    u = ((x + (plane_w / 2.0)) / plane_w) * (res_w - 1)
    y = -y  # Flip y-axis
    v = ((y + (plane_h / 2.0)) / plane_h) * (res_h - 1)

    u_int = np.round(u).astype(np.int32)
    v_int = np.round(v).astype(np.int32)

    # Clip into image bounds
    u_clipped = np.clip(u_int, 0, res_w - 1)
    v_clipped = np.clip(v_int, 0, res_h - 1)

    pix = np.vstack([u_clipped, v_clipped]).T
    return pix

