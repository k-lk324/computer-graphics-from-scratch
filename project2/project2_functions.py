import numpy as np
import cv2
from typing import Tuple
from triangle_fill import render_img

# PART A


def translate(t_vec: np.ndarray) -> np.ndarray:
    """
    Create a 4x4 affine transformation matrix for translation by t_vec.
    Input:
        t_vec: (3,) array [t_x, t_y, t_z]
    Output:
        xform: (4x4) matrix
    """
    xform = np.eye(4, dtype=np.float64)
    xform[0:3, 3] = t_vec.reshape(3)
    return xform


def rotate(axis: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray:
    """
    Create a 4x4 affine rotation matrix that rotates by 'angle' radians
    around the line passing through 'center' in the direction of 'axis'.
    Rotation is clockwise when looking along 'axis'.
    Input:
      axis:   (3,) axis vector
      angle:  scalar angle in radians (clockwise)
      center: (3,) point through which the rotation axis passes
    Output:
      xform: (4x4) affine matrix
    """
    u = axis.astype(np.float64)
    u = u / np.linalg.norm(u)

    ux, uy, uz = u
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    one_minus = 1.0 - cos_t

    # Skew-symmetric cross-product matrix of u
    u_cross = np.array(
        [[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]], dtype=np.float64
    )

    # Outer product u u^T
    u_outer = np.outer(u, u)

    # Rodrigues' rotation formula
    R3 = np.eye(3) * cos_t + one_minus * u_outer + sin_t * u_cross

    # Embed R3 in a 4x4 matrix
    R4 = np.eye(4, dtype=np.float64)
    R4[0:3, 0:3] = R3

    # Translation to/from center
    T_neg = np.eye(4, dtype=np.float64)
    T_neg[0:3, 3] = -center.reshape(3)

    T_pos = np.eye(4, dtype=np.float64)
    T_pos[0:3, 3] = center.reshape(3)

    # Compose: translate(+center) * R4 * translate(-center)
    xform = T_pos @ R4 @ T_neg
    return xform


def compose(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    """
    Compose two 4x4 affine transforms: equivalent to first applying mat2,
    then mat1. The returned matrix x satisfies:
        x . [p_hom] = mat1 . (mat2 . [p_hom]).
    Input:
      mat1, mat2: each (4x4)
    Output:
      mat = mat1 @ mat2
    """
    return mat1 @ mat2


# PART B
def world2view(pts: np.ndarray, R: np.ndarray, c0: np.ndarray) -> np.ndarray:
    """
    Transform 3D points from world frame to camera frame.
    Input:
      pts: (3xN) array of points in world coordinates
      R:   (3x3) rotation of camera frame w.r.t. world
      c0:  (3,) camera reference point in world
    Output:
      out: (Nx3) array of pts in camera frame (non-homogeneous)
    """
    shifted = pts - c0.reshape(3, 1)
    cam_pts = R @ shifted
    return cam_pts.T


# PART C


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
    z_cam = eye.astype(np.float64) - target.astype(np.float64)
    z_cam /= np.linalg.norm(z_cam)

    up_vec = up.astype(np.float64)
    x_cam = np.cross(up_vec, z_cam)
    x_cam /= np.linalg.norm(x_cam)

    y_cam = np.cross(z_cam, x_cam)

    R = np.vstack([x_cam, y_cam, z_cam])
    t = -R @ eye.reshape(3)

    return R, t.reshape(3)


# PART D


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


# PART E


def rasterize(
    pts_2d: np.ndarray, plane_w: float, plane_h: float, res_w: int, res_h: int
) -> np.ndarray:
    """
    Map 2D points on the camera plane (size = plane_w x plane_h,
    centered at (0,0)) to pixel coordinates in an image of size res_w x res_h.
    Input:
        pts_2d: (2xN) array of (x', y') in plane coordinates
        plane_w, plane_h: floats, width and height of camera plane
        res_w, res_h: ints, image resolution
    Output:
        pix: (Nx2) floating-point array of continuous pixel coordinates (u, v).
    """
    x = pts_2d[0, :]
    y = pts_2d[1, :]

    u = ((x + (plane_w / 2.0)) / plane_w) * (res_w - 1)
    v = (((plane_h / 2.0) - y) / plane_h) * (res_h - 1)

    pix = np.vstack([u, v]).T
    return pix


# PART F


def render_object(
    v_pos: np.ndarray,
    v_clr: np.ndarray,
    t_pos_idx: np.ndarray,
    v_uvs: np.ndarray,
    plane_h: float,
    plane_w: float,
    res_h: int,
    res_w: int,
    focal: float,
    eye: np.ndarray,
    up: np.ndarray,
    target: np.ndarray,
    diffuse_path: str = "stone-72_diffuse.jpg",
) -> np.ndarray:
    """
    Render a 3D object as a 2D image from a pinhole camera viewpoint,
    using texture shading (flat=t).
    Input:
        v_pos:     (Nx3) array of vertex positions in world coords
        v_clr:     (Nx3) array of per-vertex colors (unused for texture)
        t_pos_idx: (Fx3) integer array of triangle indices
        v_uvs:     (Nx2) array of UV coords (in [0..1])
        plane_h:   float, camera-plane height (world units)
        plane_w:   float, camera-plane width (world units)
        res_h:     int, image height (pixels)
        res_w:     int, image width (pixels)
        focal:     float, focal length (world units)
        eye:       (3,) camera position
        up:        (3,) up vector
        target:    (3,) camera-look-at point
        diffuse_path:  str, path to texture image
    Output:
        img: (res_h x res_w x 3) RGB image (uint8 or float32)
    """
    R_cam, t_cam = lookat(eye, up, target)
    pts_world = v_pos.T
    pts_2d_plane, depth = perspective_project(pts_world, focal, R_cam, t_cam)

    # Rasterize to pixel coords
    verts_px = rasterize(pts_2d_plane, plane_w, plane_h, res_w, res_h)

    # Load the texture image (diffuse)
    texture = cv2.imread(diffuse_path, cv2.IMREAD_COLOR)
    if texture is None:
        raise FileNotFoundError(f"Could not load texture at '{diffuse_path}'")
    # Convert to RGB float [0..255]
    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
    texture = texture.astype(np.float32)

    # Call the provided render_img routine
    # render_img returns a (512x512x3) image (white background = 255).
    # If res_h/res_w != 512, raise error.
    if (res_h != 512) or (res_w != 512):
        raise ValueError("render_img is implemented for 512x512 only.")

    faces = t_pos_idx.astype(np.int32)
    vertices_2d = verts_px.astype(np.int32)
    vcolors = v_clr.astype(np.float32)
    uvs = v_uvs.astype(np.float32)
    depth_vals = depth.astype(np.float32)

    # Always use texture shading 't'
    shaded_img = render_img(
        faces=faces,
        vertices=vertices_2d,
        vcolors=vcolors,
        uvs=uvs,
        depth=depth_vals,
        shading="t",
        textImg=texture,
    )
    # render_img returns a float32 image with values [0..255].
    # Convert to uint8
    out_img = np.clip(shaded_img, 0, 255).astype(np.uint8)

    return out_img
