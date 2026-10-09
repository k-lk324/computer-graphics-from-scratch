# project3_utils.py

from typing import List, Tuple, Union
import numpy as np
from PIL import Image
from project2_utils import lookat, perspective_project, rasterize


class MatPhong:
    def __init__(self, ka: float, kd: float, ks: float, n: float) -> None:
        """
        Initialize a Phong material with reflection coefficients:
        - ka: ambient coefficient (scalar)
        - kd: diffuse coefficient (scalar)
        - ks: specular coefficient (scalar)
        - n: Phong exponent (shininess)
        """
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.n = n


def light(
    pt: np.ndarray,
    nrm: np.ndarray,
    vclr: np.ndarray,
    cam_pos: np.ndarray,
    mat: MatPhong,
    l_pos: Union[np.ndarray, List[np.ndarray]],
    l_int: Union[np.ndarray, List[np.ndarray]],
    l_amb: np.ndarray,
) -> np.ndarray:
    """
    Calculate the reflected light intensity at point 'pt' on a surface with normal 'nrm',
    color 'vclr', camera position 'cam_pos', material 'mat', given point light(s)
    positions 'l_pos' and intensities 'l_int', and ambient light intensity 'l_amb'.

    Parameters:
    - pt: (3,) point coordinates
    - nrm: (3,) normal vector at point, assumed normalized
    - vclr: (3,) intrinsic color of the surface point (range [0,1])
    - cam_pos: (3,) camera position
    - mat: MatPhong material object
    - l_pos: (N,3) or list of (3,) light positions
    - l_int: (N,3) or list of (3,) light intensities
    - l_amb: (3,) ambient light intensity

    Returns:
    - pt_l: (3,) resulting RGB light intensity at the point
    """
    # Ensure inputs are numpy arrays and shapes
    pt = pt.reshape(3)
    nrm = nrm.reshape(3)
    vclr = vclr.reshape(3)
    cam_pos = cam_pos.reshape(3)
    l_amb = l_amb.reshape(3)

    # Normalize the normal vector to ensure correctness
    nrm_norm = np.linalg.norm(nrm)
    if nrm_norm > 0:
        nrm = nrm / nrm_norm
    else:
        # If zero normal, no lighting can be computed meaningfully
        return np.zeros(3)

    # Calculate view vector from point to camera (normalized)
    v_dir = cam_pos - pt
    v_len = np.linalg.norm(v_dir)
    if v_len > 0:
        v_dir = v_dir / v_len
    else:
        # Camera exactly at point? Avoid division by zero
        v_dir = np.zeros(3)

    # Prepare light sources list as arrays
    # Convert single light inputs to lists for uniform processing
    if isinstance(l_pos, np.ndarray) and l_pos.ndim == 1:
        l_pos = [l_pos]
    if isinstance(l_int, np.ndarray) and l_int.ndim == 1:
        l_int = [l_int]

    # Or if list but items are arrays with shape (3,), convert to single np arrays
    if isinstance(l_pos, list):
        l_pos = np.array(l_pos)
    if isinstance(l_int, list):
        l_int = np.array(l_int)

    # Initialize result with ambient component
    # Ambient light is modulated by ambient coefficient and intrinsic color
    ambient = mat.ka * l_amb

    # Accumulate diffuse and specular components from each light
    diff_spec = np.zeros(3)

    for i in range(l_pos.shape[0]):
        light_position = l_pos[i]
        light_intensity = l_int[i]

        # Calculate light direction vector from point to light (normalized)
        l_dir = light_position - pt
        l_len = np.linalg.norm(l_dir)
        if l_len > 0:
            l_dir = l_dir / l_len
        else:
            # Light at the point - avoid division by zero
            print("Warning: Light at the point, skipping light contribution.")
            continue

        # Diffuse component: max(0, n . l)
        diff_factor = max(0.0, np.dot(nrm, l_dir))

        # Reflection vector r = 2(n.l)n - l
        r = 2 * np.dot(nrm, l_dir) * nrm - l_dir
        r_len = np.linalg.norm(r)
        if r_len > 0:
            r = r / r_len

        # Specular component: max(0, r . v)^n
        spec_factor = max(0.0, np.dot(r, v_dir)) ** mat.n

        # Compute diffuse and specular intensities
        diffuse = mat.kd * diff_factor * light_intensity
        specular = mat.ks * spec_factor * light_intensity

        diff_spec += diffuse + specular

    # Final intensity is ambient + diffuse/specular, modulated by intrinsic color
    pt_l = ambient + diff_spec
    pt_l = pt_l * vclr

    # Clamp result to [0, 1]
    pt_l = np.clip(pt_l, 0.0, 1.0)

    return pt_l


def validate_mesh(pts: np.ndarray, t_pos_idx: np.ndarray) -> None:
    """
    Validate mesh dimensions, integer index types, and vertex index bounds.
    """
    if pts.ndim != 2 or pts.shape[0] != 3:
        raise ValueError(f"pts must have shape (3, Nv), got {pts.shape}")
    if t_pos_idx.ndim != 2 or t_pos_idx.shape[1] != 3:
        raise ValueError(f"t_pos_idx must have shape (Nt, 3), got {t_pos_idx.shape}")
    if not np.issubdtype(t_pos_idx.dtype, np.integer):
        raise ValueError(f"t_pos_idx must have integer dtype, got {t_pos_idx.dtype}")
    if np.any(t_pos_idx < 0):
        raise ValueError("t_pos_idx contains negative indices")
    if np.any(t_pos_idx >= pts.shape[1]):
        raise ValueError(
            f"t_pos_idx contains index out of range for mesh with {pts.shape[1]} vertices"
        )


def calc_normals(pts: np.ndarray, t_pos_idx: np.ndarray) -> np.ndarray:
    """
    Calculate normals per vertex of a triangle mesh.

    Parameters:
    - pts: (3, Nv) array of vertex coordinates
    - t_pos_idx: (Nt, 3) array of triangle vertex indices

    Returns:
    - nrm: (3, Nv) array of normalized vertex normals
    """
    validate_mesh(pts, t_pos_idx)

    num_vertices = pts.shape[1]
    normals_acc = np.zeros((3, num_vertices), dtype=np.float64)

    for triangle in t_pos_idx:
        i0, i1, i2 = triangle
        v0 = pts[:, i0]
        v1 = pts[:, i1]
        v2 = pts[:, i2]

        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)

        norm_len = np.linalg.norm(face_normal)
        if norm_len <= 1e-8:
            continue

        face_normal = face_normal / norm_len

        normals_acc[:, i0] += face_normal
        normals_acc[:, i1] += face_normal
        normals_acc[:, i2] += face_normal

    for v in range(num_vertices):
        norm_len = np.linalg.norm(normals_acc[:, v])
        if norm_len > 1e-8:
            normals_acc[:, v] /= norm_len
        else:
            normals_acc[:, v] = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    return normals_acc



def barycentric_coords(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Union[np.ndarray, None]:
    """Compute barycentric coordinates of point p with respect to triangle abc."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if np.abs(denom) < 1e-8:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=np.float32)


def interpolate_perspective_depth(
    barycentric_coords: np.ndarray,
    vertex_distances: np.ndarray,
) -> float:
    """
    Interpolate reciprocal depth in perspective projection using screen-space barycentric weights.
    d_pixel = 1 / (lambda_0 / d_0 + lambda_1 / d_1 + lambda_2 / d_2)
    """
    inv_depth: float = float(
        barycentric_coords[0] / vertex_distances[0]
        + barycentric_coords[1] / vertex_distances[1]
        + barycentric_coords[2] / vertex_distances[2]
    )
    if inv_depth <= 0.0:
        return float(np.inf)
    return 1.0 / inv_depth


def update_depth_buffer(
    depth_buffer: np.ndarray,
    x: int,
    y: int,
    pixel_depth: float,
) -> bool:
    """
    Test and update depth buffer at coordinate (y, x).
    Returns True if pixel_depth is nearer than depth_buffer[y, x].
    """
    if pixel_depth < depth_buffer[y, x]:
        depth_buffer[y, x] = pixel_depth
        return True
    return False


def shade_gouraud(
    v_pos: np.ndarray,  # (3, 2) projected 2D positions
    v_pos_3d: np.ndarray,  # (3, 3) original 3D vertex positions (world space)
    v_nrm: np.ndarray,  # (3, 3) vertex normals
    v_uvs: np.ndarray,  # (3, 2) texture coordinates
    tex: np.ndarray,  # texture image (HxWx3)
    cam_pos: np.ndarray,  # (3,) camera position
    mat: MatPhong,
    l_pos: Union[np.ndarray, List[np.ndarray]],
    l_int: Union[np.ndarray, List[np.ndarray]],
    l_amb: np.ndarray,
    img: np.ndarray,  # (res_h, res_w, 3)
    v_depth: np.ndarray,  # (3,) vertex distances along camera line of sight
    depth_buffer: np.ndarray,  # (res_h, res_w) depth buffer
) -> np.ndarray:
    res_h, res_w, _ = img.shape
    tex_h, tex_w, _ = tex.shape

    vert_colors = []
    for i in range(3):
        u, v = v_uvs[i]
        u_px = min(int(u * (tex_w - 1)), tex_w - 1)
        v_px = min(int((1 - v) * (tex_h - 1)), tex_h - 1)

        tex_color = tex[v_px, u_px] / 255.0
        nrm = v_nrm[i]
        norm_len = np.linalg.norm(nrm)
        if norm_len > 0:
            nrm = nrm / norm_len

        pt_3d = v_pos_3d[i]

        c = light(
            pt=pt_3d,
            nrm=nrm,
            vclr=tex_color,
            cam_pos=cam_pos,
            mat=mat,
            l_pos=l_pos,
            l_int=l_int,
            l_amb=l_amb,
        )
        vert_colors.append(c)
    vert_colors = np.array(vert_colors)  # shape (3, 3)

    a, b, c = v_pos[0], v_pos[1], v_pos[2]
    x_min = max(int(np.floor(np.min(v_pos[:, 0]))), 0)
    x_max = min(int(np.ceil(np.max(v_pos[:, 0]))), res_w - 1)
    y_min = max(int(np.floor(np.min(v_pos[:, 1]))), 0)
    y_max = min(int(np.ceil(np.max(v_pos[:, 1]))), res_h - 1)

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            epsilon = 1e-4
            bc = barycentric_coords(np.array([x + 0.5, y + 0.5], dtype=np.float32), a, b, c)
            if bc is None or np.any(bc < -epsilon):
                continue

            pixel_depth = interpolate_perspective_depth(bc, v_depth)
            if update_depth_buffer(depth_buffer, x, y, pixel_depth):
                color = bc @ vert_colors
                img[y, x] = color

    return img


def shade_phong(
    v_pos: np.ndarray,  # (3, 2) 2D projected positions
    v_nrm: np.ndarray,  # (3, 3) vertex normals
    v_uvs: np.ndarray,  # (3, 2) UV coords
    tex: np.ndarray,  # (H, W, 3)
    cam_pos: np.ndarray,
    mat: MatPhong,
    l_pos: Union[np.ndarray, List[np.ndarray]],
    l_int: Union[np.ndarray, List[np.ndarray]],
    l_amb: np.ndarray,
    img: np.ndarray,  # (res_h, res_w, 3)
    v_depth: np.ndarray,  # (3,) vertex distances along camera line of sight
    depth_buffer: np.ndarray,  # (res_h, res_w) depth buffer
) -> np.ndarray:
    res_h, res_w, _ = img.shape
    tex_h, tex_w, _ = tex.shape

    a, b, c = v_pos[0], v_pos[1], v_pos[2]
    x_min = max(int(np.floor(np.min(v_pos[:, 0]))), 0)
    x_max = min(int(np.ceil(np.max(v_pos[:, 0]))), res_w - 1)
    y_min = max(int(np.floor(np.min(v_pos[:, 1]))), 0)
    y_max = min(int(np.ceil(np.max(v_pos[:, 1]))), res_h - 1)

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            epsilon = 1e-4
            bc = barycentric_coords(np.array([x + 0.5, y + 0.5], dtype=np.float32), a, b, c)
            if bc is None or np.any(bc < -epsilon):
                continue

            pixel_depth = interpolate_perspective_depth(bc, v_depth)
            if update_depth_buffer(depth_buffer, x, y, pixel_depth):
                interp_nrm = bc @ v_nrm
                norm_len = np.linalg.norm(interp_nrm)
                if norm_len > 0:
                    interp_nrm /= norm_len

                interp_uv = bc @ v_uvs
                u, v = interp_uv
                u_px = min(int(u * (tex_w - 1)), tex_w - 1)
                v_px = min(int((1 - v) * (tex_h - 1)), tex_h - 1)
                tex_color = tex[v_px, u_px] / 255.0

                color = light(
                    pt=np.array([0.0, 0.0, 0.0]),
                    nrm=interp_nrm,
                    vclr=tex_color,
                    cam_pos=cam_pos,
                    mat=mat,
                    l_pos=l_pos,
                    l_int=l_int,
                    l_amb=l_amb,
                )
                img[y, x] = np.clip(color, 0, 1)

    return img


def render_object(
    v_pos: np.ndarray,  # 3 x Nv (3D vertex positions)
    v_uvs: np.ndarray,  # 2 x Nv
    t_pos_idx: np.ndarray,  # Nt x 3, 0-based indices
    tex: np.ndarray,  # H x W x 3 texture image
    plane_h: int,
    plane_w: int,
    res_h: int,
    res_w: int,
    focal: float,
    eye: np.ndarray,
    up: np.ndarray,
    target: np.ndarray,
    mat,
    l_pos: Union[np.ndarray, list],
    l_int: Union[np.ndarray, list],
    l_amb: np.ndarray,
    shader: str,
) -> np.ndarray:
    """
    Render a 3D object using a Z-buffer for depth testing.
    """
    # Compute camera basis
    z_axis = (eye - target).reshape(3)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up.reshape(3), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=0)  # 3x3
    t = -R @ eye.reshape(3)  # 3,

    # Project vertices
    verts2d, depth = perspective_project(v_pos, focal, R, t)  # 2 x Nv, (Nv,)
    verts2d = verts2d.T  # Nv x 2
    depth = depth.T  # Nv

    # Convert to pixel coords
    scale_x = res_w / plane_w
    scale_y = res_h / plane_h
    verts2d[:, 0] = (verts2d[:, 0] + plane_w / 2) * scale_x
    verts2d[:, 1] = (plane_h / 2 - verts2d[:, 1]) * scale_y
    verts2d = np.round(verts2d).astype(int)

    # Clamp to image bounds
    verts2d[:, 0] = np.clip(verts2d[:, 0], 0, res_w - 1)
    verts2d[:, 1] = np.clip(verts2d[:, 1], 0, res_h - 1)

    validate_mesh(v_pos, t_pos_idx)

    # Compute normals per vertex
    normals = calc_normals(v_pos, t_pos_idx)  # 3 x Nv

    # Initialize depth buffer with infinity
    depth_buffer = np.full((res_h, res_w), np.inf, dtype=np.float64)

    # Positive distance: d = -Z_c for negative-Z camera convention
    vertex_distances = -depth if np.all(depth < 0) else np.abs(depth)

    # Initialize image with white background
    img = np.ones((res_h, res_w, 3), dtype=np.float32)

    for triangle in t_pos_idx:
        idx = triangle  # 3 vertex indices of this triangle

        # Projected 2D positions for rasterization
        v_proj = verts2d[idx, :]  # 3 x 2 (integer pixel coords)

        # Normals of the 3 vertices (3 x 3)
        v_nrm = normals[:, idx].T  # shape: (3 vertices, 3 components)

        # Texture UV coords of the 3 vertices (3 x 2)
        v_uv = v_uvs[idx]  # shape: (3 vertices, 2 components)

        # Original 3D vertex positions of the triangle (3 x 3)
        v_pos_3d = v_pos[:, idx].T  # shape: (3 vertices, 3 components)

        # Vertex distances for depth interpolation (3,)
        v_depth = vertex_distances[idx]

        if shader == "gouraud":
            img = shade_gouraud(
                v_proj,
                v_pos_3d,
                v_nrm,
                v_uv,
                tex,
                eye.reshape(3),
                mat,
                l_pos,
                l_int,
                l_amb,
                img,
                v_depth,
                depth_buffer,
            )
        elif shader == "phong":
            img = shade_phong(
                v_proj,
                v_nrm,
                v_uv,
                tex,
                eye.reshape(3),
                mat,
                l_pos,
                l_int,
                l_amb,
                img,
                v_depth,
                depth_buffer,
            )
        else:
            raise ValueError(f"Unknown shader: {shader}")

    return img

