import numpy as np
import pytest
from project3_utils import (
    MatPhong,
    sample_texture,
    shade_gouraud,
    shade_phong,
)


def test_sample_texture_orientation() -> None:
    # 2x2 texture with distinct corner colors
    # tex[0, 0] (top-left, u=0, v=1): Red
    # tex[0, 1] (top-right, u=1, v=1): Green
    # tex[1, 0] (bottom-left, u=0, v=0): Blue
    # tex[1, 1] (bottom-right, u=1, v=0): White
    texture: np.ndarray = np.zeros((2, 2, 3), dtype=np.uint8)
    texture[0, 0] = [255, 0, 0]
    texture[0, 1] = [0, 255, 0]
    texture[1, 0] = [0, 0, 255]
    texture[1, 1] = [255, 255, 255]

    top_left = sample_texture(texture, 0.0, 1.0)
    top_right = sample_texture(texture, 1.0, 1.0)
    bottom_left = sample_texture(texture, 0.0, 0.0)
    bottom_right = sample_texture(texture, 1.0, 0.0)

    assert np.allclose(top_left, [1.0, 0.0, 0.0])
    assert np.allclose(top_right, [0.0, 1.0, 0.0])
    assert np.allclose(bottom_left, [0.0, 0.0, 1.0])
    assert np.allclose(bottom_right, [1.0, 1.0, 1.0])


def test_lighting_changes_with_surface_position() -> None:
    # 2D projected triangle covering pixel (10, 10)
    v_pos: np.ndarray = np.array([[5.0, 5.0], [20.0, 5.0], [5.0, 20.0]], dtype=np.float64)
    v_nrm: np.ndarray = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    v_uvs: np.ndarray = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    v_depth: np.ndarray = np.array([2.0, 2.0, 2.0], dtype=np.float64)

    # Position A: close to point light
    v_pos_3d_near: np.ndarray = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]], dtype=np.float64)
    # Position B: far from point light
    v_pos_3d_far: np.ndarray = np.array([[10.0, 10.0, -1.0], [11.0, 10.0, -1.0], [10.0, 11.0, -1.0]], dtype=np.float64)

    texture: np.ndarray = np.full((16, 16, 3), 255, dtype=np.uint8)
    cam_pos: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    mat: MatPhong = MatPhong(ka=0.0, kd=1.0, ks=0.0, n=1.0)
    l_pos = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
    l_int = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
    l_amb: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    img_near: np.ndarray = np.zeros((32, 32, 3), dtype=np.float32)
    depth_buf_near: np.ndarray = np.full((32, 32), np.inf, dtype=np.float64)
    res_near = shade_phong(
        v_pos, v_pos_3d_near, v_nrm, v_uvs, texture, cam_pos, mat, l_pos, l_int, l_amb, img_near, v_depth, depth_buf_near
    )

    img_far: np.ndarray = np.zeros((32, 32, 3), dtype=np.float32)
    depth_buf_far: np.ndarray = np.full((32, 32), np.inf, dtype=np.float64)
    res_far = shade_phong(
        v_pos, v_pos_3d_far, v_nrm, v_uvs, texture, cam_pos, mat, l_pos, l_int, l_amb, img_far, v_depth, depth_buf_far
    )

    # Lighting at pixel (10, 10) must differ because light direction depends on 3D surface position
    assert not np.allclose(res_near[10, 10], res_far[10, 10])


def test_gouraud_and_phong_sample_same_texture_orientation() -> None:
    # Triangle with texture gradient
    v_pos: np.ndarray = np.array([[5.0, 5.0], [25.0, 5.0], [5.0, 25.0]], dtype=np.float64)
    v_pos_3d: np.ndarray = np.array([[0.0, 0.0, -2.0], [1.0, 0.0, -2.0], [0.0, 1.0, -2.0]], dtype=np.float64)
    v_nrm: np.ndarray = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    v_uvs: np.ndarray = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    v_depth: np.ndarray = np.array([2.0, 2.0, 2.0], dtype=np.float64)

    # Gradient texture: horizontal red ramp, vertical green ramp
    texture: np.ndarray = np.zeros((32, 32, 3), dtype=np.uint8)
    for r in range(32):
        for c in range(32):
            texture[r, c] = [int(c * 255 / 31), int(r * 255 / 31), 100]

    cam_pos: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    # Pure ambient material so lighting doesn't introduce differences
    mat: MatPhong = MatPhong(ka=1.0, kd=0.0, ks=0.0, n=1.0)
    l_pos = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
    l_int = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
    l_amb: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    img_g: np.ndarray = np.zeros((32, 32, 3), dtype=np.float32)
    depth_g: np.ndarray = np.full((32, 32), np.inf, dtype=np.float64)
    res_g = shade_gouraud(
        v_pos, v_pos_3d, v_nrm, v_uvs, texture, cam_pos, mat, l_pos, l_int, l_amb, img_g, v_depth, depth_g
    )

    img_p: np.ndarray = np.zeros((32, 32, 3), dtype=np.float32)
    depth_p: np.ndarray = np.full((32, 32), np.inf, dtype=np.float64)
    res_p = shade_phong(
        v_pos, v_pos_3d, v_nrm, v_uvs, texture, cam_pos, mat, l_pos, l_int, l_amb, img_p, v_depth, depth_p
    )

    # At vertices, colors sampled by both shaders must be identical
    # Check near top-left vertex (pixel 6, 6)
    assert np.allclose(res_g[6, 6], res_p[6, 6], atol=0.05)


def test_constant_normals_uniform_material() -> None:
    v_pos: np.ndarray = np.array([[5.0, 5.0], [25.0, 5.0], [5.0, 25.0]], dtype=np.float64)
    v_pos_3d: np.ndarray = np.array([[0.0, 0.0, -10.0], [1.0, 0.0, -10.0], [0.0, 1.0, -10.0]], dtype=np.float64)
    v_nrm: np.ndarray = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    v_uvs: np.ndarray = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    v_depth: np.ndarray = np.array([10.0, 10.0, 10.0], dtype=np.float64)

    # Uniform texture
    texture: np.ndarray = np.full((16, 16, 3), 200, dtype=np.uint8)
    cam_pos: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    mat: MatPhong = MatPhong(ka=0.3, kd=0.7, ks=0.0, n=1.0)
    # Distant light source directly along Z
    l_pos = [np.array([0.5, 0.5, 0.0], dtype=np.float64)]
    l_int = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
    l_amb: np.ndarray = np.array([0.2, 0.2, 0.2], dtype=np.float64)

    img_p: np.ndarray = np.zeros((32, 32, 3), dtype=np.float32)
    depth_p: np.ndarray = np.full((32, 32), np.inf, dtype=np.float64)
    res_p = shade_phong(
        v_pos, v_pos_3d, v_nrm, v_uvs, texture, cam_pos, mat, l_pos, l_int, l_amb, img_p, v_depth, depth_p
    )

    # Pixels inside the triangle must be non-zero and finite
    inside_mask = res_p.sum(axis=2) > 0
    assert np.any(inside_mask)
    assert np.all(np.isfinite(res_p))


def test_degenerate_triangles_handled_gracefully() -> None:
    # Degenerate collinear 2D triangle
    v_pos: np.ndarray = np.array([[5.0, 5.0], [10.0, 10.0], [15.0, 15.0]], dtype=np.float64)
    v_pos_3d: np.ndarray = np.array([[0.0, 0.0, -1.0], [1.0, 1.0, -1.0], [2.0, 2.0, -1.0]], dtype=np.float64)
    v_nrm: np.ndarray = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    v_uvs: np.ndarray = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
    v_depth: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    texture: np.ndarray = np.full((16, 16, 3), 128, dtype=np.uint8)
    cam_pos: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    mat: MatPhong = MatPhong(ka=0.2, kd=0.8, ks=0.1, n=5.0)
    l_pos = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
    l_int = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
    l_amb: np.ndarray = np.array([0.1, 0.1, 0.1], dtype=np.float64)

    img: np.ndarray = np.zeros((32, 32, 3), dtype=np.float32)
    depth_buf: np.ndarray = np.full((32, 32), np.inf, dtype=np.float64)

    # Should run without crashing and output finite numbers
    res = shade_phong(
        v_pos, v_pos_3d, v_nrm, v_uvs, texture, cam_pos, mat, l_pos, l_int, l_amb, img, v_depth, depth_buf
    )
    assert np.all(np.isfinite(res))
    assert not np.any(np.isnan(res))
