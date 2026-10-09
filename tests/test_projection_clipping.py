import numpy as np
import pytest
from project3_utils import (
    MatPhong,
    render_object,
    shade_gouraud,
    shade_phong,
)


def _create_test_scene():
    texture: np.ndarray = np.full((16, 16, 3), 200, dtype=np.uint8)
    eye: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    target: np.ndarray = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    up: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    mat: MatPhong = MatPhong(ka=0.5, kd=0.5, ks=0.0, n=1.0)
    l_pos = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
    l_int = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
    l_amb: np.ndarray = np.array([0.2, 0.2, 0.2], dtype=np.float64)
    return texture, eye, target, up, mat, l_pos, l_int, l_amb


def test_triangle_entirely_inside_image() -> None:
    texture, eye, target, up, mat, l_pos, l_int, l_amb = _create_test_scene()
    # Triangle comfortably inside image at Z = -2
    v_pos: np.ndarray = np.array(
        [
            [-0.2, 0.2, 0.0],
            [-0.2, -0.2, 0.2],
            [-2.0, -2.0, -2.0],
        ],
        dtype=np.float64,
    )
    v_uvs: np.ndarray = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64)
    triangles: np.ndarray = np.array([[0, 1, 2]], dtype=np.int32)

    image = render_object(
        v_pos=v_pos,
        v_uvs=v_uvs,
        t_pos_idx=triangles,
        tex=texture,
        plane_h=2.0,
        plane_w=2.0,
        res_h=64,
        res_w=64,
        focal=1.0,
        eye=eye,
        up=up,
        target=target,
        mat=mat,
        l_pos=l_pos,
        l_int=l_int,
        l_amb=l_amb,
        shader="gouraud",
    )

    # Initial image background is white (1.0, 1.0, 1.0). Rendered pixels differ from background
    filled_pixels = np.any(image < 0.99, axis=2)
    assert np.any(filled_pixels)
    assert np.all(np.isfinite(image))


def test_triangle_partially_outside_image() -> None:
    texture, eye, target, up, mat, l_pos, l_int, l_amb = _create_test_scene()
    # Left vertex extends far to the left (X = -5.0) which projects off-screen (plane_w = 2.0)
    v_pos: np.ndarray = np.array(
        [
            [-5.0, 0.3, 0.0],
            [-0.3, -0.3, 0.3],
            [-2.0, -2.0, -2.0],
        ],
        dtype=np.float64,
    )
    v_uvs: np.ndarray = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64)
    triangles: np.ndarray = np.array([[0, 1, 2]], dtype=np.int32)

    image = render_object(
        v_pos=v_pos,
        v_uvs=v_uvs,
        t_pos_idx=triangles,
        tex=texture,
        plane_h=2.0,
        plane_w=2.0,
        res_h=64,
        res_w=64,
        focal=1.0,
        eye=eye,
        up=up,
        target=target,
        mat=mat,
        l_pos=l_pos,
        l_int=l_int,
        l_amb=l_amb,
        shader="gouraud",
    )

    # Some pixels must be filled, but not outside bounds or producing NaNs
    filled_pixels = np.any(image < 0.99, axis=2)
    assert np.any(filled_pixels)
    assert np.all(np.isfinite(image))


def test_triangle_entirely_outside_image() -> None:
    texture, eye, target, up, mat, l_pos, l_int, l_amb = _create_test_scene()
    # Entire triangle shifted to X = [10.0, 11.0, 10.5] far off the camera plane
    v_pos: np.ndarray = np.array(
        [
            [10.0, 11.0, 10.5],
            [-0.2, -0.2, 0.2],
            [-2.0, -2.0, -2.0],
        ],
        dtype=np.float64,
    )
    v_uvs: np.ndarray = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64)
    triangles: np.ndarray = np.array([[0, 1, 2]], dtype=np.int32)

    image = render_object(
        v_pos=v_pos,
        v_uvs=v_uvs,
        t_pos_idx=triangles,
        tex=texture,
        plane_h=2.0,
        plane_w=2.0,
        res_h=64,
        res_w=64,
        focal=1.0,
        eye=eye,
        up=up,
        target=target,
        mat=mat,
        l_pos=l_pos,
        l_int=l_int,
        l_amb=l_amb,
        shader="gouraud",
    )

    # No pixels should be modified (remains pure white background)
    assert np.allclose(image, 1.0)


def test_triangle_entirely_behind_camera() -> None:
    texture, eye, target, up, mat, l_pos, l_int, l_amb = _create_test_scene()
    # Vertices behind camera: camera looks along -Z, so Z = +2 is behind the camera
    v_pos: np.ndarray = np.array(
        [
            [-0.2, 0.2, 0.0],
            [-0.2, -0.2, 0.2],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )
    v_uvs: np.ndarray = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64)
    triangles: np.ndarray = np.array([[0, 1, 2]], dtype=np.int32)

    image = render_object(
        v_pos=v_pos,
        v_uvs=v_uvs,
        t_pos_idx=triangles,
        tex=texture,
        plane_h=2.0,
        plane_w=2.0,
        res_h=64,
        res_w=64,
        focal=1.0,
        eye=eye,
        up=up,
        target=target,
        mat=mat,
        l_pos=l_pos,
        l_int=l_int,
        l_amb=l_amb,
        shader="gouraud",
    )

    # Behind camera: must be rejected, no inverted projection
    assert np.allclose(image, 1.0)


def test_triangle_crossing_near_plane() -> None:
    texture, eye, target, up, mat, l_pos, l_int, l_amb = _create_test_scene()
    # Triangle straddling near plane (Z = +1.0 behind, Z = -2.0 in front)
    v_pos: np.ndarray = np.array(
        [
            [-0.5, 0.5, 0.0],
            [-0.5, -0.5, 0.5],
            [1.0, -2.0, -2.0],
        ],
        dtype=np.float64,
    )
    v_uvs: np.ndarray = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64)
    triangles: np.ndarray = np.array([[0, 1, 2]], dtype=np.int32)

    image = render_object(
        v_pos=v_pos,
        v_uvs=v_uvs,
        t_pos_idx=triangles,
        tex=texture,
        plane_h=2.0,
        plane_w=2.0,
        res_h=64,
        res_w=64,
        focal=1.0,
        eye=eye,
        up=up,
        target=target,
        mat=mat,
        l_pos=l_pos,
        l_int=l_int,
        l_amb=l_amb,
        shader="gouraud",
    )

    # Must execute safely without NaNs or infinite values
    assert np.all(np.isfinite(image))
    assert not np.any(np.isnan(image))
