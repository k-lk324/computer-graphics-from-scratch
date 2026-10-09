import numpy as np
import pytest
from project3_utils import (
    interpolate_perspective_depth,
    update_depth_buffer,
    render_object,
    MatPhong,
)


def test_interpolate_perspective_depth() -> None:
    # Equal barycentric weights at center: 1/3 each
    barycentric_coords: np.ndarray = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
    # Vertex distances: all at distance 3.0
    vertex_distances: np.ndarray = np.array([3.0, 3.0, 3.0], dtype=np.float64)
    interpolated: float = interpolate_perspective_depth(barycentric_coords, vertex_distances)
    assert np.isclose(interpolated, 3.0)

    # Varying distances: d0=2, d1=4, d2=4, weights=[0.5, 0.25, 0.25]
    # 1 / (0.5/2 + 0.25/4 + 0.25/4) = 1 / (0.25 + 0.0625 + 0.0625) = 1 / 0.375 = 2.6666...
    varying_distances: np.ndarray = np.array([2.0, 4.0, 4.0], dtype=np.float64)
    weights: np.ndarray = np.array([0.5, 0.25, 0.25], dtype=np.float64)
    expected_depth: float = 1.0 / (0.5 / 2.0 + 0.25 / 4.0 + 0.25 / 4.0)
    assert np.isclose(interpolate_perspective_depth(weights, varying_distances), expected_depth)


def test_update_depth_buffer_logic() -> None:
    depth_buffer: np.ndarray = np.full((10, 10), np.inf, dtype=np.float64)

    # First write at depth 5.0 succeeds
    first_update: bool = update_depth_buffer(depth_buffer, 2, 3, 5.0)
    assert first_update is True
    assert depth_buffer[3, 2] == 5.0

    # Farther write at depth 8.0 fails
    farther_update: bool = update_depth_buffer(depth_buffer, 2, 3, 8.0)
    assert farther_update is False
    assert depth_buffer[3, 2] == 5.0

    # Nearer write at depth 2.0 succeeds and updates buffer
    nearer_update: bool = update_depth_buffer(depth_buffer, 2, 3, 2.0)
    assert nearer_update is True
    assert depth_buffer[3, 2] == 2.0


def test_z_buffer_submission_order_invariance() -> None:
    # 6 vertices forming two identical XY triangles at different depths along Z
    # Camera at origin looking towards -Z.
    # Triangle Near: Z = -2 (distance d = 2)
    # Triangle Far:  Z = -5 (distance d = 5)
    v_pos: np.ndarray = np.array(
        [
            [-0.5, 0.5, 0.0, -0.5, 0.5, 0.0],
            [-0.5, -0.5, 0.5, -0.5, -0.5, 0.5],
            [-2.0, -2.0, -2.0, -5.0, -5.0, -5.0],
        ],
        dtype=np.float64,
    )
    v_uvs: np.ndarray = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ],
        dtype=np.float64,
    )
    # Order 1: Near (0,1,2), then Far (3,4,5)
    triangles_near_first: np.ndarray = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    # Order 2: Far (3,4,5), then Near (0,1,2)
    triangles_far_first: np.ndarray = np.array([[3, 4, 5], [0, 1, 2]], dtype=np.int32)

    # 100x100 texture where top-left is green and bottom-right is red
    texture: np.ndarray = np.full((100, 100, 3), 128, dtype=np.uint8)

    eye: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    target: np.ndarray = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    up: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    material: MatPhong = MatPhong(ka=0.5, kd=0.5, ks=0.0, n=1.0)

    l_pos = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
    l_int = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
    l_amb = np.array([0.2, 0.2, 0.2], dtype=np.float64)

    image_near_first: np.ndarray = render_object(
        v_pos=v_pos,
        v_uvs=v_uvs,
        t_pos_idx=triangles_near_first,
        tex=texture,
        plane_h=2.0,
        plane_w=2.0,
        res_h=64,
        res_w=64,
        focal=1.0,
        eye=eye,
        up=up,
        target=target,
        mat=material,
        l_pos=l_pos,
        l_int=l_int,
        l_amb=l_amb,
        shader="gouraud",
    )

    image_far_first: np.ndarray = render_object(
        v_pos=v_pos,
        v_uvs=v_uvs,
        t_pos_idx=triangles_far_first,
        tex=texture,
        plane_h=2.0,
        plane_w=2.0,
        res_h=64,
        res_w=64,
        focal=1.0,
        eye=eye,
        up=up,
        target=target,
        mat=material,
        l_pos=l_pos,
        l_int=l_int,
        l_amb=l_amb,
        shader="gouraud",
    )

    assert np.allclose(image_near_first, image_far_first)


def test_partially_overlapping_triangles() -> None:
    # Triangle A (Near): Z = -2, X in [-0.5, 0.2]
    # Triangle B (Far):  Z = -4, X in [-0.2, 0.5]
    v_pos: np.ndarray = np.array(
        [
            [-0.5, 0.2, -0.15, -0.2, 0.5, 0.15],
            [-0.5, -0.5, 0.5, -0.5, -0.5, 0.5],
            [-2.0, -2.0, -2.0, -4.0, -4.0, -4.0],
        ],
        dtype=np.float64,
    )
    v_uvs: np.array = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ],
        dtype=np.float64,
    )
    triangles_a_then_b: np.ndarray = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    triangles_b_then_a: np.ndarray = np.array([[3, 4, 5], [0, 1, 2]], dtype=np.int32)

    texture: np.ndarray = np.full((100, 100, 3), 128, dtype=np.uint8)
    eye: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    target: np.ndarray = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    up: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    material: MatPhong = MatPhong(ka=0.5, kd=0.5, ks=0.0, n=1.0)
    l_pos = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
    l_int = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
    l_amb = np.array([0.2, 0.2, 0.2], dtype=np.float64)

    image_ab: np.ndarray = render_object(
        v_pos=v_pos,
        v_uvs=v_uvs,
        t_pos_idx=triangles_a_then_b,
        tex=texture,
        plane_h=2.0,
        plane_w=2.0,
        res_h=64,
        res_w=64,
        focal=1.0,
        eye=eye,
        up=up,
        target=target,
        mat=material,
        l_pos=l_pos,
        l_int=l_int,
        l_amb=l_amb,
        shader="gouraud",
    )

    image_ba: np.ndarray = render_object(
        v_pos=v_pos,
        v_uvs=v_uvs,
        t_pos_idx=triangles_b_then_a,
        tex=texture,
        plane_h=2.0,
        plane_w=2.0,
        res_h=64,
        res_w=64,
        focal=1.0,
        eye=eye,
        up=up,
        target=target,
        mat=material,
        l_pos=l_pos,
        l_int=l_int,
        l_amb=l_amb,
        shader="gouraud",
    )

    # Acceptance criterion: Triangle submission order does not change the visible result
    assert np.allclose(image_ab, image_ba)
