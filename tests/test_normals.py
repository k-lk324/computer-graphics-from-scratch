import os
import numpy as np
import pytest
from project3_utils import calc_normals


def test_square_xy_plane_normals() -> None:
    # 4 vertices of a unit square in the XY plane
    vertex_positions: np.ndarray = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    # 2 counterclockwise triangles
    triangle_indices: np.ndarray = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=np.int32,
    )

    normals: np.ndarray = calc_normals(vertex_positions, triangle_indices)

    expected_normal: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    assert normals.shape == (3, 4)
    for vertex_index in range(4):
        assert np.allclose(normals[:, vertex_index], expected_normal)


def test_zero_based_indices_accepted() -> None:
    vertex_positions: np.ndarray = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangle_indices: np.ndarray = np.array([[0, 1, 2]], dtype=np.int32)
    normals: np.ndarray = calc_normals(vertex_positions, triangle_indices)
    assert normals.shape == (3, 3)
    assert np.all(np.isfinite(normals))


def test_out_of_range_indices_rejected() -> None:
    vertex_positions: np.ndarray = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    # Upper bound violation (index 3 for 3 vertices)
    out_of_bounds_indices: np.ndarray = np.array([[0, 1, 3]], dtype=np.int32)
    with pytest.raises(ValueError):
        calc_normals(vertex_positions, out_of_bounds_indices)

    # Negative index violation
    negative_indices: np.ndarray = np.array([[0, -1, 2]], dtype=np.int32)
    with pytest.raises(ValueError):
        calc_normals(vertex_positions, negative_indices)


def test_degenerate_triangles_produce_finite_normals() -> None:
    vertex_positions: np.ndarray = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    # Collinear / duplicate vertices forming degenerate triangles
    degenerate_indices: np.ndarray = np.array(
        [
            [0, 1, 2],  # collinear
            [0, 0, 0],  # identical vertices
        ],
        dtype=np.int32,
    )

    normals: np.ndarray = calc_normals(vertex_positions, degenerate_indices)

    assert normals.shape == (3, 3)
    assert not np.any(np.isnan(normals))
    assert np.all(np.isfinite(normals))


def test_supplied_mesh_processes_all_triangles() -> None:
    test_dir: str = os.path.dirname(os.path.abspath(__file__))
    mesh_path: str = os.path.join(test_dir, "..", "assets", "project3", "hw3.npy")
    mesh_data = np.load(mesh_path, allow_pickle=True).item()

    vertex_positions: np.ndarray = mesh_data["v_pos"]
    triangle_indices: np.ndarray = mesh_data["t_pos_idx"]

    assert vertex_positions.shape == (3, 1984)
    assert triangle_indices.shape == (960, 3)
    assert int(triangle_indices.min()) == 0
    assert int(triangle_indices.max()) == 1983

    normals: np.ndarray = calc_normals(vertex_positions, triangle_indices)

    assert normals.shape == (3, 1984)
    assert not np.any(np.isnan(normals))
    assert np.all(np.isfinite(normals))

    normal_lengths: np.ndarray = np.linalg.norm(normals, axis=0)
    assert np.allclose(normal_lengths, 1.0, atol=1e-5)
