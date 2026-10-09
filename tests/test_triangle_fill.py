import numpy as np
from triangle_fill import edge_function, vector_interp


def test_vector_interp_midpoint() -> None:
    segment_start: np.ndarray = np.array([0.0, 0.0], dtype=np.float64)
    segment_end: np.ndarray = np.array([10.0, 0.0], dtype=np.float64)
    color_start: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    color_end: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    target_coord: float = 5.0
    interpolation_dimension: int = 1  # 1 for x interpolation

    interpolated_color: np.ndarray = vector_interp(
        segment_start,
        segment_end,
        color_start,
        color_end,
        target_coord,
        interpolation_dimension,
    )
    expected_color: np.ndarray = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    assert np.allclose(interpolated_color, expected_color)


def test_edge_function_signs() -> None:
    start_vertex: np.ndarray = np.array([0.0, 0.0], dtype=np.float64)
    end_vertex: np.ndarray = np.array([5.0, 0.0], dtype=np.float64)
    point_left: np.ndarray = np.array([[2.5, 2.0]], dtype=np.float64)
    point_right: np.ndarray = np.array([[2.5, -2.0]], dtype=np.float64)

    # np.cross([5, 0], [2.5 - 0, 2 - 0]) = 5 * 2 - 0 = 10 > 0
    signed_area_left: np.ndarray = edge_function(start_vertex, end_vertex, point_left)
    signed_area_right: np.ndarray = edge_function(start_vertex, end_vertex, point_right)

    assert signed_area_left[0] > 0.0
    assert signed_area_right[0] < 0.0
