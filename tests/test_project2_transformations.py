import numpy as np
from project2_functions import (
    compose,
    lookat,
    perspective_project,
    rasterize,
    rotate,
    translate,
)


def test_translate_identity() -> None:
    zero_offset: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    result_matrix: np.ndarray = translate(zero_offset)
    expected_matrix: np.ndarray = np.eye(4, dtype=np.float64)
    assert np.allclose(result_matrix, expected_matrix)


def test_translate_point() -> None:
    offset_vector: np.ndarray = np.array([2.0, -3.0, 5.0], dtype=np.float64)
    transformation_matrix: np.ndarray = translate(offset_vector)
    homogeneous_point: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    translated_point: np.ndarray = transformation_matrix @ homogeneous_point
    expected_point: np.ndarray = np.array([3.0, -2.0, 6.0, 1.0], dtype=np.float64)
    assert np.allclose(translated_point, expected_point)


def test_rotate_z_axis_quarter_turn() -> None:
    rotation_axis: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    center_point: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    angle_radians: float = np.pi / 2.0
    rotation_matrix: np.ndarray = rotate(rotation_axis, angle_radians, center_point)

    homogeneous_point: np.ndarray = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    rotated_point: np.ndarray = rotation_matrix @ homogeneous_point
    expected_point: np.ndarray = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    assert np.allclose(rotated_point, expected_point, atol=1e-7)


def test_compose_transformations() -> None:
    offset_a: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    offset_b: np.ndarray = np.array([0.0, 2.0, 0.0], dtype=np.float64)
    matrix_a: np.ndarray = translate(offset_a)
    matrix_b: np.ndarray = translate(offset_b)
    composed_matrix: np.ndarray = compose(matrix_a, matrix_b)

    homogeneous_origin: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    transformed_point: np.ndarray = composed_matrix @ homogeneous_origin
    expected_point: np.ndarray = np.array([1.0, 2.0, 0.0, 1.0], dtype=np.float64)
    assert np.allclose(transformed_point, expected_point)


def test_lookat_orientation() -> None:
    camera_eye: np.ndarray = np.array([0.0, 0.0, 5.0], dtype=np.float64)
    up_vector: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    target_point: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    rotation_matrix, translation_vector = lookat(camera_eye, up_vector, target_point)

    assert rotation_matrix.shape == (3, 3)
    assert translation_vector.shape == (3,)
    projected_eye: np.ndarray = rotation_matrix @ camera_eye + translation_vector
    assert np.allclose(projected_eye, np.zeros(3, dtype=np.float64), atol=1e-7)


def test_perspective_project_depth() -> None:
    world_points: np.ndarray = np.array([[0.0, 2.0], [0.0, 4.0], [10.0, 20.0]], dtype=np.float64)
    focal_length: float = 5.0
    identity_rotation: np.ndarray = np.eye(3, dtype=np.float64)
    zero_translation: np.ndarray = np.zeros(3, dtype=np.float64)

    projected_2d, depth_values = perspective_project(
        world_points, focal_length, identity_rotation, zero_translation
    )

    assert projected_2d.shape == (2, 2)
    assert depth_values.shape == (2,)
    assert np.allclose(depth_values, np.array([10.0, 20.0], dtype=np.float64))
    assert np.allclose(projected_2d[:, 0], np.array([0.0, 0.0], dtype=np.float64))
    assert np.allclose(projected_2d[:, 1], np.array([0.5, 1.0], dtype=np.float64))


def test_rasterize_bounds() -> None:
    plane_points: np.ndarray = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float64)
    pixel_coordinates: np.ndarray = rasterize(
        pts_2d=plane_points,
        plane_w=2.0,
        plane_h=2.0,
        res_w=100,
        res_h=100,
    )

    assert pixel_coordinates.shape == (2, 2)
    assert np.all(pixel_coordinates >= 0)
    assert np.all(pixel_coordinates < 100)
