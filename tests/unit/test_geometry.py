import numpy as np
import pytest

from medicine_preprocess.geometry import TransformState, map_points


def test_transform_state_left_composes_current_operation() -> None:
    state = TransformState(np.eye(3, dtype=np.float64))
    crop_translation = np.array([[1, 0, -10], [0, 1, -20], [0, 0, 1]], dtype=np.float64)
    resized = np.array([[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]], dtype=np.float64)
    final = state.then(crop_translation).then(resized)
    assert np.allclose(map_points(((10, 20),), final.forward), ((0.5, 0.5),))
    assert np.allclose(final.inverse @ final.forward, np.eye(3), atol=1e-9)


def test_transform_state_copies_and_rejects_invalid_matrices() -> None:
    matrix = np.eye(3, dtype=np.float64)
    state = TransformState(matrix)
    matrix[0, 0] = 4.0
    assert state.forward[0, 0] == 1.0

    with pytest.raises(ValueError, match="finite invertible"):
        TransformState(np.zeros((3, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="finite invertible"):
        TransformState(np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.nan]], dtype=np.float64))


def test_map_points_preserves_empty_input() -> None:
    assert map_points((), np.eye(3, dtype=np.float64)) == ()
