from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from neuroscope.data.loaders import DataLoadError, load_array
from neuroscope.domain import DatasetProfile


def test_text_options_and_nonfinite_rejection(tmp_path: Path):
    p = tmp_path / "a.csv"
    p.write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
    assert load_array(p, header=True).shape == (2, 2)
    p.write_text("1,nan\n", encoding="utf-8")
    with pytest.raises(DataLoadError, match="NaN"):
        load_array(p)


def test_txt_requires_delimiter_and_rejects_empty(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("1\t2\n", encoding="utf-8")
    with pytest.raises(DataLoadError, match="delimiter"):
        load_array(p)
    assert load_array(p, delimiter="\t").tolist() == [1, 2]
    p.write_text("", encoding="utf-8")
    with pytest.raises(DataLoadError, match="empty"):
        load_array(p)


@pytest.mark.parametrize("contents", ["", "  \n\t  \n"])
def test_txt_empty_input_is_rejected_before_delimiter_check(tmp_path: Path, contents: str):
    path = tmp_path / "empty.txt"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(DataLoadError, match="TXT file contains no data"):
        load_array(path)


def test_valid_txt_still_loads_after_empty_input_check(tmp_path: Path):
    path = tmp_path / "valid.txt"
    path.write_text("1\t2\n3\t4\n", encoding="utf-8")

    np.testing.assert_array_equal(load_array(path, delimiter="\t"), [[1, 2], [3, 4]])


def test_mat_variable_selection(tmp_path: Path):
    p = tmp_path / "a.mat"
    savemat(p, {"one": np.ones(2), "two": np.zeros(3)})
    with pytest.raises(DataLoadError, match="variable_name"):
        load_array(p)
    assert load_array(p, variable_name="two").shape == (3,)


@pytest.mark.parametrize(
    ("name", "values", "expected_shape"),
    [
        ("row", np.array([[1, 2, 3]]), (3,)),
        ("column", np.array([[1], [2], [3]]), (3,)),
        ("matrix", np.arange(6).reshape(2, 3), (2, 3)),
        ("single", np.array([[7]]), (1,)),
        ("multichannel", np.arange(12).reshape(4, 3), (4, 3)),
    ],
)
def test_mat_vector_and_matrix_shapes_are_preserved(
    tmp_path: Path, name: str, values: np.ndarray, expected_shape: tuple[int, ...]
):
    path = tmp_path / f"{name}.mat"
    savemat(path, {"signal": values})

    loaded = load_array(path)

    assert loaded.shape == expected_shape
    expected = values.reshape(-1) if len(expected_shape) == 1 else values
    np.testing.assert_array_equal(loaded, expected)


def test_profile_requires_confirmed_2d_orientation(tmp_path: Path):
    with pytest.raises(ValueError, match="Confirm"):
        DatasetProfile.from_array(tmp_path / "a.npy", np.zeros((2, 3)))
    profile = DatasetProfile.from_array(tmp_path / "a.npy", np.zeros((2, 3)), channel_axis=0)
    assert profile.confirmed_fields == ["channel_axis"]
    assert not profile.metadata.sampling_rate_hz
    assert "sampling_rate_hz" in profile.unknown_fields
