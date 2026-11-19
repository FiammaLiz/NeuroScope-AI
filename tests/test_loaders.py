from pathlib import Path
import numpy as np
import pytest
from scipy.io import savemat
from neuroscope.data.loaders import load_array, DataLoadError
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


def test_mat_variable_selection(tmp_path: Path):
    p = tmp_path / "a.mat"
    savemat(p, {"one": np.ones(2), "two": np.zeros(3)})
    with pytest.raises(DataLoadError, match="variable_name"):
        load_array(p)
    assert load_array(p, variable_name="two").shape == (3,)


def test_profile_requires_confirmed_2d_orientation(tmp_path: Path):
    with pytest.raises(ValueError, match="Confirm"):
        DatasetProfile.from_array(tmp_path / "a.npy", np.zeros((2, 3)))
    profile = DatasetProfile.from_array(tmp_path / "a.npy", np.zeros((2, 3)), channel_axis=0)
    assert profile.confirmed_fields == ["channel_axis"]
    assert not profile.metadata.sampling_rate_hz
    assert "sampling_rate_hz" in profile.unknown_fields
