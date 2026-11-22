from pathlib import Path

import numpy as np

from neuroscope.domain import DatasetProfile
from neuroscope.memory.project_store import ProjectStore


def test_profile_and_history_persist(tmp_path: Path):
    profile = DatasetProfile.from_array(tmp_path / "demo.npy", np.zeros((2, 100)), channel_axis=0)
    store = ProjectStore(tmp_path / "project")
    store.save_profile(profile)
    store.append_analysis("demo", {"x": 1}, "ok")
    assert store.load_profile().shape == [2, 100]
    assert store.load_history()[0]["analysis"] == "demo"
