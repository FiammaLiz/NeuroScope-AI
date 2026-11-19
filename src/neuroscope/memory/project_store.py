from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from ..domain import DatasetProfile


class ProjectStore:
    """Small local JSON project store for the prototype."""
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.profile_path = self.root / "dataset_profile.json"
        self.history_path = self.root / "analysis_history.json"

    def save_profile(self, profile: DatasetProfile):
        self.profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    def load_profile(self) -> DatasetProfile | None:
        if not self.profile_path.exists():
            return None
        return DatasetProfile.model_validate_json(self.profile_path.read_text(encoding="utf-8"))

    def append_analysis(self, analysis_name: str, parameters: dict[str, Any], summary: str):
        history = []
        if self.history_path.exists():
            history = json.loads(self.history_path.read_text(encoding="utf-8"))
        history.append({"analysis": analysis_name, "parameters": parameters,
                        "summary": summary,
                        "timestamp": datetime.now(timezone.utc).isoformat()})
        self.history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    def load_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        return json.loads(self.history_path.read_text(encoding="utf-8"))
