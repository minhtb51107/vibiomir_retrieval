from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_dense_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data["model"]["name"] != "BAAI/bge-m3":
        raise ValueError("Phase 4 requires model.name to be exactly BAAI/bge-m3")
    if data["index"]["type"] != "IndexFlatIP":
        raise ValueError("Phase 4 exact baseline requires IndexFlatIP")
    if not data["model"]["normalize_embeddings"]:
        raise ValueError("IndexFlatIP baseline requires normalized embeddings")
    return data
