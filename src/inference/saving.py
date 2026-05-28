import json
import os
from pathlib import Path


def atomic_save_json(path: str | Path, data: dict) -> None:
    """Write data atomically via .tmp + os.replace (POSIX rename). Safe against mid-run crashes."""
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, p)
