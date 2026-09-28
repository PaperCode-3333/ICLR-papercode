"""Original shared JSON I/O helpers; no experiment semantics."""
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def atomic_json(path: Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)

def load_json(path: Path, default: Any = None) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default
