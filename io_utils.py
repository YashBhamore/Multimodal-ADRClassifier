import json
import shutil
from pathlib import Path
from typing import Any, Dict, Union
from utils.logger import log


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory if it doesn’t exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a JSON file safely."""
    path = Path(path)
    if not path.exists():
        log(f"JSON file not found: {path}", level="error")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        log(f"JSON decode error in {path}: {e}", level="error")
        return {}


def write_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2):
    """Write a dictionary to a JSON file."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    log(f"Saved JSON: {path}")


def safe_copy(src: Union[str, Path], dst: Union[str, Path]):
    """Copy a file safely, creating directories as needed."""
    src, dst = Path(src), Path(dst)
    if not src.exists():
        log(f"Cannot copy; source missing: {src}", level="warning")
        return
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    log(f"Copied: {src} -> {dst}")


def list_files(base: Union[str, Path], pattern: str = "*") -> list[Path]:
    """Return all files under a path matching pattern."""
    base = Path(base)
    if not base.exists():
        log(f"Directory not found: {base}", level="warning")
        return []
    return sorted(base.rglob(pattern))


def auto_discover(root_candidates: list[Path], filename: str) -> Union[Path, None]:
    """Find a file within multiple candidate roots."""
    for root in root_candidates:
        hits = list(Path(root).rglob(filename))
        if hits:
            log(f"Discovered file {filename} under {hits[0].parent}")
            return hits[0].parent
    log(f"File {filename} not found in any candidate root.", level="warning")
    return None
