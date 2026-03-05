"""Discover segment meshes in volpkg directory trees."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SegmentInfo:
    """Information about a discovered segment."""

    segment_id: str
    obj_path: Path
    parent_dir: Path


def discover_segments(root: str | Path) -> list[SegmentInfo]:
    """Walk a directory tree and find all OBJ mesh files.

    Understands the volpkg convention:
        volpkg/paths/<segment_id>/<segment_id>.obj

    Also finds any .obj files in flat directory structures.

    Args:
        root: Root directory to search.

    Returns:
        List of SegmentInfo for each discovered mesh.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    segments: list[SegmentInfo] = []
    seen: set[Path] = set()

    # volpkg convention: paths/<id>/<id>.obj
    paths_dir = root / "paths"
    if paths_dir.is_dir():
        for seg_dir in sorted(paths_dir.iterdir()):
            if not seg_dir.is_dir():
                continue
            for obj_file in seg_dir.glob("*.obj"):
                if obj_file not in seen:
                    seen.add(obj_file)
                    segments.append(SegmentInfo(
                        segment_id=seg_dir.name,
                        obj_path=obj_file,
                        parent_dir=seg_dir,
                    ))

    # Fallback: find all .obj files recursively
    for obj_file in sorted(root.rglob("*.obj")):
        if obj_file not in seen:
            seen.add(obj_file)
            segments.append(SegmentInfo(
                segment_id=obj_file.stem,
                obj_path=obj_file,
                parent_dir=obj_file.parent,
            ))

    return segments
