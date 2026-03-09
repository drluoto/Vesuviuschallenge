"""CT volume accessor for mesh-to-volume coordinate mapping.

Handles remote OME-Zarr access, coordinate mapping (mesh XYZ -> volume ZYX),
and neighborhood sampling for structure tensor computation.

Uses chunk-level LRU caching so that dense vertex sampling hits each remote
chunk at most once. Combined with chunk-sorted vertex ordering, this makes
processing ALL mesh vertices feasible even over remote volumes.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import zarr
import fsspec


class _ChunkCache:
    """Simple LRU cache for Zarr chunk data, keyed by chunk coordinates."""

    def __init__(self, max_chunks: int = 256):
        self._cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        self._max = max_chunks

    def get(self, key: tuple[int, int, int]) -> np.ndarray | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: tuple[int, int, int], data: np.ndarray) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max:
                self._cache.popitem(last=False)
            self._cache[key] = data


class VolumeAccessor:
    """Lazy accessor for CT volume data at mesh vertex positions.

    Maintains a chunk-level LRU cache (default 256 chunks ≈ 512 MB for
    128³ uint8 chunks) so that dense vertex sampling hits each remote
    chunk at most once.
    """

    def __init__(self, zarr_url: str, scale: int = 0, cache_chunks: int = 256):
        store = fsspec.get_mapper(zarr_url)
        group = zarr.open(store, mode='r', zarr_format=2)
        self._vol = group[str(scale)]
        self._shape = self._vol.shape  # (Z, Y, X)
        self._chunks = self._vol.chunks  # (cz, cy, cx)
        self._scale_factor = 2 ** scale
        self._cache = _ChunkCache(max_chunks=cache_chunks)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def chunks(self) -> tuple[int, int, int]:
        return self._chunks

    @property
    def scale_factor(self) -> int:
        return self._scale_factor

    def _xyz_to_zyx(self, vertex_xyz: np.ndarray) -> tuple[int, int, int]:
        """Convert mesh (X, Y, Z) to volume index (iz, iy, ix)."""
        s = self._scale_factor
        x, y, z = vertex_xyz
        return int(round(z / s)), int(round(y / s)), int(round(x / s))

    def vertex_in_bounds(self, vertex_xyz: np.ndarray, margin: int = 16) -> bool:
        """Check if vertex + margin fits in volume."""
        iz, iy, ix = self._xyz_to_zyx(vertex_xyz)
        return (margin <= iz < self._shape[0] - margin and
                margin <= iy < self._shape[1] - margin and
                margin <= ix < self._shape[2] - margin)

    def _fetch_chunk(self, cz: int, cy: int, cx: int) -> np.ndarray:
        """Fetch a single Zarr chunk by chunk index, with LRU caching."""
        key = (cz, cy, cx)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        sz, sy, sx = self._chunks
        z0, y0, x0 = cz * sz, cy * sy, cx * sx
        z1 = min(z0 + sz, self._shape[0])
        y1 = min(y0 + sy, self._shape[1])
        x1 = min(x0 + sx, self._shape[2])
        data = np.array(self._vol[z0:z1, y0:y1, x0:x1], dtype=np.float32)
        self._cache.put(key, data)
        return data

    def sample_neighborhood(
        self, vertex_xyz: np.ndarray, half_size: int = 16
    ) -> np.ndarray:
        """Fetch a cubic neighborhood centered at a mesh vertex.

        Uses chunk-level caching: if the neighborhood fits within cached
        chunks, no remote I/O is needed.

        Returns float32 array of shape (2*half_size, 2*half_size, 2*half_size).
        """
        iz, iy, ix = self._xyz_to_zyx(vertex_xyz)
        h = half_size
        sz, sy, sx = self._chunks

        # Determine which chunks this neighborhood spans
        cz0, cz1 = (iz - h) // sz, (iz + h - 1) // sz
        cy0, cy1 = (iy - h) // sy, (iy + h - 1) // sy
        cx0, cx1 = (ix - h) // sx, (ix + h - 1) // sx

        # Fast path: neighborhood fits in a single chunk
        if cz0 == cz1 and cy0 == cy1 and cx0 == cx1:
            chunk_data = self._fetch_chunk(cz0, cy0, cx0)
            # Local coords within chunk
            lz = (iz - h) - cz0 * sz
            ly = (iy - h) - cy0 * sy
            lx = (ix - h) - cx0 * sx
            return chunk_data[lz:lz + 2*h, ly:ly + 2*h, lx:lx + 2*h]

        # Slow path: assemble from multiple chunks
        result = np.zeros((2*h, 2*h, 2*h), dtype=np.float32)
        for cz in range(cz0, cz1 + 1):
            for cy in range(cy0, cy1 + 1):
                for cx in range(cx0, cx1 + 1):
                    chunk_data = self._fetch_chunk(cz, cy, cx)
                    # Global coords of this chunk
                    gz0, gy0, gx0 = cz * sz, cy * sy, cx * sx
                    # Overlap between chunk and requested region
                    oz0 = max(iz - h, gz0)
                    oy0 = max(iy - h, gy0)
                    ox0 = max(ix - h, gx0)
                    oz1 = min(iz + h, gz0 + chunk_data.shape[0])
                    oy1 = min(iy + h, gy0 + chunk_data.shape[1])
                    ox1 = min(ix + h, gx0 + chunk_data.shape[2])
                    if oz0 >= oz1 or oy0 >= oy1 or ox0 >= ox1:
                        continue
                    # Copy into result
                    result[oz0-(iz-h):oz1-(iz-h),
                           oy0-(iy-h):oy1-(iy-h),
                           ox0-(ix-h):ox1-(ix-h)] = chunk_data[
                        oz0-gz0:oz1-gz0, oy0-gy0:oy1-gy0, ox0-gx0:ox1-gx0
                    ]
        return result

    def sort_by_chunk(self, vertices_xyz: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Sort vertex indices so that nearby vertices (same chunk) are adjacent.

        This maximizes LRU cache hits when iterating over vertices.
        """
        s = self._scale_factor
        cz, cy, cx = self._chunks
        verts = vertices_xyz[indices]
        chunk_z = (verts[:, 2] / s / cz).astype(np.int64)
        chunk_y = (verts[:, 1] / s / cy).astype(np.int64)
        chunk_x = (verts[:, 0] / s / cx).astype(np.int64)
        sort_keys = chunk_z * 1_000_000 + chunk_y * 1_000 + chunk_x
        order = np.argsort(sort_keys)
        return indices[order]

    def sample_intensities(self, vertices_xyz: np.ndarray) -> np.ndarray:
        """Sample single-voxel intensities at each vertex position."""
        result = np.zeros(len(vertices_xyz), dtype=np.float32)
        for i, v in enumerate(vertices_xyz):
            iz, iy, ix = self._xyz_to_zyx(v)
            if (0 <= iz < self._shape[0] and
                0 <= iy < self._shape[1] and
                0 <= ix < self._shape[2]):
                result[i] = float(self._vol[iz, iy, ix])
        return result
