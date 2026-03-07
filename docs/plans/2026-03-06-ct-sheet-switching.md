# CT-Informed Sheet Switching Detection

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a CT-volume-aware sheet switching metric that compares mesh normals against papyrus layer normals derived from structure tensors, detecting switches that geometry-only analysis cannot.

**Architecture:** New optional metric `CTSheetSwitchingMetric` that lazily fetches CT chunks from remote OME-Zarr stores, computes structure tensors at sampled mesh vertices, and scores alignment between mesh normals and CT-derived sheet normals. Runs alongside existing geometry-only metrics. Requires `--volume` flag on CLI; gracefully skipped when not provided.

**Tech Stack:** vesuvius (structure tensor), zarr + fsspec (remote volume access), numpy (eigendecomposition), open3d (mesh), existing MetricComputer base class.

---

## Known Facts from Smoke Testing

- `StructureTensorComputer(sigma=3.0)` gives best alignment on real data
- Eigendecompose via `np.linalg.eigh` on the 6-component tensor works (vesuvius's own `eigendecompose` has shape issues)
- PHerc1667 mesh vertex coords map to volume as `vol[Z, Y, X]` where mesh vertex = `(X, Y, Z)`
- CT normal (from structure tensor) is in `(dim0, dim1, dim2)` = `(Z, Y, X)` order; reorder to `(X, Y, Z)` via `[2, 1, 0]` to match mesh normals
- 32x32x32 neighborhoods with sigma=3.0: median angle ~17 deg on good mesh regions
- Structure tensor anisotropy ratio >10^8 on real papyrus -- very clear signal
- Remote zarr access works with `fsspec.get_mapper(url)` + `zarr.open(store, 'r')`
- Volume class in vesuvius has OME-Zarr bug; use direct zarr access instead

## Open Risk: Coordinate Alignment

The "normalized" mesh showed 23 deg mean alignment (sigma=3.0). This may be because:
1. The mesh was post-processed (filename says "normalized")
2. The coordinate mapping has an offset or scale factor
3. This is genuinely what alignment looks like

**Task 1 explicitly validates this** before any metric code is written. If alignment on a known-good mesh is worse than 20 deg median, the approach needs revisiting.

---

### Task 1: Validate Coordinate Alignment

**Goal:** Confirm that mesh vertex coordinates are raw voxel indices into the CT volume by testing on multiple meshes and checking intensity at vertex positions.

**Files:**
- Create: `scripts/validate_ct_alignment.py`

**Step 1: Write validation script**

```python
"""Validate mesh-to-volume coordinate alignment.

For each mesh vertex, sample the CT intensity at that position.
If coordinates are correct, vertices should sit on high-intensity voxels
(papyrus is bright in CT). Also checks structure tensor normal alignment.
"""
import sys
import numpy as np
import open3d as o3d
import zarr
import fsspec
from vesuvius.image_proc.geometry.structure_tensor import StructureTensorComputer


def validate(mesh_path: str, zarr_url: str, n_samples: int = 200):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    print(f"Mesh: {len(verts)} vertices")
    print(f"  X: {verts[:,0].min():.0f}-{verts[:,0].max():.0f}")
    print(f"  Y: {verts[:,1].min():.0f}-{verts[:,1].max():.0f}")
    print(f"  Z: {verts[:,2].min():.0f}-{verts[:,2].max():.0f}")

    store = fsspec.get_mapper(zarr_url)
    vol = zarr.open(store, mode='r')['0']
    print(f"Volume: {vol.shape} (Z, Y, X)")

    # Check bounds
    z_max, y_max, x_max = vol.shape
    in_bounds = ((verts[:, 0] >= 0) & (verts[:, 0] < x_max) &
                 (verts[:, 1] >= 0) & (verts[:, 1] < y_max) &
                 (verts[:, 2] >= 0) & (verts[:, 2] < z_max))
    print(f"  Vertices in bounds: {in_bounds.sum()}/{len(verts)} ({in_bounds.mean():.1%})")

    if in_bounds.mean() < 0.5:
        print("ERROR: Most vertices out of bounds. Wrong volume or coordinate mapping.")
        return

    # Sample vertices in bounds
    valid_idx = np.where(in_bounds)[0]
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(valid_idx, size=min(n_samples, len(valid_idx)), replace=False)

    # Test 1: Intensity at vertex positions (should be high for papyrus)
    intensities = []
    for vi in sample_idx:
        x, y, z = verts[vi]
        ix, iy, iz = int(round(x)), int(round(y)), int(round(z))
        val = int(vol[iz, iy, ix])
        intensities.append(val)
    intensities = np.array(intensities)
    print(f"\nIntensity at vertices:")
    print(f"  Mean: {intensities.mean():.1f}, Median: {np.median(intensities):.1f}")
    print(f"  >50 (likely papyrus): {(intensities > 50).mean():.1%}")
    print(f"  >100 (definitely papyrus): {(intensities > 100).mean():.1%}")

    # Test 2: Structure tensor alignment
    stc = StructureTensorComputer(sigma=3.0, device='cpu')
    half = 16
    angles = []
    for vi in sample_idx[:50]:  # 50 vertices for speed
        x, y, z = verts[vi]
        iz, iy, ix = int(round(z)), int(round(y)), int(round(x))
        if (iz-half < 0 or iz+half > vol.shape[0] or
            iy-half < 0 or iy+half > vol.shape[1] or
            ix-half < 0 or ix+half > vol.shape[2]):
            continue
        chunk = np.array(vol[iz-half:iz+half, iy-half:iy+half, ix-half:ix+half], dtype=np.float32)
        if chunk.mean() < 5:
            continue
        st = stc.compute(chunk, as_numpy=True)
        c = half
        vals = [st[i, c, c, c] for i in range(6)]
        mat = np.array([[vals[0], vals[1], vals[2]],
                        [vals[1], vals[3], vals[4]],
                        [vals[2], vals[4], vals[5]]])
        evals, evecs = np.linalg.eigh(mat)
        ct_normal = evecs[:, 2][[2, 1, 0]]  # ZYX -> XYZ
        ct_normal /= np.linalg.norm(ct_normal) + 1e-10
        align = abs(np.dot(normals[vi], ct_normal))
        angles.append(np.degrees(np.arccos(min(align, 1.0))))

    angles = np.array(angles)
    print(f"\nNormal alignment (mesh vs CT structure tensor):")
    print(f"  n={len(angles)}, mean={angles.mean():.1f} deg, median={np.median(angles):.1f} deg")
    print(f"  <15 deg: {(angles < 15).mean():.1%}")
    print(f"  <30 deg: {(angles < 30).mean():.1%}")

    if np.median(angles) < 20:
        print("\nVERDICT: Good alignment. CT-informed detection is feasible.")
    elif np.median(angles) < 35:
        print("\nVERDICT: Moderate alignment. May work but needs investigation.")
    else:
        print("\nVERDICT: Poor alignment. Check coordinate mapping or mesh provenance.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python validate_ct_alignment.py <mesh.obj> <zarr_url>")
        sys.exit(1)
    validate(sys.argv[1], sys.argv[2])
```

**Step 2: Run on PHerc1667 mesh**

Run:
```bash
python scripts/validate_ct_alignment.py \
  data/segments/PHerc1667_20231210132040_normalized.obj \
  'https://data.aws.ash2txt.org/samples/PHerc1667/volumes/20231117161658-7.910um-53keV-masked.zarr/'
```

Expected: Intensity >50 at >80% of vertices (confirms correct coordinate mapping). Normal alignment median <25 deg.

**Decision gate:** If median alignment >35 deg, STOP and investigate coordinate mapping before proceeding. If <25 deg, proceed. If 25-35 deg, proceed but note the noise floor.

**Step 3: Commit**

```bash
git add scripts/validate_ct_alignment.py
git commit -m "Add CT alignment validation script for coordinate mapping verification"
```

---

### Task 2: Volume Accessor Module

**Goal:** A thin wrapper around zarr that handles URL construction, chunk caching, and coordinate mapping. Isolates all volume-access complexity from the metric.

**Files:**
- Create: `src/vesuvius_mesh_qa/volume.py`
- Test: `tests/test_volume.py`

**Step 1: Write the failing test**

```python
"""Tests for volume accessor."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


class TestVolumeAccessor:
    def test_sample_neighborhood_returns_correct_shape(self):
        from vesuvius_mesh_qa.volume import VolumeAccessor

        # Mock zarr array
        mock_array = MagicMock()
        mock_array.shape = (1000, 500, 500)
        mock_array.__getitem__ = MagicMock(
            return_value=np.random.rand(32, 32, 32).astype(np.float32)
        )

        accessor = VolumeAccessor.__new__(VolumeAccessor)
        accessor._vol = mock_array
        accessor._shape = mock_array.shape

        # Vertex at (250, 250, 500) -> volume index [500, 250, 250]
        chunk = accessor.sample_neighborhood(
            vertex_xyz=np.array([250.0, 250.0, 500.0]),
            half_size=16,
        )
        assert chunk.shape == (32, 32, 32)
        assert chunk.dtype == np.float32

    def test_vertex_in_bounds(self):
        from vesuvius_mesh_qa.volume import VolumeAccessor

        accessor = VolumeAccessor.__new__(VolumeAccessor)
        accessor._shape = (1000, 500, 500)

        assert accessor.vertex_in_bounds(np.array([250, 250, 500]), margin=16)
        assert not accessor.vertex_in_bounds(np.array([250, 250, 990]), margin=16)
        assert not accessor.vertex_in_bounds(np.array([5, 250, 500]), margin=16)

    def test_batch_sample_intensities(self):
        from vesuvius_mesh_qa.volume import VolumeAccessor

        mock_array = MagicMock()
        mock_array.shape = (1000, 500, 500)
        # Return a value for single voxel reads
        mock_array.__getitem__ = MagicMock(return_value=np.uint8(128))

        accessor = VolumeAccessor.__new__(VolumeAccessor)
        accessor._vol = mock_array
        accessor._shape = mock_array.shape

        vertices = np.array([[250, 250, 500], [100, 100, 200]], dtype=np.float64)
        intensities = accessor.sample_intensities(vertices)
        assert len(intensities) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_volume.py -v`
Expected: FAIL with "No module named 'vesuvius_mesh_qa.volume'"

**Step 3: Write implementation**

```python
"""CT volume accessor for mesh-to-volume coordinate mapping.

Handles remote OME-Zarr access, coordinate mapping (mesh XYZ -> volume ZYX),
and neighborhood sampling for structure tensor computation.
"""
from __future__ import annotations

import numpy as np
import zarr
import fsspec


class VolumeAccessor:
    """Lazy accessor for CT volume data at mesh vertex positions."""

    def __init__(self, zarr_url: str, scale: int = 0):
        store = fsspec.get_mapper(zarr_url)
        group = zarr.open(store, mode='r')
        self._vol = group[str(scale)]
        self._shape = self._vol.shape  # (Z, Y, X)
        self._scale_factor = 2 ** scale

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def _xyz_to_zyx(self, vertex_xyz: np.ndarray) -> tuple[int, int, int]:
        """Convert mesh (X, Y, Z) to volume index (iz, iy, ix)."""
        s = self._scale_factor
        x, y, z = vertex_xyz
        return int(round(z / s)), int(round(y / s)), int(round(x / s))

    def vertex_in_bounds(self, vertex_xyz: np.ndarray, margin: int = 16) -> bool:
        iz, iy, ix = self._xyz_to_zyx(vertex_xyz)
        return (margin <= iz < self._shape[0] - margin and
                margin <= iy < self._shape[1] - margin and
                margin <= ix < self._shape[2] - margin)

    def sample_neighborhood(
        self, vertex_xyz: np.ndarray, half_size: int = 16
    ) -> np.ndarray:
        """Fetch a cubic neighborhood centered at a mesh vertex.

        Returns float32 array of shape (2*half_size, 2*half_size, 2*half_size).
        """
        iz, iy, ix = self._xyz_to_zyx(vertex_xyz)
        h = half_size
        chunk = self._vol[iz - h : iz + h, iy - h : iy + h, ix - h : ix + h]
        return np.array(chunk, dtype=np.float32)

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
```

**Step 4: Run tests**

Run: `pytest tests/test_volume.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/vesuvius_mesh_qa/volume.py tests/test_volume.py
git commit -m "Add VolumeAccessor for CT volume access at mesh vertex positions"
```

---

### Task 3: Structure Tensor Normal Computation

**Goal:** Module that computes CT-derived papyrus normals at mesh vertex positions using structure tensor eigendecomposition.

**Files:**
- Create: `src/vesuvius_mesh_qa/ct_normals.py`
- Test: `tests/test_ct_normals.py`

**Step 1: Write the failing test**

```python
"""Tests for CT normal computation."""
import numpy as np
import pytest


class TestCTNormals:
    def test_synthetic_horizontal_sheet(self):
        """Structure tensor on a horizontal bright sheet should give Z-pointing normal."""
        from vesuvius_mesh_qa.ct_normals import compute_ct_normal

        vol = np.zeros((32, 32, 32), dtype=np.float32)
        vol[14:18, :, :] = 1.0  # horizontal sheet

        normal, anisotropy = compute_ct_normal(vol, sigma=1.0)

        # Normal should point along dim0 (Z in volume space)
        assert abs(normal[0]) > 0.9  # Z component dominant
        assert anisotropy > 0.5  # clear planar structure

    def test_synthetic_tilted_sheet(self):
        """Structure tensor on a 45-deg sheet should give tilted normal."""
        from vesuvius_mesh_qa.ct_normals import compute_ct_normal

        vol = np.zeros((32, 32, 32), dtype=np.float32)
        for i in range(32):
            z = int(16 + (i - 16) * 0.5)
            if 0 <= z < 32:
                vol[z, i, :] = 1.0

        normal, anisotropy = compute_ct_normal(vol, sigma=1.0)
        assert anisotropy > 0.5
        # Normal should have both Z and Y components
        assert abs(normal[0]) > 0.3  # Z component
        assert abs(normal[1]) > 0.3  # Y component

    def test_empty_volume_returns_zero_anisotropy(self):
        """Empty volume should have near-zero anisotropy."""
        from vesuvius_mesh_qa.ct_normals import compute_ct_normal

        vol = np.zeros((32, 32, 32), dtype=np.float32)
        normal, anisotropy = compute_ct_normal(vol, sigma=1.0)
        assert anisotropy < 0.1

    def test_batch_compute(self):
        """Batch computation returns normals and anisotropy for multiple chunks."""
        from vesuvius_mesh_qa.ct_normals import compute_ct_normals_batch

        chunks = [np.zeros((32, 32, 32), dtype=np.float32) for _ in range(3)]
        chunks[0][14:18, :, :] = 1.0  # one has structure
        chunks[1][14:18, :, :] = 1.0

        normals, anisotropies = compute_ct_normals_batch(chunks, sigma=1.0)
        assert normals.shape == (3, 3)
        assert anisotropies.shape == (3,)
        assert anisotropies[0] > 0.5
        assert anisotropies[2] < 0.1  # empty chunk
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ct_normals.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
"""Compute CT-derived papyrus surface normals from structure tensors.

Uses the vesuvius package's StructureTensorComputer to extract the dominant
orientation of papyrus layers from local CT neighborhoods. The largest
eigenvalue's eigenvector (e2) gives the sheet normal direction.

Normal is returned in volume index order (Z, Y, X). Caller must reorder
to match mesh coordinate convention.
"""
from __future__ import annotations

import numpy as np
from vesuvius.image_proc.geometry.structure_tensor import StructureTensorComputer


def compute_ct_normal(
    volume_chunk: np.ndarray,
    sigma: float = 3.0,
) -> tuple[np.ndarray, float]:
    """Compute the CT-derived surface normal at the center of a volume chunk.

    Args:
        volume_chunk: 3D float32 array (D, H, W).
        sigma: Gaussian smoothing sigma for structure tensor.

    Returns:
        normal: Unit normal vector (3,) in volume index order (Z, Y, X).
            Direction of largest eigenvalue eigenvector.
        anisotropy: Planarity score in [0, 1]. High = clear sheet structure.
    """
    stc = StructureTensorComputer(sigma=sigma, device='cpu')
    st = stc.compute(volume_chunk, as_numpy=True)  # (6, D, H, W)

    cz, cy, cx = st.shape[1] // 2, st.shape[2] // 2, st.shape[3] // 2
    components = [st[i, cz, cy, cx] for i in range(6)]
    # Layout: [Jzz, Jzy, Jzx, Jyy, Jyx, Jxx]
    mat = np.array([
        [components[0], components[1], components[2]],
        [components[1], components[3], components[4]],
        [components[2], components[4], components[5]],
    ])

    eigenvalues, eigenvectors = np.linalg.eigh(mat)
    # eigh returns sorted ascending: eigenvalues[2] is largest
    normal = eigenvectors[:, 2]  # (Z, Y, X) direction
    normal = normal / (np.linalg.norm(normal) + 1e-10)

    # Planarity: how sheet-like (lambda2 >> lambda1 ~ lambda0)
    lam = np.maximum(eigenvalues, 0)
    total = lam.sum()
    if total < 1e-12:
        return normal, 0.0
    anisotropy = float((lam[2] - lam[1]) / total)
    return normal, anisotropy


def compute_ct_normals_batch(
    chunks: list[np.ndarray],
    sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CT normals for a batch of volume chunks.

    Returns:
        normals: (N, 3) array of unit normals in volume (Z, Y, X) order.
        anisotropies: (N,) array of planarity scores.
    """
    normals = np.zeros((len(chunks), 3))
    anisotropies = np.zeros(len(chunks))
    for i, chunk in enumerate(chunks):
        normals[i], anisotropies[i] = compute_ct_normal(chunk, sigma)
    return normals, anisotropies
```

**Step 4: Run tests**

Run: `pytest tests/test_ct_normals.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/vesuvius_mesh_qa/ct_normals.py tests/test_ct_normals.py
git commit -m "Add CT normal computation via structure tensor eigendecomposition"
```

---

### Task 4: CT Sheet Switching Metric

**Goal:** The actual metric class that plugs into the existing MetricComputer system. Samples mesh vertices, fetches CT neighborhoods, computes alignment scores.

**Files:**
- Create: `src/vesuvius_mesh_qa/metrics/ct_switching.py`
- Modify: `src/vesuvius_mesh_qa/metrics/summary.py` (add optional CT metric)
- Test: `tests/test_ct_switching.py`

**Step 1: Write the failing test**

```python
"""Tests for CT-informed sheet switching metric."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestCTSheetSwitchingMetric:
    def test_well_aligned_mesh_scores_high(self):
        """When CT normals agree with mesh normals, score should be high."""
        from vesuvius_mesh_qa.metrics.ct_switching import CTSheetSwitchingMetric

        metric = CTSheetSwitchingMetric.__new__(CTSheetSwitchingMetric)
        metric.name = "ct_sheet_switching"
        metric.weight = 0.20

        # Simulate: all vertices well-aligned (angles < 20 deg)
        angles = np.random.uniform(5, 20, size=100)
        anisotropies = np.ones(100) * 0.8  # clear structure everywhere
        score = metric._compute_score(angles, anisotropies)
        assert score > 0.8

    def test_misaligned_region_scores_low(self):
        """When some CT normals disagree with mesh normals, score drops."""
        from vesuvius_mesh_qa.metrics.ct_switching import CTSheetSwitchingMetric

        metric = CTSheetSwitchingMetric.__new__(CTSheetSwitchingMetric)
        metric.name = "ct_sheet_switching"
        metric.weight = 0.20

        # 70% well-aligned, 30% badly misaligned
        angles = np.concatenate([
            np.random.uniform(5, 20, size=70),
            np.random.uniform(60, 90, size=30),
        ])
        anisotropies = np.ones(100) * 0.8
        score = metric._compute_score(angles, anisotropies)
        assert score < 0.8

    def test_low_anisotropy_vertices_ignored(self):
        """Vertices in air/void (low anisotropy) should not hurt score."""
        from vesuvius_mesh_qa.metrics.ct_switching import CTSheetSwitchingMetric

        metric = CTSheetSwitchingMetric.__new__(CTSheetSwitchingMetric)
        metric.name = "ct_sheet_switching"
        metric.weight = 0.20

        # All badly aligned but in void (low anisotropy)
        angles = np.full(100, 80.0)
        anisotropies = np.full(100, 0.05)  # no structure
        score = metric._compute_score(angles, anisotropies)
        # Should not be penalized because CT has no signal there
        assert score > 0.8
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ct_switching.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
"""CT-informed sheet switching detection.

Compares mesh surface normals against CT-derived papyrus layer normals
using structure tensor analysis. Unlike the geometry-only SheetSwitchingMetric,
this detects switches between parallel layers where mesh normals stay similar
but the mesh has jumped to an adjacent papyrus sheet.

Requires a zarr volume URL. Gracefully returns a neutral score if
CT data is unavailable.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

from vesuvius_mesh_qa.metrics.base import MetricComputer, MetricResult
from vesuvius_mesh_qa.volume import VolumeAccessor
from vesuvius_mesh_qa.ct_normals import compute_ct_normal


class CTSheetSwitchingMetric(MetricComputer):
    """Detect sheet switching by comparing mesh normals to CT structure tensor normals.

    For each sampled vertex:
    1. Fetch a 32^3 CT neighborhood
    2. Compute structure tensor -> largest eigenvector = papyrus normal
    3. Measure angle between mesh normal and CT normal
    4. Weight by anisotropy (ignore vertices in void/air)

    Vertices with high anisotropy and large angular deviation are
    likely on a different layer than the CT suggests.
    """

    name: str = "ct_sheet_switching"
    weight: float = 0.20
    _n_samples: int = 500
    _half_size: int = 16
    _sigma: float = 3.0
    _misalignment_threshold_deg: float = 45.0
    _anisotropy_threshold: float = 0.1

    def __init__(self, volume_accessor: VolumeAccessor):
        self._accessor = volume_accessor

    def compute(self, mesh: o3d.geometry.TriangleMesh) -> MetricResult:
        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)

        if len(vertices) == 0:
            return MetricResult(
                name=self.name, score=0.0, weight=self.weight,
                details={"error": "mesh has no vertices"},
            )

        # Filter to vertices in bounds
        in_bounds = np.array([
            self._accessor.vertex_in_bounds(v, margin=self._half_size)
            for v in vertices
        ])
        valid_idx = np.where(in_bounds)[0]

        if len(valid_idx) == 0:
            return MetricResult(
                name=self.name, score=0.5, weight=self.weight,
                details={"error": "no vertices in volume bounds", "n_in_bounds": 0},
            )

        # Sample vertices
        rng = np.random.default_rng(42)
        n_sample = min(self._n_samples, len(valid_idx))
        sample_idx = rng.choice(valid_idx, size=n_sample, replace=False)

        angles = []
        anisotropies = []
        problem_vertices = []

        for vi in sample_idx:
            chunk = self._accessor.sample_neighborhood(vertices[vi], self._half_size)
            if chunk.mean() < 5:  # masked/empty region
                continue

            ct_normal_zyx, aniso = compute_ct_normal(chunk, self._sigma)
            ct_normal_xyz = ct_normal_zyx[[2, 1, 0]]  # ZYX -> XYZ

            alignment = abs(np.dot(normals[vi], ct_normal_xyz))
            angle_deg = float(np.degrees(np.arccos(min(alignment, 1.0))))

            angles.append(angle_deg)
            anisotropies.append(aniso)

            if aniso > self._anisotropy_threshold and angle_deg > self._misalignment_threshold_deg:
                problem_vertices.append(int(vi))

        if not angles:
            return MetricResult(
                name=self.name, score=0.5, weight=self.weight,
                details={"error": "no valid CT samples", "n_sampled": 0},
            )

        angles = np.array(angles)
        anisotropies = np.array(anisotropies)
        score = self._compute_score(angles, anisotropies)

        return MetricResult(
            name=self.name,
            score=score,
            weight=self.weight,
            details={
                "n_sampled": len(angles),
                "mean_angle_deg": float(angles.mean()),
                "median_angle_deg": float(np.median(angles)),
                "fraction_misaligned": float(
                    ((angles > self._misalignment_threshold_deg) &
                     (anisotropies > self._anisotropy_threshold)).mean()
                ),
                "mean_anisotropy": float(anisotropies.mean()),
                "n_problem_vertices": len(problem_vertices),
            },
        )

    def _compute_score(self, angles: np.ndarray, anisotropies: np.ndarray) -> float:
        """Score based on fraction of high-anisotropy vertices with good alignment.

        Only vertices with clear CT structure (anisotropy > threshold) count.
        Of those, the score is the fraction within the alignment threshold.
        """
        structured = anisotropies > self._anisotropy_threshold
        if structured.sum() == 0:
            return 1.0  # no structure to judge against

        misaligned = (angles > self._misalignment_threshold_deg) & structured
        fraction_bad = misaligned.sum() / structured.sum()
        return float(np.clip(1.0 - fraction_bad, 0.0, 1.0))
```

**Step 4: Run tests**

Run: `pytest tests/test_ct_switching.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/vesuvius_mesh_qa/metrics/ct_switching.py tests/test_ct_switching.py
git commit -m "Add CT-informed sheet switching metric using structure tensor alignment"
```

---

### Task 5: Integrate CT Metric into CLI

**Goal:** Add `--volume` CLI flag that enables the CT metric alongside existing geometry-only metrics.

**Files:**
- Modify: `src/vesuvius_mesh_qa/metrics/summary.py`
- Modify: `src/vesuvius_mesh_qa/cli.py`
- Modify: `pyproject.toml` (add zarr, fsspec to dependencies)

**Step 1: Update pyproject.toml dependencies**

Add to `dependencies` list:
```
"zarr>=2.16",
"fsspec>=2023.6",
"aiohttp>=3.8",
"vesuvius>=0.2",
```

**Step 2: Modify summary.py to support optional CT metric**

Add after the existing `compute_all_metrics` function:

```python
def compute_all_metrics(
    mesh: o3d.geometry.TriangleMesh,
    weight_overrides: dict[str, float] | None = None,
    on_progress: callable | None = None,
    volume_url: str | None = None,
) -> list[MetricResult]:
    """Compute all metrics on a mesh.

    Args:
        mesh: Open3D triangle mesh with normals computed.
        weight_overrides: Optional dict of {metric_name: new_weight}.
        on_progress: Optional callback(metric_name, index, total).
        volume_url: Optional zarr volume URL for CT-informed metrics.
    """
    metrics: list[MetricComputer] = [cls() for cls in DEFAULT_METRICS]

    if volume_url:
        from vesuvius_mesh_qa.volume import VolumeAccessor
        from vesuvius_mesh_qa.metrics.ct_switching import CTSheetSwitchingMetric
        accessor = VolumeAccessor(volume_url)
        metrics.append(CTSheetSwitchingMetric(accessor))

    n_metrics = len(metrics)
    results = []
    for i, metric in enumerate(metrics):
        if weight_overrides and metric.name in weight_overrides:
            metric.weight = weight_overrides[metric.name]
        if on_progress:
            on_progress(metric.name, i, n_metrics)
        result = metric.compute(mesh)
        results.append(result)
        gc.collect()
    return results
```

**Step 3: Add --volume flag to CLI**

In `cli.py`, add to the `score` command:

```python
@click.option("--volume", type=str, default=None,
              help="OME-Zarr volume URL for CT-informed sheet switching detection")
```

And pass `volume_url=volume` to `compute_all_metrics()`.

Update the progress bar total from hardcoded `6` to `7` when volume is provided.

**Step 4: Run existing tests to verify no regression**

Run: `pytest tests/ -v`
Expected: All 19 existing tests PASS (CT metric is only activated when `volume_url` is provided)

**Step 5: Commit**

```bash
git add pyproject.toml src/vesuvius_mesh_qa/metrics/summary.py src/vesuvius_mesh_qa/cli.py
git commit -m "Integrate CT sheet switching metric into CLI with --volume flag"
```

---

### Task 6: Integration Test on Real Data

**Goal:** Run the full tool with `--volume` on the PHerc1667 mesh and verify the CT metric produces meaningful output.

**Files:** None created (manual verification)

**Step 1: Run with CT metric enabled**

```bash
mesh-qa score data/segments/PHerc1667_20231210132040_normalized.obj \
  --volume 'https://data.aws.ash2txt.org/samples/PHerc1667/volumes/20231117161658-7.910um-53keV-masked.zarr/'
```

Expected: 7 metrics in output table. CT metric shows `mean_angle_deg`, `fraction_misaligned`, `n_sampled`. Score should be reasonable (>0.5 for a presumably decent mesh).

**Step 2: Run without CT metric (regression check)**

```bash
mesh-qa score data/segments/PHerc1667_20231210132040_normalized.obj
```

Expected: Same 6 metrics as before, no errors. Identical scores to pre-change behavior.

**Step 3: Run JSON output**

```bash
mesh-qa score data/segments/PHerc1667_20231210132040_normalized.obj \
  --volume 'https://data.aws.ash2txt.org/samples/PHerc1667/volumes/20231117161658-7.910um-53keV-masked.zarr/' \
  --format json
```

Expected: Valid JSON with ct_sheet_switching metric included.

**Step 4: Commit any fixes needed, then tag**

```bash
git add -A && git commit -m "Verify CT metric integration on real PHerc1667 data"
```

---

### Task 7: Update README and Documentation

**Goal:** Document the CT-informed metric in README, including usage, limitations, and what it detects that geometry-only cannot.

**Files:**
- Modify: `README.md`

**Step 1: Add CT metric documentation**

Add a new section after "Sheet Switching Detection":

```markdown
### CT-Informed Sheet Switching (Optional)

When a zarr volume URL is provided, an additional metric compares mesh normals
against CT-derived papyrus layer normals using structure tensor analysis:

\`\`\`bash
mesh-qa score segment.obj \
  --volume 'https://data.aws.ash2txt.org/samples/PHerc1667/volumes/20231117161658-7.910um-53keV-masked.zarr/'
\`\`\`

**Algorithm:**
1. Sample 500 mesh vertices
2. Fetch 32^3 CT neighborhood at each vertex position (lazy zarr access)
3. Compute 3D structure tensor (Holoborodko derivative kernels, sigma=3.0)
4. Extract largest eigenvector = papyrus sheet normal direction
5. Measure angular deviation between mesh normal and CT normal
6. Score = fraction of structured vertices (anisotropy > 0.1) with alignment < 45 deg

**What this catches:** Parallel-layer sheet switches where the mesh jumps to an
adjacent papyrus sheet but normals stay similar. The geometry-only detector
misses these because it only sees angular anomalies. The CT detector catches them
because the structure tensor reveals the actual papyrus orientation at each point.

**Requirements:** Network access to the scroll's zarr volume (fetched lazily,
typically 50-500 MB of chunks). Adds ~1-5 minutes to scoring time.
```

**Step 2: Update metrics table**

Add row:
```
| `ct_sheet_switching` | 0.20 | **CT-informed layer alignment** (requires `--volume`) |
```

**Step 3: Commit**

```bash
git add README.md
git commit -m "Document CT-informed sheet switching metric in README"
```

---

## Summary

| Task | What | Dependencies | Time estimate |
|------|------|-------------|---------------|
| 1 | Validate coordinate alignment | None | 15 min |
| 2 | Volume accessor module | None | 15 min |
| 3 | Structure tensor normal computation | Task 2 | 15 min |
| 4 | CT sheet switching metric | Tasks 2, 3 | 20 min |
| 5 | CLI integration | Tasks 2, 3, 4 | 10 min |
| 6 | Integration test on real data | Task 5 | 10 min (+ network time) |
| 7 | README update | Task 6 | 5 min |

**Decision gate at Task 1:** If coordinate alignment validation fails (median angle >35 deg on a known-good mesh), stop and debug the coordinate mapping before proceeding with Tasks 2-7.

**Dependency graph:** Task 1 is independent (validation script). Tasks 2 and 3 are independent of each other. Task 4 depends on 2+3. Task 5 depends on 4. Tasks 6 and 7 depend on 5.
