"""Fiber coherence metric for parallel-layer sheet switch detection.

Detects sheet switches between tightly packed parallel papyrus layers by
analyzing fiber orientation consistency along the mesh surface.

Papyrus has crossed reed strips — horizontal and vertical fibers bonded
together. Each layer has a unique fiber pattern at any position. When a
mesh switches layers, the dominant fiber orientation changes abruptly.

Two methods for obtaining fiber orientation:
- Structure tensor (default): lightweight, no model needed
- nnUNet fiber model (optional): highest accuracy, requires --fiber-model

Both methods feed into the same scoring algorithm: compare fiber features
between topologically adjacent mesh vertices, flag abrupt changes.
"""

from __future__ import annotations


import numpy as np
import open3d as o3d
from scipy import sparse

from vesuvius_mesh_qa.metrics.base import MetricComputer, MetricResult
from vesuvius_mesh_qa.volume import VolumeAccessor


def _compute_fiber_orientation_structure_tensor(
    volume: VolumeAccessor,
    vertices: np.ndarray,
    sample_indices: np.ndarray,
    half_size: int = 16,
    sigma: float = 1.5,
    rho: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute fiber orientation at sampled vertices using structure tensor.

    For stripe-like papyrus fibers, the eigendecomposition of the 3D
    structure tensor gives:
    - Smallest eigenvalue eigenvector = fiber long axis direction
    - Largest eigenvalue eigenvector = sheet normal

    We extract the smallest eigenvector and classify it as horizontal
    or vertical based on its angle to the Z-axis.

    Args:
        volume: CT volume accessor.
        vertices: (N, 3) vertex positions.
        sample_indices: Indices of vertices to sample.
        half_size: Half-size of the CT patch to extract.
        sigma: Gaussian prefilter sigma for derivative computation.
        rho: Gaussian window sigma for tensor smoothing.

    Returns:
        fiber_dirs: (M, 3) normalized fiber direction vectors (smallest eigenvector).
        fiber_class: (M,) integer array — 0=background/invalid, 1=horizontal, 2=vertical.
        anisotropy: (M,) float array — eigenvalue ratio as confidence measure.
    """
    try:
        import structure_tensor as st
    except ImportError:
        # Fall back to manual computation
        return _compute_fiber_orientation_manual(
            volume, vertices, sample_indices, half_size, sigma, rho
        )

    n_samples = len(sample_indices)
    fiber_dirs = np.zeros((n_samples, 3), dtype=np.float64)
    fiber_class = np.zeros(n_samples, dtype=np.int32)
    anisotropy = np.zeros(n_samples, dtype=np.float64)

    for i, vi in enumerate(sample_indices):
        vertex = vertices[vi]
        if not volume.vertex_in_bounds(vertex, margin=half_size):
            continue

        chunk = volume.sample_neighborhood(vertex, half_size=half_size)
        if chunk.size == 0 or np.std(chunk) < 1e-6:
            continue

        # Compute 3D structure tensor
        # structure_tensor.structure_tensor_3d returns S as (6, *shape)
        # representing the unique elements of the symmetric 3x3 tensor
        S = st.structure_tensor_3d(chunk.astype(np.float64), sigma, rho)

        # Average the tensor over the patch to get one tensor per vertex
        S_mean = np.array([s.mean() for s in S])

        # Reconstruct 3x3 symmetric matrix from 6 unique elements
        # Order: S[0]=Szz, S[1]=Szy, S[2]=Szx, S[3]=Syy, S[4]=Syx, S[5]=Sxx
        T = np.array([
            [S_mean[0], S_mean[1], S_mean[2]],
            [S_mean[1], S_mean[3], S_mean[4]],
            [S_mean[2], S_mean[4], S_mean[5]],
        ])

        eigenvalues, eigenvectors = np.linalg.eigh(T)
        # eigenvalues are in ascending order
        # Smallest eigenvalue eigenvector = fiber direction (column 0)
        fiber_dir_zyx = eigenvectors[:, 0]

        # Convert ZYX to XYZ
        fiber_dir_xyz = fiber_dir_zyx[[2, 1, 0]]
        norm = np.linalg.norm(fiber_dir_xyz)
        if norm > 1e-10:
            fiber_dir_xyz /= norm

        fiber_dirs[i] = fiber_dir_xyz

        # Classify: horizontal vs vertical by angle to Z-axis
        # Z-axis in mesh coords is [0, 0, 1]
        z_dot = abs(fiber_dir_xyz[2])
        # threshold 1/sqrt(2) ≈ 0.707 (45°)
        if z_dot > 0.707:
            fiber_class[i] = 2  # vertical (fiber aligned with Z)
        else:
            fiber_class[i] = 1  # horizontal (fiber in XY plane)

        # Anisotropy: ratio of largest to smallest eigenvalue
        if eigenvalues[0] > 1e-10:
            anisotropy[i] = eigenvalues[2] / eigenvalues[0]
        else:
            anisotropy[i] = 1.0

    return fiber_dirs, fiber_class, anisotropy


def _compute_fiber_orientation_manual(
    volume: VolumeAccessor,
    vertices: np.ndarray,
    sample_indices: np.ndarray,
    half_size: int = 16,
    sigma: float = 1.5,
    rho: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback: compute structure tensor manually with scipy."""
    from scipy.ndimage import gaussian_filter

    n_samples = len(sample_indices)
    fiber_dirs = np.zeros((n_samples, 3), dtype=np.float64)
    fiber_class = np.zeros(n_samples, dtype=np.int32)
    anisotropy = np.zeros(n_samples, dtype=np.float64)

    for i, vi in enumerate(sample_indices):
        vertex = vertices[vi]
        if not volume.vertex_in_bounds(vertex, margin=half_size):
            continue

        chunk = volume.sample_neighborhood(vertex, half_size=half_size)
        if chunk.size == 0 or np.std(chunk) < 1e-6:
            continue

        # Smooth
        smoothed = gaussian_filter(chunk.astype(np.float64), sigma=sigma)

        # Compute gradients (ZYX order)
        gz, gy, gx = np.gradient(smoothed)

        # Structure tensor elements (averaged over patch with Gaussian window)
        Szz = gaussian_filter(gz * gz, sigma=rho).mean()
        Szy = gaussian_filter(gz * gy, sigma=rho).mean()
        Szx = gaussian_filter(gz * gx, sigma=rho).mean()
        Syy = gaussian_filter(gy * gy, sigma=rho).mean()
        Syx = gaussian_filter(gy * gx, sigma=rho).mean()
        Sxx = gaussian_filter(gx * gx, sigma=rho).mean()

        T = np.array([
            [Szz, Szy, Szx],
            [Szy, Syy, Syx],
            [Szx, Syx, Sxx],
        ])

        eigenvalues, eigenvectors = np.linalg.eigh(T)
        fiber_dir_zyx = eigenvectors[:, 0]
        fiber_dir_xyz = fiber_dir_zyx[[2, 1, 0]]
        norm = np.linalg.norm(fiber_dir_xyz)
        if norm > 1e-10:
            fiber_dir_xyz /= norm

        fiber_dirs[i] = fiber_dir_xyz

        z_dot = abs(fiber_dir_xyz[2])
        fiber_class[i] = 2 if z_dot > 0.707 else 1

        if eigenvalues[0] > 1e-10:
            anisotropy[i] = eigenvalues[2] / eigenvalues[0]
        else:
            anisotropy[i] = 1.0

    return fiber_dirs, fiber_class, anisotropy


def _build_vertex_adjacency_sparse(
    triangles: np.ndarray, n_vertices: int
) -> sparse.csr_matrix:
    """Build sparse vertex adjacency matrix from triangles."""
    edges_a = np.column_stack([triangles[:, 0], triangles[:, 1]])
    edges_b = np.column_stack([triangles[:, 1], triangles[:, 2]])
    edges_c = np.column_stack([triangles[:, 2], triangles[:, 0]])
    all_edges = np.vstack([edges_a, edges_b, edges_c])

    rows = np.concatenate([all_edges[:, 0], all_edges[:, 1]])
    cols = np.concatenate([all_edges[:, 1], all_edges[:, 0]])
    data = np.ones(len(rows), dtype=np.float32)
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    return adj


class FiberCoherenceMetric(MetricComputer):
    """Detect sheet switches via fiber orientation discontinuities.

    On a correct single-layer mesh, the dominant fiber orientation
    (horizontal vs vertical) is locally consistent — you're on the same
    face of the same papyrus sheet. A sheet switch to an adjacent layer
    causes the fiber pattern to change abruptly.

    This catches tightly-packed parallel-layer switches that geometry
    and winding angle metrics cannot detect.

    Requires CT volume access (--volume).
    """

    name: str = "fiber_coherence"
    weight: float = 0.10

    def __init__(
        self,
        volume_accessor: VolumeAccessor,
        *,
        n_samples: int = 300,
        half_size: int = 16,
        n_rings: int = 4,
    ) -> None:
        self._volume = volume_accessor
        self._n_samples = n_samples
        self._half_size = half_size
        self._n_rings = n_rings

    def compute(self, mesh: o3d.geometry.TriangleMesh) -> MetricResult:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        n_vertices = len(vertices)

        if len(triangles) == 0:
            return MetricResult(
                name=self.name, score=0.0, weight=self.weight,
                details={"error": "mesh has no triangles"},
            )

        # Select spatially distributed sample vertices
        in_bounds = np.array([
            self._volume.vertex_in_bounds(v, margin=self._half_size)
            for v in vertices
        ])
        valid_indices = np.where(in_bounds)[0]

        if len(valid_indices) < 10:
            return MetricResult(
                name=self.name, score=1.0, weight=self.weight,
                details={"n_sampled": 0, "note": "too few in-bounds vertices"},
            )

        rng = np.random.default_rng(42)
        if len(valid_indices) > self._n_samples:
            sample_indices = rng.choice(valid_indices, self._n_samples, replace=False)
        else:
            sample_indices = valid_indices

        # Compute fiber orientation at each sampled vertex
        fiber_dirs, fiber_class, anisotropy = (
            _compute_fiber_orientation_structure_tensor(
                self._volume, vertices, sample_indices,
                half_size=self._half_size,
            )
        )

        # Build neighborhood for comparison
        # Use multi-ring adjacency for wider spatial context
        adj = _build_vertex_adjacency_sparse(triangles, n_vertices)
        adj_k = adj
        for _ in range(self._n_rings - 1):
            adj_k = adj_k.dot(adj)
            adj_k.data[:] = 1.0

        # For each sampled vertex, compare fiber features to neighbors
        sample_idx_map = {int(v): i for i, v in enumerate(sample_indices)}

        n_compared = 0
        n_class_flips = 0
        n_direction_discontinuities = 0
        flip_vertices: list[int] = []

        for i, vi in enumerate(sample_indices):
            if fiber_class[i] == 0:  # invalid
                continue
            if anisotropy[i] < 2.0:  # low confidence — skip
                continue

            # Get neighbors that are also sampled
            row_start = adj_k.indptr[vi]
            row_end = adj_k.indptr[vi + 1]
            neighbor_verts = adj_k.indices[row_start:row_end]

            neighbor_samples = [
                sample_idx_map[int(nv)]
                for nv in neighbor_verts
                if int(nv) in sample_idx_map and int(nv) != vi
            ]

            if not neighbor_samples:
                continue

            # Compare fiber class
            my_class = fiber_class[i]
            neighbor_classes = fiber_class[neighbor_samples]
            valid_neighbors = neighbor_classes > 0
            if not np.any(valid_neighbors):
                continue

            n_compared += 1
            neighbor_valid_classes = neighbor_classes[valid_neighbors]

            # Class flip: majority of neighbors have different class
            n_same = np.sum(neighbor_valid_classes == my_class)
            n_diff = np.sum(neighbor_valid_classes != my_class)
            if n_diff > n_same:
                n_class_flips += 1
                flip_vertices.append(int(vi))

            # Direction discontinuity: fiber direction differs significantly
            my_dir = fiber_dirs[i]
            neighbor_dirs = fiber_dirs[neighbor_samples]
            valid_dir_mask = np.linalg.norm(neighbor_dirs, axis=1) > 0.5
            if np.any(valid_dir_mask):
                valid_ndirs = neighbor_dirs[valid_dir_mask]
                # Use absolute dot product (fiber direction is ambiguous ±)
                dots = np.abs(np.sum(my_dir * valid_ndirs, axis=1))
                dots = np.clip(dots, 0.0, 1.0)
                mean_alignment = float(np.mean(dots))
                if mean_alignment < 0.5:  # >60° average misalignment
                    n_direction_discontinuities += 1
                    if int(vi) not in flip_vertices:
                        flip_vertices.append(int(vi))

        # Score
        if n_compared == 0:
            score = 1.0
            flip_fraction = 0.0
        else:
            total_flags = n_class_flips + n_direction_discontinuities
            flip_fraction = total_flags / n_compared
            # Linear mapping: 0% flagged = 1.0, 10% flagged = 0.0
            score = float(np.clip(1.0 - flip_fraction / 0.10, 0.0, 1.0))

        # Map flip vertices to faces
        problem_faces = []
        if flip_vertices:
            flip_set = set(flip_vertices)
            for fi, tri in enumerate(triangles):
                if int(tri[0]) in flip_set or int(tri[1]) in flip_set or int(tri[2]) in flip_set:
                    problem_faces.append(fi)

        return MetricResult(
            name=self.name,
            score=score,
            weight=self.weight,
            details={
                "n_sampled": len(sample_indices),
                "n_compared": n_compared,
                "n_class_flips": n_class_flips,
                "n_direction_discontinuities": n_direction_discontinuities,
                "flip_fraction": flip_fraction,
                "mean_anisotropy": float(np.mean(anisotropy[anisotropy > 0])) if np.any(anisotropy > 0) else 0.0,
            },
            problem_faces=np.array(sorted(problem_faces), dtype=np.int64) if problem_faces else None,
        )
