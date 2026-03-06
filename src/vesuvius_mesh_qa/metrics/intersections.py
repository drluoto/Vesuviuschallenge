"""Self-intersection detection for mesh quality assessment.

Uses random sampling with triangle-triangle intersection testing (Moller 1997)
for accurate self-intersection detection on large meshes.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from vesuvius_mesh_qa.metrics.base import MetricComputer, MetricResult


def _coplanar_tri_tri_2d(
    a0: np.ndarray, a1: np.ndarray, a2: np.ndarray,
    b0: np.ndarray, b1: np.ndarray, b2: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """2D overlap test for coplanar triangles using SAT with in-plane edge normals.

    Projects triangles onto the dominant plane (drop the axis most aligned with
    the face normal) and tests 6 separating axes (edge perpendiculars in 2D).

    Args:
        a0, a1, a2: Shape (3,) - vertices of triangle A.
        b0, b1, b2: Shape (N, 3) - vertices of N candidate triangles B.
        normal: Shape (3,) - shared face normal direction.

    Returns:
        Boolean array (N,) - True where triangles overlap in 2D.
    """
    n = len(b0)

    # Choose projection plane: drop axis most aligned with normal
    abs_n = np.abs(normal)
    drop = np.argmax(abs_n)
    keep = [i for i in range(3) if i != drop]

    # Project to 2D
    a0_2d = a0[keep]  # (2,)
    a1_2d = a1[keep]
    a2_2d = a2[keep]
    b0_2d = b0[:, keep]  # (N, 2)
    b1_2d = b1[:, keep]
    b2_2d = b2[:, keep]

    separated = np.zeros(n, dtype=bool)

    # Test 6 edge-perpendicular axes (3 from A, 3 from B)
    a_edges = [a1_2d - a0_2d, a2_2d - a1_2d, a0_2d - a2_2d]
    a_verts_2d = np.array([a0_2d, a1_2d, a2_2d])  # (3, 2)

    for edge in a_edges:
        # Perpendicular in 2D: (dx, dy) -> (-dy, dx)
        axis = np.array([-edge[1], edge[0]])
        if np.dot(axis, axis) < 1e-12:
            continue
        pa = a_verts_2d @ axis  # (3,)
        pb0 = b0_2d @ axis  # (N,)
        pb1 = b1_2d @ axis
        pb2 = b2_2d @ axis
        a_min, a_max = pa.min(), pa.max()
        b_min = np.minimum(np.minimum(pb0, pb1), pb2)
        b_max = np.maximum(np.maximum(pb0, pb1), pb2)
        separated |= (a_max <= b_min) | (b_max <= a_min)

    b_edges_list = [
        (b1_2d - b0_2d),  # (N, 2)
        (b2_2d - b1_2d),
        (b0_2d - b2_2d),
    ]
    for be in b_edges_list:
        if np.all(separated):
            break
        # Per-candidate perpendicular axis: (-dy, dx)
        axis = np.stack([-be[:, 1], be[:, 0]], axis=1)  # (N, 2)
        lens_sq = np.sum(axis * axis, axis=1)
        valid = lens_sq > 1e-12
        if not np.any(valid):
            continue
        # Project A vertices: (3, 2) dot (N, 2) -> we need (N, 3)
        pa = a_verts_2d @ axis.T  # (3, N)
        pa_min = pa.min(axis=0)  # (N,)
        pa_max = pa.max(axis=0)
        pb0 = np.sum(b0_2d * axis, axis=1)
        pb1 = np.sum(b1_2d * axis, axis=1)
        pb2 = np.sum(b2_2d * axis, axis=1)
        pb_min = np.minimum(np.minimum(pb0, pb1), pb2)
        pb_max = np.maximum(np.maximum(pb0, pb1), pb2)
        separated |= ((pa_max <= pb_min) | (pb_max <= pa_min)) & valid

    return ~separated


def _tri_tri_intersect_batch(tri_a: np.ndarray, tri_b: np.ndarray) -> np.ndarray:
    """Test if triangle A intersects each of N triangles in B.

    Implements the Moller (1997) algorithm:
    1. Plane rejection: if all vertices of one triangle are on the same side
       of the other triangle's plane, no intersection.
    2. Coplanar case: 2D SAT with edge perpendiculars.
    3. General case: compute intersection intervals on the line where the two
       planes meet; overlap means intersection.

    Args:
        tri_a: Shape (3, 3) - vertices of triangle A.
        tri_b: Shape (N, 3, 3) - vertices of N candidate triangles.

    Returns:
        Boolean array of shape (N,) — True where intersection detected.
    """
    n = len(tri_b)
    if n == 0:
        return np.array([], dtype=bool)

    a0, a1, a2 = tri_a[0], tri_a[1], tri_a[2]  # each (3,)
    b0 = tri_b[:, 0]  # (N, 3)
    b1 = tri_b[:, 1]
    b2 = tri_b[:, 2]

    # --- Step 1: Plane of triangle A ---
    ea01 = a1 - a0
    ea02 = a2 - a0
    na = np.cross(ea01, ea02)  # (3,) normal of A
    da = np.dot(na, a0)  # plane equation: na . x = da

    # Signed distances of B vertices to plane of A
    db0 = np.dot(b0, na) - da  # (N,)
    db1 = np.dot(b1, na) - da
    db2 = np.dot(b2, na) - da

    # Reject if all B vertices on same side of A's plane
    all_pos_b = (db0 > 1e-8) & (db1 > 1e-8) & (db2 > 1e-8)
    all_neg_b = (db0 < -1e-8) & (db1 < -1e-8) & (db2 < -1e-8)
    rejected = all_pos_b | all_neg_b

    # --- Step 2: Plane of each triangle B ---
    eb01 = b1 - b0  # (N, 3)
    eb02 = b2 - b0
    nb = np.cross(eb01, eb02)  # (N, 3)
    db_plane = np.sum(nb * b0, axis=1)  # (N,)

    # Signed distances of A vertices to plane of each B
    da0 = np.dot(nb, a0) - db_plane  # (N,) broadcast: nb (N,3) . a0 (3,) -> sum
    da1 = np.dot(nb, a1) - db_plane
    da2 = np.dot(nb, a2) - db_plane

    all_pos_a = (da0 > 1e-8) & (da1 > 1e-8) & (da2 > 1e-8)
    all_neg_a = (da0 < -1e-8) & (da1 < -1e-8) & (da2 < -1e-8)
    rejected |= all_pos_a | all_neg_a

    result = np.zeros(n, dtype=bool)

    # --- Step 3: Coplanar case ---
    coplanar = (~rejected) & (np.abs(db0) < 1e-8) & (np.abs(db1) < 1e-8) & (np.abs(db2) < 1e-8)
    cop_idx = np.where(coplanar)[0]
    if len(cop_idx) > 0:
        result[cop_idx] = _coplanar_tri_tri_2d(
            a0, a1, a2,
            b0[cop_idx], b1[cop_idx], b2[cop_idx],
            na,
        )

    # --- Step 4: General (non-coplanar, non-rejected) case ---
    general = (~rejected) & (~coplanar)
    gen_idx = np.where(general)[0]
    if len(gen_idx) > 0:
        result[gen_idx] = _general_intersection(
            a0, a1, a2, na,
            b0[gen_idx], b1[gen_idx], b2[gen_idx], nb[gen_idx],
            da0[gen_idx], da1[gen_idx], da2[gen_idx],
            db0[gen_idx], db1[gen_idx], db2[gen_idx],
        )

    return result


def _general_intersection(
    a0, a1, a2, na,
    b0, b1, b2, nb,
    da0, da1, da2,
    db0, db1, db2,
) -> np.ndarray:
    """Compute intersection intervals on the plane-plane intersection line.

    For each triangle pair, compute where each triangle intersects the line
    formed by the two planes meeting. If the intervals overlap, the triangles
    intersect.

    Returns:
        Boolean array (N,) - True where triangles intersect.
    """
    n = len(b0)

    # Direction of intersection line
    D = np.cross(na, nb)  # (N, 3) — na is (3,), nb is (N,3)

    # Project vertices onto the intersection line direction.
    # Use the component with largest absolute value for numerical stability.
    abs_D = np.abs(D)
    # For each candidate, pick the dominant axis
    proj_axis = np.argmax(abs_D, axis=1)  # (N,)

    # Project A vertices onto dominant axis of D
    pa0 = np.full(n, a0[0])
    pa1 = np.full(n, a1[0])
    pa2 = np.full(n, a2[0])
    mask_y = proj_axis == 1
    mask_z = proj_axis == 2
    pa0[mask_y] = a0[1]; pa1[mask_y] = a1[1]; pa2[mask_y] = a2[1]
    pa0[mask_z] = a0[2]; pa1[mask_z] = a1[2]; pa2[mask_z] = a2[2]

    pb0_vals = np.where(proj_axis == 0, b0[:, 0], np.where(proj_axis == 1, b0[:, 1], b0[:, 2]))
    pb1_vals = np.where(proj_axis == 0, b1[:, 0], np.where(proj_axis == 1, b1[:, 1], b1[:, 2]))
    pb2_vals = np.where(proj_axis == 0, b2[:, 0], np.where(proj_axis == 1, b2[:, 1], b2[:, 2]))

    # Compute interval for triangle A on intersection line
    # A vertex is "on the line side" if its signed distance to B's plane changes sign
    # The interval endpoints are where the triangle edges cross the plane
    t_a_min, t_a_max = _compute_interval(pa0, pa1, pa2, da0, da1, da2)
    t_b_min, t_b_max = _compute_interval(pb0_vals, pb1_vals, pb2_vals, db0, db1, db2)

    # Check interval overlap (strict)
    return (t_a_min < t_b_max) & (t_b_min < t_a_max)


def _compute_interval(p0, p1, p2, d0, d1, d2):
    """Compute the interval [t_min, t_max] where a triangle crosses a plane.

    Given projected positions p0,p1,p2 on the intersection line and signed
    distances d0,d1,d2 to the other triangle's plane, compute the two
    intersection points (where edges cross the plane).

    Each of p0,p1,p2,d0,d1,d2 has shape (N,).
    Returns (t_min, t_max) each shape (N,).
    """
    n = len(p0)
    t_min = np.full(n, np.inf)
    t_max = np.full(n, -np.inf)

    # For each edge, if the two endpoints have different signs of d,
    # compute the crossing point by linear interpolation.
    edges = [(0, 1), (0, 2), (1, 2)]
    ps = [p0, p1, p2]
    ds = [d0, d1, d2]

    for i, j in edges:
        pi, pj = ps[i], ps[j]
        di, dj = ds[i], ds[j]
        # Edge crosses if signs differ (or one is zero)
        denom = di - dj
        # Avoid division by zero
        valid = np.abs(denom) > 1e-12
        safe_denom = np.where(valid, denom, 1.0)
        t = np.where(valid, pi + (pj - pi) * di / safe_denom, pi)
        # This edge contributes a crossing point when signs differ
        crosses = (di * dj) <= 0  # includes zero
        t_min = np.where(crosses & (t < t_min), t, t_min)
        t_max = np.where(crosses & (t > t_max), t, t_max)

    # If a vertex is exactly on the plane (d=0), it's a crossing point too
    for i in range(3):
        on_plane = np.abs(ds[i]) < 1e-8
        t_min = np.where(on_plane & (ps[i] < t_min), ps[i], t_min)
        t_max = np.where(on_plane & (ps[i] > t_max), ps[i], t_max)

    return t_min, t_max


def _check_intersections_vectorized(
    vertices: np.ndarray,
    triangles: np.ndarray,
    sample_size: int = 2_000,
    neighbors_per_sample: int = 30,
) -> tuple[int, float, bool]:
    """Detect self-intersections by sampling faces and testing nearby non-adjacent faces.

    Uses Moller triangle-triangle intersection test instead of AABB overlap
    for accurate results on curved 3D surfaces.
    """
    n_faces = len(triangles)
    was_subsampled = n_faces > sample_size

    # Compute per-face data
    tri_verts = vertices[triangles]  # (F, 3, 3)
    centroids = tri_verts.mean(axis=1)  # (F, 3)

    # Build vertex-to-face adjacency vectorized
    flat_verts = triangles.ravel()  # (F*3,)
    flat_faces = np.repeat(np.arange(n_faces), 3)  # (F*3,)
    n_verts = vertices.shape[0]
    sort_idx = np.argsort(flat_verts)
    sorted_verts = flat_verts[sort_idx]
    sorted_faces = flat_faces[sort_idx]
    vert_starts = np.searchsorted(sorted_verts, np.arange(n_verts), side='left')
    vert_ends = np.searchsorted(sorted_verts, np.arange(n_verts), side='right')

    def _get_adjacent_faces(face_idx: int) -> set[int]:
        adj = {face_idx}
        for vi in triangles[face_idx]:
            s, e = vert_starts[vi], vert_ends[vi]
            for fi in sorted_faces[s:e]:
                adj.add(fi)
        return adj

    # Sample faces
    rng = np.random.default_rng(42)
    if was_subsampled:
        sample_indices = rng.choice(n_faces, sample_size, replace=False)
    else:
        sample_indices = np.arange(n_faces)

    # Build KD-tree on centroids for fast nearest-neighbor lookup
    tree = cKDTree(centroids)

    intersecting_faces: set[int] = set()
    n_intersecting_pairs = 0

    k = neighbors_per_sample + 20  # buffer for adjacency exclusion

    for si in sample_indices:
        si = int(si)
        _, neighbor_indices = tree.query(centroids[si], k=min(k, n_faces))

        adjacent = _get_adjacent_faces(si)
        candidates = np.array([idx for idx in neighbor_indices if idx not in adjacent])

        if len(candidates) == 0:
            continue

        candidates = candidates[:neighbors_per_sample]

        intersects = _tri_tri_intersect_batch(tri_verts[si], tri_verts[candidates])

        n_hits = int(intersects.sum())
        if n_hits > 0:
            n_intersecting_pairs += n_hits
            intersecting_faces.add(si)
            for ci in candidates[intersects]:
                intersecting_faces.add(int(ci))

    n_tested = len(sample_indices)
    intersection_fraction = len(intersecting_faces) / max(n_tested, 1)

    if was_subsampled:
        scale = n_faces / n_tested
        n_intersecting_pairs = int(n_intersecting_pairs * scale)

    return n_intersecting_pairs, intersection_fraction, was_subsampled


class SelfIntersectionMetric(MetricComputer):
    """Detects self-intersecting triangles in a mesh.

    Uses spatial nearest-neighbor search with Moller triangle-triangle
    intersection testing for accurate detection on curved 3D surfaces.

    Score mapping:
        0% intersecting faces -> 1.0
        5%+ intersecting faces -> 0.0
    """

    name: str = "self_intersections"
    weight: float = 0.15

    def compute(self, mesh: o3d.geometry.TriangleMesh) -> MetricResult:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        if len(triangles) == 0:
            return MetricResult(
                name=self.name, score=1.0, weight=self.weight,
                details={
                    "n_intersecting_pairs": 0,
                    "intersection_fraction": 0.0,
                    "was_subsampled": False,
                    "n_faces": 0,
                },
            )

        n_intersecting_pairs, intersection_fraction, was_subsampled = (
            _check_intersections_vectorized(vertices, triangles)
        )

        clamped = float(np.clip(intersection_fraction, 0.0, 0.05))
        score = 1.0 - clamped * 20.0

        return MetricResult(
            name=self.name,
            score=score,
            weight=self.weight,
            details={
                "n_intersecting_pairs": n_intersecting_pairs,
                "intersection_fraction": intersection_fraction,
                "was_subsampled": was_subsampled,
                "n_faces": len(triangles),
            },
        )
