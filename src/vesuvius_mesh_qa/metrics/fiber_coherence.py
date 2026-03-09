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


def _compute_fiber_orientation_nnunet(
    volume: VolumeAccessor,
    vertices: np.ndarray,
    sample_indices: np.ndarray,
    model_path: str,
    half_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute fiber orientation using bruniss nnUNet fiber segmentation models.

    bruniss trained two separate binary segmentation models:
    - Dataset040_newHorizontals: detects horizontal fibers (bg=0, fiber=1)
    - Dataset041_newVerticals: detects vertical fibers (bg=0, fiber=1)

    We run both models on each CT patch and combine:
    - Neither detects fiber → class 0 (background)
    - Only hz model detects → class 1 (horizontal)
    - Only vt model detects → class 2 (vertical)
    - Both detect → take higher-confidence prediction

    model_path should point to the base directory containing both
    Dataset040_newHorizontals/ and Dataset041_newVerticals/ subdirectories.
    Each subdirectory contains the nnUNet trainer folder structure.

    Args:
        volume: CT volume accessor.
        vertices: (N, 3) vertex positions.
        sample_indices: Indices of vertices to sample.
        model_path: Path to base directory containing both model folders.
        half_size: Half-size of the CT patch for inference.

    Returns:
        fiber_dirs: (M, 3) placeholder direction vectors.
        fiber_class: (M,) integer array — 0=background, 1=horizontal, 2=vertical.
        confidence: (M,) float array — softmax probability of predicted class.
    """
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        import torch
    except ImportError:
        raise ImportError(
            "nnUNet fiber model requires 'nnunetv2' and 'torch'. "
            "Install with: pip install nnunetv2 torch"
        )

    import os

    n_samples = len(sample_indices)
    fiber_dirs = np.zeros((n_samples, 3), dtype=np.float64)
    fiber_class = np.zeros(n_samples, dtype=np.int32)
    confidence = np.zeros(n_samples, dtype=np.float64)

    # Find model directories
    hz_model_dir = _find_nnunet_model_dir(model_path, "Dataset040")
    vt_model_dir = _find_nnunet_model_dir(model_path, "Dataset041")

    if hz_model_dir is None and vt_model_dir is None:
        raise FileNotFoundError(
            f"No nnUNet model directories found in {model_path}. "
            "Expected Dataset040_newHorizontals/ and/or Dataset041_newVerticals/ "
            "subdirectories with nnUNet trainer folders."
        )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Extract all patches first (shared between both models)
    patches = []
    patch_valid = []
    for i, vi in enumerate(sample_indices):
        vertex = vertices[vi]
        if not volume.vertex_in_bounds(vertex, margin=half_size):
            patches.append(None)
            patch_valid.append(False)
            continue
        chunk = volume.sample_neighborhood(vertex, half_size=half_size)
        if chunk.size == 0 or chunk.max() == 0:
            # Skip empty patches (outside scan or in masked region)
            patches.append(None)
            patch_valid.append(False)
            continue
        patches.append(chunk)
        patch_valid.append(True)

    # Run horizontal model on all patches
    hz_classes = np.zeros(n_samples, dtype=np.int32)
    hz_probs = np.zeros(n_samples, dtype=np.float64)

    if hz_model_dir is not None:
        hz_predictor = _make_nnunet_predictor(hz_model_dir, device)
        for i in range(n_samples):
            if not patch_valid[i]:
                continue
            cls, prob = _predict_center_class(hz_predictor, patches[i])
            hz_classes[i] = cls
            hz_probs[i] = prob
        del hz_predictor
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        import gc
        gc.collect()

    # Run vertical model on all patches
    vt_classes = np.zeros(n_samples, dtype=np.int32)
    vt_probs = np.zeros(n_samples, dtype=np.float64)

    if vt_model_dir is not None:
        vt_predictor = _make_nnunet_predictor(vt_model_dir, device)
        for i in range(n_samples):
            if not patch_valid[i]:
                continue
            cls, prob = _predict_center_class(vt_predictor, patches[i])
            vt_classes[i] = cls
            vt_probs[i] = prob
        del vt_predictor
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        import gc
        gc.collect()

    # Combine: hz fiber=1 → class 1, vt fiber=1 → class 2
    for i in range(n_samples):
        if not patch_valid[i]:
            continue

        hz_is_fiber = hz_classes[i] == 1
        vt_is_fiber = vt_classes[i] == 1

        if hz_is_fiber and vt_is_fiber:
            # Both detected — take higher confidence
            if hz_probs[i] >= vt_probs[i]:
                fiber_class[i] = 1  # horizontal
                confidence[i] = hz_probs[i]
            else:
                fiber_class[i] = 2  # vertical
                confidence[i] = vt_probs[i]
        elif hz_is_fiber:
            fiber_class[i] = 1
            confidence[i] = hz_probs[i]
        elif vt_is_fiber:
            fiber_class[i] = 2
            confidence[i] = vt_probs[i]
        else:
            fiber_class[i] = 0  # background
            confidence[i] = max(hz_probs[i], vt_probs[i])

        # Set direction based on class
        if fiber_class[i] == 1:
            fiber_dirs[i] = [1.0, 0.0, 0.0]
        elif fiber_class[i] == 2:
            fiber_dirs[i] = [0.0, 0.0, 1.0]

    return fiber_dirs, fiber_class, confidence


def _find_nnunet_model_dir(base_path: str, dataset_prefix: str) -> str | None:
    """Find the nnUNet trainer directory for a dataset within base_path.

    Searches for directories matching the dataset prefix (e.g., "Dataset040")
    and returns the path to the trainer subdirectory containing fold_0/.
    """
    import os

    for entry in os.listdir(base_path):
        if entry.startswith(dataset_prefix):
            dataset_dir = os.path.join(base_path, entry)
            if not os.path.isdir(dataset_dir):
                continue
            # Find the trainer subdirectory (prefer 16G over 40G for memory)
            trainer_dirs = []
            for sub in os.listdir(dataset_dir):
                sub_path = os.path.join(dataset_dir, sub)
                fold_path = os.path.join(sub_path, "fold_0")
                if os.path.isdir(sub_path) and os.path.isdir(fold_path):
                    trainer_dirs.append(sub_path)
            # Prefer 16G variant for lower memory usage
            for td in trainer_dirs:
                if "16G" in td and "SkeletonRecall" not in td:
                    return td
            # Fall back to any available
            if trainer_dirs:
                return trainer_dirs[0]
    return None


def _make_nnunet_predictor(model_dir: str, device):
    """Create and initialize an nnUNetPredictor from a model directory."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=device,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth",
    )
    return predictor


def _predict_center_class(
    predictor, chunk: np.ndarray
) -> tuple[int, float]:
    """Run nnUNet prediction on a chunk and return center voxel class + probability."""
    patch = chunk.astype(np.float32)[np.newaxis, ...]  # (1, Z, Y, X)
    properties = {"spacing": [1.0, 1.0, 1.0]}

    seg, probs = predictor.predict_single_npy_array(
        patch, properties, save_or_return_probabilities=True,
    )

    cz, cy, cx = seg.shape[0] // 2, seg.shape[1] // 2, seg.shape[2] // 2
    center_class = int(seg[cz, cy, cx])
    center_prob = float(probs[center_class, cz, cy, cx]) if probs is not None else 1.0
    return center_class, center_prob


def _compute_fiber_orientation_predictions(
    volume: VolumeAccessor,
    vertices: np.ndarray,
    sample_indices: np.ndarray,
    predictions_url: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-computed fiber predictions from a Zarr store.

    For scrolls where nnUNet inference has already been run, pre-computed
    per-voxel fiber class labels are stored as Zarr arrays. We just sample
    at vertex positions — no inference needed.

    Args:
        volume: CT volume accessor (used for coordinate mapping).
        vertices: (N, 3) vertex positions.
        sample_indices: Indices of vertices to sample.
        predictions_url: URL/path to pre-computed fiber prediction Zarr.

    Returns:
        fiber_dirs: (M, 3) placeholder direction vectors.
        fiber_class: (M,) integer array — 0=background, 1=horizontal, 2=vertical.
        confidence: (M,) float array — 1.0 for all (pre-computed = high confidence).
    """
    import zarr

    store = zarr.open(predictions_url, mode="r")
    # Expect a 3D array of class labels (0/1/2)
    if isinstance(store, zarr.Group):
        # Try common array names
        for name in ["labels", "predictions", "0"]:
            if name in store:
                pred_array = store[name]
                break
        else:
            pred_array = store[list(store.array_keys())[0]]
    else:
        pred_array = store

    n_samples = len(sample_indices)
    fiber_dirs = np.zeros((n_samples, 3), dtype=np.float64)
    fiber_class = np.zeros(n_samples, dtype=np.int32)
    confidence = np.ones(n_samples, dtype=np.float64)

    for i, vi in enumerate(sample_indices):
        vertex = vertices[vi]
        # Convert vertex position to voxel coordinates
        voxel = np.round(vertex).astype(int)
        # Bounds check
        if any(v < 0 for v in voxel) or any(
            voxel[d] >= pred_array.shape[d] for d in range(3)
        ):
            continue

        cls_val = int(pred_array[voxel[0], voxel[1], voxel[2]])
        fiber_class[i] = cls_val
        if cls_val == 1:
            fiber_dirs[i] = [1.0, 0.0, 0.0]
        elif cls_val == 2:
            fiber_dirs[i] = [0.0, 0.0, 1.0]

    return fiber_dirs, fiber_class, confidence


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
        fiber_model_path: str | None = None,
        fiber_predictions_url: str | None = None,
    ) -> None:
        self._volume = volume_accessor
        self._n_samples = n_samples
        self._half_size = half_size
        self._n_rings = n_rings
        self._fiber_model_path = fiber_model_path
        self._fiber_predictions_url = fiber_predictions_url

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
        # When using nnUNet, limit samples since inference is slow (~12s/patch on MPS)
        n_samples = self._n_samples
        if self._fiber_model_path and n_samples > 50:
            n_samples = 50

        # Oversample 3x and spot-check for non-zero CT data.
        # Masked volumes have large empty regions where in-bounds vertices
        # map to all-zero data — we skip these upfront.
        oversample = min(n_samples * 3, len(valid_indices))
        if len(valid_indices) > oversample:
            candidate_indices = rng.choice(valid_indices, oversample, replace=False)
        else:
            candidate_indices = valid_indices.copy()

        # Filter to vertices with actual CT data (non-zero patches)
        live_indices = []
        for vi in candidate_indices:
            patch = self._volume.sample_neighborhood(vertices[vi], half_size=4)
            if patch.max() > 0:
                live_indices.append(vi)
            if len(live_indices) >= n_samples:
                break
        sample_indices = np.array(live_indices, dtype=np.intp) if live_indices else candidate_indices[:n_samples]

        # Compute fiber orientation using best available method:
        # 1. Pre-computed predictions (fastest, highest quality)
        # 2. nnUNet model (high accuracy, requires torch)
        # 3. Structure tensor (lightweight default)
        method = "structure_tensor"
        # confidence_threshold: minimum confidence to include a vertex in comparison.
        # For nnUNet/predictions: softmax probability (0-1), threshold at 0.5.
        # For structure tensor: eigenvalue anisotropy ratio — real papyrus CT
        # has anisotropy very close to 1.0, so we use 0.0 (no filtering).
        # Non-zero patch filtering already excludes empty regions.
        confidence_threshold = 0.0  # default for structure tensor (no filtering)
        if self._fiber_predictions_url:
            method = "predictions"
            confidence_threshold = 0.5
            fiber_dirs, fiber_class, confidence = (
                _compute_fiber_orientation_predictions(
                    self._volume, vertices, sample_indices,
                    self._fiber_predictions_url,
                )
            )
        elif self._fiber_model_path:
            try:
                method = "nnunet"
                confidence_threshold = 0.5
                fiber_dirs, fiber_class, confidence = (
                    _compute_fiber_orientation_nnunet(
                        self._volume, vertices, sample_indices,
                        self._fiber_model_path,
                        half_size=self._half_size,
                    )
                )
            except ImportError:
                method = "structure_tensor (nnunet unavailable)"
                fiber_dirs, fiber_class, confidence = (
                    _compute_fiber_orientation_structure_tensor(
                        self._volume, vertices, sample_indices,
                        half_size=self._half_size,
                    )
                )
        else:
            fiber_dirs, fiber_class, confidence = (
                _compute_fiber_orientation_structure_tensor(
                    self._volume, vertices, sample_indices,
                    half_size=self._half_size,
                )
            )

        # Compare each sample to its K nearest sampled neighbors in 3D space.
        # This is more robust than mesh adjacency for sparse sampling on large meshes.
        from scipy.spatial import KDTree

        # Build KD-tree from sampled vertex positions
        sample_positions = vertices[sample_indices]
        # Filter to valid (classified) samples for comparison
        valid_mask = (fiber_class > 0)
        for i in range(len(sample_indices)):
            if confidence[i] < confidence_threshold:
                valid_mask[i] = False

        valid_sample_idx = np.where(valid_mask)[0]

        n_compared = 0
        n_class_flips = 0
        n_direction_discontinuities = 0
        flip_vertices: list[int] = []

        if len(valid_sample_idx) >= 2:
            valid_positions = sample_positions[valid_sample_idx]
            tree = KDTree(valid_positions)
            k_neighbors = min(6, len(valid_sample_idx) - 1)

            for local_i, global_i in enumerate(valid_sample_idx):
                vi = int(sample_indices[global_i])
                pos = valid_positions[local_i]
                # Query k+1 nearest (includes self)
                dists, nn_idx = tree.query(pos, k=k_neighbors + 1)
                # Exclude self
                neighbor_local = [j for j in nn_idx if j != local_i][:k_neighbors]
                neighbor_global = [valid_sample_idx[j] for j in neighbor_local]

                if not neighbor_global:
                    continue

                # Compare fiber class
                my_class = fiber_class[global_i]
                neighbor_classes = fiber_class[neighbor_global]

                n_compared += 1

                # Class flip: majority of neighbors have different class
                n_same = int(np.sum(neighbor_classes == my_class))
                n_diff = int(np.sum(neighbor_classes != my_class))
                if n_diff > n_same:
                    n_class_flips += 1
                    flip_vertices.append(vi)

                # Direction discontinuity: fiber direction differs significantly
                my_dir = fiber_dirs[global_i]
                neighbor_dirs = fiber_dirs[neighbor_global]
                valid_dir_mask = np.linalg.norm(neighbor_dirs, axis=1) > 0.5
                if np.any(valid_dir_mask):
                    valid_ndirs = neighbor_dirs[valid_dir_mask]
                    dots = np.abs(np.sum(my_dir * valid_ndirs, axis=1))
                    dots = np.clip(dots, 0.0, 1.0)
                    mean_alignment = float(np.mean(dots))
                    if mean_alignment < 0.5:  # >60° average misalignment
                        n_direction_discontinuities += 1
                        if vi not in flip_vertices:
                            flip_vertices.append(vi)

        # Score
        if n_compared == 0:
            score = 1.0
            flip_fraction = 0.0
        else:
            total_flags = n_class_flips + n_direction_discontinuities
            flip_fraction = total_flags / n_compared
            # Linear mapping: 0% flagged = 1.0, 25% flagged = 0.0
            # Structure tensor classification is noisy (~5-10% baseline error),
            # so we use a wider threshold. Real sheet switches cause 30%+ flips.
            score = float(np.clip(1.0 - flip_fraction / 0.25, 0.0, 1.0))

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
                "method": method,
                "n_sampled": len(sample_indices),
                "n_compared": n_compared,
                "n_class_flips": n_class_flips,
                "n_direction_discontinuities": n_direction_discontinuities,
                "flip_fraction": flip_fraction,
                "mean_confidence": float(np.mean(confidence[confidence > 0])) if np.any(confidence > 0) else 0.0,
                "sample_indices": sample_indices,
                "fiber_class": fiber_class,
            },
            problem_faces=np.array(sorted(problem_faces), dtype=np.int64) if problem_faces else None,
        )
