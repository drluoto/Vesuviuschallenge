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
