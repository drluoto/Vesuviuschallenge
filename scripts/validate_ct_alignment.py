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
