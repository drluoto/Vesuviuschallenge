"""Test smaller chunk sizes and better scoring."""
import numpy as np
import open3d as o3d
from vesuvius_mesh_qa.volume import VolumeAccessor
from vesuvius_mesh_qa.ct_normals import compute_ct_normal

MESH = "data/bruniss/manual/20240413132301.obj"
VOLUME = "https://data.aws.ash2txt.org/samples/PHerc1667/volumes/20231117161658-7.910um-53keV-masked.zarr/"

mesh = o3d.io.read_triangle_mesh(MESH)
mesh.compute_vertex_normals()
vertices = np.asarray(mesh.vertices)
normals = np.asarray(mesh.vertex_normals)
vol = VolumeAccessor(VOLUME)

rng = np.random.default_rng(42)
in_bounds = np.array([vol.vertex_in_bounds(v, margin=16) for v in vertices])
valid = np.where(in_bounds)[0]
sample = rng.choice(valid, 50, replace=False)

print("Half-size sweep (sigma=3.0):")
for half_size in [4, 6, 8, 10, 12, 16]:
    test_angles = []
    for vi in sample:
        v = vertices[vi]
        mn = normals[vi]
        mn = mn / (np.linalg.norm(mn) + 1e-10)

        # Check bounds for this half_size
        if not vol.vertex_in_bounds(v, margin=half_size):
            continue

        chunk = vol.sample_neighborhood(v, half_size=half_size)
        # Adjust sigma relative to chunk size
        sigma = min(3.0, half_size / 3.0)
        ct_zyx, aniso = compute_ct_normal(chunk, sigma=sigma)
        ct_xyz = ct_zyx[[2, 1, 0]]
        ct_norm = np.linalg.norm(ct_xyz)
        if ct_norm > 1e-10:
            ct_xyz = ct_xyz / ct_norm
        dot = np.abs(np.dot(mn, ct_xyz))
        dot = np.clip(dot, 0.0, 1.0)
        test_angles.append(float(np.degrees(np.arccos(dot))))
    a = np.array(test_angles)
    print(f"  half={half_size} ({half_size*2}³ chunk, ~{half_size*2*7.91:.0f}µm), sigma={sigma:.1f}: "
          f"n={len(a)}, median={np.median(a):.1f}°, mean={np.mean(a):.1f}°, "
          f"<20°: {(a<20).sum()}/{len(a)}, >45°: {(a>45).sum()}/{len(a)}")

# Now test half_size=16 with different sigma values
print("\nSigma sweep at half_size=16 (32³ chunk):")
for sigma in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    test_angles = []
    for vi in sample:
        v = vertices[vi]
        mn = normals[vi]
        mn = mn / (np.linalg.norm(mn) + 1e-10)
        chunk = vol.sample_neighborhood(v, half_size=16)
        ct_zyx, aniso = compute_ct_normal(chunk, sigma=sigma)
        ct_xyz = ct_zyx[[2, 1, 0]]
        ct_norm = np.linalg.norm(ct_xyz)
        if ct_norm > 1e-10:
            ct_xyz = ct_xyz / ct_norm
        dot = np.abs(np.dot(mn, ct_xyz))
        dot = np.clip(dot, 0.0, 1.0)
        test_angles.append(float(np.degrees(np.arccos(dot))))
    a = np.array(test_angles)
    print(f"  sigma={sigma}: median={np.median(a):.1f}°, mean={np.mean(a):.1f}°, "
          f"<20°: {(a<20).sum()}/{len(a)}, >45°: {(a>45).sum()}/{len(a)}")

# Test: half_size=8 with sigma sweep
print("\nSigma sweep at half_size=8 (16³ chunk, ~127µm):")
for sigma in [1.0, 1.5, 2.0, 2.5, 3.0]:
    test_angles = []
    for vi in sample:
        v = vertices[vi]
        mn = normals[vi]
        mn = mn / (np.linalg.norm(mn) + 1e-10)
        chunk = vol.sample_neighborhood(v, half_size=8)
        ct_zyx, aniso = compute_ct_normal(chunk, sigma=sigma)
        ct_xyz = ct_zyx[[2, 1, 0]]
        ct_norm = np.linalg.norm(ct_xyz)
        if ct_norm > 1e-10:
            ct_xyz = ct_xyz / ct_norm
        dot = np.abs(np.dot(mn, ct_xyz))
        dot = np.clip(dot, 0.0, 1.0)
        test_angles.append(float(np.degrees(np.arccos(dot))))
    a = np.array(test_angles)
    print(f"  sigma={sigma}: median={np.median(a):.1f}°, mean={np.mean(a):.1f}°, "
          f"<20°: {(a<20).sum()}/{len(a)}, >45°: {(a>45).sum()}/{len(a)}")
