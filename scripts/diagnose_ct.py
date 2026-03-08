"""Diagnose CT alignment issue by inspecting individual vertices."""
import numpy as np
import open3d as o3d
from vesuvius_mesh_qa.volume import VolumeAccessor
from vesuvius_mesh_qa.ct_normals import compute_ct_normal

MESH = "data/bruniss/manual/20240413132301.obj"
VOLUME = "https://data.aws.ash2txt.org/samples/PHerc1667/volumes/20231117161658-7.910um-53keV-masked.zarr/"

print("Loading mesh...")
mesh = o3d.io.read_triangle_mesh(MESH)
mesh.compute_vertex_normals()
vertices = np.asarray(mesh.vertices)
normals = np.asarray(mesh.vertex_normals)

print("Connecting to volume...")
vol = VolumeAccessor(VOLUME)
print(f"  Volume shape (Z,Y,X): {vol.shape}")
print(f"  Mesh vertex range: X=[{vertices[:,0].min():.0f},{vertices[:,0].max():.0f}] "
      f"Y=[{vertices[:,1].min():.0f},{vertices[:,1].max():.0f}] "
      f"Z=[{vertices[:,2].min():.0f},{vertices[:,2].max():.0f}]")

# Sample 20 vertices from the interior
rng = np.random.default_rng(42)
in_bounds = np.array([vol.vertex_in_bounds(v, margin=16) for v in vertices])
valid = np.where(in_bounds)[0]
sample = rng.choice(valid, min(20, len(valid)), replace=False)

print(f"\n{'idx':>6} | {'mesh_normal (XYZ)':>30} | {'ct_normal (ZYX)':>30} | {'ct_normal (XYZ)':>30} | {'angle':>6} | {'aniso':>6}")
print("-" * 130)

angles = []
for vi in sample:
    v = vertices[vi]
    mn = normals[vi]
    mn = mn / (np.linalg.norm(mn) + 1e-10)

    chunk = vol.sample_neighborhood(v, half_size=16)
    ct_zyx, aniso = compute_ct_normal(chunk, sigma=3.0)

    # Try different reorderings
    ct_xyz = ct_zyx[[2, 1, 0]]  # current: ZYX -> XYZ
    ct_norm = np.linalg.norm(ct_xyz)
    if ct_norm > 1e-10:
        ct_xyz_n = ct_xyz / ct_norm
    else:
        ct_xyz_n = ct_xyz

    dot = np.abs(np.dot(mn, ct_xyz_n))
    dot = np.clip(dot, 0.0, 1.0)
    angle = float(np.degrees(np.arccos(dot)))
    angles.append(angle)

    print(f"{vi:6d} | [{mn[0]:8.4f} {mn[1]:8.4f} {mn[2]:8.4f}] | "
          f"[{ct_zyx[0]:8.4f} {ct_zyx[1]:8.4f} {ct_zyx[2]:8.4f}] | "
          f"[{ct_xyz_n[0]:8.4f} {ct_xyz_n[1]:8.4f} {ct_xyz_n[2]:8.4f}] | "
          f"{angle:6.1f} | {aniso:6.3f}")

print(f"\nMedian angle: {np.median(angles):.1f}°  Mean: {np.mean(angles):.1f}°")

# Now test: what if we DON'T reorder? (leave as ZYX)
print("\n--- Test: skip ZYX->XYZ reorder (use CT normal as-is) ---")
angles2 = []
for vi in sample:
    v = vertices[vi]
    mn = normals[vi]
    mn = mn / (np.linalg.norm(mn) + 1e-10)
    chunk = vol.sample_neighborhood(v, half_size=16)
    ct_zyx, aniso = compute_ct_normal(chunk, sigma=3.0)
    ct_norm = np.linalg.norm(ct_zyx)
    if ct_norm > 1e-10:
        ct_n = ct_zyx / ct_norm
    else:
        ct_n = ct_zyx
    dot = np.abs(np.dot(mn, ct_n))
    dot = np.clip(dot, 0.0, 1.0)
    angle = float(np.degrees(np.arccos(dot)))
    angles2.append(angle)
print(f"Median angle (no reorder): {np.median(angles2):.1f}°  Mean: {np.mean(angles2):.1f}°")

# Test: what if mesh is XYZ but CT is already XYZ (no reorder needed)?
# Test: smallest eigenvector instead of largest
print("\n--- Test: use SMALLEST eigenvector (sheet-parallel direction) ---")
angles3 = []
for vi in sample:
    v = vertices[vi]
    mn = normals[vi]
    mn = mn / (np.linalg.norm(mn) + 1e-10)
    chunk = vol.sample_neighborhood(v, half_size=16)

    from vesuvius.image_proc.geometry.structure_tensor import StructureTensorComputer
    stc = StructureTensorComputer(sigma=3.0, device='cpu')
    st = stc.compute(chunk, as_numpy=True)
    cz, cy, cx = st.shape[1]//2, st.shape[2]//2, st.shape[3]//2
    components = [st[i, cz, cy, cx] for i in range(6)]
    mat = np.array([[components[0], components[1], components[2]],
                    [components[1], components[3], components[4]],
                    [components[2], components[4], components[5]]])
    eigenvalues, eigenvectors = np.linalg.eigh(mat)
    # Smallest eigenvector = eigenvectors[:, 0]
    ct_small = eigenvectors[:, 0]
    ct_small_xyz = ct_small[[2, 1, 0]]
    ct_norm = np.linalg.norm(ct_small_xyz)
    if ct_norm > 1e-10:
        ct_small_xyz = ct_small_xyz / ct_norm
    dot = np.abs(np.dot(mn, ct_small_xyz))
    dot = np.clip(dot, 0.0, 1.0)
    angle = float(np.degrees(np.arccos(dot)))
    angles3.append(angle)
print(f"Median angle (smallest eigvec): {np.median(angles3):.1f}°  Mean: {np.mean(angles3):.1f}°")

# Test: middle eigenvector
print("\n--- Test: use MIDDLE eigenvector ---")
angles4 = []
for vi in sample:
    v = vertices[vi]
    mn = normals[vi]
    mn = mn / (np.linalg.norm(mn) + 1e-10)
    chunk = vol.sample_neighborhood(v, half_size=16)
    stc = StructureTensorComputer(sigma=3.0, device='cpu')
    st = stc.compute(chunk, as_numpy=True)
    cz, cy, cx = st.shape[1]//2, st.shape[2]//2, st.shape[3]//2
    components = [st[i, cz, cy, cx] for i in range(6)]
    mat = np.array([[components[0], components[1], components[2]],
                    [components[1], components[3], components[4]],
                    [components[2], components[4], components[5]]])
    eigenvalues, eigenvectors = np.linalg.eigh(mat)
    ct_mid = eigenvectors[:, 1]
    ct_mid_xyz = ct_mid[[2, 1, 0]]
    ct_norm = np.linalg.norm(ct_mid_xyz)
    if ct_norm > 1e-10:
        ct_mid_xyz = ct_mid_xyz / ct_norm
    dot = np.abs(np.dot(mn, ct_mid_xyz))
    dot = np.clip(dot, 0.0, 1.0)
    angle = float(np.degrees(np.arccos(dot)))
    angles4.append(angle)
print(f"Median angle (middle eigvec): {np.median(angles4):.1f}°  Mean: {np.mean(angles4):.1f}°")

# Test: largest eigvec WITHOUT reorder
print("\n--- Test: largest eigvec, NO ZYX->XYZ reorder ---")
angles5 = []
for vi in sample:
    v = vertices[vi]
    mn = normals[vi]
    mn = mn / (np.linalg.norm(mn) + 1e-10)
    chunk = vol.sample_neighborhood(v, half_size=16)
    stc = StructureTensorComputer(sigma=3.0, device='cpu')
    st = stc.compute(chunk, as_numpy=True)
    cz, cy, cx = st.shape[1]//2, st.shape[2]//2, st.shape[3]//2
    components = [st[i, cz, cy, cx] for i in range(6)]
    mat = np.array([[components[0], components[1], components[2]],
                    [components[1], components[3], components[4]],
                    [components[2], components[4], components[5]]])
    eigenvalues, eigenvectors = np.linalg.eigh(mat)
    ct_big = eigenvectors[:, 2]  # largest, kept as ZYX
    ct_norm = np.linalg.norm(ct_big)
    if ct_norm > 1e-10:
        ct_big = ct_big / ct_norm
    dot = np.abs(np.dot(mn, ct_big))
    dot = np.clip(dot, 0.0, 1.0)
    angle = float(np.degrees(np.arccos(dot)))
    angles5.append(angle)
print(f"Median angle (largest, no reorder): {np.median(angles5):.1f}°  Mean: {np.mean(angles5):.1f}°")

# Print summary
print("\n=== SUMMARY ===")
print(f"Largest eigvec + ZYX->XYZ reorder (current):  median={np.median(angles):.1f}°")
print(f"Largest eigvec, no reorder:                    median={np.median(angles5):.1f}°")
print(f"No reorder (ct_zyx as-is):                     median={np.median(angles2):.1f}°")
print(f"Smallest eigvec + ZYX->XYZ reorder:            median={np.median(angles3):.1f}°")
print(f"Middle eigvec + ZYX->XYZ reorder:              median={np.median(angles4):.1f}°")
print("Lowest median = best configuration")
