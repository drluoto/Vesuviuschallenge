"""Deeper diagnosis: check if bad angles correlate with mesh curvature or CT intensity."""
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
sample = rng.choice(valid, min(100, len(valid)), replace=False)

angles = []
intensities = []
anisotropies = []
eigenvalue_ratios = []

from vesuvius.image_proc.geometry.structure_tensor import StructureTensorComputer

for i, vi in enumerate(sample):
    v = vertices[vi]
    mn = normals[vi]
    mn = mn / (np.linalg.norm(mn) + 1e-10)

    chunk = vol.sample_neighborhood(v, half_size=16)
    center_intensity = chunk[16, 16, 16]
    intensities.append(float(center_intensity))

    stc = StructureTensorComputer(sigma=3.0, device='cpu')
    st = stc.compute(chunk, as_numpy=True)
    cz, cy, cx = st.shape[1]//2, st.shape[2]//2, st.shape[3]//2
    components = [st[i_c, cz, cy, cx] for i_c in range(6)]
    mat = np.array([[components[0], components[1], components[2]],
                    [components[1], components[3], components[4]],
                    [components[2], components[4], components[5]]])
    eigenvalues, eigenvectors = np.linalg.eigh(mat)

    lam = np.maximum(eigenvalues, 0)
    total = lam.sum()
    aniso = float((lam[2] - lam[1]) / total) if total > 1e-12 else 0.0
    anisotropies.append(aniso)
    eigenvalue_ratios.append(lam[2] / (lam[1] + 1e-12) if lam[1] > 0 else 0)

    ct_zyx = eigenvectors[:, 2]
    ct_xyz = ct_zyx[[2, 1, 0]]
    ct_norm = np.linalg.norm(ct_xyz)
    if ct_norm > 1e-10:
        ct_xyz = ct_xyz / ct_norm

    dot = np.abs(np.dot(mn, ct_xyz))
    dot = np.clip(dot, 0.0, 1.0)
    angle = float(np.degrees(np.arccos(dot)))
    angles.append(angle)

angles = np.array(angles)
intensities = np.array(intensities)
anisotropies = np.array(anisotropies)
eigenvalue_ratios = np.array(eigenvalue_ratios)

# Split into good and bad
good = angles < 20
mid = (angles >= 20) & (angles < 45)
bad = angles >= 45

print(f"Total: {len(angles)}, Good (<20°): {good.sum()}, Mid (20-45°): {mid.sum()}, Bad (>45°): {bad.sum()}")
print(f"\nGood vertices: mean_intensity={intensities[good].mean():.1f}, mean_aniso={anisotropies[good].mean():.3f}, mean_eig_ratio={eigenvalue_ratios[good].mean():.1f}")
print(f"Mid vertices:  mean_intensity={intensities[mid].mean():.1f}, mean_aniso={anisotropies[mid].mean():.3f}, mean_eig_ratio={eigenvalue_ratios[mid].mean():.1f}")
if bad.sum() > 0:
    print(f"Bad vertices:  mean_intensity={intensities[bad].mean():.1f}, mean_aniso={anisotropies[bad].mean():.3f}, mean_eig_ratio={eigenvalue_ratios[bad].mean():.1f}")

print(f"\nAngle distribution:")
for threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    frac = (angles < threshold).sum() / len(angles)
    print(f"  <{threshold}°: {frac:.1%}")

# Check if higher anisotropy threshold helps
print(f"\nEffect of anisotropy threshold:")
for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    mask = np.array(anisotropies) > thresh
    if mask.sum() > 0:
        filtered = angles[mask]
        print(f"  aniso>{thresh}: n={mask.sum()}, median_angle={np.median(filtered):.1f}°, mean={np.mean(filtered):.1f}°")

# Check sigma sensitivity
print(f"\nSigma sensitivity (on 20 vertices):")
for sigma in [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]:
    test_angles = []
    for vi in sample[:20]:
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
    print(f"  sigma={sigma}: median={np.median(test_angles):.1f}°, mean={np.mean(test_angles):.1f}°")
