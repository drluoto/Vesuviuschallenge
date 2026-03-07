"""Dense CT alignment heatmap visualizer.

Samples many vertices, computes CT alignment angle at each,
interpolates to full mesh, renders continuous heatmap.
"""
import sys
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from vesuvius_mesh_qa.volume import VolumeAccessor
from vesuvius_mesh_qa.ct_normals import compute_ct_normal


def compute_angles(mesh, volume, n_samples=2000, half_size=16, sigma=3.0):
    """Compute CT alignment angles at sampled vertices."""
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)

    # Find in-bounds vertices
    in_bounds = np.array([volume.vertex_in_bounds(v, margin=half_size) for v in vertices])
    valid = np.where(in_bounds)[0]
    print(f"  {len(valid)}/{len(vertices)} vertices in bounds")

    if len(valid) == 0:
        return np.array([]), np.array([]), np.array([])

    # Sample
    rng = np.random.default_rng(42)
    if len(valid) > n_samples:
        sample_idx = rng.choice(valid, n_samples, replace=False)
    else:
        sample_idx = valid

    print(f"  Sampling {len(sample_idx)} vertices...")

    angles = np.full(len(sample_idx), np.nan)
    anisotropies = np.zeros(len(sample_idx))

    for i, vi in enumerate(sample_idx):
        if i % 200 == 0:
            print(f"    {i}/{len(sample_idx)}...")

        v = vertices[vi]
        mn = normals[vi]
        mn_norm = np.linalg.norm(mn)
        if mn_norm < 1e-10:
            angles[i] = 0.0
            continue
        mn = mn / mn_norm

        chunk = volume.sample_neighborhood(v, half_size=half_size)
        ct_normal_zyx, aniso = compute_ct_normal(chunk, sigma=sigma)
        ct_xyz = ct_normal_zyx[[2, 1, 0]]
        ct_norm = np.linalg.norm(ct_xyz)
        if ct_norm < 1e-10:
            angles[i] = 0.0
            anisotropies[i] = aniso
            continue
        ct_xyz = ct_xyz / ct_norm

        dot = np.abs(np.dot(mn, ct_xyz))
        dot = np.clip(dot, 0.0, 1.0)
        angles[i] = float(np.degrees(np.arccos(dot)))
        anisotropies[i] = aniso

    return sample_idx, angles, anisotropies


def interpolate_to_full_mesh(vertices, sample_idx, sample_values, k=6):
    """Interpolate sampled values to all vertices using IDW."""
    sampled_pos = vertices[sample_idx]
    tree = cKDTree(sampled_pos)

    dists, indices = tree.query(vertices, k=k)

    # Inverse distance weighting
    # Avoid division by zero for exact matches
    weights = 1.0 / np.maximum(dists, 1e-10)
    weighted_vals = np.sum(weights * sample_values[indices], axis=1)
    total_weights = np.sum(weights, axis=1)

    return weighted_vals / total_weights


def angle_to_color(angles, vmin=0, vmax=70):
    """Map angles to green-yellow-red color gradient."""
    t = np.clip((angles - vmin) / (vmax - vmin), 0, 1)

    colors = np.zeros((len(t), 3))
    # Green (0°) -> Yellow (35°) -> Red (70°)
    # Green to yellow: R goes 0->1, G stays 1
    # Yellow to red: R stays 1, G goes 1->0
    mask_low = t <= 0.5
    mask_high = ~mask_low

    colors[mask_low, 0] = t[mask_low] * 2       # R: 0 -> 1
    colors[mask_low, 1] = 0.9                     # G: stays high
    colors[mask_low, 2] = 0.1                     # B: low

    colors[mask_high, 0] = 1.0                    # R: stays 1
    colors[mask_high, 1] = 0.9 * (1 - (t[mask_high] - 0.5) * 2)  # G: 0.9 -> 0
    colors[mask_high, 2] = 0.1                    # B: low

    return colors


def render_views(mesh, output_base):
    """Render mesh from multiple angles."""
    views = [
        ("top", [0, 0, -1], [0, -1, 0]),
        ("front", [0, -1, 0], [0, 0, -1]),
        ("angle1", [0.5, -0.5, -0.7], [0, 0, -1]),
        ("angle2", [-0.5, -0.3, -0.8], [0, 0, -1]),
    ]

    for name, front, up in views:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1600, height=1000, visible=False)
        vis.add_geometry(mesh)

        vc = vis.get_view_control()
        vc.set_front(front)
        vc.set_up(up)
        vc.set_zoom(0.4)

        vis.poll_events()
        vis.update_renderer()

        path = f"{output_base}_{name}.png"
        vis.capture_screen_image(path)
        vis.destroy_window()
        print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="CT alignment heatmap")
    parser.add_argument("mesh_path", help="Path to OBJ mesh")
    parser.add_argument("--volume", required=True, help="Zarr volume URL")
    parser.add_argument("--samples", type=int, default=2000, help="Number of vertices to sample")
    parser.add_argument("--output", default=None, help="Output base name (without extension)")
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--interactive", action="store_true", help="Open interactive viewer")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.mesh_path.replace('.obj', '_heatmap')

    print(f"Loading {args.mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(args.mesh_path)
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    print(f"  {len(vertices)} vertices, {len(mesh.triangles)} faces")

    print(f"Connecting to volume...")
    volume = VolumeAccessor(args.volume)
    print(f"  Volume shape: {volume.shape}")

    print(f"Computing CT alignment angles ({args.samples} samples)...")
    sample_idx, angles, anisotropies = compute_angles(
        mesh, volume, n_samples=args.samples, sigma=args.sigma
    )

    if len(sample_idx) == 0:
        print("No in-bounds vertices. Check coordinate mapping.")
        sys.exit(1)

    # Mask out low-anisotropy samples (no clear structure)
    # Set their angle to 0 so they don't pollute interpolation
    low_aniso = anisotropies < 0.1
    angles_for_interp = angles.copy()
    angles_for_interp[low_aniso] = 0.0

    print(f"  Structured vertices: {(~low_aniso).sum()}/{len(angles)}")
    structured_angles = angles[~low_aniso]
    if len(structured_angles) > 0:
        print(f"  Angle stats (structured only): mean={structured_angles.mean():.1f}° median={np.median(structured_angles):.1f}° max={structured_angles.max():.1f}°")

    print("Interpolating to full mesh...")
    full_angles = interpolate_to_full_mesh(vertices, sample_idx, angles_for_interp, k=8)

    print("Coloring mesh...")
    colors = angle_to_color(full_angles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Save PLY
    ply_path = args.output + ".ply"
    o3d.io.write_triangle_mesh(ply_path, mesh)
    print(f"  Saved {ply_path}")

    # Render views
    print("Rendering views...")
    render_views(mesh, args.output)

    if args.interactive:
        print("Opening interactive viewer...")
        o3d.visualization.draw_geometries([mesh], window_name="CT Alignment Heatmap")

    print("Done.")


if __name__ == "__main__":
    main()
