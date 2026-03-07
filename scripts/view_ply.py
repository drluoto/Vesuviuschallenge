"""Render colored PLY mesh to a PNG image using matplotlib."""
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def render_ply(ply_path: str, output_png: str, elev: float = 30, azim: float = -60):
    mesh = o3d.io.read_triangle_mesh(ply_path)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)  # per-vertex RGB [0,1]

    # Compute per-face colors by averaging vertex colors
    face_colors = colors[triangles].mean(axis=1)

    # Build polygon collection
    polys = vertices[triangles]

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    pc = Poly3DCollection(polys, facecolors=face_colors, edgecolors='none', linewidths=0.0)
    ax.add_collection3d(pc)

    # Set axis limits
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2
    span = (maxs - mins).max() / 2
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(ply_path.split('/')[-1].replace('.ply', ''), fontsize=14)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=[180/255, 220/255, 180/255], label='Healthy'),
        Patch(facecolor=[0, 1, 1], label='CT misalignment'),
        Patch(facecolor=[1, 0, 0], label='Sheet switching'),
        Patch(facecolor=[0, 128/255, 1], label='Noise'),
        Patch(facecolor=[1, 0, 1], label='Self-intersection'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_png}")

if __name__ == "__main__":
    ply = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else ply.replace('.ply', '.png')
    render_ply(ply, out)
