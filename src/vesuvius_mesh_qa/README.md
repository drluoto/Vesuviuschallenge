# vesuvius-mesh-qa

Automated mesh quality scoring for Vesuvius Challenge segmentation meshes.

Scores segment meshes on 6 quality metrics — including **sheet switching detection** — so the community can prioritize which segments are worth reviewing. No GPU required.

## Quick Start

```bash
# Install
cd /path/to/Vesuviuschallenge
pip install -e .

# Score a single mesh
mesh-qa score path/to/segment.obj

# Score with JSON output
mesh-qa score path/to/segment.obj --format json

# Batch score all segments in a volpkg
mesh-qa batch path/to/volpkg/paths/ -o rankings.csv
```

## Example Output

```
          Mesh Quality Report: 20231210121321_intermediate_mesh.obj
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric             ┃ Score ┃ Weight ┃ Weighted ┃ Details                  ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ triangle_quality   │ 0.977 │   0.10 │    0.098 │ aspect_ratio_score=1.00  │
│ topology           │ 0.999 │   0.10 │    0.100 │ manifold_score=1.000     │
│ normal_consistency │ 0.981 │   0.15 │    0.147 │ mean_dihedral=7.7deg     │
│ sheet_switching    │ 0.995 │   0.30 │    0.299 │ n_switch_regions=121     │
│ self_intersections │ 1.000 │   0.15 │    0.150 │ intersection_frac=0.000  │
│ noise              │ 0.975 │   0.10 │    0.098 │ n_outliers=676           │
└────────────────────┴───────┴────────┴──────────┴──────────────────────────┘

  Aggregate Score: 0.990  Grade: A
```

## Metrics

| Metric | Weight | What it measures |
|--------|--------|-----------------|
| `triangle_quality` | 0.10 | Aspect ratios, minimum angles, area uniformity |
| `topology` | 0.10 | Manifoldness, connected components, boundary edges |
| `normal_consistency` | 0.15 | Smoothness of adjacent face normals |
| `sheet_switching` | 0.30 | **Detects layer-jumping failures** (the key metric) |
| `self_intersections` | 0.15 | Self-overlapping geometry (Moller triangle intersection) |
| `noise` | 0.10 | Statistical outlier vertices (spikes) |

### Sheet Switching Detection

The most valuable metric. Detects regions where a segmentation mesh incorrectly jumps between adjacent papyrus layers — the primary failure mode in automated scroll segmentation.

**Algorithm:**
1. Build sparse face adjacency matrix
2. Compute 8-ring neighborhood via repeated squaring (adj² → adj⁴ → adj⁸)
3. Smooth normals by averaging over the 8-ring neighborhood
4. Flag faces where the actual normal deviates >35° from the smoothed normal
5. Cluster flagged faces into connected components (BFS)
6. Filter clusters <20 faces (noise)
7. Score = 1 - (flagged area / total area)

The wide 8-ring neighborhood is necessary because sheet switches create transition zones that are locally consistent — a smaller neighborhood (e.g. 3-ring) would match the transition's own normals and miss the problem entirely.

Reports the number, size, and centroid of each detected switch region.

### Self-Intersection Detection

Uses the Moller (1997) triangle-triangle intersection algorithm instead of AABB overlap. Samples 2,000 faces, tests each against its 30 nearest non-adjacent neighbors via KD-tree lookup. The Moller algorithm handles coplanar triangles (2D SAT) and general intersections (plane-plane interval overlap) correctly, avoiding false positives on curved 3D surfaces where bounding boxes naturally overlap.

## Batch Mode

Score all segments in a volpkg directory tree. Outputs a CSV ranked by quality (worst first = review priority):

```bash
mesh-qa batch /path/to/scroll1.volpkg/paths/ -o rankings.csv
```

The tool auto-discovers OBJ files following the volpkg convention (`paths/<segment_id>/<segment_id>.obj`) and also handles flat directories.

## Custom Weights

Override metric weights via JSON:

```bash
mesh-qa score segment.obj --weights '{"sheet_switching": 0.5, "noise": 0.05}'
```

## Requirements

- Python 3.10+
- open3d, trimesh, numpy, scipy, click, rich, pandas
- No GPU required — pure geometry + statistics

## License

MIT
