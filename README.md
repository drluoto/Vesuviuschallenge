# vesuvius-mesh-qa

Automated mesh quality scoring for [Vesuvius Challenge](https://scrollprize.org) segmentation meshes.

Scores segment meshes on 6 quality metrics — including **sheet switching detection** — so the community can prioritize which segments are worth reviewing. No GPU required.

## Why

The Vesuvius Challenge virtual unwrapping pipeline produces thousands of mesh segments. Not all are usable — some have sheet switching (layer jumps), self-intersections, noisy vertices, or topological defects. Manually reviewing every segment is impractical.

`vesuvius-mesh-qa` automates this triage: score meshes, rank them, and focus human review on the segments that matter.

## Quick Start

```bash
# Install
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
└────────────────────┴───────┴────────┴──────────┘

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

Each metric produces a score from 0.0 (worst) to 1.0 (best). The aggregate score is the weighted average. Letter grades: A (>0.9), B (>0.75), C (>0.6), D (>0.4), F.

### Sheet Switching Detection

The most valuable metric. Detects regions where a segmentation mesh incorrectly jumps between adjacent papyrus layers — the primary failure mode in automated scroll segmentation.

Uses two complementary detectors:

**Detector 1: Normal deviation** — catches switches that create angular bends (e.g. when layers diverge):
1. Compute 8-ring neighborhood via repeated squaring (adj^2 -> adj^4 -> adj^8)
2. Smooth normals by averaging over the 8-ring neighborhood
3. Flag faces where normal deviates >35 degrees from smoothed normal

**Detector 2: Edge length outliers** — catches switches between tightly packed parallel layers (the common case):
1. Compute max edge length per face
2. Compare to 4-ring neighborhood mean using MAD-based z-scores
3. Flag faces with z-score > 2.0 (abnormally long edges from stretching between layers)

The union of both detectors is clustered into connected components (BFS), filtered for clusters >= 20 faces, and scored by flagged area fraction. Each detected region is tagged with which detector found it (`normal`, `edge_length`, or `both`).

### Self-Intersection Detection

Uses the Moller (1997) triangle-triangle intersection algorithm instead of AABB overlap. Samples 2,000 faces, tests each against its 30 nearest non-adjacent neighbors via KD-tree lookup. The Moller algorithm handles coplanar triangles (2D SAT) and general intersections (plane-plane interval overlap) correctly, avoiding false positives on curved 3D surfaces where bounding boxes naturally overlap.

### Noise Detection

Mesh-aware statistical outlier detection. Excludes vertices within 3 hops of mesh boundaries before running Open3D's statistical outlier removal — boundary neighborhoods have fewer point cloud neighbors and would otherwise produce false positives.

## Visualization

Export a colored PLY mesh highlighting problem regions:

```bash
mesh-qa score segment.obj --visualize problems.ply
```

Open the PLY in MeshLab, CloudCompare, or any 3D viewer. Color coding:
- **Green** — no issues
- **Red** — sheet switching
- **Magenta** — self-intersections
- **Orange** — poor triangle quality
- **Blue** — noise/spikes
- **Yellow** — normal inconsistency
- **Purple** — topology issues

## Batch Mode

Score all segments in a directory tree. Outputs a CSV ranked by quality (worst first = review priority):

```bash
mesh-qa batch /path/to/scroll1.volpkg/paths/ -o rankings.csv
```

The tool auto-discovers OBJ files following the volpkg convention (`paths/<segment_id>/<segment_id>.obj`) and also handles flat directories.

## JSON Output

Machine-readable output for integration with other tools:

```bash
mesh-qa score segment.obj --format json
```

JSON output includes scroll ID, segment ID (auto-detected from path/filename), per-metric scores with details, and problem face indices for visualization.

## Custom Weights

Override metric weights via JSON:

```bash
mesh-qa score segment.obj --weights '{"sheet_switching": 0.5, "noise": 0.05}'
```

## Validated On

Tested on real scroll segments from the Vesuvius Challenge dataset:

| Scroll | Segment | Faces | Score | Grade |
|--------|---------|-------|-------|-------|
| PHerc0332 | 20240711124827 | 340K | 0.993 | A |
| PHerc0332 | 20231210121321 | 1.1M | 0.990 | A |

Self-intersection detection cross-validated against Open3D's exhaustive `is_self_intersecting()` method.

## Requirements

- Python 3.10+
- open3d, trimesh, numpy, scipy, click, rich, pandas
- No GPU required — pure geometry and statistics
- Runs in <30s on 1M+ face meshes (Apple M2)

## License

MIT
