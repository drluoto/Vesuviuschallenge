# vesuvius-mesh-qa

Automated mesh quality scoring for [Vesuvius Challenge](https://scrollprize.org) segmentation meshes.

Scores segment meshes on 6 geometry metrics — plus an optional **CT-informed sheet switching detector** that compares mesh normals against actual papyrus layer orientation from the CT volume. No GPU required.

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

# Score with CT-informed sheet switching (requires volume URL)
mesh-qa score path/to/segment.obj \
  --volume 'https://data.aws.ash2txt.org/samples/PHerc1667/volumes/20231117161658-7.910um-53keV-masked.zarr/'

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
| `ct_sheet_switching` | 0.20 | **CT-informed layer alignment** (optional, requires `--volume`) |

Each metric produces a score from 0.0 (worst) to 1.0 (best). The aggregate score is the weighted average. Letter grades: A (>0.9), B (>0.75), C (>0.6), D (>0.4), F.

### Sheet Switching Detection

The most valuable metric. Detects regions where a segmentation mesh incorrectly jumps between adjacent papyrus layers — the primary failure mode in automated scroll segmentation.

**Algorithm:**
1. Build sparse face adjacency matrix
2. Compute 8-ring neighborhood via repeated squaring (adj^2 -> adj^4 -> adj^8)
3. Smooth normals by averaging over the 8-ring neighborhood
4. Flag faces where the actual normal deviates >35 degrees from the smoothed normal
5. Cluster flagged faces into connected components (BFS)
6. Filter clusters <20 faces (noise)
7. Score = 1 - (flagged area / total area)

The wide 8-ring neighborhood is necessary because sheet switches create transition zones that are locally consistent — a smaller neighborhood (e.g. 3-ring) would match the transition's own normals and miss the problem entirely.

**Limitation:** This detects angular surface anomalies, which catches sheet switches where layers diverge at an angle. It cannot detect switches between tightly packed parallel layers where normals stay similar — for that, use the CT-informed metric below.

### CT-Informed Sheet Switching (Optional)

When a zarr volume URL is provided via `--volume`, an additional metric compares mesh normals against CT-derived papyrus layer normals using 3D structure tensor analysis. This catches the hard case: parallel-layer switches invisible to geometry-only analysis.

**Algorithm:**
1. Sample 500 mesh vertices
2. Fetch 32x32x32 CT neighborhood at each vertex (lazy remote zarr access)
3. Compute 3D structure tensor using Holoborodko derivative kernels (sigma=3.0)
4. Extract largest eigenvector = papyrus sheet normal from CT data
5. Measure angular deviation between mesh normal and CT-derived normal
6. Only count vertices with clear sheet structure (anisotropy > 0.1)
7. Score = fraction of structured vertices aligned within 45 degrees

**What this catches that geometry-only cannot:** When a mesh jumps to an adjacent papyrus layer but the layers are nearly parallel, the mesh normals look fine — the geometry-only detector sees nothing wrong. But the CT structure tensor reveals the actual papyrus orientation at each point. If the mesh normal disagrees with the CT-derived normal, the mesh is on the wrong layer.

**Requirements:** Network access to the scroll's OME-Zarr volume (chunks fetched lazily, typically 50-500 MB). Adds ~2-5 minutes to scoring time. Available sample volumes at `https://data.aws.ash2txt.org/samples/`.

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
| PHerc1667 | 20231210132040 | 544K | 0.950 | A | (with CT metric)

Self-intersection detection cross-validated against Open3D's exhaustive `is_self_intersecting()` method. CT-informed metric validated by comparing structure tensor normals against mesh normals at 500 sampled vertices (median alignment 22°, consistent with correctly-segmented papyrus).

## Requirements

- Python 3.10+
- open3d, trimesh, numpy, scipy, click, rich, pandas
- Optional: zarr, fsspec, vesuvius (for CT-informed metric)
- No GPU required — pure geometry and statistics
- Geometry metrics: <30s on 1M+ face meshes (Apple M2)
- CT metric: +2-5 min (network I/O for remote zarr chunks)

## License

MIT
