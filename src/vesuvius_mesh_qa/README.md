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
                    Mesh Quality Report: 20231001000000.obj
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric             ┃ Score ┃ Weight ┃ Weighted ┃ Details                  ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ triangle_quality   │ 0.950 │   0.10 │    0.095 │ aspect_ratio_score=0.98  │
│ topology           │ 0.900 │   0.10 │    0.090 │ n_components=1           │
│ normal_consistency │ 0.920 │   0.15 │    0.138 │ mean_dihedral=8.5deg     │
│ sheet_switching    │ 0.750 │   0.30 │    0.225 │ n_switch_regions=2       │
│ self_intersections │ 0.980 │   0.15 │    0.147 │ intersection_frac=0.001  │
│ noise              │ 0.950 │   0.10 │    0.095 │ n_outliers=12            │
└────────────────────┴───────┴────────┴──────────┴──────────────────────────┘

  Aggregate Score: 0.877  Grade: B
```

## Metrics

| Metric | Weight | What it measures |
|--------|--------|-----------------|
| `triangle_quality` | 0.10 | Aspect ratios, minimum angles, area uniformity |
| `topology` | 0.10 | Manifoldness, connected components, boundary edges |
| `normal_consistency` | 0.15 | Smoothness of adjacent face normals |
| `sheet_switching` | 0.30 | **Detects layer-jumping failures** (the key metric) |
| `self_intersections` | 0.15 | Self-overlapping geometry |
| `noise` | 0.10 | Statistical outlier vertices (spikes) |

### Sheet Switching Detection

The most valuable metric. Detects regions where a segmentation mesh incorrectly jumps between adjacent papyrus layers — the primary failure mode in automated scroll segmentation.

**Algorithm:**
1. Compute smoothed normals using 3-ring face neighborhoods (via sparse matrix power)
2. Flag faces where the actual normal deviates >45 degrees from the smoothed normal
3. Cluster flagged faces into connected components
4. Filter clusters <20 faces (noise)
5. Score = 1 - (flagged area / total area)

Reports the number, size, and centroid of each detected switch region.

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
