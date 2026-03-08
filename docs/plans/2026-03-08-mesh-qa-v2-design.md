# vesuvius-mesh-qa v2: Design Document

## Problem Statement

Automated segmentation of Herculaneum papyrus scrolls produces thousands of mesh segments. These segments have varying quality — from perfect single-layer surfaces to meshes riddled with self-intersections, topology errors, and the dreaded "sheet switching" where the surface jumps between adjacent papyrus layers.

Currently, quality review is manual. Segmenters (bruniss, spelufo, etc.) visually inspect rendered segments and surface volumes in tools like Khartes, VC3D, or Crackle Viewer. There is no automated triage, no batch ranking, and no standardized quality score.

vesuvius-mesh-qa v1 solves the geometry QA part: self-intersections, topology, noise, normal consistency, and angular sheet switching detection. It cleanly separates known-good from known-bad segments on bruniss's data.

**What v1 cannot do:** detect parallel-layer sheet switches — the hard case where tightly packed layers run parallel and normals look identical. This is the primary unsolved quality problem in the community.

## Design Principles

1. **Don't reinvent detection — integrate the gold standards.** The community has spent years developing sheet switch prevention (winding angles, fiber analysis, topology scoring). We should use their methods, not compete with them.

2. **Tiered activation.** Not everyone has a CT volume URL. Not everyone knows their scroll's umbilicus. The tool must be useful with zero context and progressively more powerful as context is provided.

3. **The tool IS the product.** Our contribution is the unified QA interface: score, triage, review. The individual metrics can come from anywhere. The value is in combining them with an interactive human review workflow.

4. **Honest about what we can and can't detect.** Report confidence alongside scores. Flag what needs human review rather than pretending automation is sufficient.

---

## Architecture: Three Tiers

```
Tier 1: Geometry Only          Tier 2: + CT Volume          Tier 3: + Scroll Context
(no external data)              (--volume URL)               (--volume + --umbilicus)

  triangle_quality               All Tier 1 metrics          All Tier 1+2 metrics
  topology                       + ct_normal_alignment       + winding_angle_consistency
  normal_consistency             + texture_coherence
  sheet_switching (angular)
  self_intersections
  noise

  ~15s on 1M faces              +2-5min (network I/O)        +30s (local compute)
```

### Tier 1: Geometry Only (current, working)

No external data needed. Fast. Already validated.

| Metric | Weight | Status |
|--------|--------|--------|
| triangle_quality | 0.10 | Done, working |
| topology | 0.10 | Done, working |
| normal_consistency | 0.10 | Done, working |
| sheet_switching (angular) | 0.25 | Done, working. Catches angular switches only. |
| self_intersections | 0.20 | Done, working. Strongest differentiator in validation. |
| noise | 0.10 | Done, working |

**What this catches:** self-intersecting geometry, non-manifold topology, noisy vertices, angular sheet switches where the surface bends sharply.

**What this misses:** parallel-layer switches, subtle texture discontinuities.

**Weight change from v1:** sheet_switching reduced from 0.30 to 0.25, normal_consistency from 0.15 to 0.10. This reflects that the angular detection has known false positives at mesh boundaries (validated in our cross-section analysis). The freed weight is redistributed when higher tiers activate.

### Tier 2: CT Volume Context (requires --volume)

Adds metrics that sample the CT volume at mesh vertex positions. Requires network access to OME-Zarr store.

| Metric | Weight | Status |
|--------|--------|--------|
| ct_normal_alignment | 0.05 | Exists (ct_sheet_switching). High baseline noise (~25-30°). Keep at low weight. |
| texture_coherence | 0.10 | **NEW.** See below. |

**texture_coherence (NEW):**

The key insight from the research: parallel-layer switches are detectable by texture discontinuity. Adjacent papyrus layers have different fiber patterns. When a mesh jumps layers, the CT texture sampled at the surface changes abruptly.

Algorithm:
1. Sample N vertices along the mesh surface (e.g., 500).
2. At each vertex, extract a small CT surface patch (e.g., 16x16 voxels in the tangent plane).
3. Compute local texture features: gradient magnitude statistics, dominant orientation via Gabor or structure tensor, local contrast.
4. Build a mesh-topology-aware neighbor graph (8-ring, same as angular sheet switching).
5. For each vertex, compare its texture features to its neighborhood average.
6. Flag vertices where texture features deviate significantly from their neighbors.
7. Score = 1 - (fraction of flagged area).

**Why this works for parallel-layer switches:** Even when normals are identical, the actual papyrus texture differs between layers. One layer might have horizontal fibers at a given location while the adjacent layer has vertical fibers. The texture features capture this.

**Why this is better than our existing CT metric:** The current ct_normal_alignment compares mesh normals to CT structure tensor normals — but parallel layers have the same structure tensor normal. The texture_coherence metric instead looks at the actual texture content, which differs between layers.

**Complexity:** Moderate. Requires sampling CT patches (similar infrastructure to existing CT metric). Feature computation is fast (numpy). The hard part is defining "texture features" that are discriminative. Starting with: (1) gradient magnitude histogram, (2) dominant gradient direction, (3) local intensity variance.

**Risk:** Texture might be too noisy at 7-8µm resolution to reliably distinguish layers. Needs validation on confirmed sheet-switch segments.

### Tier 3: Scroll Context (requires --volume + --umbilicus)

Adds metrics that use knowledge of the scroll's geometry — specifically the umbilicus (center axis).

| Metric | Weight | Status |
|--------|--------|--------|
| winding_angle_consistency | 0.15 | **NEW.** See below. |

**winding_angle_consistency (NEW):**

This adapts ThaumatoAnakalyptor's winding angle concept for post-hoc QA. The core idea: on a correct single-layer mesh, the winding angle (angular position around the scroll's center axis) varies smoothly. A sheet switch creates a discontinuity.

**How ThaumatoAnakalyptor actually computes winding angles** (from reading the source code — `split_mesh.py`, `mesh_quality.py`):

The umbilicus is NOT a 2D point — it's a **3D curve** through the scroll center, stored as a text file of (x, y, z) samples. `MeshSplitter` loads this and creates scipy interpolation functions to get the umbilicus (x, z) position at any Y-coordinate (note: ThaumatoAnakalyptor uses Y as the scroll axis, with coordinate transform `vertices[:,[1,2,0]]` + offset of 500).

Winding angles are computed via **BFS traversal**, not simple atan2:
1. `compute_uv_with_bfs(start_vertex)` — breadth-first walk from a seed vertex
2. For each edge (v1 → v2), compute `angle_between_vertices(v1, v2)`: project both vertices to 2D plane relative to umbilicus at each vertex's Y-level, compute atan2 angle difference, normalize to ±180°
3. Accumulate: `new_angle = current_angle + angle_diff`
4. Store per-vertex: `(winding_angle, radial_distance_from_umbilicus)`

The BFS accumulation is critical — it gives consistent winding angles that increase monotonically around the scroll. Simple per-vertex atan2 would give angles modulo 360° with no way to distinguish winding 5 from winding 6.

**Our adaptation for standalone QA** (no reference mesh needed):

Algorithm:
1. User provides umbilicus data (either a text file path or simplified (x, y) for scrolls with straight center axes).
2. Run BFS from an arbitrary seed vertex, accumulating winding angles exactly as ThaumatoAnakalyptor does.
3. On a correct single-layer mesh, the winding angle field is smooth — gradients across edges are small and consistent.
4. Compute winding angle gradient across mesh edges. On a good mesh, adjacent vertices have similar winding angles.
5. Flag edges where the winding angle jumps abruptly (e.g., >15° between adjacent vertices — one winding spans ~18° for a 20-winding scroll).
6. Cluster flagged edges into regions.
7. Score = 1 - (fraction of flagged area).

**Important distinction:** ThaumatoAnakalyptor's `mesh_quality()` function is a COMPARISON metric — it aligns winding angles between an input mesh and a ground truth mesh using `align_winding_angles()`. We DON'T need that. We only need the smoothness check: does the winding angle field on THIS mesh have discontinuities? That's a standalone property.

**Why this catches parallel-layer switches:** Adjacent layers are at different radial distances from the umbilicus AND at different accumulated winding angles. A layer jump creates a step in the winding angle field — detectable even when normals are identical.

**What's needed from the user:** Umbilicus data for the scroll. Available in ThaumatoAnakalyptor datasets and community tools. For scrolls with straight axes, a simple (x, z) center point suffices (we generate a constant umbilicus curve).

**Complexity:** Moderate. The BFS traversal and angle accumulation are straightforward to implement (~100 lines). The coordinate transform (axis reorder, offset) must match ThaumatoAnakalyptor conventions for the umbilicus data to work. Edge case: disconnected mesh components need separate BFS seeds.

**Risk:** Medium. The BFS accumulation means errors can propagate — if one edge has a large legitimate angle change (sharp mesh turn), the accumulated angle downstream could be off. ThaumatoAnakalyptor handles this in their graph solver with spring constants and global optimization. For QA, we only need LOCAL smoothness (gradient check), not globally correct winding numbers, so this is less of a concern.

---

## Weight Redistribution

Weights must sum to 1.0 across active metrics. When higher tiers activate, geometry-only metrics give up some weight.

**Tier 1 only (6 metrics):**
```
triangle_quality:    0.10
topology:            0.10
normal_consistency:  0.10
sheet_switching:     0.30
self_intersections:  0.25
noise:               0.15
                     ----
                     1.00
```

**Tier 1 + 2 (8 metrics):**
```
triangle_quality:    0.10
topology:            0.10
normal_consistency:  0.10
sheet_switching:     0.20  (reduced — texture_coherence covers similar ground better)
self_intersections:  0.20
noise:               0.10
ct_normal_alignment: 0.05
texture_coherence:   0.15
                     ----
                     1.00
```

**Tier 1 + 2 + 3 (9 metrics):**
```
triangle_quality:       0.05
topology:               0.10
normal_consistency:     0.05
sheet_switching:        0.15  (further reduced)
self_intersections:     0.15
noise:                  0.05
ct_normal_alignment:    0.05
texture_coherence:      0.15
winding_angle:          0.25  (the heavy hitter)
                        ----
                        1.00
```

Winding angle gets the highest weight in Tier 3 because it's the most direct and reliable detector of actual sheet switches (both angular AND parallel-layer).

Users can still override any weight with `--weights '{...}'`.

---

## CLI Design

```bash
# Tier 1: Geometry only (fast, no dependencies)
mesh-qa score segment.obj

# Tier 2: Add CT volume context
mesh-qa score segment.obj --volume 'https://...zarr/'

# Tier 3: Add scroll context (umbilicus)
mesh-qa score segment.obj \
  --volume 'https://...zarr/' \
  --umbilicus 3200,3200

# Interactive review (works at any tier)
mesh-qa score segment.obj --review review.html
mesh-qa score segment.obj --volume URL --umbilicus 3200,3200 --review review.html

# Batch mode (works at any tier)
mesh-qa batch path/to/segments/ -o rankings.csv
mesh-qa batch path/to/segments/ -o rankings.csv --volume URL --umbilicus 3200,3200
```

The `--umbilicus` flag accepts either:
- A path to a ThaumatoAnakalyptor-format umbilicus text file (rows of x, y, z coordinates along the scroll axis)
- A simple `x,z` pair for scrolls with straight center axes (we generate a constant umbilicus curve)

```bash
# Using a scroll config file
mesh-qa score segment.obj --scroll-config scrolls/PHerc1667.json
```

Where the config contains:
```json
{
  "volume_url": "https://...zarr/",
  "umbilicus": "path/to/umbilicus.txt",
  "umbilicus_simple": [3200, 3200],
  "estimated_windings": 20,
  "layer_spacing_um": 150,
  "coordinate_transform": {"axis_order": [1, 2, 0], "offset": 500}
}
```

---

## HTML Review Viewer Enhancements

The v2 viewer should show tier-appropriate diagnostics:

**Tier 1:** Current functionality — metric colors, deviation heatmap, cluster fly-to, cross-section charts, boundary warnings.

**Tier 2:** Add texture coherence heatmap (toggle). Show CT-sampled texture patches at flagged locations so the human reviewer can see what the texture discontinuity looks like.

**Tier 3:** Add winding angle visualization — color the mesh by winding angle (rainbow gradient around the scroll). Sheet switches appear as color discontinuities. This is immediately intuitive for domain experts: the scroll should be a smooth gradient, any color jump = layer jump.

---

## Implementation Priority

### Phase 1: Winding Angle Metric (highest impact, moderate effort)
- Pure geometry + umbilicus data (text file or simple center point)
- Most direct sheet switch detector — adapted from ThaumatoAnakalyptor's BFS winding angle traversal
- No ML, no CT volume, no network I/O
- Catches BOTH angular and parallel-layer switches
- Implementation: BFS traversal (~100 lines), coordinate transform handling, umbilicus loading/interpolation
- Need ThaumatoAnakalyptor umbilicus files for validation scrolls
- Can validate on our existing test data once umbilicus data is obtained
- **Effort: 2-3 days** (BFS implementation + coordinate system alignment + validation)

### Phase 2: Texture Coherence Metric (high impact, moderate effort)
- Requires CT volume (existing infrastructure)
- Needs careful feature engineering or experimentation
- Potentially catches cases that winding angle misses (e.g., at scroll center where winding angle is ambiguous)
- Requires validation on confirmed sheet-switch data
- **Effort: 3-5 days**

### Phase 3: Review Viewer Improvements (high impact for adoption)
- Winding angle rainbow visualization
- Texture patch viewer at flagged regions
- Export shareable review bundles for community feedback
- **Effort: 2-3 days**

### Phase 4: Scroll Config System + Community Polish
- Per-scroll config files with umbilicus, volume URL, metadata
- Pre-configured for PHerc1667, PHerc1447, PHerc0332, etc.
- Integration guide for ThaumatoAnakalyptor / Surface Tracer pipelines
- Monthly Progress Prize submission
- **Effort: 1-2 days**

---

## What We Explicitly Don't Build

- **Fiber segmentation model.** bruniss has this ($20K prize-winning work). If we want fiber-based detection, we integrate his model, not build our own.
- **Winding angle graph solver.** ThaumatoAnakalyptor has this. We use the concept (winding angle consistency check) but don't replicate the full graph optimization.
- **Surface detection ML.** The Kaggle competition handles this. We QA the output, not generate it.
- **Betti Matching / TopoScore.** Requires a reference surface (ground truth). We're scoring standalone meshes with no reference. Our topology metric already captures what's possible without a reference (manifoldness, components, boundary edges).

---

## Open Questions

1. **Umbilicus data per scroll** — ThaumatoAnakalyptor stores these as text files (rows of x, y, z). Available in their datasets. Need to confirm: does the community have umbilicus files for all active scrolls (PHerc1667, PHerc1447, PHerc0332)? Do Khartes/VC3D use a different format?

2. **Texture features** — what's the minimum set that distinguishes adjacent layers? Structure tensor orientation alone doesn't work (validated). Need to experiment with: Gabor filter banks, local binary patterns, gradient histograms, raw patch comparison.

3. **Validation data** — we still need confirmed sheet-switch segments to validate the new metrics. The winding angle metric can be validated against ThaumatoAnakalyptor's own winding angle assignments if we can access that data.

4. **Weight tuning** — current weights are educated guesses. Once we have labeled data (good/bad segments with known failure modes), we should tune weights to maximize separation.

---

## Success Criteria

The tool succeeds if:
1. A segmenter can run `mesh-qa batch` on 1000 segments and get a ranked list where the bottom 10% actually contains the worst segments (validated by human review).
2. The HTML review page lets a domain expert confirm or deny flagged sheet switches in under 30 seconds per cluster.
3. The winding angle metric catches at least some parallel-layer switches that the geometry-only metrics miss (even one confirmed case is a win).
4. The community finds it useful enough to adopt (measured by: bruniss/Julian/Hendrik try it and give feedback).
