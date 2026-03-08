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
  normal_consistency             + fiber_coherence
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
| fiber_coherence | 0.10 | **NEW.** See below. |

**fiber_coherence (NEW) — the key to detecting tightly-packed parallel-layer switches:**

This metric addresses the hardest unsolved case: sheet switches between tightly packed parallel layers where normals are identical and winding angle differences are tiny. Neither geometry analysis nor winding angles can reliably catch these. But **fiber patterns can**.

#### Why papyrus fiber patterns solve this

Papyrus is manufactured from crossed reed strips — one layer of horizontal fibers, one layer of vertical fibers, bonded together. This creates a distinctive two-layer texture visible in micro-CT at 7-8µm resolution (individual fibers are ~100-200µm diameter = 12-25 voxels). Critically:

1. Each papyrus sheet has **two faces** with different dominant fiber orientations (horizontal vs vertical).
2. When the scroll is rolled, the fiber pattern at any given (x, y, z) position is **unique to that specific layer**. The layer above and below have their own distinct patterns at that position.
3. On a correct single-layer mesh, the fiber orientation sampled along the surface changes **gradually** — you're following one continuous sheet with slowly varying fiber patterns.
4. On a sheet switch, the fiber pattern changes **abruptly** — you've jumped to a different layer with a completely different fiber layout at that location.

This works even when layers are <100µm apart, because the fiber pattern difference is a **material property of the papyrus**, not a geometric property of the layer spacing.

#### How to detect fiber orientation: two methods

From studying the community's actual tools and codebases, there are two proven approaches for computing fiber orientation from CT data. Both work. They differ in accuracy, dependencies, and hardware requirements.

**Method A: Structure tensor analysis (lightweight, no model needed)**

The `structure-tensor` PyPI package (MIT license, GPU-capable via CuPy) computes 3D structure tensors directly from CT voxel data. For stripe-like structures like papyrus fibers, the eigendecomposition gives:
- **Smallest eigenvalue eigenvector** = fiber long axis direction (the fiber runs along this)
- **Middle eigenvalue eigenvector** = in-sheet plane, perpendicular to fiber
- **Largest eigenvalue eigenvector** = normal to papyrus sheet surface

This is the same mathematical basis that Villa's `get_fiber_directions.py` uses internally (Gaussian prefilter σ=1.5, Holoborodko derivative kernels, outer product → 3×3 tensor, Gaussian window smoothing σ=2.0, eigendecomposition).

To classify fiber orientation (horizontal vs vertical): compute the dot product of the smallest eigenvector with the Z-axis. Villa's `hz-vt-generator.py` uses the threshold `|dot(principal_axis, z_axis)| > 1/√2` (~45°).

- **Pro**: Lightweight (numpy/scipy or the `structure-tensor` package), fast, no model download, works on CPU
- **Con**: Noisy at 7-8µm — our existing CT metric already uses structure tensor normals and has ~25-30° baseline noise. However, that metric compares structure tensor normal vs mesh normal (a cross-domain comparison). Here we'd compare structure tensor features vs *neighboring* structure tensor features (same-domain comparison), which should be more discriminative.
- **Dependencies**: `structure-tensor` (pip install) or numpy/scipy
- **Compute**: ~seconds per 500 sample points

**Method B: bruniss nnUNet fiber segmentation model (highest accuracy)**

bruniss's $20K prize-winning fiber detection model is a standard nnUNetv2 trained on micro-CT data. It is the community's gold standard for fiber detection.

**Model details** (from reading the actual codebase and training configs):
- **Architecture**: `PlainConvUNet` (nnUNet default 3D UNet), 6 encoder stages, features [32, 64, 128, 256, 320, 320]
- **Training data**: Dataset004_fiber3 — 6 manually labeled volumes (labels created in Dragonfly 3D)
- **Input**: Raw 3D micro-CT, single channel, uint16. Z-score normalized (mean=7534, std=16854)
- **Patch size**: [64, 160, 224] voxels (ZYX), batch size 2
- **Output**: Per-voxel class labels — 0=background, 1=horizontal fibers, 2=vertical fibers
- **Checkpoint**: ~238 MB (`checkpoint_final.pth`)
- **Inference**: Standard `nnUNetPredictor` API, supports MPS (Apple Silicon)

**Model weights availability**:
- Original URL (README link) returns 404
- Available at: `dl.ash2txt.org/community-uploads/bruniss/old_models/Dataset004-3d_mask-hz-only/`
- Newer models exist (Dataset040_newHorizontals, Dataset041_newVerticals) using ResidualEncoderUNet (~781MB each) — these are separate horizontal/vertical binary models with 41 training samples, but are too large for 8GB M2

**Programmatic inference** (from reading nnUNetv2 source):
```python
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch, numpy as np

predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=False,       # disable TTA to save memory on M2
    perform_everything_on_device=False,  # forced False on non-CUDA
    device=torch.device('mps'),          # Apple Silicon
)
predictor.initialize_from_trained_model_folder(
    'path/to/Dataset004_fiber3/nnUNetTrainer__nnUNetPlans__3d_fullres',
    use_folds=(0,), checkpoint_name='checkpoint_final.pth'
)

# Predict on a CT patch: shape (1, Z, Y, X), channels-first
ct_patch = np.zeros((1, 64, 160, 224), dtype=np.float32)
properties = {'spacing': [7.91, 7.91, 7.91]}
seg, probs = predictor.predict_single_npy_array(
    ct_patch, properties, save_or_return_probabilities=True
)
# seg: (64, 160, 224) with values {0, 1, 2}
# probs: (3, 64, 160, 224) softmax probabilities per class
```

- **Pro**: Highest accuracy — proven on this exact data, won $20K prize. Per-voxel classification, not noisy orientation estimates.
- **Con**: Heavy dependencies (nnunetv2, PyTorch ~2GB install), 238MB model download, ~6-8GB memory for inference (tight on M2 8GB), slower (~minutes for a full segment)
- **Dependencies**: `nnunetv2`, `torch` (with MPS support ≥1.12), model weights
- **Compute**: ~1-5 min per segment depending on patch count

**Newer option — Villa multi-task 3D UNet**:
Villa's `segmentation/models/multi-task-3d-unet/` has a more sophisticated ResEnc UNet with squeeze-and-excitation blocks that can jointly predict surfaces, fiber orientations, normals, and ink. However, this requires more infrastructure and there are no publicly available pre-trained weights specifically for fiber detection. Not recommended for initial integration.

#### Which method to use: decision framework

The right answer depends on what's available at runtime:

```
Has --fiber-model flag with path to nnUNet weights?
  → YES: Use Method B (nnUNet). Highest accuracy.
  → NO: Use Method A (structure tensor). Still effective for same-domain comparison.
```

Both methods feed into the **same scoring algorithm** (below). The difference is input quality, not algorithm design.

#### Scoring algorithm (shared by both methods)

1. **Sample N vertices** along the mesh surface (e.g., 500, spatially distributed).
2. **At each vertex**, extract a CT subvolume centered on the vertex position. Size:
   - Method A: 32³ voxels (for structure tensor computation)
   - Method B: [64, 160, 224] voxels (nnUNet patch size) — but we batch nearby vertices into shared patches to avoid redundant inference
3. **Compute fiber orientation at each vertex**:
   - Method A: Structure tensor eigendecomposition → smallest eigenvector → classify as horizontal/vertical by angle to Z-axis (threshold: 45°). Also extract anisotropy (eigenvalue ratio) as confidence.
   - Method B: nnUNet prediction → per-voxel labels (0/1/2) → at vertex position, read dominant class and probability. Also extract the local fiber *direction* from the probability gradient.
4. **Build mesh-topology-aware neighbor graph** (8-ring, same as angular sheet switching metric).
5. **For each vertex, compare its fiber features to its topological neighbors**:
   - Primary signal: **fiber class flip** — does the dominant fiber orientation (horizontal vs vertical) change abruptly between adjacent mesh regions? On a correct single-layer mesh, the dominant class should be locally consistent (you're on the same face of the same sheet). A flip = probable sheet switch.
   - Secondary signal: **fiber direction discontinuity** — even within the same class (e.g., both horizontal), the actual fiber direction vector should vary smoothly. An abrupt direction change between neighbors suggests a layer jump.
   - Confidence weighting: weight each vertex's contribution by anisotropy (Method A) or class probability (Method B). Low-confidence vertices (isotropic regions, low probability) contribute less to the score.
6. **Flag vertices** where fiber features deviate significantly from their topological neighborhood.
7. **Cluster flagged vertices** into contiguous regions (same algorithm as angular sheet switching).
8. **Score** = 1 - (fraction of flagged mesh area, weighted by confidence).

#### Why this catches tightly-packed parallel-layer switches

Consider two papyrus layers 100µm apart, running perfectly parallel:
- Their surface normals are identical → geometry metrics see nothing
- Their winding angles may differ by only ~7° (for 50-winding regions) → winding angle metric may miss it
- But at the same (x,y,z) position, layer A has horizontal fibers while layer B has vertical fibers (or vice versa) → **the fiber class at that point is different**

When the mesh switches from layer A to layer B, the fiber class sampled at the surface flips from horizontal-dominant to vertical-dominant. This flip is detectable regardless of layer spacing, because it's a material property of the papyrus sheet, not a geometric property of the scroll structure.

#### Why structure tensor comparison works better here than in our existing CT metric

Our existing `ct_normal_alignment` metric compares the **largest** structure tensor eigenvector (sheet normal) against the mesh face normal. This is a cross-domain comparison (CT-derived orientation vs geometry-derived orientation) and has ~25-30° baseline noise.

The fiber_coherence metric instead compares the **smallest** eigenvector (fiber direction) of **neighboring vertices against each other**. This is a same-domain comparison — both values come from the same measurement method applied to nearby positions. Systematic biases cancel out. What matters is the *change* between neighbors, not the absolute value. This is fundamentally more robust to noise.

#### Hardware constraints and mitigations

**Apple M2, 8GB unified memory:**
- Method A (structure tensor): No issue. 32³ patches × 500 vertices = trivial memory.
- Method B (nnUNet): Tight but feasible with these settings:
  - `use_mirroring=False` (disable test-time augmentation, halves memory)
  - `use_folds=(0,)` (single fold, not ensemble)
  - `perform_everything_on_device=False` (forced on MPS anyway)
  - Batch nearby vertices into shared patches to minimize inference calls
  - Original model (238MB) fits; newer models (781MB) do not

**Fallback**: If nnUNet inference fails due to memory, automatically fall back to Method A with a warning. The scoring algorithm is the same; only the input quality degrades.

#### Pre-computed fiber predictions

For scrolls where bruniss or the community has already run fiber inference, pre-computed prediction volumes may be available (e.g., `hz-cc-6.zarr`, `vt-cc-6.zarr` on dl.ash2txt.org). If a `--fiber-predictions` flag points to these, we skip inference entirely and just sample the existing labels. This is the fastest and most accurate path.

```bash
# Use pre-computed fiber predictions (fastest, no inference needed)
mesh-qa score segment.obj --volume URL --fiber-predictions path/to/predictions.zarr

# Use bruniss nnUNet model (highest accuracy, runs inference)
mesh-qa score segment.obj --volume URL --fiber-model path/to/nnUNet_results/Dataset004_fiber3/

# No fiber model available (fall back to structure tensor)
mesh-qa score segment.obj --volume URL
```

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
sheet_switching:     0.20  (reduced — fiber_coherence covers similar ground better)
self_intersections:  0.20
noise:               0.10
ct_normal_alignment: 0.05
fiber_coherence:   0.15
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
fiber_coherence:      0.15
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

**Tier 2:** Add fiber coherence visualization: color mesh by dominant fiber class (horizontal=blue, vertical=red). Sheet switches appear as abrupt color flips between adjacent regions. Overlay flagged fiber-flip boundaries. If nnUNet predictions are available, show per-vertex class probabilities as confidence.

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

### Phase 2: Fiber Coherence Metric (highest impact for parallel-layer switches)
- **Method A (structure tensor)**: Implement first. Uses `structure-tensor` PyPI package or numpy/scipy directly. Lightweight, no model download. Compare smallest eigenvector (fiber direction) between mesh-adjacent vertices. Same-domain comparison is more robust than our existing cross-domain CT metric.
- **Method B (nnUNet)**: Add as optional enhancement. Requires bruniss's pre-trained weights (~238MB download), nnunetv2 package, PyTorch. Highest accuracy. Gated behind `--fiber-model` flag.
- **Pre-computed predictions**: Support `--fiber-predictions` flag for scrolls where inference has already been done (fastest path).
- Catches the hardest case: tightly packed parallel layers where geometry and winding angle fail
- Requires validation on confirmed sheet-switch segments
- **Effort: Phase 2a (structure tensor): 2-3 days. Phase 2b (nnUNet integration): 3-5 days.**

### Phase 3: Review Viewer Improvements (high impact for adoption)
- Winding angle rainbow visualization (color mesh by accumulated winding angle — discontinuities = layer jumps)
- Fiber orientation heatmap (color mesh by dominant fiber class — hz=blue, vt=red, flips highlighted in yellow)
- Fiber class flip overlay showing exact boundaries where fiber orientation changes abruptly
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

- **Training our own fiber segmentation model.** bruniss has the $20K prize-winning nnUNet model. We integrate his pre-trained weights, not train our own. If better models emerge (Villa multi-task UNet, newer bruniss datasets), we swap them in.
- **Winding angle graph solver.** ThaumatoAnakalyptor has this. We use the concept (winding angle consistency check) but don't replicate the full graph optimization.
- **Surface detection ML.** The Kaggle competition handles this. We QA the output, not generate it.
- **Betti Matching / TopoScore.** Requires a reference surface (ground truth). We're scoring standalone meshes with no reference. Our topology metric already captures what's possible without a reference (manifoldness, components, boundary edges).

---

## Open Questions

1. **Umbilicus data per scroll** — ThaumatoAnakalyptor stores these as text files (rows of x, y, z). Available in their datasets. Need to confirm: does the community have umbilicus files for all active scrolls (PHerc1667, PHerc1447, PHerc0332)? Do Khartes/VC3D use a different format?

2. **Structure tensor vs nnUNet for fiber coherence** — our existing CT metric showed structure tensor normals have ~25-30° baseline noise for cross-domain comparison (CT normal vs mesh normal). But the fiber_coherence metric does same-domain comparison (neighbor vs neighbor), which should be more robust. Need to validate: is structure tensor fiber direction consistent enough between adjacent vertices on a good mesh? If not, nnUNet is the minimum viable approach. bruniss's original model weights URL is dead — need to confirm availability at `dl.ash2txt.org/community-uploads/bruniss/old_models/` or ask bruniss directly on Discord.

3. **Validation data** — we still need confirmed sheet-switch segments to validate the new metrics. The winding angle metric can be validated against ThaumatoAnakalyptor's own winding angle assignments if we can access that data.

4. **Weight tuning** — current weights are educated guesses. Once we have labeled data (good/bad segments with known failure modes), we should tune weights to maximize separation.

---

## Success Criteria

The tool succeeds if:
1. A segmenter can run `mesh-qa batch` on 1000 segments and get a ranked list where the bottom 10% actually contains the worst segments (validated by human review).
2. The HTML review page lets a domain expert confirm or deny flagged sheet switches in under 30 seconds per cluster.
3. The winding angle metric catches at least some parallel-layer switches that the geometry-only metrics miss (even one confirmed case is a win).
4. The community finds it useful enough to adopt (measured by: bruniss/Julian/Hendrik try it and give feedback).
