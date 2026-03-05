# Vesuvius Challenge Project

## Project Goal
Win prize money on scrollprize.org by contributing to reading the Herculaneum papyrus scrolls buried by Mount Vesuvius in 79 AD.

## Challenge Overview
The Vesuvius Challenge uses machine learning, computer vision, and computational geometry to digitally unwrap and read ancient carbonized papyri from 3D micro-CT scans without physically damaging them.

### Three Core Problems
1. **3D Surface Detection** - Semantic segmentation of scroll layers from volumetric CT data (Kaggle competition, $200k prize pool)
2. **Geometric Reconstruction (Segmentation)** - Mapping surfaces and extracting readable sheets from detected layers
3. **Ink Detection** - Identifying carbon ink patterns on extracted surfaces using ML (7 x $60k First Letters/Title prizes)

### Active Prizes (2026)
- **3D Surface Detection (Kaggle)**: $200,000 total (1st: $60k, 2nd: $40k, 3rd: $30k...)
- **First Letters Prize**: $60,000 per scroll for 10+ legible letters in 4cm² area (Scrolls 2-3)
- **First Title Prize**: $60,000 per scroll for discovering titles (Scrolls 1-3)
- **Monthly Progress Prizes**: $1k-$20k for open-source contributions

### Virtual Unwrapping Pipeline
1. Scan scrolls with synchrotron micro-CT → 3D volumetric data (OME-Zarr / TIFF stacks)
2. ML-based surface detection (nnUNet semantic segmentation)
3. Geometry processing to extract sheet meshes
4. Sample surface volumes around the mesh
5. ML ink detection on extracted surfaces

### Three Segmentation Approaches
- **Spiral Fitting** (top-down): Global optimization fitting canonical 3D spiral
- **Surface Tracer** (bottom-up): Iterative local mesh expansion from seed patches
- **Thaumato Anakalyptor** (bottom-up): Dense point-cloud extraction + graph-based stitching

### Key Unsolved Problem
**Sheet switching** — where fitted surfaces incorrectly jump between wrapping layers. This is the main bottleneck preventing full automation.

### Ink Detection
- Herculaneum ink and papyrus have similar densities → can't threshold, need ML
- Training: fragment surfaces with IR photography as ground truth labels
- Models learn subtle textural patterns from 3D surface volumes
- Hallucination mitigation required (max window: 0.5x0.5mm)

## Data
- **Volumes**: 3D micro-CT reconstructions (OME-Zarr or TIFF stacks)
- **Segments**: Extracted papyrus surfaces as geometry + surface-aligned texture volumes
- **Representations/Predictions**: ML-predicted surfaces and ink detections
- **Access**: `s3://vesuvius-challenge-open-data/` or `https://data.aws.ash2txt.org/samples/`
- **License**: CC-BY-NC 4.0 (EduceLab-Scrolls Dataset)

## Key Resources
- **Villa monorepo**: github.com/ScrollPrize/villa (Python + C libraries, segmentation tools, ink detection)
- **Python library**: `vesuvius` package for CT data access
- **Tutorials**: scrollprize.org/tutorial (ink detection: /tutorial5, unwrapping: /unwrapping)
- **Discord**: discord.gg/V4fJhvtaQn
- **Data browser**: scrollprize.org/data

## Tech Stack
- Python (primary)
- PyTorch for ML models
- nnUNet for semantic segmentation
- CUDA for GPU acceleration
- OME-Zarr for volumetric data

## Project Structure
```
/data/              - Downloaded scroll data (gitignored)
/notebooks/         - Exploration and experiment notebooks
/src/               - Source code
/models/            - Trained model checkpoints (gitignored)
/outputs/           - Generated outputs and submissions (gitignored)
```

## Conventions
- Use Python 3.10+
- Format with ruff
- Type hints on public APIs
- Experiments tracked in notebooks with clear descriptions
- Large files (data, models, outputs) are gitignored

## Submission Requirements
- Programmatically generated images (no manual annotations)
- Scale bar indicating 1cm
- 3D position coordinates within scroll
- Reproducible methodology documentation
- Hallucination mitigation strategy
- No training-prediction data overlap
- Winners must open-source under permissive license
