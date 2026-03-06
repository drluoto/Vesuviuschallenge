"""Tests for CT normal computation."""
import numpy as np
import pytest


class TestCTNormals:
    def test_synthetic_horizontal_sheet(self):
        """Structure tensor on a horizontal bright sheet should give Z-pointing normal."""
        from vesuvius_mesh_qa.ct_normals import compute_ct_normal

        vol = np.zeros((32, 32, 32), dtype=np.float32)
        vol[14:18, :, :] = 1.0  # horizontal sheet

        normal, anisotropy = compute_ct_normal(vol, sigma=1.0)

        # Normal should point along dim0 (Z in volume space)
        assert abs(normal[0]) > 0.9  # Z component dominant
        assert anisotropy > 0.5  # clear planar structure

    def test_synthetic_tilted_sheet(self):
        """Structure tensor on a 45-deg sheet should give tilted normal."""
        from vesuvius_mesh_qa.ct_normals import compute_ct_normal

        vol = np.zeros((32, 32, 32), dtype=np.float32)
        for i in range(32):
            z = int(16 + (i - 16) * 0.5)
            if 0 <= z < 32:
                vol[z, i, :] = 1.0

        normal, anisotropy = compute_ct_normal(vol, sigma=1.0)
        assert anisotropy > 0.5
        # Normal should have both Z and Y components
        assert abs(normal[0]) > 0.3  # Z component
        assert abs(normal[1]) > 0.3  # Y component

    def test_empty_volume_returns_zero_anisotropy(self):
        """Empty volume should have near-zero anisotropy."""
        from vesuvius_mesh_qa.ct_normals import compute_ct_normal

        vol = np.zeros((32, 32, 32), dtype=np.float32)
        normal, anisotropy = compute_ct_normal(vol, sigma=1.0)
        assert anisotropy < 0.1

    def test_batch_compute(self):
        """Batch computation returns normals and anisotropy for multiple chunks."""
        from vesuvius_mesh_qa.ct_normals import compute_ct_normals_batch

        chunks = [np.zeros((32, 32, 32), dtype=np.float32) for _ in range(3)]
        chunks[0][14:18, :, :] = 1.0  # one has structure
        chunks[1][14:18, :, :] = 1.0

        normals, anisotropies = compute_ct_normals_batch(chunks, sigma=1.0)
        assert normals.shape == (3, 3)
        assert anisotropies.shape == (3,)
        assert anisotropies[0] > 0.5
        assert anisotropies[2] < 0.1  # empty chunk
