"""Tests for volume accessor."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


class TestVolumeAccessor:
    def test_sample_neighborhood_returns_correct_shape(self):
        from vesuvius_mesh_qa.volume import VolumeAccessor

        # Mock zarr array
        mock_array = MagicMock()
        mock_array.shape = (1000, 500, 500)
        mock_array.__getitem__ = MagicMock(
            return_value=np.random.rand(32, 32, 32).astype(np.float32)
        )

        accessor = VolumeAccessor.__new__(VolumeAccessor)
        accessor._vol = mock_array
        accessor._shape = mock_array.shape
        accessor._scale_factor = 1

        # Vertex at (250, 250, 500) -> volume index [500, 250, 250]
        chunk = accessor.sample_neighborhood(
            vertex_xyz=np.array([250.0, 250.0, 500.0]),
            half_size=16,
        )
        assert chunk.shape == (32, 32, 32)
        assert chunk.dtype == np.float32

    def test_vertex_in_bounds(self):
        from vesuvius_mesh_qa.volume import VolumeAccessor

        accessor = VolumeAccessor.__new__(VolumeAccessor)
        accessor._shape = (1000, 500, 500)
        accessor._scale_factor = 1

        assert accessor.vertex_in_bounds(np.array([250, 250, 500]), margin=16)
        assert not accessor.vertex_in_bounds(np.array([250, 250, 990]), margin=16)
        assert not accessor.vertex_in_bounds(np.array([5, 250, 500]), margin=16)

    def test_batch_sample_intensities(self):
        from vesuvius_mesh_qa.volume import VolumeAccessor

        mock_array = MagicMock()
        mock_array.shape = (1000, 500, 500)
        # Return a value for single voxel reads
        mock_array.__getitem__ = MagicMock(return_value=np.uint8(128))

        accessor = VolumeAccessor.__new__(VolumeAccessor)
        accessor._vol = mock_array
        accessor._shape = mock_array.shape
        accessor._scale_factor = 1

        vertices = np.array([[250, 250, 500], [100, 100, 200]], dtype=np.float64)
        intensities = accessor.sample_intensities(vertices)
        assert len(intensities) == 2

    def test_xyz_to_zyx_applies_scale_factor(self):
        from vesuvius_mesh_qa.volume import VolumeAccessor

        accessor = VolumeAccessor.__new__(VolumeAccessor)
        accessor._shape = (500, 250, 250)
        accessor._scale_factor = 2  # scale=1 means 2^1=2

        # Vertex (100, 200, 300) with scale factor 2:
        # ix = round(100/2) = 50, iy = round(200/2) = 100, iz = round(300/2) = 150
        iz, iy, ix = accessor._xyz_to_zyx(np.array([100.0, 200.0, 300.0]))
        assert iz == 150
        assert iy == 100
        assert ix == 50
