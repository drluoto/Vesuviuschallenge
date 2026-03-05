"""Dataset for ink detection on Herculaneum scroll fragments.

Loads surface volume layers (multi-channel CT slices) and binary ink labels.
Based on the Grand Prize winning approach from villa/ink-detection.

Memory-efficient: loads patches lazily from disk to support 8GB RAM machines.
"""

from pathlib import Path

import numpy as np
import torch
import tifffile
from PIL import Image
from torch.utils.data import Dataset

# Default layer range matching the Grand Prize approach
DEFAULT_Z_START = 15
DEFAULT_Z_END = 45  # 30 layers


def find_layer_paths(fragment_dir: Path, z_start: int, z_end: int) -> list[Path]:
    """Find surface volume layer files for the given z range."""
    layers_dir = fragment_dir / "surface_volume"
    paths = []
    for i in range(z_start, z_end):
        for ext in (".tif", ".jpg", ".png"):
            path = layers_dir / f"{i:02d}{ext}"
            if path.exists():
                paths.append(path)
                break
        else:
            raise FileNotFoundError(f"No layer file found for index {i} in {layers_dir}")
    return paths


def read_layer(path: Path) -> np.ndarray:
    """Read a single surface volume layer as float32, clipped to [0, 200]."""
    if path.suffix == ".tif":
        img = tifffile.imread(str(path))
        if img.ndim == 3:
            img = img[:, :, 0]  # take first channel if multi-channel
        img = img.astype(np.float32)
    else:
        img = np.array(Image.open(path).convert("L"), dtype=np.float32)
    return np.clip(img, 0, 200)


def read_fragment(fragment_dir: str | Path, z_start: int = DEFAULT_Z_START, z_end: int = DEFAULT_Z_END) -> np.ndarray:
    """Read all surface volume layers into memory. Use only if RAM allows."""
    fragment_dir = Path(fragment_dir)
    paths = find_layer_paths(fragment_dir, z_start, z_end)
    images = [read_layer(p) for p in paths]
    return np.stack(images, axis=0)


def read_mask(fragment_dir: str | Path) -> np.ndarray:
    """Read the fragment mask (valid region). Normalized to [0, 1]."""
    fragment_dir = Path(fragment_dir)
    for name in ("mask.png", "mask.tif"):
        path = fragment_dir / name
        if path.exists():
            mask = np.array(Image.open(path).convert("L"), dtype=np.float32)
            if mask.max() > 1:
                mask = mask / 255.0
            return mask
    raise FileNotFoundError(f"No mask found in {fragment_dir}")


def read_ink_label(fragment_dir: str | Path) -> np.ndarray:
    """Read the ink label (ground truth from IR photography).

    Labels may be binary (0/1) or grayscale (0/255) — both are normalized to [0, 1].
    """
    fragment_dir = Path(fragment_dir)
    for name in ("inklabels.png", "inklabels.tif", "ink_labels.png"):
        path = fragment_dir / name
        if path.exists():
            label = np.array(Image.open(path).convert("L"), dtype=np.float32)
            if label.max() > 1:
                label = label / 255.0
            return label
    raise FileNotFoundError(f"No ink label found in {fragment_dir}")


class FragmentDataset(Dataset):
    """Memory-efficient patch dataset for ink detection training.

    Loads only mask and labels into memory (~15MB each for Frag3).
    Reads volume patches lazily from disk via tifffile.
    """

    def __init__(
        self,
        fragment_dirs: list[str | Path],
        patch_size: int = 64,
        stride: int | None = None,
        z_start: int = DEFAULT_Z_START,
        z_end: int = DEFAULT_Z_END,
        transform=None,
    ):
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.z_start = z_start
        self.z_end = z_end
        self.transform = transform

        self.fragments = []
        self.patches = []  # (fragment_idx, y, x)

        for i, fdir in enumerate(fragment_dirs):
            fdir = Path(fdir)
            mask = read_mask(fdir)
            label = read_ink_label(fdir)
            layer_paths = find_layer_paths(fdir, z_start, z_end)

            self.fragments.append({
                "layer_paths": layer_paths,
                "mask": mask,
                "label": label,
                "path": fdir,
            })

            h, w = mask.shape
            for y in range(0, h - patch_size + 1, self.stride):
                for x in range(0, w - patch_size + 1, self.stride):
                    patch_mask = mask[y : y + patch_size, x : x + patch_size]
                    if patch_mask.mean() > 0.5:
                        self.patches.append((i, y, x))

        print(f"FragmentDataset: {len(self.fragments)} fragments, {len(self.patches)} patches")

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> dict:
        frag_idx, y, x = self.patches[idx]
        frag = self.fragments[frag_idx]
        ps = self.patch_size

        # Read only the patch region from each layer (lazy disk I/O)
        layers = []
        for path in frag["layer_paths"]:
            if path.suffix == ".tif":
                # tifffile can read a crop efficiently for some TIF formats
                full = tifffile.imread(str(path))
                if full.ndim == 3:
                    full = full[:, :, 0]
                patch = full[y : y + ps, x : x + ps].astype(np.float32)
            else:
                img = Image.open(path).convert("L")
                patch = np.array(img.crop((x, y, x + ps, y + ps)), dtype=np.float32)
            layers.append(np.clip(patch, 0, 200))

        volume_patch = np.stack(layers, axis=0) / 200.0  # normalize to [0, 1]
        label_patch = frag["label"][y : y + ps, x : x + ps].copy()

        if self.transform:
            transformed = self.transform(image=volume_patch.transpose(1, 2, 0), mask=label_patch)
            volume_patch = transformed["image"].transpose(2, 0, 1)
            label_patch = transformed["mask"]

        return {
            "volume": torch.from_numpy(volume_patch).float(),
            "label": torch.from_numpy(label_patch).float().unsqueeze(0),
        }


class PreloadedFragmentDataset(Dataset):
    """Loads all data into RAM. Only use if you have enough memory.

    For Frag3 with 10 layers: ~1.5GB. For 30 layers: ~4.5GB.
    """

    def __init__(
        self,
        fragment_dirs: list[str | Path],
        patch_size: int = 64,
        stride: int | None = None,
        z_start: int = DEFAULT_Z_START,
        z_end: int = DEFAULT_Z_END,
        transform=None,
    ):
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.transform = transform

        self.fragments = []
        self.patches = []

        for i, fdir in enumerate(fragment_dirs):
            fdir = Path(fdir)
            volume = read_fragment(fdir, z_start, z_end)
            mask = read_mask(fdir)
            label = read_ink_label(fdir)
            self.fragments.append({"volume": volume, "mask": mask, "label": label})

            h, w = mask.shape
            for y in range(0, h - patch_size + 1, self.stride):
                for x in range(0, w - patch_size + 1, self.stride):
                    if mask[y : y + patch_size, x : x + patch_size].mean() > 0.5:
                        self.patches.append((i, y, x))

        print(f"PreloadedFragmentDataset: {len(self.fragments)} fragments, {len(self.patches)} patches")

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> dict:
        frag_idx, y, x = self.patches[idx]
        frag = self.fragments[frag_idx]
        ps = self.patch_size

        volume_patch = frag["volume"][:, y : y + ps, x : x + ps].copy() / 200.0
        label_patch = frag["label"][y : y + ps, x : x + ps].copy()

        if self.transform:
            transformed = self.transform(image=volume_patch.transpose(1, 2, 0), mask=label_patch)
            volume_patch = transformed["image"].transpose(2, 0, 1)
            label_patch = transformed["mask"]

        return {
            "volume": torch.from_numpy(volume_patch).float(),
            "label": torch.from_numpy(label_patch).float().unsqueeze(0),
        }


def get_fragment_dirs(data_root: str | Path) -> list[Path]:
    """Find all fragment directories under data_root that have the required files."""
    data_root = Path(data_root)
    dirs = []
    for d in sorted(data_root.iterdir()):
        if d.is_dir() and (d / "surface_volume").exists():
            dirs.append(d)
    return dirs
