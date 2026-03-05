"""Inference pipeline for ink detection.

Runs a trained model on fragment surface volumes,
producing a full-resolution ink prediction map using sliding window
with Gaussian-weighted tile blending.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.data.dataset import read_fragment, read_mask
from src.models.ink_detector import InkDetectorUNet, InkDetectorLite
from src.utils.device import get_device


def make_gaussian_kernel(size: int, sigma: float | None = None) -> np.ndarray:
    """Create 2D Gaussian weighting kernel for tile blending."""
    if sigma is None:
        sigma = size / 4
    ax = np.linspace(-size / 2, size / 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.max()


@torch.no_grad()
def predict_fragment(
    model: nn.Module,
    fragment_dir: str | Path,
    patch_size: int = 64,
    stride: int | None = None,
    z_start: int = 15,
    z_end: int = 45,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> np.ndarray:
    """Run inference on a fragment, returning full-resolution prediction map."""
    if device is None:
        device = get_device()
    if stride is None:
        stride = patch_size // 2

    model.to(device)
    model.eval()

    volume = read_fragment(fragment_dir, z_start, z_end)
    mask = read_mask(fragment_dir)

    _, h, w = volume.shape
    volume = volume / 200.0

    pred_sum = np.zeros((h, w), dtype=np.float64)
    weight_sum = np.zeros((h, w), dtype=np.float64)
    gaussian = make_gaussian_kernel(patch_size)

    patches = []
    coords = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            if mask[y : y + patch_size, x : x + patch_size].mean() > 0.3:
                patch = volume[:, y : y + patch_size, x : x + patch_size]
                patches.append(patch)
                coords.append((y, x))

    for i in range(0, len(patches), batch_size):
        batch = np.stack(patches[i : i + batch_size])
        batch_tensor = torch.from_numpy(batch).float().to(device)
        preds = torch.sigmoid(model(batch_tensor)).cpu().numpy()[:, 0]

        for j, (y, x) in enumerate(coords[i : i + batch_size]):
            pred_sum[y : y + patch_size, x : x + patch_size] += preds[j] * gaussian
            weight_sum[y : y + patch_size, x : x + patch_size] += gaussian

    prediction = np.divide(pred_sum, weight_sum, where=weight_sum > 0, out=np.zeros_like(pred_sum))
    prediction = (prediction * mask).clip(0, 1)

    return prediction


def load_model(checkpoint_path: str | Path, device: torch.device | None = None) -> nn.Module:
    """Load a trained model from checkpoint."""
    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]

    in_channels = args.get("z_end", 45) - args.get("z_start", 15)
    if args.get("model") == "unet":
        model = InkDetectorUNet(in_channels=in_channels, encoder_name=args.get("encoder", "resnet34"))
    else:
        model = InkDetectorLite(in_channels=in_channels)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Run ink detection inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--fragment-dir", type=str, required=True, help="Path to fragment directory")
    parser.add_argument("--output", type=str, default="outputs/prediction.png", help="Output path")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)
    print("Model loaded")

    prediction = predict_fragment(
        model,
        args.fragment_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        device=device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_img = Image.fromarray((prediction * 255).astype(np.uint8))
    pred_img.save(output_path)
    print(f"Prediction saved to {output_path}")


if __name__ == "__main__":
    main()
