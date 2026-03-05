"""Training loop for ink detection models."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.dataset import FragmentDataset, PreloadedFragmentDataset, get_fragment_dirs
from src.models.ink_detector import InkDetectorUNet, InkDetectorLite
from src.utils.device import get_device


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss, matching the Grand Prize approach."""

    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)

        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice_loss = 1 - (2 * intersection + 1) / (pred_sigmoid.sum() + target.sum() + 1)

        return self.dice_weight * dice_loss + (1 - self.dice_weight) * bce_loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        volume = batch["volume"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        pred = model(volume)
        loss = criterion(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        volume = batch["volume"].to(device)
        label = batch["label"].to(device)

        pred = model(volume)
        loss = criterion(pred, label)
        total_loss += loss.item()

        all_preds.append(torch.sigmoid(pred).cpu().numpy())
        all_labels.append(label.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    binary_preds = (preds > 0.5).astype(np.float32)
    intersection = (binary_preds * labels).sum()
    dice = (2 * intersection) / (binary_preds.sum() + labels.sum() + 1e-8)

    return {
        "loss": total_loss / len(loader),
        "dice": dice,
    }


def main():
    parser = argparse.ArgumentParser(description="Train ink detection model")
    parser.add_argument("--data-dir", type=str, default="data/fragments", help="Path to fragment data")
    parser.add_argument("--model", type=str, default="lite", choices=["unet", "lite"], help="Model architecture")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Encoder for UNet model")
    parser.add_argument("--patch-size", type=int, default=64, help="Training patch size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for checkpoints")
    parser.add_argument("--z-start", type=int, default=15, help="Start z-layer")
    parser.add_argument("--z-end", type=int, default=45, help="End z-layer")
    parser.add_argument("--preload", action="store_true", help="Preload all data into RAM (faster but needs more memory)")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    fragment_dirs = get_fragment_dirs(args.data_dir)
    if not fragment_dirs:
        print(f"No fragments found in {args.data_dir}")
        print("Download fragment data first. See: scripts/download_data.sh")
        return

    print(f"Found {len(fragment_dirs)} fragments: {[d.name for d in fragment_dirs]}")

    in_channels = args.z_end - args.z_start
    DatasetClass = PreloadedFragmentDataset if args.preload else FragmentDataset
    dataset = DatasetClass(
        fragment_dirs=fragment_dirs,
        patch_size=args.patch_size,
        z_start=args.z_start,
        z_end=args.z_end,
    )
    print(f"Total patches: {len(dataset)}")

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.model == "unet":
        model = InkDetectorUNet(in_channels=in_channels, encoder_name=args.encoder)
    else:
        model = InkDetectorLite(in_channels=in_channels)

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} ({param_count:,} parameters)")

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = run_validation(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} | "
            f"val_dice: {val_metrics['dice']:.4f} | "
            f"lr: {scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "dice": best_dice, "args": vars(args)},
                output_dir / "best_model.pt",
            )
            print(f"  -> Saved best model (dice: {best_dice:.4f})")

    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": args.epochs, "dice": val_metrics["dice"], "args": vars(args)},
        output_dir / "final_model.pt",
    )
    print(f"\nTraining complete. Best dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
