# %% [markdown]
# # 01 - Explore Vesuvius Challenge Data
#
# Initial exploration of fragment data and the vesuvius library.
# Run this notebook to verify your setup is working.

# %%
import sys
sys.path.insert(0, "..")

import torch
import numpy as np
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")

from src.utils.device import get_device
device = get_device()
print(f"Using device: {device}")

# %%
# Test that models can be instantiated and run
from src.models.ink_detector import InkDetectorLite, InkDetectorUNet

# Lite model - good for M2 development
model_lite = InkDetectorLite(in_channels=30).to(device)
param_count = sum(p.numel() for p in model_lite.parameters())
print(f"InkDetectorLite: {param_count:,} parameters")

# Test forward pass with random data
x = torch.randn(1, 30, 64, 64).to(device)
with torch.no_grad():
    y = model_lite(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Output range: [{y.min().item():.3f}, {y.max().item():.3f}]")

# %%
# UNet model - more powerful, needs more memory
model_unet = InkDetectorUNet(in_channels=30, encoder_name="resnet34").to(device)
param_count = sum(p.numel() for p in model_unet.parameters())
print(f"InkDetectorUNet (resnet34): {param_count:,} parameters")

with torch.no_grad():
    y = model_unet(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Output range: [{y.min().item():.3f}, {y.max().item():.3f}]")

# %%
# Test the vesuvius library (if installed)
try:
    import vesuvius
    print(f"vesuvius library version: {vesuvius.__version__}")
    print("Library available - can stream data directly from S3")
except ImportError:
    print("vesuvius library not installed. Install with: pip install vesuvius")

# %% [markdown]
# ## Next Steps
#
# 1. Download fragment data using `scripts/download_fragments.sh`
#    or use the vesuvius library to stream data
# 2. Load and visualize surface volume layers
# 3. Train a model with `python -m src.training.train`
# 4. Run inference with `python -m src.inference.predict`
