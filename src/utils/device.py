import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16
    # MPS supports float16 but bfloat16 can be more stable
    return torch.float32
