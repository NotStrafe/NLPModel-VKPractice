from __future__ import annotations


def resolve_device(device: str | None = "auto") -> str | None:
    if device and device != "auto":
        return device

    try:
        import torch
    except ImportError:
        return None

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
