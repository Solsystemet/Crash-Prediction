"""Device detection utilities for hardware-agnostic training.

This module provides utilities for detecting the best available compute device
(CUDA/ROCm GPU or CPU) in a hardware-agnostic way.

PyTorch's torch.cuda API works for both NVIDIA CUDA and AMD ROCm, as ROCm
provides a HIP-based compatibility layer. This module adds proper detection
and logging to help users understand which backend is being used.
"""

import torch


def get_best_device() -> str:
    """Get the best available compute device.

    Checks for GPU availability (CUDA for NVIDIA, ROCm/HIP for AMD) and
    falls back to CPU if no GPU is available.

    Returns:
        Device string: "cuda" if GPU available, "cpu" otherwise.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device_info() -> dict[str, str | int | bool]:
    """Get detailed information about the available compute devices.

    Returns:
        Dictionary with device information including:
        - device: The device string ("cuda" or "cpu")
        - device_name: Name of the device (GPU name or "CPU")
        - gpu_available: Whether a GPU is available
        - backend: The backend being used ("cuda", "rocm", or "cpu")
        - device_count: Number of available GPUs
    """
    info: dict[str, str | int | bool] = {
        "device": "cpu",
        "device_name": "CPU",
        "gpu_available": False,
        "backend": "cpu",
        "device_count": 0,
    }

    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu_available"] = True
        info["device_count"] = torch.cuda.device_count()

        # Get device name
        try:
            info["device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            info["device_name"] = "Unknown GPU"

        # Detect backend (CUDA vs ROCm)
        # ROCm uses HIP which is exposed through torch.cuda
        # We can detect ROCm by checking the device name or torch version
        device_name = str(info["device_name"]).lower()
        torch_version = torch.__version__.lower()

        if "rocm" in torch_version or "hip" in torch_version:
            info["backend"] = "rocm"
        elif any(
            amd in device_name
            for amd in ["amd", "radeon", "mi50", "mi100", "mi200", "mi250", "mi300"]
        ):
            info["backend"] = "rocm"
        else:
            info["backend"] = "cuda"

    return info


def print_device_info() -> str:
    """Print and return information about the compute device being used.

    Returns:
        The device string that will be used for training.
    """
    info = get_device_info()

    if info["gpu_available"]:
        backend_name = "ROCm (AMD)" if info["backend"] == "rocm" else "CUDA (NVIDIA)"
        print(f"GPU Available: {info['device_name']}")
        print(f"Backend: {backend_name}")
        print(f"Device count: {info['device_count']}")
    else:
        print("No GPU available, using CPU")
        print("Note: Training on CPU may be significantly slower")

    return str(info["device"])


def get_device_for_config() -> str:
    """Get the device string suitable for use in training configs.

    This is a simple wrapper that returns the device string without printing.

    Returns:
        Device string: "cuda" if GPU available, "cpu" otherwise.
    """
    return get_best_device()
