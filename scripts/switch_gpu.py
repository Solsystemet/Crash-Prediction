#!/usr/bin/env python3
"""Switch PyTorch between CUDA and ROCm backends.

Usage:
    python scripts/switch_gpu.py cuda   # For NVIDIA GPUs
    python scripts/switch_gpu.py rocm   # For AMD GPUs

Then run: uv sync --reinstall
"""

import sys
from pathlib import Path


def switch_backend(backend: str) -> None:
    """Switch the PyTorch backend in pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()

    if backend == "cuda":
        old = 'torch = { index = "pytorch-rocm" }'
        new = 'torch = { index = "pytorch-cu124" }'
        old2 = 'torchvision = { index = "pytorch-rocm" }'
        new2 = 'torchvision = { index = "pytorch-cu124" }'
        print("Switching to CUDA (NVIDIA GPU)")
    elif backend == "rocm":
        old = 'torch = { index = "pytorch-cu124" }'
        new = 'torch = { index = "pytorch-rocm" }'
        old2 = 'torchvision = { index = "pytorch-cu124" }'
        new2 = 'torchvision = { index = "pytorch-rocm" }'
        print("Switching to ROCm (AMD GPU)")
    else:
        print(f"Unknown backend: {backend}")
        print("Usage: python scripts/switch_gpu.py [cuda|rocm]")
        sys.exit(1)

    if old not in content:
        print(f"Already using {backend} backend (or pyproject.toml format changed)")
        return

    content = content.replace(old, new).replace(old2, new2)
    pyproject_path.write_text(content)

    print(f"Updated pyproject.toml to use {backend}")
    print("\nNow run:")
    print("  uv sync --reinstall")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/switch_gpu.py [cuda|rocm]")
        sys.exit(1)

    switch_backend(sys.argv[1].lower())
