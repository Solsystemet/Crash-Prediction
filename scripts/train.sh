#!/bin/bash
# Training script with AMD GPU support for gfx1102 (RDNA3)
# This sets the required environment variable for ROCm compatibility

export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Suppress harmless amdgpu.ids warnings
export AMD_LOG_LEVEL=0

# Run training with all arguments passed through
exec uv run python -m training.main_aggregate "$@"
