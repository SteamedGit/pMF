#!/bin/bash
# GPU environment for pMF. Works on Blackwell (sm_120: RTX PRO / RTX 50-series laptop
# used for local debugging) AND Hopper (sm_90: H200 cluster). Needs a recent NVIDIA
# driver (>= 560, i.e. a CUDA 12.8+ capable driver).
#
# Why this differs from the old TPU install (scripts/install.sh):
#   * jax 0.10.x ships CUDA 12.9 wheels (cudnn 9.24, cublas 12.9) that have Blackwell
#     kernels. The old jax 0.4.x pins have NO sm_120 support -> that was the
#     "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR" crash.
#   * torch is installed CPU-only. It is used solely for the ImageNet DataLoader,
#     ConvNeXt weight loading, and a CPU tensor in FID -- never on the GPU. A CUDA
#     torch would bundle its own cudnn/cublas and clobber jax's (that is what broke
#     the previous env: torch pulled cudnn 9.1 over jax's).
#   * TensorFlow / keras / clu are dropped: nothing in the repo imports them.
set -euo pipefail

uv venv --python 3.11 --clear

# --- JAX with CUDA 12 (pulls matching cudnn/cublas/nccl). Blackwell + Hopper capable.
uv pip install "jax[cuda12]==0.10.2"

# --- CPU-only PyTorch (keeps all CUDA libraries out of the env).
uv pip install "torch==2.13.0" "torchvision==0.28.0" \
  --index-url https://download.pytorch.org/whl/cpu

# --- JAX ecosystem + project dependencies (all pure-python / CPU; no CUDA libs).
uv pip install \
  "flax==0.12.7" "optax==0.2.8" "orbax-checkpoint==0.12.1" "chex==0.1.92" \
  "ml-collections==1.1.0" "ml-dtypes==0.5.4" "tensorstore==0.1.84" \
  lpips_j transformers wandb pillow requests tqdm pyyaml absl-py

# --- Quick sanity check: matmul (cublas) + conv (cudnn) on the GPU.
uv run python - <<'PY'
import jax, jax.numpy as jnp
from jax import lax
print("jax", jax.__version__, "| devices:", jax.devices())
x = jnp.ones((512, 512)); assert float((x @ x)[0, 0]) == 512.0
img = jnp.ones((4, 32, 32, 3)); k = jnp.ones((3, 3, 3, 8))
lax.conv_general_dilated(img, k, (1, 1), "SAME",
    dimension_numbers=("NHWC", "HWIO", "NHWC")).block_until_ready()
print("GPU matmul + cudnn conv OK")
PY
