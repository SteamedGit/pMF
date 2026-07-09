#!/bin/bash

uv venv --python 3.11
uv pip install jax[cuda12]==0.4.27 jaxlib==0.4.27 flax==0.10.4 pillow clu tensorflow==2.15.0 keras==2.15.0
uv pip install torch==2.4.0 torchvision
uv pip install orbax-checkpoint==0.6.4 ml-dtypes==0.5.0 tensorstore==0.1.67
uv pip install wandb lpips_j optax ml-collections
uv pip install safetensors
