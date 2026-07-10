from __future__ import annotations

import jax.numpy as jnp

import torch
from torch import Tensor


def resize_torch_grid_sample(x, out_h=299, out_w=299):
    assert x.ndim == 4, x.shape
    B, H, W, C = x.shape
    x = x.astype(jnp.float32)

    ys = (jnp.arange(out_h, dtype=jnp.float32) + 0.5) * (H / out_h) - 0.5
    y0 = jnp.floor(ys).astype(jnp.int32)
    y1 = y0 + 1
    wy = (ys - y0.astype(jnp.float32)).reshape(1, out_h, 1, 1)

    row0 = jnp.take(x, y0, axis=1, mode="clip")
    row1 = jnp.take(x, y1, axis=1, mode="clip")
    tmp = row0 * (1.0 - wy) + row1 * wy

    xs = (jnp.arange(out_w, dtype=jnp.float32) + 0.5) * (W / out_w) - 0.5
    x0 = jnp.floor(xs).astype(jnp.int32)
    x1 = x0 + 1
    wx = (xs - x0.astype(jnp.float32)).reshape(1, 1, out_w, 1)

    col0 = jnp.take(tmp, x0, axis=2, mode="clip")
    col1 = jnp.take(tmp, x1, axis=2, mode="clip")
    out = col0 * (1.0 - wx) + col1 * wx

    return out


# was missing from pMF got it from: https://github.com/Lyy-iiis/imeanflow/blob/main/utils/jax_fid/resize.py
def forward(
    img: Tensor,
) -> Tensor:

    _0 = torch.nn.functional.affine_grid
    _1 = torch.nn.functional.grid_sample
    (
        batch_size,
        channels,
        height,
        width,
    ) = img.shape
    x = img
    theta = torch.eye(2, 3, dtype=None, layout=None)
    _3 = torch.select(torch.select(theta, 0, 0), 0, 2)
    _4 = torch.select(torch.select(theta, 0, 0), 0, 0)
    _5 = torch.div(_4, width)
    _6 = torch.select(torch.select(theta, 0, 0), 0, 0)
    _7 = torch.add(_3, torch.sub(_5, torch.div(_6, 299)))
    _8 = torch.select(torch.select(theta, 0, 1), 0, 2)
    _9 = torch.select(torch.select(theta, 0, 1), 0, 1)
    _10 = torch.div(_9, height)
    _11 = torch.select(torch.select(theta, 0, 1), 0, 1)
    _12 = torch.add(_8, torch.sub(_10, torch.div(_11, 299)))
    _13 = torch.unsqueeze(theta, 0)
    theta0 = _13.repeat([batch_size, 1, 1])
    grid = _0(
        theta0,
        [batch_size, channels, 299, 299],
        False,
    )
    x0 = _1(
        x,
        grid,
        "bilinear",
        "border",
        False,
    )
    x1 = torch.sub(x0, 128)
    x2 = torch.div(x1, 128)

    return x2
