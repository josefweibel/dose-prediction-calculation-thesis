"""
MedNeXt model implementation with diffusion support.

This module contains:
- MedNeXt: Standard MedNeXt architecture for medical image segmentation
- MedNeXtDiffusion: Extended version with time step injection and conditioning for diffusion models
- Helper classes: TimeEmbedding, Swish, TimeEmbeddingProjection
- build_mednext_model: Factory function for creating models

The MedNeXt architecture is based on the MedNeXt paper and provides a ConvNeXt-based
approach to medical image segmentation with 2D/3D support.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional

from nnunet_mednext.network_architecture.mednextv1.blocks import MedNeXtBlock, MedNeXtDownBlock, OutBlock
from model_cache import DeepCache
from mednext_fixed_blocks import MedNeXtUpBlock


class MedNeXt(nn.Module):
  def __init__(
    self,
    in_channels: int,
    n_channels: int,
    n_classes: int,
    exp_r: int = 4,  # Expansion ratio as in Swin Transformers
    kernel_size: int = 7,  # Ofcourse can test kernel_size
    enc_kernel_size: int = None,
    dec_kernel_size: int = None,
    deep_supervision: bool = False,  # Can be used to test deep supervision
    do_res: bool = False,  # Can be used to individually test residual connection
    do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
    checkpoint_style: bool = None,  # Either inside block or outside block
    block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],  # Can be used to test staging ratio:
    # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
    norm_type='group',
    dim='3d',  # 2d or 3d
    grn=False,
  ):
    super().__init__()

    self.do_ds = deep_supervision
    assert checkpoint_style in [None, 'outside_block']
    self.inside_block_checkpointing = False
    self.outside_block_checkpointing = False
    if checkpoint_style == 'outside_block':
      self.outside_block_checkpointing = True
    assert dim in ['2d', '3d']

    if kernel_size is not None:
      enc_kernel_size = kernel_size
      dec_kernel_size = kernel_size

    if dim == '2d':
      conv = nn.Conv2d
    elif dim == '3d':
      conv = nn.Conv3d

    self.stem = conv(in_channels, n_channels, kernel_size=1)
    if type(exp_r) == int:
      exp_r = [exp_r for i in range(len(block_counts))]

    self.enc_block_0 = nn.Sequential(
      *[
        MedNeXtBlock(
          in_channels=n_channels,
          out_channels=n_channels,
          exp_r=exp_r[0],
          kernel_size=enc_kernel_size,
          do_res=do_res,
          norm_type=norm_type,
          dim=dim,
          grn=grn,
        )
        for i in range(block_counts[0])
      ]
    )

    self.down_0 = MedNeXtDownBlock(
      in_channels=n_channels,
      out_channels=2 * n_channels,
      exp_r=exp_r[1],
      kernel_size=enc_kernel_size,
      do_res=do_res_up_down,
      norm_type=norm_type,
      dim=dim,
    )

    self.enc_block_1 = nn.Sequential(
      *[
        MedNeXtBlock(
          in_channels=n_channels * 2,
          out_channels=n_channels * 2,
          exp_r=exp_r[1],
          kernel_size=enc_kernel_size,
          do_res=do_res,
          norm_type=norm_type,
          dim=dim,
          grn=grn,
        )
        for i in range(block_counts[1])
      ]
    )

    self.down_1 = MedNeXtDownBlock(
      in_channels=2 * n_channels,
      out_channels=4 * n_channels,
      exp_r=exp_r[2],
      kernel_size=enc_kernel_size,
      do_res=do_res_up_down,
      norm_type=norm_type,
      dim=dim,
      grn=grn,
    )

    self.enc_block_2 = nn.Sequential(
      *[
        MedNeXtBlock(
          in_channels=n_channels * 4,
          out_channels=n_channels * 4,
          exp_r=exp_r[2],
          kernel_size=enc_kernel_size,
          do_res=do_res,
          norm_type=norm_type,
          dim=dim,
          grn=grn,
        )
        for i in range(block_counts[2])
      ]
    )

    self.down_2 = MedNeXtDownBlock(
      in_channels=4 * n_channels,
      out_channels=8 * n_channels,
      exp_r=exp_r[3],
      kernel_size=enc_kernel_size,
      do_res=do_res_up_down,
      norm_type=norm_type,
      dim=dim,
      grn=grn,
    )

    self.enc_block_3 = nn.Sequential(
      *[
        MedNeXtBlock(
          in_channels=n_channels * 8,
          out_channels=n_channels * 8,
          exp_r=exp_r[3],
          kernel_size=enc_kernel_size,
          do_res=do_res,
          norm_type=norm_type,
          dim=dim,
          grn=grn,
        )
        for i in range(block_counts[3])
      ]
    )

    self.down_3 = MedNeXtDownBlock(
      in_channels=8 * n_channels,
      out_channels=16 * n_channels,
      exp_r=exp_r[4],
      kernel_size=enc_kernel_size,
      do_res=do_res_up_down,
      norm_type=norm_type,
      dim=dim,
      grn=grn,
    )

    self.bottleneck = nn.Sequential(
      *[
        MedNeXtBlock(
          in_channels=n_channels * 16,
          out_channels=n_channels * 16,
          exp_r=exp_r[4],
          kernel_size=dec_kernel_size,
          do_res=do_res,
          norm_type=norm_type,
          dim=dim,
          grn=grn,
        )
        for i in range(block_counts[4])
      ]
    )

    self.up_3 = MedNeXtUpBlock(
      in_channels=16 * n_channels,
      out_channels=8 * n_channels,
      exp_r=exp_r[5],
      kernel_size=dec_kernel_size,
      do_res=do_res_up_down,
      norm_type=norm_type,
      dim=dim,
      grn=grn,
    )

    self.dec_block_3 = nn.Sequential(
      *[
        MedNeXtBlock(
          in_channels=n_channels * 8,
          out_channels=n_channels * 8,
          exp_r=exp_r[5],
          kernel_size=dec_kernel_size,
          do_res=do_res,
          norm_type=norm_type,
          dim=dim,
          grn=grn,
        )
        for i in range(block_counts[5])
      ]
    )

    self.up_2 = MedNeXtUpBlock(
      in_channels=8 * n_channels,
      out_channels=4 * n_channels,
      exp_r=exp_r[6],
      kernel_size=dec_kernel_size,
      do_res=do_res_up_down,
      norm_type=norm_type,
      dim=dim,
      grn=grn,
    )

    self.dec_block_2 = nn.Sequential(
      *[
        MedNeXtBlock(
          in_channels=n_channels * 4,
          out_channels=n_channels * 4,
          exp_r=exp_r[6],
          kernel_size=dec_kernel_size,
          do_res=do_res,
          norm_type=norm_type,
          dim=dim,
          grn=grn,
        )
        for i in range(block_counts[6])
      ]
    )

    self.up_1 = MedNeXtUpBlock(
      in_channels=4 * n_channels,
      out_channels=2 * n_channels,
      exp_r=exp_r[7],
      kernel_size=dec_kernel_size,
      do_res=do_res_up_down,
      norm_type=norm_type,
      dim=dim,
      grn=grn,
    )

    self.dec_block_1 = nn.Sequential(
      *[
        MedNeXtBlock(
          in_channels=n_channels * 2,
          out_channels=n_channels * 2,
          exp_r=exp_r[7],
          kernel_size=dec_kernel_size,
          do_res=do_res,
          norm_type=norm_type,
          dim=dim,
          grn=grn,
        )
        for i in range(block_counts[7])
      ]
    )

    self.up_0 = MedNeXtUpBlock(
      in_channels=2 * n_channels,
      out_channels=n_channels,
      exp_r=exp_r[8],
      kernel_size=dec_kernel_size,
      do_res=do_res_up_down,
      norm_type=norm_type,
      dim=dim,
      grn=grn,
    )

    self.dec_block_0 = nn.Sequential(
      *[
        MedNeXtBlock(
          in_channels=n_channels,
          out_channels=n_channels,
          exp_r=exp_r[8],
          kernel_size=dec_kernel_size,
          do_res=do_res,
          norm_type=norm_type,
          dim=dim,
          grn=grn,
        )
        for i in range(block_counts[8])
      ]
    )

    self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)

    # Used to fix PyTorch checkpointing bug
    self.dummy_tensor = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    if deep_supervision:
      self.out_1 = OutBlock(in_channels=n_channels * 2, n_classes=n_classes, dim=dim)
      self.out_2 = OutBlock(in_channels=n_channels * 4, n_classes=n_classes, dim=dim)
      self.out_3 = OutBlock(in_channels=n_channels * 8, n_classes=n_classes, dim=dim)
      self.out_4 = OutBlock(in_channels=n_channels * 16, n_classes=n_classes, dim=dim)

    self.block_counts = block_counts

  def iterative_checkpoint(self, sequential_block, x):
    """
    This simply forwards x through each block of the sequential_block while
    using gradient_checkpointing. This implementation is designed to bypass
    the following issue in PyTorch's gradient checkpointing:
    https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
    """
    for l in sequential_block:
      x = checkpoint.checkpoint(l, x, self.dummy_tensor)
    return x

  def forward(self, x):
    x = self.stem(x)
    if self.outside_block_checkpointing:
      x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
      x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor)
      x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
      x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor)
      x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
      x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor)
      x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
      x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor)

      x = self.iterative_checkpoint(self.bottleneck, x)
      if self.do_ds:
        x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)

      x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
      dec_x = x_res_3 + x_up_3
      x = self.iterative_checkpoint(self.dec_block_3, dec_x)
      if self.do_ds:
        x_ds_3 = checkpoint.checkpoint(self.out_3, x, self.dummy_tensor)
      del x_res_3, x_up_3

      x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor)
      dec_x = x_res_2 + x_up_2
      x = self.iterative_checkpoint(self.dec_block_2, dec_x)
      if self.do_ds:
        x_ds_2 = checkpoint.checkpoint(self.out_2, x, self.dummy_tensor)
      del x_res_2, x_up_2

      x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor)
      dec_x = x_res_1 + x_up_1
      x = self.iterative_checkpoint(self.dec_block_1, dec_x)
      if self.do_ds:
        x_ds_1 = checkpoint.checkpoint(self.out_1, x, self.dummy_tensor)
      del x_res_1, x_up_1

      x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor)
      dec_x = x_res_0 + x_up_0
      x = self.iterative_checkpoint(self.dec_block_0, dec_x)
      del x_res_0, x_up_0, dec_x

      x = checkpoint.checkpoint(self.out_0, x, self.dummy_tensor)

    else:
      x_res_0 = self.enc_block_0(x)
      x = self.down_0(x_res_0)
      x_res_1 = self.enc_block_1(x)
      x = self.down_1(x_res_1)
      x_res_2 = self.enc_block_2(x)
      x = self.down_2(x_res_2)
      x_res_3 = self.enc_block_3(x)
      x = self.down_3(x_res_3)

      x = self.bottleneck(x)
      if self.do_ds:
        x_ds_4 = self.out_4(x)

      x_up_3 = self.up_3(x)
      dec_x = x_res_3 + x_up_3
      x = self.dec_block_3(dec_x)

      if self.do_ds:
        x_ds_3 = self.out_3(x)
      del x_res_3, x_up_3

      x_up_2 = self.up_2(x)
      dec_x = x_res_2 + x_up_2
      x = self.dec_block_2(dec_x)
      if self.do_ds:
        x_ds_2 = self.out_2(x)
      del x_res_2, x_up_2

      x_up_1 = self.up_1(x)
      dec_x = x_res_1 + x_up_1
      x = self.dec_block_1(dec_x)
      if self.do_ds:
        x_ds_1 = self.out_1(x)
      del x_res_1, x_up_1

      x_up_0 = self.up_0(x)
      dec_x = x_res_0 + x_up_0
      x = self.dec_block_0(dec_x)
      del x_res_0, x_up_0, dec_x

      x = self.out_0(x)

    if self.do_ds:
      return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
    else:
      return x


class TimeEmbedding(nn.Module):
  """Sinusoidal time embedding for diffusion models."""

  def __init__(self, embed_dim, scale=30.0):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :].to(x.device) * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Swish(nn.Module):
  """Swish activation function."""

  def forward(self, x):
    return x * torch.sigmoid(x)


class TimeEmbeddingProjection(nn.Module):
  """Projects time embeddings to match feature dimensions at each level."""

  def __init__(self, in_size, out_size):
    super().__init__()
    self.act = Swish()
    self.proj = nn.Linear(in_size, out_size)

  def forward(self, x):
    x = self.act(x)
    return self.proj(x)


class MedNeXtDiffusion(MedNeXt):
  """
  MedNeXt extended for diffusion models with time step injection and conditioning.

  This class extends MedNeXt to add:
  - Time step embedding injection at multiple encoder/decoder levels
  - Conditioning through structure inputs
  - Compatible forward signature with diffusion training

  Args:
      in_channels: Number of input channels (noisy dose)
      n_channels: Base number of feature channels
      n_classes: Number of output channels
      in_struc_channels: Number of structure conditioning channels (default: 5)
      exp_r: Expansion ratio(s) for MedNeXt blocks
      kernel_size: Kernel size for convolutions
      deep_supervision: Whether to use deep supervision
      do_res: Whether to use residual connections
      do_res_up_down: Whether to use residual connections in up/down blocks
      checkpoint_style: Gradient checkpointing style
      block_counts: Number of blocks at each level
      norm_type: Normalization type
      dim: '2d' or '3d'
      grn: Whether to use global response normalization
  """

  def __init__(
    self,
    in_channels: int,
    n_channels: int,
    n_classes: int,
    in_struc_channels: int = 5,
    exp_r: int = 4,
    kernel_size: int = 7,
    enc_kernel_size: int = None,
    dec_kernel_size: int = None,
    deep_supervision: bool = False,
    do_res: bool = False,
    do_res_up_down: bool = False,
    checkpoint_style: bool = None,
    block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],
    norm_type='group',
    dim='2d',
    grn=False,
  ):
    # Initialize parent MedNeXt with input channels for noisy dose
    super().__init__(
      in_channels=in_channels,
      n_channels=n_channels,
      n_classes=n_classes,
      exp_r=exp_r,
      kernel_size=kernel_size,
      enc_kernel_size=enc_kernel_size,
      dec_kernel_size=dec_kernel_size,
      deep_supervision=deep_supervision,
      do_res=do_res,
      do_res_up_down=do_res_up_down,
      checkpoint_style=checkpoint_style,
      block_counts=block_counts,
      norm_type=norm_type,
      dim=dim,
      grn=grn,
    )

    # Time embedding (use fixed dim to avoid issues with small n_channels)
    tdim = max(n_channels * 4, 128)
    self.time_embedding = TimeEmbedding(tdim)

    # Create time projections for encoder levels (0-3 and bottleneck)
    # Match the channel counts in MedNeXt encoder
    self.t_proj_enc_0 = TimeEmbeddingProjection(tdim, n_channels * 1)  # enc_block_0
    self.t_proj_enc_1 = TimeEmbeddingProjection(tdim, n_channels * 2)  # enc_block_1
    self.t_proj_enc_2 = TimeEmbeddingProjection(tdim, n_channels * 4)  # enc_block_2
    self.t_proj_enc_3 = TimeEmbeddingProjection(tdim, n_channels * 8)  # enc_block_3
    self.t_proj_bottleneck = TimeEmbeddingProjection(tdim, n_channels * 16)  # bottleneck

    # Create time projections for decoder levels (3-0)
    # Match the channel counts in MedNeXt decoder
    self.t_proj_dec_3 = TimeEmbeddingProjection(tdim, n_channels * 8)  # dec_block_3
    self.t_proj_dec_2 = TimeEmbeddingProjection(tdim, n_channels * 4)  # dec_block_2
    self.t_proj_dec_1 = TimeEmbeddingProjection(tdim, n_channels * 2)  # dec_block_1
    self.t_proj_dec_0 = TimeEmbeddingProjection(tdim, n_channels * 1)  # dec_block_0

    # Structure conditioning head
    conv = nn.Conv2d if dim == '2d' else nn.Conv3d
    self.head_struc = conv(in_struc_channels, in_channels, kernel_size=3, stride=1, padding=1)

  def forward(self, x, structure, t, cache: Optional[DeepCache] = None):
    """
    Forward pass with time step injection and conditioning.

    Args:
          x: Input noisy image [B, C, H, W] or [B, C, D, H, W]
          structure: Structure conditioning input [B, S, H, W] or [B, S, D, H, W]
          t: Time step tensor [B]
          cache: Optional DeepCache for intermediate features

    Returns:
          Predicted noise [B, C, H, W] or [B, C, D, H, W]
    """
    # Generate time embeddings
    t_emb = self.time_embedding(t)

    # Generate time projections for each level
    t_enc_0 = self.t_proj_enc_0(t_emb)[:, :, None, None]
    t_enc_1 = self.t_proj_enc_1(t_emb)[:, :, None, None]
    t_enc_2 = self.t_proj_enc_2(t_emb)[:, :, None, None]
    t_enc_3 = self.t_proj_enc_3(t_emb)[:, :, None, None]
    t_bottleneck = self.t_proj_bottleneck(t_emb)[:, :, None, None]

    t_dec_3 = self.t_proj_dec_3(t_emb)[:, :, None, None]
    t_dec_2 = self.t_proj_dec_2(t_emb)[:, :, None, None]
    t_dec_1 = self.t_proj_dec_1(t_emb)[:, :, None, None]
    t_dec_0 = self.t_proj_dec_0(t_emb)[:, :, None, None]

    # Add extra dimension for 3D
    if self.stem.weight.ndim == 5:  # 3D case
      t_enc_0 = t_enc_0.unsqueeze(-1)
      t_enc_1 = t_enc_1.unsqueeze(-1)
      t_enc_2 = t_enc_2.unsqueeze(-1)
      t_enc_3 = t_enc_3.unsqueeze(-1)
      t_bottleneck = t_bottleneck.unsqueeze(-1)
      t_dec_3 = t_dec_3.unsqueeze(-1)
      t_dec_2 = t_dec_2.unsqueeze(-1)
      t_dec_1 = t_dec_1.unsqueeze(-1)
      t_dec_0 = t_dec_0.unsqueeze(-1)

    # Process structure conditioning and add to input
    structure_feat = cache.get_or_compute('head_struc', lambda: self.head_struc(structure))
    x = x + structure_feat

    # Modified forward pass from MedNeXt with time injection
    x = self.stem(x)

    if self.outside_block_checkpointing:
      x_res_0 = cache.get_or_compute('enc_block_0', lambda: self.iterative_checkpoint(self.enc_block_0, x)) + t_enc_0  # Inject time
      x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor)

      x_res_1 = cache.get_or_compute('enc_block_1', lambda: self.iterative_checkpoint(self.enc_block_1, x)) + t_enc_1  # Inject time
      x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor)

      x_res_2 = cache.get_or_compute('enc_block_2', lambda: self.iterative_checkpoint(self.enc_block_2, x)) + t_enc_2  # Inject time
      x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor)

      x_res_3 = cache.get_or_compute('enc_block_3', lambda: self.iterative_checkpoint(self.enc_block_3, x)) + t_enc_3  # Inject time
      x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor)

      x = cache.get_or_compute('bottleneck', lambda: self.iterative_checkpoint(self.bottleneck, x)) + t_bottleneck  # Inject time
      if self.do_ds:
        x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)

      cache_key = 'dec_block_3'
      if cache.should_use(cache_key):
        x = cache.get(cache_key)
      else:
        x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
        dec_x = x_res_3 + x_up_3
        x = self.iterative_checkpoint(self.dec_block_3, dec_x)
        cache.set(cache_key, x)
      x = x + t_dec_3  # Inject time after cache

      if self.do_ds:
        x_ds_3 = checkpoint.checkpoint(self.out_3, x, self.dummy_tensor)
      del x_res_3

      cache_key = 'dec_block_2'
      if cache.should_use(cache_key):
        x = cache.get(cache_key)
      else:
        x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor)
        dec_x = x_res_2 + x_up_2
        x = self.iterative_checkpoint(self.dec_block_2, dec_x)
        cache.set(cache_key, x)
      x = x + t_dec_2  # Inject time after cache

      if self.do_ds:
        x_ds_2 = checkpoint.checkpoint(self.out_2, x, self.dummy_tensor)
      del x_res_2

      cache_key = 'dec_block_1'
      if cache.should_use(cache_key):
        x = cache.get(cache_key)
      else:
        x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor)
        dec_x = x_res_1 + x_up_1
        x = self.iterative_checkpoint(self.dec_block_1, dec_x)
        cache.set(cache_key, x)
      x = x + t_dec_1  # Inject time after cache

      if self.do_ds:
        x_ds_1 = checkpoint.checkpoint(self.out_1, x, self.dummy_tensor)
      del x_res_1

      cache_key = 'dec_block_0'
      if cache.should_use(cache_key):
        x = cache.get(cache_key)
      else:
        x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor)
        dec_x = x_res_0 + x_up_0
        x = self.iterative_checkpoint(self.dec_block_0, dec_x)
        cache.set(cache_key, x)
      x = x + t_dec_0  # Inject time after cache

      del x_res_0

      x = checkpoint.checkpoint(self.out_0, x, self.dummy_tensor)

    else:
      x_res_0 = cache.get_or_compute('enc_block_0', lambda: self.enc_block_0(x)) + t_enc_0  # Inject time
      x = self.down_0(x_res_0)

      x_res_1 = cache.get_or_compute('enc_block_1', lambda: self.enc_block_1(x)) + t_enc_1  # Inject time
      x = self.down_1(x_res_1)

      x_res_2 = cache.get_or_compute('enc_block_2', lambda: self.enc_block_2(x)) + t_enc_2  # Inject time
      x = self.down_2(x_res_2)

      x_res_3 = cache.get_or_compute('enc_block_3', lambda: self.enc_block_3(x)) + t_enc_3  # Inject time
      x = self.down_3(x_res_3)

      x = cache.get_or_compute('bottleneck', lambda: self.bottleneck(x)) + t_bottleneck  # Inject time
      if self.do_ds:
        x_ds_4 = self.out_4(x)

      cache_key = 'dec_block_3'
      if cache.should_use(cache_key):
        x = cache.get(cache_key)
      else:
        x_up_3 = self.up_3(x)
        dec_x = x_res_3 + x_up_3
        x = self.dec_block_3(dec_x)
        cache.set(cache_key, x)
      x = x + t_dec_3  # Inject time after cache

      if self.do_ds:
        x_ds_3 = self.out_3(x)
      del x_res_3

      cache_key = 'dec_block_2'
      if cache.should_use(cache_key):
        x = cache.get(cache_key)
      else:
        x_up_2 = self.up_2(x)
        dec_x = x_res_2 + x_up_2
        x = self.dec_block_2(dec_x)
        cache.set(cache_key, x)
      x = x + t_dec_2  # Inject time after cache

      if self.do_ds:
        x_ds_2 = self.out_2(x)
      del x_res_2

      cache_key = 'dec_block_1'
      if cache.should_use(cache_key):
        x = cache.get(cache_key)
      else:
        x_up_1 = self.up_1(x)
        dec_x = x_res_1 + x_up_1
        x = self.dec_block_1(dec_x)
        cache.set(cache_key, x)
      x = x + t_dec_1  # Inject time after cache

      if self.do_ds:
        x_ds_1 = self.out_1(x)
      del x_res_1

      cache_key = 'dec_block_0'
      if cache.should_use(cache_key):
        x = cache.get(cache_key)
      else:
        x_up_0 = self.up_0(x)
        dec_x = x_res_0 + x_up_0
        x = self.dec_block_0(dec_x)
        cache.set(cache_key, x)
      x = x + t_dec_0  # Inject time after cache

      del x_res_0

      x = self.out_0(x)

    if self.do_ds:
      return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
    else:
      return x


def build_mednext_model(model_config, in_channels, for_diffusion=False):
  """
  Build a MedNeXt model for either standard or diffusion use.

  Args:
      model_config: Configuration dictionary containing MedNeXt settings
      in_channels: Number of input channels
      for_diffusion: If True, returns MedNeXtDiffusion with time/condition support

  Returns:
      MedNeXt or MedNeXtDiffusion model instance
  """
  config = model_config['mednext']
  model_id = config.get('id', 'medium').lower()

  common_params = {
    'in_channels': 1 if for_diffusion else in_channels,
    'n_channels': 32,
    'n_classes': 1,
    'kernel_size': config['kernel_size'],
    'deep_supervision': config['deep_supervision'],
    'do_res': True,
    'do_res_up_down': True,
    'dim': '2d',
  }

  # Model-specific parameters
  if model_id == 'small':
    model_params = {
      **common_params,
      'exp_r': 2,
      'block_counts': [2, 2, 2, 2, 2, 2, 2, 2, 2],
    }
  elif model_id == 'base':
    model_params = {
      **common_params,
      'exp_r': [2, 3, 4, 4, 4, 4, 4, 3, 2],
      'block_counts': [2, 2, 2, 2, 2, 2, 2, 2, 2],
    }
  elif model_id == 'medium':
    model_params = {
      **common_params,
      'exp_r': [2, 3, 4, 4, 4, 4, 4, 3, 2],
      'block_counts': [3, 4, 4, 4, 4, 4, 4, 4, 3],
      'checkpoint_style': 'outside_block',
    }
  elif model_id == 'large':
    model_params = {
      **common_params,
      'exp_r': [3, 4, 8, 8, 8, 8, 8, 4, 3],
      'block_counts': [3, 4, 8, 8, 8, 8, 8, 4, 3],
      'checkpoint_style': 'outside_block',
    }
  else:
    raise ValueError(f"Unknown MedNeXt model id: {model_id}")

  if for_diffusion:
    return MedNeXtDiffusion(
      in_struc_channels=in_channels,
      **model_params
    )
  else:
    return MedNeXt(**model_params)
