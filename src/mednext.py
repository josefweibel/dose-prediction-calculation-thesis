"""
MedNeXt: Modern Efficient Deep Network for Medical Image Segmentation

This file contains:
1. Original MedNeXt components (LayerNorm, MedNeXtBlock, MedNeXtDownBlock,
   MedNeXtUpBlock, OutBlock, MedNeXt)
2. Diffusion model extensions (marked with "ADDED/MODIFIED FOR DIFFUSION")

CHANGES FOR DIFFUSION MODEL SUPPORT:
================================================================================
The following modifications enable MedNeXt to be used as a backbone for
diffusion-based dose prediction models:

1. NEW CLASS: SinusoidalPositionEmbedding
   - Encodes diffusion timesteps into continuous representations
   - Located at the top of the file after imports

2. MODIFIED CLASS: MedNeXtBlock
   - Added time_emb_dim parameter to __init__
   - Added time_emb parameter to forward()
   - Added timestep conditioning via Adaptive Group Normalization (AdaGN)
   - Applies learned scale and shift based on timestep embeddings

3. MODIFIED CLASS: MedNeXtDownBlock
   - Added time_emb_dim parameter and passes it to parent MedNeXtBlock
   - Added time_emb parameter to forward() and passes it to parent

4. MODIFIED CLASS: MedNeXtUpBlock
   - Added time_emb_dim parameter and passes it to parent MedNeXtBlock
   - Added time_emb parameter to forward() and passes it to parent

5. MODIFIED CLASS: MedNeXt
   - Added time_emb_dim parameter (stored but not used in base class)
   - Maintains backward compatibility with non-diffusion usage

6. NEW CLASS: MedNeXtDiffusion
   - Complete diffusion-compatible wrapper
   - 5-level encoder-decoder U-Net with MedNeXt blocks
   - Timestep conditioning throughout the network
   - Compatible interface: forward(x_noisy, condition, timesteps, cache)
   - Concatenation-based conditioning (noisy dose + structural info)

7. MODIFIED FUNCTION: build_mednext_model(model_config, structure_selector, for_diffusion)
   - Unified builder function for both direct prediction and diffusion models
   - Supports model size variants: 'small', 'base', 'medium', 'large'
   - Uses structure_selector for both input channels and conditioning
   - Compatible with diffusion training framework
   - Similar pattern to build_beamdiff and build_spdiffdose

USAGE:
================================================================================
from mednext import build_mednext_model

# For diffusion models
model_config = {
  'mednext': {
    'id': 'medium',  # or 'small', 'base', 'large'
    'kernel_size': 7,
    'deep_supervision': False,
    'dim': '2d',
    'grn': False
  }
}

model = build_mednext_model(model_config, structure_selector, for_diffusion=True)
noise_pred = model(x_noisy, condition, timesteps)

# For direct prediction models
model = build_mednext_model(model_config, structure_selector, for_diffusion=False)
dose_pred = model(structural_info)

# Override specific parameters
model_config = {
  'mednext': {
    'id': 'medium',
    'base_channels': 64,  # Override default
    'kernel_size': 5,
    'exp_r': [2,3,4,4,4,4,4,3,2],  # Custom expansion ratios
    'block_counts': [3,4,4,4,4,4,4,4,3],  # Custom block counts
  }
}

model = build_mednext_model(model_config, structure_selector, for_diffusion=True)

MODEL SIZES:
- small: 32 channels, exp_r=2, blocks=[2,2,2,2,2,2,2,2,2]
- base: 32 channels, exp_r=[2,3,4,4,4,4,4,3,2], blocks=[2,2,2,2,2,2,2,2,2]
- medium: 32 channels, exp_r=[2,3,4,4,4,4,4,3,2], blocks=[3,4,4,4,4,4,4,4,3]
- large: 32 channels, exp_r=[3,4,8,8,8,8,8,4,3], blocks=[3,4,8,8,8,8,8,4,3]

All modifications are clearly marked in the code with comments like:
  - "ADDED FOR DIFFUSION: ..."
  - "MODIFIED FOR DIFFUSION: ..."
  - "NEW CLASS FOR DIFFUSION: ..."
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math


# ============================================================================
# DIFFUSION MODEL SUPPORT - NEW ADDITION
# ============================================================================

class SinusoidalPositionEmbedding(nn.Module):
  """
  Sinusoidal position encoding for timestep embedding.

  ADDED FOR DIFFUSION: Encodes diffusion timesteps into continuous representations.
  """

  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, timesteps):
    device = timesteps.device
    half_dim = self.dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
    embeddings = timesteps[:, None] * embeddings[None, :]
    embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    return embeddings

# ============================================================================
# END DIFFUSION MODEL SUPPORT
# ============================================================================


class LayerNorm(nn.Module):
  """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
  The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
  shape (batch_size, height, width, channels) while channels_first corresponds to inputs
  with shape (batch_size, channels, height, width).
  """
  def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(normalized_shape))        # beta
    self.bias = nn.Parameter(torch.zeros(normalized_shape))         # gamma
    self.eps = eps
    self.data_format = data_format
    if self.data_format not in ["channels_last", "channels_first"]:
      raise NotImplementedError
    self.normalized_shape = (normalized_shape, )

  def forward(self, x, dummy_tensor=False):
    if self.data_format == "channels_last":
      return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    elif self.data_format == "channels_first":
      u = x.mean(1, keepdim=True)
      s = (x - u).pow(2).mean(1, keepdim=True)
      x = (x - u) / torch.sqrt(s + self.eps)
      x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
      return x


class MedNeXtBlock(nn.Module):

  def __init__(self,
              in_channels:int,
              out_channels:int,
              exp_r:int=4,
              kernel_size:int=7,
              do_res:int=True,
              norm_type:str = 'group',
              n_groups:int or None = None,
              dim = '3d',
              grn = False,
              time_emb_dim = None  # MODIFIED FOR DIFFUSION: Added timestep embedding support
              ):

    super().__init__()

    self.do_res = do_res

    assert dim in ['2d', '3d']
    self.dim = dim
    if self.dim == '2d':
      conv = nn.Conv2d
    elif self.dim == '3d':
      conv = nn.Conv3d

    # First convolution layer with DepthWise Convolutions
    self.conv1 = conv(
      in_channels = in_channels,
      out_channels = in_channels,
      kernel_size = kernel_size,
      stride = 1,
      padding = kernel_size//2,
      groups = in_channels if n_groups is None else n_groups,
    )

    # Normalization Layer. GroupNorm is used by default.
    if norm_type=='group':
      self.norm = nn.GroupNorm(
        num_groups=in_channels,
        num_channels=in_channels
        )
    elif norm_type=='layer':
      self.norm = LayerNorm(
        normalized_shape=in_channels,
        data_format='channels_first'
        )

    # =========================================================================
    # ADDED FOR DIFFUSION: Timestep embedding projection
    # Projects timestep embeddings to scale and shift parameters for AdaGN
    # (Adaptive Group Normalization), similar to BeamDiff and SP-DiffDose
    # =========================================================================
    if time_emb_dim is not None:
      self.time_emb_proj = nn.Linear(time_emb_dim, in_channels * 2)
    else:
      self.time_emb_proj = None
    # =========================================================================

    # Second convolution (Expansion) layer with Conv3D 1x1x1
    self.conv2 = conv(
      in_channels = in_channels,
      out_channels = exp_r*in_channels,
      kernel_size = 1,
      stride = 1,
      padding = 0
    )

    # GeLU activations
    self.act = nn.GELU()

    # Third convolution (Compression) layer with Conv3D 1x1x1
    self.conv3 = conv(
      in_channels = exp_r*in_channels,
      out_channels = out_channels,
      kernel_size = 1,
      stride = 1,
      padding = 0
    )

    self.grn = grn
    if grn:
      if dim == '3d':
        self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1,1), requires_grad=True)
        self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1,1), requires_grad=True)
      elif dim == '2d':
        self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)
        self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)


  def forward(self, x, dummy_tensor=None, time_emb=None):  # MODIFIED FOR DIFFUSION: Added time_emb parameter

    x1 = x
    x1 = self.conv1(x1)
    x1 = self.norm(x1)

    # =========================================================================
    # ADDED FOR DIFFUSION: Apply timestep conditioning (adaptive scale and shift)
    # This implements Adaptive Group Normalization (AdaGN):
    #   norm(x) * (1 + scale(t)) + shift(t)
    # where scale and shift are learned functions of the timestep embedding
    # =========================================================================
    if time_emb is not None and self.time_emb_proj is not None:
      time_emb_proj = self.time_emb_proj(time_emb)
      while len(time_emb_proj.shape) < len(x1.shape):
        time_emb_proj = time_emb_proj[..., None]
      scale, shift = time_emb_proj.chunk(2, dim=1)
      x1 = x1 * (1 + scale) + shift
    # =========================================================================

    x1 = self.act(self.conv2(x1))
    if self.grn:
      # gamma, beta: learnable affine transform parameters
      # X: input of shape (N,C,H,W,D)
      if self.dim == '3d':
        gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
      elif self.dim == '2d':
        gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
      nx = gx / (gx.mean(dim=1, keepdim=True)+1e-6)
      x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
    x1 = self.conv3(x1)
    if self.do_res:
      x1 = x + x1
    return x1


class MedNeXtDownBlock(MedNeXtBlock):

  def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
              do_res=False, norm_type = 'group', dim='3d', grn=False,
              time_emb_dim=None):  # MODIFIED FOR DIFFUSION: Added time_emb_dim parameter

    super().__init__(in_channels, out_channels, exp_r, kernel_size,
                    do_res = False, norm_type = norm_type, dim=dim,
                    grn=grn, time_emb_dim=time_emb_dim)  # MODIFIED FOR DIFFUSION: Pass through time_emb_dim

    if dim == '2d':
      conv = nn.Conv2d
    elif dim == '3d':
      conv = nn.Conv3d
    self.resample_do_res = do_res
    if do_res:
      self.res_conv = conv(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = 1,
        stride = 2
      )

    self.conv1 = conv(
      in_channels = in_channels,
      out_channels = in_channels,
      kernel_size = kernel_size,
      stride = 2,
      padding = kernel_size//2,
      groups = in_channels,
    )

  def forward(self, x, dummy_tensor=None, time_emb=None):  # MODIFIED FOR DIFFUSION: Added time_emb parameter

    x1 = super().forward(x, dummy_tensor, time_emb)  # MODIFIED FOR DIFFUSION: Pass through time_emb

    if self.resample_do_res:
      res = self.res_conv(x)
      x1 = x1 + res

    return x1


class MedNeXtUpBlock(MedNeXtBlock):

  def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
              do_res=False, norm_type = 'group', dim='3d', grn = False,
              time_emb_dim=None):  # MODIFIED FOR DIFFUSION: Added time_emb_dim parameter
    super().__init__(in_channels, out_channels, exp_r, kernel_size,
                     do_res=False, norm_type = norm_type, dim=dim,
                     grn=grn, time_emb_dim=time_emb_dim)  # MODIFIED FOR DIFFUSION: Pass through time_emb_dim

    self.resample_do_res = do_res

    self.dim = dim
    if dim == '2d':
      conv = nn.ConvTranspose2d
    elif dim == '3d':
      conv = nn.ConvTranspose3d
    if do_res:
      self.res_conv = conv(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = 1,
        stride = 2
        )

    self.conv1 = conv(
      in_channels = in_channels,
      out_channels = in_channels,
      kernel_size = kernel_size,
      stride = 2,
      padding = kernel_size//2,
      groups = in_channels,
    )


  def forward(self, x, dummy_tensor=None, time_emb=None):  # MODIFIED FOR DIFFUSION: Added time_emb parameter

    x1 = super().forward(x, dummy_tensor, time_emb)  # MODIFIED FOR DIFFUSION: Pass through time_emb
    # Asymmetry but necessary to match shape

    if self.dim == '2d':
      x1 = torch.nn.functional.pad(x1, (1,0,1,0))
    elif self.dim == '3d':
      x1 = torch.nn.functional.pad(x1, (1,0,1,0,1,0))

    if self.resample_do_res:
      res = self.res_conv(x)
      if self.dim == '2d':
        res = torch.nn.functional.pad(res, (1,0,1,0))
      elif self.dim == '3d':
        res = torch.nn.functional.pad(res, (1,0,1,0,1,0))
      x1 = x1 + res

    return x1


class OutBlock(nn.Module):

  def __init__(self, in_channels, n_classes, dim):
    super().__init__()

    if dim == '2d':
      conv = nn.ConvTranspose2d
    elif dim == '3d':
      conv = nn.ConvTranspose3d
    self.conv_out = conv(in_channels, n_classes, kernel_size=1)

  def forward(self, x, dummy_tensor=None):
    return self.conv_out(x)


class MedNeXt(nn.Module):

  def __init__(self,
    in_channels: int,
    n_channels: int,
    n_classes: int,
    exp_r: int = 4,                            # Expansion ratio as in Swin Transformers
    kernel_size: int = 7,                      # Ofcourse can test kernel_size
    enc_kernel_size: int = None,
    dec_kernel_size: int = None,
    deep_supervision: bool = False,             # Can be used to test deep supervision
    do_res: bool = False,                       # Can be used to individually test residual connection
    do_res_up_down: bool = False,             # Additional 'res' connection on up and down convs
    checkpoint_style: bool = None,            # Either inside block or outside block
    block_counts: list = [2,2,2,2,2,2,2,2,2], # Can be used to test staging ratio:
                                        # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
    norm_type = 'group',
    dim = '3d',                                # 2d or 3d
    grn = False,
    time_emb_dim = None                        # MODIFIED FOR DIFFUSION: Timestep embedding dim for diffusion models
  ):

    super().__init__()

    self.do_ds = deep_supervision
    self.time_emb_dim = time_emb_dim  # MODIFIED FOR DIFFUSION: Store time_emb_dim (not used in base MedNeXt)
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

    self.enc_block_0 = nn.Sequential(*[
      MedNeXtBlock(
        in_channels=n_channels,
        out_channels=n_channels,
        exp_r=exp_r[0],
        kernel_size=enc_kernel_size,
        do_res=do_res,
        norm_type=norm_type,
        dim=dim,
        grn=grn
        )
      for i in range(block_counts[0])]
    )

    self.down_0 = MedNeXtDownBlock(
      in_channels=n_channels,
      out_channels=2*n_channels,
      exp_r=exp_r[1],
      kernel_size=enc_kernel_size,
      do_res=do_res_up_down,
      norm_type=norm_type,
      dim=dim,
      grn=grn,
    )

    self.enc_block_1 = nn.Sequential(*[
      MedNeXtBlock(
        in_channels=n_channels*2,
        out_channels=n_channels*2,
        exp_r=exp_r[1],
        kernel_size=enc_kernel_size,
        do_res=do_res,
        norm_type=norm_type,
        dim=dim,
        grn=grn
        )
      for i in range(block_counts[1])]
    )

    self.down_1 = MedNeXtDownBlock(
      in_channels=2*n_channels,
      out_channels=4*n_channels,
      exp_r=exp_r[2],
      kernel_size=enc_kernel_size,
      do_res=do_res_up_down,
      norm_type=norm_type,
      dim=dim,
      grn=grn
    )

    self.enc_block_2 = nn.Sequential(*[
      MedNeXtBlock(
        in_channels=n_channels*4,
        out_channels=n_channels*4,
        exp_r=exp_r[2],
        kernel_size=enc_kernel_size,
        do_res=do_res,
        norm_type=norm_type,
        dim=dim,
        grn=grn
        )
      for i in range(block_counts[2])]
    )

    self.down_2 = MedNeXtDownBlock(
      in_channels=4*n_channels,
      out_channels=8*n_channels,
      exp_r=exp_r[3],
      kernel_size=enc_kernel_size,
      do_res=do_res_up_down,
      norm_type=norm_type,
      dim=dim,
      grn=grn
    )

    self.enc_block_3 = nn.Sequential(*[
      MedNeXtBlock(
        in_channels=n_channels*8,
        out_channels=n_channels*8,
        exp_r=exp_r[3],
        kernel_size=enc_kernel_size,
        do_res=do_res,
        norm_type=norm_type,
        dim=dim,
        grn=grn
        )
      for i in range(block_counts[3])]
    )

    self.down_3 = MedNeXtDownBlock(
      in_channels=8*n_channels,
      out_channels=16*n_channels,
      exp_r=exp_r[4],
      kernel_size=enc_kernel_size,
      do_res=do_res_up_down,
      norm_type=norm_type,
      dim=dim,
      grn=grn
    )

    self.bottleneck = nn.Sequential(*[
      MedNeXtBlock(
        in_channels=n_channels*16,
        out_channels=n_channels*16,
        exp_r=exp_r[4],
        kernel_size=dec_kernel_size,
        do_res=do_res,
        norm_type=norm_type,
        dim=dim,
        grn=grn
        )
      for i in range(block_counts[4])]
    )

    self.up_3 = MedNeXtUpBlock(
      in_channels=16*n_channels,
      out_channels=8*n_channels,
      exp_r=exp_r[5],
      kernel_size=dec_kernel_size,
      do_res=do_res_up_down,
      norm_type=norm_type,
      dim=dim,
      grn=grn
    )

    self.dec_block_3 = nn.Sequential(*[
      MedNeXtBlock(
        in_channels=n_channels*8,
        out_channels=n_channels*8,
        exp_r=exp_r[5],
        kernel_size=dec_kernel_size,
        do_res=do_res,
        norm_type=norm_type,
        dim=dim,
        grn=grn
        )
      for i in range(block_counts[5])]
    )

    self.up_2 = MedNeXtUpBlock(
      in_channels=8*n_channels,
      out_channels=4*n_channels,
      exp_r=exp_r[6],
      kernel_size=dec_kernel_size,
      do_res=do_res_up_down,
      norm_type=norm_type,
      dim=dim,
      grn=grn
    )

    self.dec_block_2 = nn.Sequential(*[
      MedNeXtBlock(
        in_channels=n_channels*4,
        out_channels=n_channels*4,
        exp_r=exp_r[6],
        kernel_size=dec_kernel_size,
        do_res=do_res,
        norm_type=norm_type,
        dim=dim,
        grn=grn
        )
      for i in range(block_counts[6])]
    )

    self.up_1 = MedNeXtUpBlock(
      in_channels=4*n_channels,
      out_channels=2*n_channels,
      exp_r=exp_r[7],
      kernel_size=dec_kernel_size,
      do_res=do_res_up_down,
      norm_type=norm_type,
      dim=dim,
      grn=grn
    )

    self.dec_block_1 = nn.Sequential(*[
      MedNeXtBlock(
        in_channels=n_channels*2,
        out_channels=n_channels*2,
        exp_r=exp_r[7],
        kernel_size=dec_kernel_size,
        do_res=do_res,
        norm_type=norm_type,
        dim=dim,
        grn=grn
        )
      for i in range(block_counts[7])]
    )

    self.up_0 = MedNeXtUpBlock(
      in_channels=2*n_channels,
      out_channels=n_channels,
      exp_r=exp_r[8],
      kernel_size=dec_kernel_size,
      do_res=do_res_up_down,
      norm_type=norm_type,
      dim=dim,
      grn=grn
    )

    self.dec_block_0 = nn.Sequential(*[
      MedNeXtBlock(
        in_channels=n_channels,
        out_channels=n_channels,
        exp_r=exp_r[8],
        kernel_size=dec_kernel_size,
        do_res=do_res,
        norm_type=norm_type,
        dim=dim,
        grn=grn
        )
      for i in range(block_counts[8])]
    )

    self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)

    # Used to fix PyTorch checkpointing bug
    self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

    if deep_supervision:
      self.out_1 = OutBlock(in_channels=n_channels*2, n_classes=n_classes, dim=dim)
      self.out_2 = OutBlock(in_channels=n_channels*4, n_classes=n_classes, dim=dim)
      self.out_3 = OutBlock(in_channels=n_channels*8, n_classes=n_classes, dim=dim)
      self.out_4 = OutBlock(in_channels=n_channels*16, n_classes=n_classes, dim=dim)

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


# ============================================================================
# DIFFUSION MODEL WRAPPER - NEW ADDITION
# ============================================================================

class MedNeXtDiffusion(nn.Module):
  """
  MedNeXt-based Diffusion Model Backend.

  NEW CLASS FOR DIFFUSION: This is a complete diffusion-compatible wrapper
  that uses MedNeXt blocks in a U-Net architecture with timestep conditioning.

  Uses MedNeXt blocks (depthwise convolutions + expansion/compression) in a U-Net
  architecture with timestep conditioning for diffusion-based dose prediction.

  Key Features:
  - MedNeXt blocks with adaptive timestep conditioning (scale and shift)
  - 5-level encoder-decoder architecture with skip connections
  - Sinusoidal timestep embeddings
  - Concatenation-based conditioning (noisy dose + structural info)
  - Compatible with standard diffusion training frameworks

  Architecture:
  - Input: Noisy dose (1 ch) + Condition (N ch) → concatenated
  - Encoder: 5 levels with MedNeXt blocks + downsampling
  - Bottleneck: MedNeXt blocks at lowest resolution
  - Decoder: 5 levels with MedNeXt blocks + upsampling + skip connections
  - Output: Predicted noise (1 ch)

  Interface (compatible with DiffusionModelTrainerModule):
    forward(x_noisy, condition, timesteps, cache=None)
      Args:
        - x_noisy: Noisy dose map [B, 1, H, W]
        - condition: Structural info (CT + PTV + OARs, etc.) [B, C, H, W]
        - timesteps: Diffusion timesteps [B]
        - cache: DeepCache (unused, for compatibility)
      Returns:
        - Predicted noise [B, 1, H, W]

  Usage:
    from mednext import build_mednext

    config = {
      'base_channels': 32,
      'kernel_size': 7,
      'block_counts': [3,4,4,4,4,4,4,4,3],
      'exp_r': [2,3,4,4,4,4,4,3,2],
      'do_res': True,
      'do_res_up_down': True,
      'grn': False,
      'dim': '2d'
    }

    model = build_mednext(config, structure_selector)
  """

  def __init__(
    self,
    structure_selector,
    base_channels=32,
    exp_r=[2,3,4,4,4,4,4,3,2],
    kernel_size=7,
    deep_supervision=False,  # Not used yet, reserved for future
    do_res=True,
    do_res_up_down=True,
    block_counts=[3,4,4,4,4,4,4,4,3],
    checkpoint_style='outside_block',  # Not used yet, reserved for future
    dim='2d',
    grn=False
  ):
    super().__init__()

    self.structure_selector = structure_selector
    in_channels_cond = structure_selector.total_channels

    # Timestep embedding
    time_embed_dim = base_channels * 4
    self.time_embed = nn.Sequential(
      SinusoidalPositionEmbedding(base_channels),
      nn.Linear(base_channels, time_embed_dim),
      nn.SiLU(),
      nn.Linear(time_embed_dim, time_embed_dim)
    )

    # Combined input: noisy dose (1 channel) + condition (in_channels_cond)
    in_channels_total = 1 + in_channels_cond

    # For now, we'll create a simpler time-conditioned U-Net using MedNeXt blocks
    # This avoids modifying all the existing block instantiations in MedNeXt class

    if dim == '2d':
      conv = nn.Conv2d
    elif dim == '3d':
      conv = nn.Conv3d

    # Initial conv
    self.conv_in = conv(in_channels_total, base_channels, kernel_size=3, padding=1)

    if type(exp_r) == int:
      exp_r = [exp_r for i in range(len(block_counts))]

    # Build encoder
    self.encoder_blocks = nn.ModuleList()
    self.down_blocks = nn.ModuleList()

    channels = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16]

    for level in range(5):
      # All blocks at this level work with out_ch
      out_ch = channels[level]

      # Encoder blocks at this level
      num_blocks = block_counts[level] if level < len(block_counts) else block_counts[-1]

      blocks = nn.ModuleList([
        MedNeXtBlock(
          in_channels=out_ch,
          out_channels=out_ch,
          exp_r=exp_r[level] if level < len(exp_r) else exp_r[-1],
          kernel_size=kernel_size,
          do_res=do_res,
          norm_type='group',
          dim=dim,
          grn=grn,
          time_emb_dim=time_embed_dim
        )
        for _ in range(num_blocks)
      ])
      self.encoder_blocks.append(blocks)

      # Downsample (except last level) - changes both spatial and channel dimensions
      if level < 4:
        next_ch = channels[level + 1]
        self.down_blocks.append(MedNeXtDownBlock(
          in_channels=out_ch,
          out_channels=next_ch,
          exp_r=exp_r[level+1] if level+1 < len(exp_r) else exp_r[-1],
          kernel_size=kernel_size,
          do_res=do_res_up_down,
          norm_type='group',
          dim=dim,
          grn=grn,
          time_emb_dim=time_embed_dim
        ))
      else:
        self.down_blocks.append(nn.Identity())

    # Bottleneck
    self.bottleneck = nn.ModuleList([
      MedNeXtBlock(
        in_channels=channels[4],
        out_channels=channels[4],
        exp_r=exp_r[4] if 4 < len(exp_r) else exp_r[-1],
        kernel_size=kernel_size,
        do_res=do_res,
        norm_type='group',
        dim=dim,
        grn=grn,
        time_emb_dim=time_embed_dim
      )
      for _ in range(block_counts[4] if 4 < len(block_counts) else block_counts[-1])
    ])

    # Build decoder
    self.decoder_blocks = nn.ModuleList()
    self.up_blocks = nn.ModuleList()

    for level in range(4, -1, -1):
      cur_ch = channels[level]

      # Decoder blocks at this level (process at current resolution)
      num_blocks = block_counts[level+5] if level+5 < len(block_counts) else block_counts[-1]
      blocks = nn.ModuleList([
        MedNeXtBlock(
          in_channels=cur_ch,
          out_channels=cur_ch,
          exp_r=exp_r[level+4] if level+4 < len(exp_r) else exp_r[-1],
          kernel_size=kernel_size,
          do_res=do_res,
          norm_type='group',
          dim=dim,
          grn=grn,
          time_emb_dim=time_embed_dim
        )
        for _ in range(num_blocks)
      ])
      self.decoder_blocks.append(blocks)

      # Upsample (except last level) - reduces channels for next level
      if level > 0:
        next_ch = channels[level - 1]
        self.up_blocks.append(MedNeXtUpBlock(
          in_channels=cur_ch,
          out_channels=next_ch,
          exp_r=exp_r[level+4] if level+4 < len(exp_r) else exp_r[-1],
          kernel_size=kernel_size,
          do_res=do_res_up_down,
          norm_type='group',
          dim=dim,
          grn=grn,
          time_emb_dim=time_embed_dim
        ))
      else:
        self.up_blocks.append(nn.Identity())

    # Output
    self.conv_out = conv(base_channels, 1, kernel_size=1)

  def forward(self, x_noisy, condition, timesteps, cache=None):  # pylint: disable=unused-argument
    """
    Forward pass compatible with DiffusionModelTrainerModule.

    Args:
        x_noisy: Noisy dose map [B, 1, H, W]
        condition: Structural information [B, C, H, W]
        timesteps: Diffusion timesteps [B]
        cache: DeepCache (unused, for compatibility)

    Returns:
        Predicted noise [B, 1, H, W]
    """
    # Timestep embedding
    t_emb = self.time_embed(timesteps)

    # Concatenate noisy dose and condition
    x = torch.cat([x_noisy, condition], dim=1)

    # Initial conv
    x = self.conv_in(x)

    # Encoder
    skip_connections = []
    for level in range(5):
      # Apply blocks
      for block in self.encoder_blocks[level]:
        x = block(x, time_emb=t_emb)

      # Store for skip connection
      skip_connections.append(x)

      # Downsample
      if level < 4:
        x = self.down_blocks[level](x, time_emb=t_emb)

    # Bottleneck
    for block in self.bottleneck:
      x = block(x, time_emb=t_emb)

    # Decoder
    for level in range(5):
      # Skip connection
      skip = skip_connections[4 - level]
      x = x + skip

      # Apply blocks
      for block in self.decoder_blocks[level]:
        x = block(x, time_emb=t_emb)

      # Upsample
      if level < 4:
        x = self.up_blocks[level](x, time_emb=t_emb)

    # Output
    x = self.conv_out(x)

    return x


# ============================================================================
# END DIFFUSION MODEL WRAPPER
# ============================================================================


def build_mednext_model(model_config, structure_selector, out_channels=1):
  """
  Build MedNeXt model from configuration.

  MODIFIED: Unified builder function that supports both direct prediction (MedNeXt)
  and diffusion models (MedNeXtDiffusion) with model size variants (small, base, medium, large).

  Args:
      model_config: Full model configuration dict
      structure_selector: StructureSelector instance (provides input channels and conditioning)
      out_channels: Number of output channels (default: 1, only 1 supported for diffusion)

  Returns:
      MedNeXt or MedNeXtDiffusion model
  """
  config = model_config['mednext']
  model_id = config.get('id', 'medium').lower()
  kernel_size = config.get('kernel_size', 7)
  deep_supervision = config.get('deep_supervision', False)
  dim = config.get('dim', '2d')
  grn = config.get('grn', False)

  # Define model size configurations
  # Format: (base_channels, exp_r, block_counts)
  size_configs = {
    'small': (32, 2, [2,2,2,2,2,2,2,2,2]),
    'base': (32, [2,3,4,4,4,4,4,3,2], [2,2,2,2,2,2,2,2,2]),
    'medium': (32, [2,3,4,4,4,4,4,3,2], [3,4,4,4,4,4,4,4,3]),
    'large': (32, [3,4,8,8,8,8,8,4,3], [3,4,8,8,8,8,8,4,3]),
  }

  if model_id not in size_configs:
    raise ValueError(f"Unknown model_id: {model_id}. Choose from: {list(size_configs.keys())}")

  base_channels, exp_r, block_counts = size_configs[model_id]

  # Allow config overrides
  base_channels = config.get('base_channels', base_channels)
  exp_r = config.get('exp_r', exp_r)
  block_counts = config.get('block_counts', block_counts)

  # Get input channels from structure_selector
  in_channels = structure_selector.total_channels

  # Build diffusion model
  if 'diffusion' in model_config:
    if out_channels != 1:
      raise ValueError("MedNeXtDiffusion only supports out_channels=1 for noise prediction")

    model = MedNeXtDiffusion(
      structure_selector=structure_selector,
      base_channels=base_channels,
      exp_r=exp_r,
      kernel_size=kernel_size,
      deep_supervision=deep_supervision,
      do_res=config.get('do_res', True),
      do_res_up_down=config.get('do_res_up_down', True),
      block_counts=block_counts,
      checkpoint_style=config.get('checkpoint_style', 'outside_block'),
      dim=dim,
      grn=grn
    )
    return model

  # Build direct prediction model
  checkpoint_style = 'outside_block' if model_id in ['medium', 'large'] else None

  return MedNeXt(
    in_channels=in_channels,
    n_channels=base_channels,
    n_classes=out_channels,
    exp_r=exp_r,
    kernel_size=kernel_size,
    deep_supervision=deep_supervision,
    do_res=True,
    do_res_up_down=True,
    block_counts=block_counts,
    checkpoint_style=checkpoint_style,
    dim=dim,
    grn=grn
  )
