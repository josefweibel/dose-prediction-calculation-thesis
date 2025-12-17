"""
Fixed MedNeXt blocks that avoid checkerboard artifacts from transposed convolutions.

The fix replaces ConvTranspose with bilinear upsampling + regular convolution.
This is a well-known solution to checkerboard artifacts in generative models.

Reference: https://distill.pub/2016/deconv-checkerboard/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedNeXtUpBlock(nn.Module):
  """
  Fixed MedNeXtUpBlock that avoids checkerboard artifacts by replacing
  transposed convolution with bilinear upsample + regular convolution.

  This implementation replicates the original MedNeXtUpBlock behavior but uses
  interpolation + regular conv instead of transposed conv to avoid artifacts.
  """

  def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
               do_res=False, norm_type='group', dim='2d', grn=False):
    super().__init__()

    self.do_res = False  # Internal residual in MedNeXt block pattern
    self.resample_do_res = do_res  # Residual for the upsampling operation
    self.dim = dim
    self.grn = grn

    assert dim in ['2d', '3d']

    if dim == '2d':
      conv = nn.Conv2d
    elif dim == '3d':
      conv = nn.Conv3d

    # First convolution layer with DepthWise Convolutions (regular stride=1)
    self.conv1 = conv(
      in_channels=in_channels,
      out_channels=in_channels,
      kernel_size=kernel_size,
      stride=1,  # Regular conv, not transposed
      padding=kernel_size//2,
      groups=in_channels,
    )

    # Normalization Layer. GroupNorm is used by default.
    if norm_type == 'group':
      self.norm = nn.GroupNorm(
        num_groups=in_channels,
        num_channels=in_channels
      )
    elif norm_type == 'layer':
      # Assuming LayerNorm exists in the codebase
      from nnunet_mednext.network_architecture.mednextv1.blocks import LayerNorm
      self.norm = LayerNorm(
        normalized_shape=in_channels,
        data_format='channels_first'
      )

    # Second convolution (Expansion) layer
    self.conv2 = conv(
      in_channels=in_channels,
      out_channels=exp_r * in_channels,
      kernel_size=1,
      stride=1,
      padding=0
    )

    # GeLU activation
    self.act = nn.GELU()

    # Third convolution (Compression) layer
    self.conv3 = conv(
      in_channels=exp_r * in_channels,
      out_channels=out_channels,
      kernel_size=1,
      stride=1,
      padding=0
    )

    # GRN parameters if enabled
    if grn:
      if dim == '3d':
        self.grn_beta = nn.Parameter(torch.zeros(1, exp_r*in_channels, 1, 1, 1), requires_grad=True)
        self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r*in_channels, 1, 1, 1), requires_grad=True)
      elif dim == '2d':
        self.grn_beta = nn.Parameter(torch.zeros(1, exp_r*in_channels, 1, 1), requires_grad=True)
        self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r*in_channels, 1, 1), requires_grad=True)

    # Optional residual connection for upsampling
    if do_res:
      self.res_conv = conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1  # Regular conv, not transposed
      )

  def forward(self, x, dummy_tensor=None):
    # Store original for potential residual connection
    x_orig = x

    # Upsample FIRST using bilinear/trilinear interpolation
    if self.dim == '2d':
      x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    elif self.dim == '3d':
      x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)

    # Apply MedNeXt block operations (from MedNeXtBlock.forward)
    x1 = x
    x1 = self.conv1(x1)
    x1 = self.act(self.conv2(self.norm(x1)))

    if self.grn:
      # Global Response Normalization
      if self.dim == '3d':
        gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
      elif self.dim == '2d':
        gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
      nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
      x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1

    x1 = self.conv3(x1)

    # Add internal residual if enabled (typically False for up/down blocks)
    if self.do_res:
      x1 = x + x1

    # Add upsampling residual connection if enabled
    if self.resample_do_res:
      if self.dim == '2d':
        res = F.interpolate(x_orig, scale_factor=2, mode='bilinear', align_corners=False)
      elif self.dim == '3d':
        res = F.interpolate(x_orig, scale_factor=2, mode='trilinear', align_corners=False)
      res = self.res_conv(res)
      x1 = x1 + res

    return x1
