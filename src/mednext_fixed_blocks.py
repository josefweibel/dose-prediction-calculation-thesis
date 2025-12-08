"""
Fixed MedNeXt blocks that avoid checkerboard artifacts from transposed convolutions.

The fix replaces ConvTranspose with bilinear upsampling + regular convolution.
This is a well-known solution to checkerboard artifacts in generative models.

Reference: https://distill.pub/2016/deconv-checkerboard/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunet_mednext.network_architecture.mednextv1.blocks import MedNeXtBlock


class MedNeXtUpBlock(MedNeXtBlock):
  """
  Fixed MedNeXtUpBlock that avoids checkerboard artifacts by replacing
  transposed convolution with bilinear upsample + regular convolution.
  """

  def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
               do_res=False, norm_type='group', dim='2d', grn=False):
    super().__init__(in_channels, out_channels, exp_r, kernel_size,
                     do_res=False, norm_type=norm_type, dim=dim,
                     grn=grn)

    self.resample_do_res = do_res

    self.dim = dim
    if dim == '2d':
      conv = nn.Conv2d
    elif dim == '3d':
      conv = nn.Conv3d

    # Optional residual connection using regular conv (not transposed)
    if do_res:
      self.res_conv = conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1  # Changed from 2
      )

    # Replace transposed conv with regular depthwise conv
    # Upsampling will be done separately using bilinear interpolation
    self.conv1 = conv(
      in_channels=in_channels,
      out_channels=in_channels,
      kernel_size=kernel_size,
      stride=1,  # Changed from 2
      padding=kernel_size//2,
      groups=in_channels,
    )


  def forward(self, x, dummy_tensor=None):
    # Store original input for residual connection
    x_orig = x

    # Upsample FIRST using bilinear interpolation
    if self.dim == '2d':
      x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    elif self.dim == '3d':
      x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)

    # Then apply the MedNeXtBlock operations
    x1 = super().forward(x)

    # NO asymmetric padding needed since we're using bilinear upsample
    # The output size is already correct

    if self.resample_do_res:
      # Upsample residual connection from ORIGINAL input
      if self.dim == '2d':
        res = F.interpolate(x_orig, scale_factor=2, mode='bilinear', align_corners=False)
      elif self.dim == '3d':
        res = F.interpolate(x_orig, scale_factor=2, mode='trilinear', align_corners=False)
      res = self.res_conv(res)
      x1 = x1 + res

    return x1
