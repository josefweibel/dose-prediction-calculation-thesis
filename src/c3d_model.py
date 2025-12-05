"""
Cascade 3D U-Net (C3D) for Dose Prediction
Based on the paper: "A cascade 3D U-Net for dose prediction in radiotherapy"
Liu et al., Medical Physics 2021

This file contains both 3D and 2D implementations:
- C3D: Original 3D implementation for volumetric data
- C3D_2D: 2D adaptation for slice-based processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegularConvBlock(nn.Module):
  """Regular convolution block: 3x3x3 Conv (stride=1) -> IN -> ReLU"""
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.norm = nn.InstanceNorm3d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    return self.relu(self.norm(self.conv(x)))


class DownsamplingBlock(nn.Module):
  """Down-sampling block: 3x3x3 Conv (stride=2) -> IN -> ReLU"""
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    self.norm = nn.InstanceNorm3d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    return self.relu(self.norm(self.conv(x)))


class UpsamplingBlock(nn.Module):
  """Up-sampling block: Trilinear upsampling -> 3x3x3 Conv (stride=1) -> IN -> ReLU"""
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
    self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.norm = nn.InstanceNorm3d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.upsample(x)
    x = self.conv(x)
    x = self.norm(x)
    return self.relu(x)


class UNet3D(nn.Module):
  """
  3D U-Net architecture as described in the C3D paper.

  Args:
    in_channels: Number of input channels
    out_channels: Number of output channels (feature maps for intermediate model)
    initial_features: Initial number of feature maps (16 for Model A, 32 for Model B)
  """
  def __init__(self, in_channels, out_channels, initial_features=16):
    super().__init__()

    features = initial_features

    # Encoder path (down-sampling)
    # Level 1: 128x128x128
    self.enc1_conv1 = RegularConvBlock(in_channels, features)
    self.enc1_conv2 = RegularConvBlock(features, features)
    self.enc1_down = DownsamplingBlock(features, features * 2)

    # Level 2: 64x64x64
    self.enc2_conv1 = RegularConvBlock(features * 2, features * 2)
    self.enc2_conv2 = RegularConvBlock(features * 2, features * 2)
    self.enc2_down = DownsamplingBlock(features * 2, features * 4)

    # Level 3: 32x32x32
    self.enc3_conv1 = RegularConvBlock(features * 4, features * 4)
    self.enc3_conv2 = RegularConvBlock(features * 4, features * 4)
    self.enc3_down = DownsamplingBlock(features * 4, features * 8)

    # Level 4: 16x16x16
    self.enc4_conv1 = RegularConvBlock(features * 8, features * 8)
    self.enc4_conv2 = RegularConvBlock(features * 8, features * 8)
    self.enc4_down = DownsamplingBlock(features * 8, features * 16)

    # Bottleneck: 8x8x8
    self.bottleneck_conv1 = RegularConvBlock(features * 16, features * 16)
    self.bottleneck_conv2 = RegularConvBlock(features * 16, features * 16)

    # Decoder path (up-sampling)
    # Level 4: 16x16x16
    self.dec4_up = UpsamplingBlock(features * 16, features * 8)
    self.dec4_conv1 = RegularConvBlock(features * 16, features * 8)  # After concat
    self.dec4_conv2 = RegularConvBlock(features * 8, features * 8)

    # Level 3: 32x32x32
    self.dec3_up = UpsamplingBlock(features * 8, features * 4)
    self.dec3_conv1 = RegularConvBlock(features * 8, features * 4)  # After concat
    self.dec3_conv2 = RegularConvBlock(features * 4, features * 4)

    # Level 2: 64x64x64
    self.dec2_up = UpsamplingBlock(features * 4, features * 2)
    self.dec2_conv1 = RegularConvBlock(features * 4, features * 2)  # After concat
    self.dec2_conv2 = RegularConvBlock(features * 2, features * 2)

    # Level 1: 128x128x128
    self.dec1_up = UpsamplingBlock(features * 2, features)
    self.dec1_conv1 = RegularConvBlock(features * 2, features)  # After concat
    self.dec1_conv2 = RegularConvBlock(features, features)

    # Output layer: 1x1x1 conv
    self.output = nn.Conv3d(features, out_channels, kernel_size=1, stride=1)

  def forward(self, x):
    # Encoder
    # Level 1
    enc1 = self.enc1_conv1(x)
    enc1 = self.enc1_conv2(enc1)
    enc1_down = self.enc1_down(enc1)

    # Level 2
    enc2 = self.enc2_conv1(enc1_down)
    enc2 = self.enc2_conv2(enc2)
    enc2_down = self.enc2_down(enc2)

    # Level 3
    enc3 = self.enc3_conv1(enc2_down)
    enc3 = self.enc3_conv2(enc3)
    enc3_down = self.enc3_down(enc3)

    # Level 4
    enc4 = self.enc4_conv1(enc3_down)
    enc4 = self.enc4_conv2(enc4)
    enc4_down = self.enc4_down(enc4)

    # Bottleneck
    bottleneck = self.bottleneck_conv1(enc4_down)
    bottleneck = self.bottleneck_conv2(bottleneck)

    # Decoder
    # Level 4
    dec4 = self.dec4_up(bottleneck)
    dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection
    dec4 = self.dec4_conv1(dec4)
    dec4 = self.dec4_conv2(dec4)

    # Level 3
    dec3 = self.dec3_up(dec4)
    dec3 = torch.cat([dec3, enc3], dim=1)  # Skip connection
    dec3 = self.dec3_conv1(dec3)
    dec3 = self.dec3_conv2(dec3)

    # Level 2
    dec2 = self.dec2_up(dec3)
    dec2 = torch.cat([dec2, enc2], dim=1)  # Skip connection
    dec2 = self.dec2_conv1(dec2)
    dec2 = self.dec2_conv2(dec2)

    # Level 1
    dec1 = self.dec1_up(dec2)
    dec1 = torch.cat([dec1, enc1], dim=1)  # Skip connection
    dec1 = self.dec1_conv1(dec1)
    dec1 = self.dec1_conv2(dec1)

    # Output
    output = self.output(dec1)

    return output


class C3D(nn.Module):
  """
  Cascade 3D U-Net (C3D) model for dose prediction.

  The architecture consists of two cascaded 3D U-Net models:
  - Model A: Predicts coarse dose distribution (16 feature maps output)
  - Model B: Refines the prediction using Model A's output

  Input channels:
  - Model A: 9 channels (1 PTV + 7 OARs + 1 CT)
  - Model B: 25 channels (9 original + 16 from Model A)

  Args:
    in_channels: Number of input channels (default: 9 for OpenKBP dataset)
    out_channels: Number of output channels (default: 1 for dose prediction)
  """
  def __init__(self, in_channels=9, out_channels=1):
    super().__init__()

    # Model A: Initial coarse prediction
    # Input: 9 channels, Output: 16 feature maps
    self.model_a = UNet3D(
      in_channels=in_channels,
      out_channels=16,
      initial_features=16
    )

    # Output layer for Model A (for deep supervision)
    self.model_a_output = nn.Conv3d(16, out_channels, kernel_size=1, stride=1)

    # Model B: Refinement model
    # Input: 9 original + 16 from Model A = 25 channels
    # Output: 1 dose map
    self.model_b = UNet3D(
      in_channels=in_channels + 16,
      out_channels=out_channels,
      initial_features=32
    )

  def forward(self, x):
    """
    Forward pass through the cascade network.

    Args:
      x: Input tensor of shape (B, 9, D, H, W)
         where channels are: [PTV, OAR1-7, CT]

    Returns:
      If training: (output_a, output_b) for deep supervision
      If inference: output_b (final prediction)
    """
    # Model A: Coarse prediction
    features_a = self.model_a(x)
    output_a = self.model_a_output(features_a)

    # Concatenate original input with Model A features
    model_b_input = torch.cat([x, features_a], dim=1)

    # Model B: Refined prediction
    output_b = self.model_b(model_b_input)

    # Apply ReLU to ensure non-negative dose values
    output_a = F.relu(output_a)
    output_b = F.relu(output_b)

    if self.training:
      # Return both outputs for deep supervision during training
      return output_a, output_b
    else:
      # Return only final output during inference
      return output_b


class C3DLoss(nn.Module):
  """
  Loss function for C3D with deep supervision.

  Combines losses from both Model A and Model B:
  L = α * L₁(P_A, GT) + L₁(P_B, GT)

  where:
  - P_A: prediction from Model A
  - P_B: prediction from Model B (final output)
  - GT: ground truth
  - α: weight for Model A loss (default: 0.5)
  - L₁: Mean Absolute Error (MAE)

  Only voxels with non-zero dose in ground truth are included.
  """
  def __init__(self, alpha=0.5):
    super().__init__()
    self.alpha = alpha

  def forward(self, pred_a, pred_b, target):
    """
    Compute the cascade loss.

    Args:
      pred_a: Prediction from Model A (B, 1, D, H, W)
      pred_b: Prediction from Model B (B, 1, D, H, W)
      target: Ground truth dose (B, 1, D, H, W)

    Returns:
      Combined loss value
    """
    # Create mask for non-zero dose regions
    mask = target > 0

    if mask.sum() == 0:
      # No non-zero dose voxels, return zero loss
      return torch.tensor(0.0, device=target.device, requires_grad=True)

    # Compute MAE only on non-zero dose regions
    loss_a = torch.abs(pred_a[mask] - target[mask]).mean()
    loss_b = torch.abs(pred_b[mask] - target[mask]).mean()

    # Combined loss with deep supervision
    total_loss = self.alpha * loss_a + loss_b

    return total_loss


def create_c3d_model(in_channels=9, out_channels=1):
  """
  Factory function to create a C3D model.

  Args:
    in_channels: Number of input channels (default: 9)
    out_channels: Number of output channels (default: 1)

  Returns:
    C3D model instance
  """
  return C3D(in_channels=in_channels, out_channels=out_channels)


# ============================================================================
# 2D VERSION - For slice-based processing
# ============================================================================

class RegularConvBlock2D(nn.Module):
  """Regular 2D convolution block: 3x3 Conv (stride=1) -> IN -> ReLU"""
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.norm = nn.InstanceNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    return self.relu(self.norm(self.conv(x)))


class DownsamplingBlock2D(nn.Module):
  """Down-sampling 2D block: 3x3 Conv (stride=2) -> IN -> ReLU"""
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    self.norm = nn.InstanceNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    return self.relu(self.norm(self.conv(x)))


class UpsamplingBlock2D(nn.Module):
  """Up-sampling 2D block: Bilinear upsampling -> 3x3 Conv (stride=1) -> IN -> ReLU"""
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.norm = nn.InstanceNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.upsample(x)
    x = self.conv(x)
    x = self.norm(x)
    return self.relu(x)


class UNet2D(nn.Module):
  """
  2D U-Net architecture adapted from the C3D paper.

  Args:
    in_channels: Number of input channels
    out_channels: Number of output channels (feature maps for intermediate model)
    initial_features: Initial number of feature maps (16 for Model A, 32 for Model B)
  """
  def __init__(self, in_channels, out_channels, initial_features=16):
    super().__init__()

    features = initial_features

    # Encoder path (down-sampling)
    # Level 1: 128x128
    self.enc1_conv1 = RegularConvBlock2D(in_channels, features)
    self.enc1_conv2 = RegularConvBlock2D(features, features)
    self.enc1_down = DownsamplingBlock2D(features, features * 2)

    # Level 2: 64x64
    self.enc2_conv1 = RegularConvBlock2D(features * 2, features * 2)
    self.enc2_conv2 = RegularConvBlock2D(features * 2, features * 2)
    self.enc2_down = DownsamplingBlock2D(features * 2, features * 4)

    # Level 3: 32x32
    self.enc3_conv1 = RegularConvBlock2D(features * 4, features * 4)
    self.enc3_conv2 = RegularConvBlock2D(features * 4, features * 4)
    self.enc3_down = DownsamplingBlock2D(features * 4, features * 8)

    # Level 4: 16x16
    self.enc4_conv1 = RegularConvBlock2D(features * 8, features * 8)
    self.enc4_conv2 = RegularConvBlock2D(features * 8, features * 8)
    self.enc4_down = DownsamplingBlock2D(features * 8, features * 16)

    # Bottleneck: 8x8
    self.bottleneck_conv1 = RegularConvBlock2D(features * 16, features * 16)
    self.bottleneck_conv2 = RegularConvBlock2D(features * 16, features * 16)

    # Decoder path (up-sampling)
    # Level 4: 16x16
    self.dec4_up = UpsamplingBlock2D(features * 16, features * 8)
    self.dec4_conv1 = RegularConvBlock2D(features * 16, features * 8)  # After concat
    self.dec4_conv2 = RegularConvBlock2D(features * 8, features * 8)

    # Level 3: 32x32
    self.dec3_up = UpsamplingBlock2D(features * 8, features * 4)
    self.dec3_conv1 = RegularConvBlock2D(features * 8, features * 4)  # After concat
    self.dec3_conv2 = RegularConvBlock2D(features * 4, features * 4)

    # Level 2: 64x64
    self.dec2_up = UpsamplingBlock2D(features * 4, features * 2)
    self.dec2_conv1 = RegularConvBlock2D(features * 4, features * 2)  # After concat
    self.dec2_conv2 = RegularConvBlock2D(features * 2, features * 2)

    # Level 1: 128x128
    self.dec1_up = UpsamplingBlock2D(features * 2, features)
    self.dec1_conv1 = RegularConvBlock2D(features * 2, features)  # After concat
    self.dec1_conv2 = RegularConvBlock2D(features, features)

    # Output layer: 1x1 conv
    self.output = nn.Conv2d(features, out_channels, kernel_size=1, stride=1)

  def forward(self, x):
    # Encoder
    # Level 1
    enc1 = self.enc1_conv1(x)
    enc1 = self.enc1_conv2(enc1)
    enc1_down = self.enc1_down(enc1)

    # Level 2
    enc2 = self.enc2_conv1(enc1_down)
    enc2 = self.enc2_conv2(enc2)
    enc2_down = self.enc2_down(enc2)

    # Level 3
    enc3 = self.enc3_conv1(enc2_down)
    enc3 = self.enc3_conv2(enc3)
    enc3_down = self.enc3_down(enc3)

    # Level 4
    enc4 = self.enc4_conv1(enc3_down)
    enc4 = self.enc4_conv2(enc4)
    enc4_down = self.enc4_down(enc4)

    # Bottleneck
    bottleneck = self.bottleneck_conv1(enc4_down)
    bottleneck = self.bottleneck_conv2(bottleneck)

    # Decoder
    # Level 4
    dec4 = self.dec4_up(bottleneck)
    dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection
    dec4 = self.dec4_conv1(dec4)
    dec4 = self.dec4_conv2(dec4)

    # Level 3
    dec3 = self.dec3_up(dec4)
    dec3 = torch.cat([dec3, enc3], dim=1)  # Skip connection
    dec3 = self.dec3_conv1(dec3)
    dec3 = self.dec3_conv2(dec3)

    # Level 2
    dec2 = self.dec2_up(dec3)
    dec2 = torch.cat([dec2, enc2], dim=1)  # Skip connection
    dec2 = self.dec2_conv1(dec2)
    dec2 = self.dec2_conv2(dec2)

    # Level 1
    dec1 = self.dec1_up(dec2)
    dec1 = torch.cat([dec1, enc1], dim=1)  # Skip connection
    dec1 = self.dec1_conv1(dec1)
    dec1 = self.dec1_conv2(dec1)

    # Output
    output = self.output(dec1)

    return output


class C3D_2D(nn.Module):
  """
  Cascade 2D U-Net model for slice-based dose prediction.

  Adapted from the C3D paper for 2D slice processing.
  The architecture maintains the cascade mechanism but processes
  2D slices instead of 3D volumes.

  Input channels:
  - Model A: variable (typically 10-15 for structure masks + CT)
  - Model B: original_channels + 16 from Model A

  Args:
    in_channels: Number of input channels
    out_channels: Number of output channels (default: 1 for dose prediction)
  """
  def __init__(self, in_channels=10, out_channels=1):
    super().__init__()

    # Model A: Initial coarse prediction
    # Input: in_channels, Output: 16 feature maps
    self.model_a = UNet2D(
      in_channels=in_channels,
      out_channels=16,
      initial_features=16
    )

    # Output layer for Model A (for deep supervision)
    self.model_a_output = nn.Conv2d(16, out_channels, kernel_size=1, stride=1)

    # Model B: Refinement model
    # Input: original + 16 from Model A
    # Output: 1 dose map
    self.model_b = UNet2D(
      in_channels=in_channels + 16,
      out_channels=out_channels,
      initial_features=32
    )

  def forward(self, x):
    """
    Forward pass through the cascade network.

    Args:
      x: Input tensor of shape (B, C, H, W)
         where channels are: [structures, CT]

    Returns:
      If training: (output_a, output_b) for deep supervision
      If inference: output_b (final prediction)
    """
    # Model A: Coarse prediction
    features_a = self.model_a(x)
    output_a = self.model_a_output(features_a)

    # Concatenate original input with Model A features
    model_b_input = torch.cat([x, features_a], dim=1)

    # Model B: Refined prediction
    output_b = self.model_b(model_b_input)

    # Apply ReLU to ensure non-negative dose values
    output_a = F.relu(output_a)
    output_b = F.relu(output_b)

    if self.training:
      # Return both outputs for deep supervision during training
      return output_a, output_b
    else:
      # Return only final output during inference
      return output_b


def create_c3d_2d_model(in_channels=10, out_channels=1):
  """
  Factory function to create a C3D_2D model.

  Args:
    in_channels: Number of input channels (default: 10)
    out_channels: Number of output channels (default: 1)

  Returns:
    C3D_2D model instance
  """
  return C3D_2D(in_channels=in_channels, out_channels=out_channels)
