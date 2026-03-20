"""
DeepLabv3+ for Dose Distribution Prediction
Based on the paper: "Dose prediction using a deep neural network for accelerated
planning of rectal cancer radiotherapy" by Song et al., Radiotherapy and Oncology 2020

Paper Specifications:
- Network: DeepLabv3+ with ResNet-50 encoder
- Processing: 2D slice-by-slice manner (each CT slice predicts corresponding dose slice)
- Input: CT scans + ROI contouring (PTV, OARs, body region)
- Output: Dose distribution for corresponding ROI
- Loss: Mean Square Error (MSE) + Perceptual loss (to preserve dose gradient)

Training Configuration (from paper):
- Batch size: 10
- Total epochs: 100
- Optimizer: Stochastic Gradient Descent (SGD)
- Initial learning rate: 0.01
- LR schedule: multiplied by (1 - t/T)^0.9 for epoch t
- Framework: PyTorch

Architecture Components:
- ResNet-50 encoder: Feature extraction with atrous convolutions
- ASPP module: Multi-scale contextual information capture
- Decoder with concat operator: Multi-level feature integration
- Perceptual loss: Preserves dose gradient features in fall-off regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class AtrousSeparableConv2d(nn.Module):
  """Atrous Separable Convolution for efficient feature extraction"""
  def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, bias=False):
    super().__init__()
    self.depthwise = nn.Conv2d(
      in_channels, in_channels, kernel_size=kernel_size,
      dilation=dilation, padding=dilation, groups=in_channels, bias=bias
    )
    self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.depthwise(x)
    x = self.pointwise(x)
    x = self.bn(x)
    return self.relu(x)


class ASPPModule(nn.Module):
  """
  Atrous Spatial Pyramid Pooling module
  Captures multi-scale contextual information using parallel atrous convolutions
  with different dilation rates
  """
  def __init__(self, in_channels, out_channels=256, atrous_rates=(6, 12, 18)):
    super().__init__()

    modules = []
    # 1x1 convolution
    modules.append(nn.Sequential(
      nn.Conv2d(in_channels, out_channels, 1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    ))

    # Atrous convolutions with different rates
    for rate in atrous_rates:
      modules.append(AtrousSeparableConv2d(in_channels, out_channels, dilation=rate))

    # Image-level features (global average pooling)
    self.global_avg_pool = nn.Sequential(
      nn.AdaptiveAvgPool2d(1),
      nn.Conv2d(in_channels, out_channels, 1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )

    self.convs = nn.ModuleList(modules)

    # Projection after concatenation
    self.project = nn.Sequential(
      nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Dropout(0.5)
    )

  def forward(self, x):
    res = []
    for conv in self.convs:
      res.append(conv(x))

    # Global features
    global_features = self.global_avg_pool(x)
    global_features = F.interpolate(
      global_features, size=x.shape[2:], mode='bilinear', align_corners=False
    )
    res.append(global_features)

    # Concatenate all features
    res = torch.cat(res, dim=1)
    return self.project(res)


class DecoderModule(nn.Module):
  """
  Decoder module with skip connections
  Aggregates encoder features with decoder features using concatenation
  """
  def __init__(self, low_level_channels, num_classes=1, separate_heads=False, var_head_init_bias=-2.0):
    super().__init__()
    self.separate_heads = separate_heads and num_classes >= 2

    # Reduce channels from low-level features
    self.conv1 = nn.Sequential(
      nn.Conv2d(low_level_channels, 48, 1, bias=False),
      nn.BatchNorm2d(48),
      nn.ReLU(inplace=True)
    )

    # Decoder layers after concatenation
    self.conv2 = nn.Sequential(
      nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Dropout(0.5),
      nn.Conv2d(256, 256, 3, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Dropout(0.1)
    )

    if self.separate_heads:
      self.mean_head = nn.Conv2d(256, 1, 1)
      self.var_head = nn.Conv2d(256, 1, 1)
      nn.init.constant_(self.var_head.bias, var_head_init_bias)
      if num_classes > 2:
        self.extra_head = nn.Conv2d(256, num_classes - 2, 1)
      else:
        self.extra_head = None
    else:
      self.conv3 = nn.Conv2d(256, num_classes, 1)

  def forward(self, encoder_features, low_level_features):
    # Reduce low-level feature channels
    low_level_features = self.conv1(low_level_features)

    # Upsample encoder features to match low-level features size
    encoder_features = F.interpolate(
      encoder_features, size=low_level_features.shape[2:],
      mode='bilinear', align_corners=False
    )

    # Concatenate
    x = torch.cat([encoder_features, low_level_features], dim=1)

    # Decode
    x = self.conv2(x)

    if self.separate_heads:
      parts = [self.mean_head(x), self.var_head(x)]
      if self.extra_head is not None:
        parts.append(self.extra_head(x))
      return torch.cat(parts, dim=1)
    else:
      return self.conv3(x)


class ResNet50Encoder(nn.Module):
  """
  ResNet-50 encoder for feature extraction
  Returns multi-scale features from different layers
  """
  def __init__(self, in_channels=3, pretrained=False):
    super().__init__()

    # Load ResNet-50
    if pretrained:
      weights = ResNet50_Weights.DEFAULT
      resnet = resnet50(weights=weights)
    else:
      resnet = resnet50(weights=None)

    # Modify first conv layer if input channels != 3
    if in_channels != 3:
      self.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
      )
      # Initialize with appropriate strategy
      nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
    else:
      self.conv1 = resnet.conv1

    self.bn1 = resnet.bn1
    self.relu = resnet.relu
    self.maxpool = resnet.maxpool

    # ResNet layers
    self.layer1 = resnet.layer1  # Output: 256 channels, 1/4 scale
    self.layer2 = resnet.layer2  # Output: 512 channels, 1/8 scale
    self.layer3 = resnet.layer3  # Output: 1024 channels, 1/16 scale
    self.layer4 = resnet.layer4  # Output: 2048 channels, 1/32 scale (output stride 16 with dilation)

    # Modify layer4 to use atrous convolutions (output stride = 16)
    self._modify_layer4_for_atrous()

  def _modify_layer4_for_atrous(self):
    """Modify layer4 to use dilated convolutions for maintaining resolution"""
    for n, m in self.layer4.named_modules():
      if 'conv2' in n:
        m.dilation = (2, 2)
        m.padding = (2, 2)
        m.stride = (1, 1)
      elif 'downsample.0' in n:
        m.stride = (1, 1)

  def forward(self, x):
    # Initial layers
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    # Encoder layers
    x1 = self.layer1(x)    # Low-level features (1/4 scale)
    x2 = self.layer2(x1)   # (1/8 scale)
    x3 = self.layer3(x2)   # (1/16 scale)
    x4 = self.layer4(x3)   # High-level features (1/16 scale with atrous conv)

    return x1, x4  # Return low-level and high-level features


class DeepLabv3Plus(nn.Module):
  """
  DeepLabv3+ model for dose distribution prediction

  Architecture:
  1. ResNet-50 encoder extracts multi-scale features
  2. ASPP module captures contextual information at multiple scales
  3. Decoder aggregates encoder features using concatenation
  4. Output is upsampled to input resolution

  Args:
    in_channels: Number of input channels (CT + ROI masks)
    num_classes: Number of output channels (1 for dose prediction)
    pretrained_encoder: Whether to use pretrained ResNet-50 weights
  """
  def __init__(self, in_channels=3, num_classes=1, pretrained_encoder=False,
               separate_heads=False, var_head_init_bias=-2.0):
    super().__init__()

    # Encoder
    self.encoder = ResNet50Encoder(in_channels=in_channels, pretrained=pretrained_encoder)

    # ASPP module
    self.aspp = ASPPModule(in_channels=2048, out_channels=256)

    # Decoder
    self.decoder = DecoderModule(
      low_level_channels=256, num_classes=num_classes,
      separate_heads=separate_heads, var_head_init_bias=var_head_init_bias
    )

  def forward(self, x):
    """
    Forward pass

    Args:
      x: Input tensor (B, C, H, W) where C includes CT scan and ROI masks

    Returns:
      Predicted dose distribution (B, 1, H, W)
    """
    input_shape = x.shape[-2:]

    # Encoder
    low_level_features, high_level_features = self.encoder(x)

    # ASPP
    aspp_features = self.aspp(high_level_features)

    # Decoder
    output = self.decoder(aspp_features, low_level_features)

    # Upsample to input resolution
    output = F.interpolate(
      output, size=input_shape, mode='bilinear', align_corners=False
    )

    parts = [F.relu(output[:, 0:1, :, :])]

    if output.shape[1] > 1:
      parts.append(F.softplus(output[:, 1:2, :, :]) + 0.01)
      if output.shape[1] > 2:
        parts.append(output[:, 2:, :, :])

    output = torch.cat(parts, dim=1)

    return output


class PerceptualLoss(nn.Module):
  """
  Perceptual loss for preserving dose gradient information

  Based on the paper reference [18]: "Perceptual losses for real-time style transfer
  and super-resolution" (Johnson et al., ECCV 2016)

  This implementation uses a lightweight feature extractor to compute perceptual loss
  in feature space, which helps preserve the dose gradient features, especially in
  dose fall-off regions.

  Note: The paper doesn't specify the exact perceptual loss architecture, so we use
  a simple VGG-style feature extractor. The features are computed and frozen during
  training to provide consistent gradient signals.
  """
  def __init__(self):
    super().__init__()

    # Simple feature extractor (VGG-style)
    # Using a lightweight architecture suitable for single-channel dose maps
    self.features = nn.Sequential(
      # Block 1
      nn.Conv2d(1, 64, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, 2),

      # Block 2
      nn.Conv2d(64, 128, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, 2),

      # Block 3
      nn.Conv2d(128, 256, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, 3, padding=1),
      nn.ReLU(inplace=True),
    )

    # Initialize weights with standard initialization
    self._initialize_weights()

    # Freeze feature extractor - it should not be trained
    # This provides consistent perceptual features throughout training
    for param in self.features.parameters():
      param.requires_grad = False

  def _initialize_weights(self):
    """Initialize weights using standard initialization"""
    for m in self.features.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)

  def forward(self, pred, target):
    """
    Compute perceptual loss

    Args:
      pred: Predicted dose (B, 1, H, W)
      target: Target dose (B, 1, H, W)

    Returns:
      Perceptual loss value
    """
    # Extract features
    pred_features = self.features(pred)
    target_features = self.features(target)

    # Compute L2 distance in feature space
    loss = F.mse_loss(pred_features, target_features)

    return loss


class DeepLabv3PlusLoss(nn.Module):
  """
  Combined loss function for DeepLabv3+ dose prediction

  As stated in the paper: "the perceptual loss is introduced as a part of the network
  loss function" to preserve dose gradient information.

  L_total = L_MSE + λ * L_perceptual

  where:
  - L_MSE: Mean Squared Error on dose values (primary loss as stated in paper)
  - L_perceptual: Perceptual loss to preserve dose gradients
  - λ: weight for perceptual loss (paper doesn't specify exact value, using 0.1 as reasonable default)
  """
  def __init__(self, perceptual_weight=0.1, use_perceptual=True):
    super().__init__()
    self.perceptual_weight = perceptual_weight
    self.use_perceptual = use_perceptual

    if use_perceptual:
      self.perceptual_loss = PerceptualLoss()

  def forward(self, pred, target):
    """
    Compute combined loss

    Args:
      pred: Predicted dose (B, 1, H, W)
      target: Target dose (B, 1, H, W)

    Returns:
      Total loss value (and optionally loss dict)
    """
    # MSE loss (as specified in paper: "loss function used in the optimization was mean square error")
    mse_loss = F.mse_loss(pred, target)

    if self.use_perceptual and self.perceptual_weight > 0:
      # Perceptual loss
      perceptual = self.perceptual_loss(pred, target)
      total_loss = mse_loss + self.perceptual_weight * perceptual
      return total_loss, {'mse': mse_loss.item(), 'perceptual': perceptual.item()}
    else:
      return mse_loss, {'mse': mse_loss.item()}


def build_deeplabv3plus(config, structure_selector, out_channels=1):
  """
  Factory function to build DeepLabv3+ model

  Args:
    config: Model configuration dict with 'deeplabv3plus' key
    structure_selector: Structure selector defining input channels
    out_channels: Number of output channels (default: 1 for dose prediction)

  Returns:
    DeepLabv3Plus model instance
  """
  deeplabv3plus_config = config.get('deeplabv3plus', {})

  n_input_channels = structure_selector.total_channels
  pretrained = deeplabv3plus_config.get('pretrained_encoder', False)
  separate_heads = deeplabv3plus_config.get('separate_heads', False)
  var_head_init_bias = deeplabv3plus_config.get('var_head_init_bias', -2.0)

  model = DeepLabv3Plus(
    in_channels=n_input_channels,
    num_classes=out_channels,
    pretrained_encoder=pretrained,
    separate_heads=separate_heads,
    var_head_init_bias=var_head_init_bias,
  )

  return model
