"""
HD U-net (Hierarchically Densely connected U-net) implementation.

Based on: "Artificial intelligence based treatment planning of radiotherapy for
locally advanced breast cancer" by van de Sande et al. (2021)
Physics and Imaging in Radiation Oncology 20 (2021) 111-116

Architecture follows Figure 2 from the paper with the following specifications:
- Growth rate: 16
- Dense blocks with 2 layers per level, 4 at bottleneck
- 3x3 zero-padded convolutions with ReLU
- Downsampling: 3x3 convolution with 2x2 stride
- Upsampling: 3x3 transpose convolution with 2x2 stride
- MSE loss, Adam optimizer with lr=0.0001
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
  """
  Single dense layer: 3x3 zero-padded convolution + ReLU.
  Output is concatenated with input (dense connection).
  """
  def __init__(self, in_channels, growth_rate):
    super().__init__()
    self.conv = nn.Conv2d(
      in_channels,
      growth_rate,
      kernel_size=3,
      padding=1,  # zero padding to maintain spatial size
      bias=True
    )
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    out = self.relu(self.conv(x))
    # Dense connection: concatenate input and output
    return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
  """
  Dense block containing multiple dense layers.
  Each layer receives all preceding feature maps as input.
  """
  def __init__(self, in_channels, growth_rate, num_layers):
    super().__init__()
    self.layers = nn.ModuleList()
    current_channels = in_channels

    for _ in range(num_layers):
      self.layers.append(DenseLayer(current_channels, growth_rate))
      current_channels += growth_rate

    self.out_channels = current_channels

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


class TransitionDown(nn.Module):
  """
  Transition down layer: 3x3 convolution with 2x2 stride + ReLU.
  Reduces spatial dimensions by half and adds growth_rate features.
  """
  def __init__(self, in_channels, growth_rate):
    super().__init__()
    self.conv = nn.Conv2d(
      in_channels,
      in_channels + growth_rate,
      kernel_size=3,
      stride=2,
      padding=1,
      bias=True
    )
    self.relu = nn.ReLU(inplace=True)
    self.out_channels = in_channels + growth_rate

  def forward(self, x):
    return self.relu(self.conv(x))


class TransitionUp(nn.Module):
  """
  Transition up layer: 3x3 transpose convolution with 2x2 stride + ReLU.
  Increases spatial dimensions by 2x.
  """
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.ConvTranspose2d(
      in_channels,
      out_channels,
      kernel_size=3,
      stride=2,
      padding=1,
      output_padding=1,
      bias=True
    )
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    return self.relu(self.conv(x))


class HDUNet(nn.Module):
  """
  Hierarchically Densely connected U-net (HD U-net).

  Architecture based on Figure 2 from van de Sande et al. (2021):

  Encoder path (with feature counts for 5-channel input):
  - Level 0 (256x256): 5 → 21 → 37 (2 dense convs)
  - Level 1 (128x128): 53 → 69 → 85 (2 dense convs)
  - Level 2 (64x64): 101 → 117 → 133 (2 dense convs)
  - Level 3 (32x32): 149 → 165 → 181 (2 dense convs)
  - Bottleneck (16x16): 197 → 213 → 229 → 245 → 261 (4 dense convs)

  Decoder path mirrors encoder with skip connections via max pooling.

  Args:
    in_channels: Number of input channels (5 in paper: PTV, Heart, Lungs,
           Humerus_PRV10, External-PTV as distance maps)
    out_channels: Number of output channels (1 for dose prediction)
    growth_rate: Number of features added by each dense layer (16 in paper)
    num_dense_layers: Number of dense layers per block at each level
    num_bottleneck_layers: Number of dense layers in bottleneck
  """
  def __init__(
    self,
    in_channels=5,
    out_channels=1,
    growth_rate=16,
    num_dense_layers=2,
    num_bottleneck_layers=4
  ):
    super().__init__()

    self.in_channels = in_channels
    self.growth_rate = growth_rate

    # Encoder
    # Level 0: 5 → 21 → 37
    self.enc_block0 = DenseBlock(in_channels, growth_rate, num_dense_layers)
    enc0_out = self.enc_block0.out_channels  # 37

    # Transition down 0→1: 37 → 53
    self.trans_down0 = TransitionDown(enc0_out, growth_rate)
    trans0_out = self.trans_down0.out_channels  # 53

    # Level 1: 53 → 69 → 85
    self.enc_block1 = DenseBlock(trans0_out, growth_rate, num_dense_layers)
    enc1_out = self.enc_block1.out_channels  # 85

    # Transition down 1→2: 85 → 101
    self.trans_down1 = TransitionDown(enc1_out, growth_rate)
    trans1_out = self.trans_down1.out_channels  # 101

    # Level 2: 101 → 117 → 133
    self.enc_block2 = DenseBlock(trans1_out, growth_rate, num_dense_layers)
    enc2_out = self.enc_block2.out_channels  # 133

    # Transition down 2→3: 133 → 149
    self.trans_down2 = TransitionDown(enc2_out, growth_rate)
    trans2_out = self.trans_down2.out_channels  # 149

    # Level 3: 149 → 165 → 181
    self.enc_block3 = DenseBlock(trans2_out, growth_rate, num_dense_layers)
    enc3_out = self.enc_block3.out_channels  # 181

    # Transition down 3→bottleneck: 181 → 197
    self.trans_down3 = TransitionDown(enc3_out, growth_rate)
    trans3_out = self.trans_down3.out_channels  # 197

    # Bottleneck: 197 → 213 → 229 → 245 → 261
    self.bottleneck = DenseBlock(trans3_out, growth_rate, num_bottleneck_layers)
    bottleneck_out = self.bottleneck.out_channels  # 261

    # Max pooling for skip connections (2x2)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    # Decoder
    # Transition up bottleneck→3: features from bottleneck upsampled
    # Skip connection adds max-pooled enc3 features
    self.trans_up3 = TransitionUp(bottleneck_out, bottleneck_out)
    # After concat with skip: bottleneck_out + enc3_out
    dec3_in = bottleneck_out + enc3_out  # 261 + 181 = 442
    self.dec_block3 = DenseBlock(dec3_in, growth_rate, num_dense_layers)
    dec3_out = self.dec_block3.out_channels  # 442 + 32 = 474

    # Transition up 3→2
    self.trans_up2 = TransitionUp(dec3_out, dec3_out)
    dec2_in = dec3_out + enc2_out  # 474 + 133 = 607
    self.dec_block2 = DenseBlock(dec2_in, growth_rate, num_dense_layers)
    dec2_out = self.dec_block2.out_channels  # 607 + 32 = 639

    # Transition up 2→1
    self.trans_up1 = TransitionUp(dec2_out, dec2_out)
    dec1_in = dec2_out + enc1_out  # 639 + 85 = 724
    self.dec_block1 = DenseBlock(dec1_in, growth_rate, num_dense_layers)
    dec1_out = self.dec_block1.out_channels  # 724 + 32 = 756

    # Transition up 1→0
    self.trans_up0 = TransitionUp(dec1_out, dec1_out)
    dec0_in = dec1_out + enc0_out  # 756 + 37 = 793
    self.dec_block0 = DenseBlock(dec0_in, growth_rate, num_dense_layers)
    dec0_out = self.dec_block0.out_channels  # 793 + 32 = 825

    # Final output layer: 1x1 convolution to get single channel output
    self.final_conv = nn.Conv2d(dec0_out, out_channels, kernel_size=1, bias=True)

    # Store encoder output channels for reference
    self._enc_channels = [enc0_out, enc1_out, enc2_out, enc3_out, bottleneck_out]

  def forward(self, x):
    # Encoder path
    # Level 0
    enc0 = self.enc_block0(x)  # 37 channels

    # Transition down and Level 1
    x = self.trans_down0(enc0)  # 53 channels
    enc1 = self.enc_block1(x)   # 85 channels

    # Transition down and Level 2
    x = self.trans_down1(enc1)  # 101 channels
    enc2 = self.enc_block2(x)   # 133 channels

    # Transition down and Level 3
    x = self.trans_down2(enc2)  # 149 channels
    enc3 = self.enc_block3(x)   # 181 channels

    # Transition down and Bottleneck
    x = self.trans_down3(enc3)  # 197 channels
    x = self.bottleneck(x)    # 261 channels

    # Decoder path with skip connections
    # Level 3 decoder
    x = self.trans_up3(x)
    # Skip connection: max pool enc3 to match decoder spatial size, then concat
    skip3 = enc3  # Already at correct spatial size after trans_up
    x = torch.cat([x, skip3], dim=1)
    x = self.dec_block3(x)

    # Level 2 decoder
    x = self.trans_up2(x)
    skip2 = enc2
    x = torch.cat([x, skip2], dim=1)
    x = self.dec_block2(x)

    # Level 1 decoder
    x = self.trans_up1(x)
    skip1 = enc1
    x = torch.cat([x, skip1], dim=1)
    x = self.dec_block1(x)

    # Level 0 decoder
    x = self.trans_up0(x)
    skip0 = enc0
    x = torch.cat([x, skip0], dim=1)
    x = self.dec_block0(x)

    # Final output
    x = self.final_conv(x)

    return x


def build_hdunet(model_config, structure_selector):
  """
  Build HD U-net model from configuration.

  Args:
    model_config: Model configuration dictionary
    structure_selector: Structure selector for determining input channels

  Returns:
    HDUNet model instance
  """
  hdunet_config = model_config.get('hdunet', {})

  n_input_channels = structure_selector.total_channels

  model = HDUNet(
    in_channels=n_input_channels,
    out_channels=1,
    growth_rate=hdunet_config.get('growth_rate', 16),
    num_dense_layers=hdunet_config.get('num_dense_layers', 2),
    num_bottleneck_layers=hdunet_config.get('num_bottleneck_layers', 4)
  )

  return model
