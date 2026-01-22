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
  def __init__(self, in_channels, growth_rate):
    super().__init__()
    self.norm = nn.BatchNorm2d(in_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv = nn.Conv2d(
      in_channels, growth_rate,
      kernel_size=3, padding=1, bias=False
    )

  def forward(self, x):
    out = self.conv(self.relu(self.norm(x)))
    return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
  def __init__(self, in_channels, growth_rate, num_layers):
    super().__init__()
    self.layers = nn.ModuleList()
    channels = in_channels

    for _ in range(num_layers):
      self.layers.append(DenseLayer(channels, growth_rate))
      channels += growth_rate

    self.out_channels = channels

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


class TransitionDown(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.norm = nn.BatchNorm2d(in_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv = nn.Conv2d(
      in_channels, out_channels,
      kernel_size=3, stride=2, padding=1, bias=False
    )

  def forward(self, x):
    return self.conv(self.relu(self.norm(x)))


class TransitionUp(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.ConvTranspose2d(
      in_channels, out_channels,
      kernel_size=3, stride=2, padding=1, output_padding=1,
      bias=False
    )

  def forward(self, x):
    return self.conv(x)

class ChannelReduce(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

  def forward(self, x):
    return self.conv(x)


class HDUNet(nn.Module):
  def __init__(
    self,
    in_channels=5,
    out_channels=1,
    growth_rate=16,
    n_dense=2,
    n_bottleneck=4,
    base_channels=32
  ):
    super().__init__()

    # ---------- Encoder ----------
    self.enc0 = DenseBlock(in_channels, growth_rate, n_dense)
    c0 = self.enc0.out_channels

    self.down0 = TransitionDown(c0, base_channels)
    self.enc1 = DenseBlock(base_channels, growth_rate, n_dense)
    c1 = self.enc1.out_channels

    self.down1 = TransitionDown(c1, base_channels * 2)
    self.enc2 = DenseBlock(base_channels * 2, growth_rate, n_dense)
    c2 = self.enc2.out_channels

    self.down2 = TransitionDown(c2, base_channels * 4)
    self.enc3 = DenseBlock(base_channels * 4, growth_rate, n_dense)
    c3 = self.enc3.out_channels

    self.down3 = TransitionDown(c3, base_channels * 8)
    self.bottleneck = DenseBlock(base_channels * 8, growth_rate, n_bottleneck)
    cb = self.bottleneck.out_channels

    # ---------- Decoder ----------
    self.up3 = TransitionUp(cb, base_channels * 4)
    self.reduce3 = ChannelReduce(base_channels * 4 + c3, base_channels * 4)
    self.dec3 = DenseBlock(base_channels * 4, growth_rate, n_dense)

    self.up2 = TransitionUp(self.dec3.out_channels, base_channels * 2)
    self.reduce2 = ChannelReduce(base_channels * 2 + c2, base_channels * 2)
    self.dec2 = DenseBlock(base_channels * 2, growth_rate, n_dense)

    self.up1 = TransitionUp(self.dec2.out_channels, base_channels)
    self.reduce1 = ChannelReduce(base_channels + c1, base_channels)
    self.dec1 = DenseBlock(base_channels, growth_rate, n_dense)

    self.up0 = TransitionUp(self.dec1.out_channels, base_channels)
    self.reduce0 = ChannelReduce(base_channels + c0, base_channels)
    self.dec0 = DenseBlock(base_channels, growth_rate, n_dense)

    self.final = nn.Conv2d(self.dec0.out_channels, out_channels, kernel_size=1)

  def forward(self, x):
    e0 = self.enc0(x)
    e1 = self.enc1(self.down0(e0))
    e2 = self.enc2(self.down1(e1))
    e3 = self.enc3(self.down2(e2))

    b = self.bottleneck(self.down3(e3))

    d3 = self.up3(b)
    d3 = self.dec3(self.reduce3(torch.cat([d3, e3], dim=1)))

    d2 = self.up2(d3)
    d2 = self.dec2(self.reduce2(torch.cat([d2, e2], dim=1)))

    d1 = self.up1(d2)
    d1 = self.dec1(self.reduce1(torch.cat([d1, e1], dim=1)))

    d0 = self.up0(d1)
    d0 = self.dec0(self.reduce0(torch.cat([d0, e0], dim=1)))

    return self.final(d0)


def build_hdunet(model_config, structure_selector, out_channels=1):
  """
  Build HD U-net model from configuration.

  Args:
    model_config: Model configuration dictionary
    structure_selector: Structure selector for determining input channels
    out_channels: Number of output channels (default: 1 for dose prediction)

  Returns:
    HDUNet model instance
  """
  hdunet_config = model_config.get('hdunet', {})

  n_input_channels = structure_selector.total_channels

  model = HDUNet(
    in_channels=n_input_channels,
    out_channels=out_channels,
    growth_rate=hdunet_config.get('growth_rate', 16),
    n_dense=hdunet_config.get('num_dense_layers', 2),
    n_bottleneck=hdunet_config.get('num_bottleneck_layers', 4),
    base_channels=hdunet_config.get('base_channels', 32),
  )

  return model
