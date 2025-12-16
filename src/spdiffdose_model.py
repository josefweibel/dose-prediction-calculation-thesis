"""
SP-DiffDose: A Conditional Diffusion Model for Radiation Dose Prediction
Based on Multi-Scale Fusion of Anatomical Structures, Guided by SwinTransformer and Projector

Based on: "SP-DiffDose: A Conditional Diffusion Model for Radiation Dose Prediction
Based on Multi-Scale Fusion of Anatomical Structures, Guided by SwinTransformer and Projector"
Fu et al., IEEE Transactions, 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SinusoidalPositionEmbedding(nn.Module):
  """Sinusoidal position encoding for timestep embedding."""

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


class TimestepEmbedding(nn.Module):
  """Convert timestep to embedding."""

  def __init__(self, timestep_dim, embed_dim):
    super().__init__()
    self.linear_1 = nn.Linear(timestep_dim, embed_dim)
    self.act = nn.SiLU()
    self.linear_2 = nn.Linear(embed_dim, embed_dim)

  def forward(self, t):
    t_emb = self.linear_1(t)
    t_emb = self.act(t_emb)
    t_emb = self.linear_2(t_emb)
    return t_emb


class WindowAttention(nn.Module):
  """
  Window-based Multi-Head Self Attention (W-MSA) module with relative position bias.
  Based on Swin Transformer.
  """

  def __init__(self, dim, window_size, num_heads):
    super().__init__()
    self.dim = dim
    self.window_size = window_size  # (Wh, Ww)
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.scale = head_dim ** -0.5

    # Define a parameter table of relative position bias
    self.relative_position_bias_table = nn.Parameter(
      torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
    )

    # Get pair-wise relative position index for each token inside the window
    coords_h = torch.arange(self.window_size[0])
    coords_w = torch.arange(self.window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += self.window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    self.register_buffer("relative_position_index", relative_position_index)

    self.qkv = nn.Linear(dim, dim * 3)
    self.proj = nn.Linear(dim, dim)

    nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

  def forward(self, x):
    """
    Args:
        x: input features with shape of (num_windows*B, N, C)
    """
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
      self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    attn = attn + relative_position_bias.unsqueeze(0)

    attn = F.softmax(attn, dim=-1)

    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    return x


def window_partition(x, window_size):
  """
  Args:
      x: (B, H, W, C)
      window_size: window size

  Returns:
      windows: (num_windows*B, window_size, window_size, C)
  """
  B, H, W, C = x.shape
  x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
  windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
  return windows


def window_reverse(windows, window_size, H, W):
  """
  Args:
      windows: (num_windows*B, window_size, window_size, C)
      window_size: Window size
      H: Height of image
      W: Width of image

  Returns:
      x: (B, H, W, C)
  """
  B = int(windows.shape[0] / (H * W / window_size / window_size))
  x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
  x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
  return x


class SwinTransformerBlock(nn.Module):
  """
  Swin Transformer Block.

  Args:
      dim: Number of input channels.
      num_heads: Number of attention heads.
      window_size: Window size.
      shift_size: Shift size for SW-MSA.
      mlp_ratio: Ratio of mlp hidden dim to embedding dim.
  """

  def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.):
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.window_size = window_size
    self.shift_size = shift_size
    self.mlp_ratio = mlp_ratio

    self.norm1 = nn.LayerNorm(dim)
    self.attn = WindowAttention(
      dim, window_size=(self.window_size, self.window_size), num_heads=num_heads
    )

    self.norm2 = nn.LayerNorm(dim)
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_hidden_dim),
      nn.GELU(),
      nn.Linear(mlp_hidden_dim, dim)
    )

  def forward(self, x):
    """
    Args:
        x: Input tensor (B, C, H, W)
    Returns:
        Output tensor (B, C, H, W)
    """
    B, C, H, W = x.shape
    shortcut = x

    # Reshape to (B, H, W, C) for window attention
    x = x.permute(0, 2, 3, 1).contiguous()

    # Pad if needed
    pad_l = pad_t = 0
    pad_r = (self.window_size - W % self.window_size) % self.window_size
    pad_b = (self.window_size - H % self.window_size) % self.window_size
    if pad_r > 0 or pad_b > 0:
      x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
    _, Hp, Wp, _ = x.shape

    # Cyclic shift
    if self.shift_size > 0:
      shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
      shifted_x = x

    # Partition windows
    x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

    # W-MSA/SW-MSA
    attn_windows = self.attn(self.norm1(x_windows))  # nW*B, window_size*window_size, C

    # Merge windows
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B, Hp, Wp, C

    # Reverse cyclic shift
    if self.shift_size > 0:
      attn_out = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
      attn_out = shifted_x

    # Remove padding
    if pad_r > 0 or pad_b > 0:
      attn_out = attn_out[:, :H, :W, :].contiguous()

    # Add residual connection after attention (CRITICAL FIX)
    x = x + attn_out

    # FFN with residual
    x = x + self.mlp(self.norm2(x))

    # Reshape back to (B, C, H, W)
    x = x.permute(0, 3, 1, 2).contiguous()

    return x + shortcut


class ConvBlock(nn.Module):
  """Convolutional block with GroupNorm and SiLU activation."""

  def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=8):
    super().__init__()
    self.norm1 = nn.GroupNorm(groups, in_channels)
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    self.norm2 = nn.GroupNorm(groups, out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    self.act = nn.SiLU()

    # Timestep embedding projection
    if time_emb_dim is not None:
      self.time_emb_proj = nn.Linear(time_emb_dim, in_channels * 2)
    else:
      self.time_emb_proj = None

    # Residual connection
    if in_channels != out_channels:
      self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    else:
      self.residual_conv = nn.Identity()

  def forward(self, x, time_emb=None):
    residual = self.residual_conv(x)

    # First conv block
    h = self.norm1(x)

    # Apply timestep conditioning
    if time_emb is not None and self.time_emb_proj is not None:
      time_emb = self.time_emb_proj(time_emb)
      while len(time_emb.shape) < len(h.shape):
        time_emb = time_emb[..., None]
      scale, shift = time_emb.chunk(2, dim=1)
      h = h * (1 + scale) + shift

    h = self.act(h)
    h = self.conv1(h)

    # Second conv block
    h = self.norm2(h)
    h = self.act(h)
    h = self.conv2(h)

    return h + residual


class Projector(nn.Module):
  """
  Projector module for optimizing fused features.
  MLPdown -> GELU -> MLPup
  Uses fixed ratio of 0.5 for dimension reduction.
  """

  def __init__(self, channels):
    super().__init__()
    hidden_channels = channels // 2
    self.mlp_down = nn.Linear(channels, hidden_channels)
    self.gelu = nn.GELU()
    self.mlp_up = nn.Linear(hidden_channels, channels)

  def forward(self, x):
    """
    Args:
        x: Input features (B, C, H, W)
    Returns:
        Output features (B, C, H, W)
    """
    B, C, H, W = x.shape
    identity = x  # Save for residual connection

    # Reshape to (B, H, W, C) for linear layers
    x = x.permute(0, 2, 3, 1).contiguous()

    # Apply projector
    x = self.mlp_down(x)
    x = self.gelu(x)
    x = self.mlp_up(x)

    # Reshape back to (B, C, H, W)
    x = x.permute(0, 3, 1, 2).contiguous()

    # Add residual connection (CRITICAL FIX)
    return x + identity


class CrossAttentionFusion(nn.Module):
  """Cross-attention mechanism for fusing structural and noise features."""

  def __init__(self, channels, num_heads=8):
    super().__init__()
    self.channels = channels
    self.num_heads = num_heads

    # Linear transformations for Q, K, V
    self.query_proj = nn.Linear(channels, channels)
    self.key_proj = nn.Linear(channels, channels)
    self.value_proj = nn.Linear(channels, channels)

    self.scale = (channels // num_heads) ** -0.5
    self.out_proj = nn.Linear(channels, channels)

  def forward(self, x_noise, x_struct):
    """
    Args:
        x_noise: Noise features (B, C, H, W) - Query
        x_struct: Structural features (B, C, H, W) - Key and Value
    Returns:
        Fused features (B, C, H, W)
    """
    B, C, H, W = x_noise.shape

    # Reshape to (B, H*W, C)
    x_noise_flat = x_noise.flatten(2).transpose(1, 2)  # B, H*W, C
    x_struct_flat = x_struct.flatten(2).transpose(1, 2)  # B, H*W, C

    # Compute Q, K, V
    Q = self.query_proj(x_noise_flat)  # B, H*W, C
    K = self.key_proj(x_struct_flat)   # B, H*W, C
    V = self.value_proj(x_struct_flat) # B, H*W, C

    # Reshape for multi-head attention
    Q = Q.reshape(B, H * W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    K = K.reshape(B, H * W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    V = V.reshape(B, H * W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

    # Attention
    attn = (Q @ K.transpose(-2, -1)) * self.scale
    attn = F.softmax(attn, dim=-1)

    # Apply attention to values
    out = attn @ V  # B, num_heads, H*W, head_dim
    out = out.permute(0, 2, 1, 3).reshape(B, H * W, C)
    out = self.out_proj(out)

    # Reshape back to (B, C, H, W)
    out = out.transpose(1, 2).reshape(B, C, H, W)

    # Add residual connection from noise features (CRITICAL FIX)
    # Preserves gradient flow and original noise information
    return out + x_noise


class EncoderStage(nn.Module):
  """Encoder stage with convolutional blocks."""

  def __init__(self, in_channels, out_channels, time_emb_dim, num_blocks=3):
    super().__init__()
    self.blocks = nn.ModuleList([
      ConvBlock(
        in_channels if i == 0 else out_channels,
        out_channels,
        time_emb_dim
      ) for i in range(num_blocks)
    ])

  def forward(self, x, time_emb):
    for block in self.blocks:
      x = block(x, time_emb)
    return x


class SwinEncoderStage(nn.Module):
  """Encoder stage with SwinTransformer blocks with optional timestep conditioning."""

  def __init__(self, in_channels, out_channels, time_emb_dim, num_blocks=3, window_size=7, num_heads=8):
    super().__init__()
    # First adjust channels if needed
    if in_channels != out_channels:
      self.channel_adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    else:
      self.channel_adjust = nn.Identity()

    self.blocks = nn.ModuleList()
    self.time_emb_projs = nn.ModuleList() if time_emb_dim is not None else None

    for i in range(num_blocks):
      # Alternate between W-MSA and SW-MSA
      shift_size = 0 if (i % 2 == 0) else window_size // 2
      self.blocks.append(
        SwinTransformerBlock(
          dim=out_channels,
          num_heads=num_heads,
          window_size=window_size,
          shift_size=shift_size
        )
      )

      # Add timestep conditioning projection for each block (for denoising network)
      if time_emb_dim is not None:
        self.time_emb_projs.append(nn.Linear(time_emb_dim, out_channels * 2))

  def forward(self, x, time_emb):
    x = self.channel_adjust(x)

    for i, block in enumerate(self.blocks):
      # Apply Swin block (which includes LayerNorm internally)
      x = block(x)

      # Apply timestep conditioning AFTER the block (FIXED)
      # This way it's not erased by internal LayerNorm
      if time_emb is not None and self.time_emb_projs is not None:
        # Project time embedding to get scale and shift parameters
        t_emb = self.time_emb_projs[i](time_emb)
        # Reshape for broadcasting: (B, C*2) -> (B, C*2, 1, 1)
        while len(t_emb.shape) < len(x.shape):
          t_emb = t_emb[..., None]
        scale, shift = t_emb.chunk(2, dim=1)
        # Apply adaptive modulation after the block output
        x = x * (1 + scale) + shift

    return x


class DecoderStage(nn.Module):
  """Decoder stage with convolutional blocks."""

  def __init__(self, in_channels, out_channels, time_emb_dim, num_blocks=3):
    super().__init__()
    self.blocks = nn.ModuleList([
      ConvBlock(
        in_channels if i == 0 else out_channels,
        out_channels,
        time_emb_dim
      ) for i in range(num_blocks)
    ])

  def forward(self, x, time_emb):
    for block in self.blocks:
      x = block(x, time_emb)
    return x


class StructureEncoder(nn.Module):
  """
  Structure Encoder Network
  Extracts anatomical information from CT, PTV, OARs (and optionally beam fields).
  Uses Conv-ResNet blocks for early layers and Swin-ResNet blocks for deeper layers.
  """

  def __init__(
    self,
    in_channels,
    base_channels=64,
    channel_mult=(1, 2, 2, 4, 4),
    num_blocks=3,
    window_size=7,
    num_heads=8
  ):
    super().__init__()

    # Initial convolution
    self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

    # Encoder stages
    self.encoder_stages = nn.ModuleList()
    self.downsamples = nn.ModuleList()

    channels = [base_channels * m for m in channel_mult]
    in_ch = base_channels

    for level, out_ch in enumerate(channels):
      # First level uses Conv-ResNet, deeper levels use Swin-ResNet
      if level == 0:
        stage = EncoderStage(in_ch, out_ch, time_emb_dim=None, num_blocks=num_blocks)
      else:
        stage = SwinEncoderStage(in_ch, out_ch, time_emb_dim=None, num_blocks=num_blocks,
                                 window_size=window_size, num_heads=num_heads)

      self.encoder_stages.append(stage)

      # Downsample (except last level)
      if level < len(channels) - 1:
        self.downsamples.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
      else:
        self.downsamples.append(nn.Identity())

      in_ch = out_ch

  def forward(self, x):
    """
    Args:
        x: Structural input (CT + PTV + OARs + optionally beam fields) [B, C, H, W]
    Returns:
        List of feature maps at different scales
    """
    features = []

    x = self.conv_in(x)

    for stage, downsample in zip(self.encoder_stages, self.downsamples):
      x = stage(x, time_emb=None)
      features.append(x)
      x = downsample(x)

    return features


class DenoisingNetwork(nn.Module):
  """
  Denoising Network for SP-DiffDose

  Architecture combining convolution and SwinTransformer:
  - Early layers: Convolutional blocks
  - Deeper layers: SwinTransformer blocks
  - Fusion: Element-wise addition for early layers, cross-attention for last 2 layers
  - Projector after each fusion layer
  """

  def __init__(
    self,
    structure_selector,
    base_channels=64,
    channel_mult=(1, 2, 2, 4, 4, 8),  # Following paper: 64, 128, 128, 256, 256, 512
    num_blocks=3,
    window_size=7,
    num_heads=8,
    use_projector=True
  ):
    super().__init__()

    self.use_projector = use_projector
    self.structure_selector = structure_selector
    self.num_levels = len(channel_mult)

    # Timestep embedding
    time_embed_dim = base_channels * 4
    self.time_embed = nn.Sequential(
      SinusoidalPositionEmbedding(base_channels),
      nn.Linear(base_channels, time_embed_dim),
      nn.SiLU(),
      nn.Linear(time_embed_dim, time_embed_dim)
    )

    # Structure encoder
    in_channels_struct = structure_selector.total_channels
    self.structure_encoder = StructureEncoder(
      in_channels=in_channels_struct,
      base_channels=base_channels,
      channel_mult=channel_mult,
      num_blocks=num_blocks,
      window_size=window_size,
      num_heads=num_heads
    )

    # Initial convolution for noisy input
    self.conv_in_noisy = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)

    # Encoder stages for noisy image
    self.encoder_stages = nn.ModuleList()
    self.downsamples = nn.ModuleList()

    channels = [base_channels * m for m in channel_mult]
    in_ch = base_channels

    for level, out_ch in enumerate(channels):
      # First 2 levels use Conv blocks, deeper levels (2-5) use Swin blocks
      if level < 2:  # First 2 levels use convolution
        stage = EncoderStage(in_ch, out_ch, time_embed_dim, num_blocks)
      else:  # Levels 2-5 use SwinTransformer
        stage = SwinEncoderStage(in_ch, out_ch, time_embed_dim, num_blocks, window_size, num_heads)

      self.encoder_stages.append(stage)

      # Downsample (except last level)
      if level < len(channels) - 1:
        self.downsamples.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
      else:
        self.downsamples.append(nn.Identity())

      in_ch = out_ch

    # Fusion modules
    self.fusion_modules = nn.ModuleList()
    self.projectors = nn.ModuleList() if use_projector else None

    for level, ch in enumerate(channels):
      # Last 2 layers use cross-attention, others use addition
      if level >= len(channels) - 2:
        self.fusion_modules.append(CrossAttentionFusion(ch, num_heads))
      else:
        self.fusion_modules.append(None)  # Will use addition

      # Projector after each fusion
      if use_projector:
        self.projectors.append(Projector(ch))

    # Middle blocks (2 Swin blocks with timestep conditioning)
    self.middle = nn.ModuleList()
    self.middle_time_projs = nn.ModuleList()
    for _ in range(2):
      self.middle.append(SwinTransformerBlock(channels[-1], num_heads, window_size))
      self.middle_time_projs.append(nn.Linear(time_embed_dim, channels[-1] * 2))

    # Decoder stages
    self.decoder_stages = nn.ModuleList()
    self.upsamples = nn.ModuleList()

    reversed_channels = list(reversed(channels))
    for level, out_ch in enumerate(reversed_channels):
      next_ch = reversed_channels[level + 1] if level + 1 < len(reversed_channels) else base_channels

      self.decoder_stages.append(
        DecoderStage(out_ch, next_ch, time_embed_dim, num_blocks)
      )

      # Upsample (except last level)
      if level < len(reversed_channels) - 1:
        self.upsamples.append(nn.ConvTranspose2d(next_ch, next_ch, kernel_size=4, stride=2, padding=1))
      else:
        self.upsamples.append(nn.Identity())

    # Output
    self.norm_out = nn.GroupNorm(8, base_channels)
    self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

  def forward(self, x_noisy, condition, timesteps, cache=None):
    """
    Args:
        x_noisy: Noisy dose map [B, 1, H, W]
        condition: Structural information (CT + PTV + OARs + optionally beam fields) [B, C, H, W]
        timesteps: Diffusion timesteps [B]
        cache: DeepCache (unused, for compatibility)

    Returns:
        Predicted noise [B, 1, H, W]
    """
    # Timestep embedding
    t_emb = self.time_embed(timesteps)

    # Extract structural features
    struct_features = self.structure_encoder(condition)

    # Initial convolution for noisy input
    h = self.conv_in_noisy(x_noisy)

    # Encoder with fusion - track spatial dimensions at each level
    encoder_outputs = []
    spatial_dims = []

    for level in range(self.num_levels):
      # Process noisy image
      h = self.encoder_stages[level](h, t_emb)

      # Fuse with structural features
      struct_feat = struct_features[level]

      # Use cross-attention for last 2 layers, addition for others
      if self.fusion_modules[level] is not None:
        # Cross-attention fusion
        fused = self.fusion_modules[level](h, struct_feat)
      else:
        # Element-wise addition
        fused = h + struct_feat

      # Apply projector
      if self.use_projector:
        fused = self.projectors[level](fused)

      # Update h with fused features
      h = fused

      # Store for skip connections and spatial dimensions
      encoder_outputs.append(h)
      spatial_dims.append((h.shape[2], h.shape[3]))

      # Downsample
      h = self.downsamples[level](h)

    # Middle blocks with timestep conditioning
    for i, block in enumerate(self.middle):
      # Apply Swin block first (contains LayerNorm)
      h = block(h)

      # Apply timestep conditioning AFTER the block (FIXED)
      # This way it's not erased by internal LayerNorm
      t_emb_proj = self.middle_time_projs[i](t_emb)
      while len(t_emb_proj.shape) < len(h.shape):
        t_emb_proj = t_emb_proj[..., None]
      scale, shift = t_emb_proj.chunk(2, dim=1)
      h = h * (1 + scale) + shift

    # Decoder - upsample to match encoder spatial dimensions
    for level in range(len(self.decoder_stages)):
      # No skip connections in SP-DiffDose (based on the paper)
      h = self.decoder_stages[level](h, t_emb)

      # Use transposed conv for upsampling
      h = self.upsamples[level](h)

      # Ensure exact spatial size match with corresponding encoder level
      # Decoder level i should match encoder level (num_levels - 1 - i - 1) = (num_levels - 2 - i)
      # because after the last encoder level we downsample before middle blocks
      if level < len(self.decoder_stages) - 1:  # Skip the last level which has Identity upsample
        encoder_level = self.num_levels - 2 - level
        if encoder_level >= 0 and encoder_level < len(spatial_dims):
          target_h, target_w = spatial_dims[encoder_level]
          if h.shape[2] != target_h or h.shape[3] != target_w:
            h = F.interpolate(h, size=(target_h, target_w), mode='bilinear', align_corners=False)

    # Output
    h = self.norm_out(h)
    h = F.silu(h)
    h = self.conv_out(h)

    return h


def build_spdiffdose(config, structure_selector):
  """
  Build SP-DiffDose model from configuration.

  Default values correspond to the best configuration from the paper (Table II, last row):
  - SwinTransformer: Enabled (in deeper layers 2-5)
  - Structure Encoder: Enabled
  - Projector: Enabled
  - Base channels: 64
  - Channel multipliers: [1, 2, 2, 4, 4, 8] giving 64→128→128→256→256→512 progression
  - Window size: 7
  - Num heads: 8
  - Num blocks: 3 per stage

  Args:
      config: SP-DiffDose configuration dict
      structure_selector: StructureSelector instance for extracting components from conditioning tensor

  Returns:
      DenoisingNetwork model
  """
  model = DenoisingNetwork(
    structure_selector=structure_selector,
    base_channels=config.get('base_channels', 64),
    channel_mult=tuple(config.get('channel_mult', [1, 2, 2, 4, 4, 8])),
    num_blocks=config.get('num_blocks', 3),
    window_size=config.get('window_size', 7),
    num_heads=config.get('num_heads', 8),
    use_projector=config.get('use_projector', True)
  )
  return model
