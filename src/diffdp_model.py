"""
DiffDP: Radiotherapy Dose Prediction via a Diffusion Model

Based on: "DiffDP: Radiotherapy Dose Prediction via a Diffusion Model"
Feng et al., 2023

Key Features:
1. Diffusion-based dose prediction using DDPM framework
2. Structure encoder to extract anatomical information from CT + PTV + OARs
3. Noise predictor based on UNet architecture with cross-attention fusion
4. Element-wise addition for early fusion, cross-attention for deeper levels
5. Addresses over-smoothing problem by using diffusion model without L1/L2 loss

Architecture:
- Structure Encoder: 5 levels with ResBlock and Down blocks
- Noise Predictor: 6-level UNet with encoder, bottleneck, and decoder
- Fusion: First 2 levels use element-wise addition, last 3 use cross-attention
- Timestep conditioning via adaptive group normalization

Following the paper strictly:
- Forward process: Gradually add noise to dose distribution (Equations 1-4)
- Reverse process: Iteratively denoise from Gaussian noise (Equations 5-8)
- Structure encoder pre-trained with L1 loss
- Noise predictor trained to predict added noise (Equation 9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


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


class ConvBlock(nn.Module):
  """
  Convolutional block with GroupNorm and Swish activation.
  Used in ResBlock as described in Section 2.3 of the paper.
  """

  def __init__(self, in_channels, out_channels, groups=8):
    super().__init__()
    self.norm = nn.GroupNorm(groups, in_channels)
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    self.act = nn.SiLU()  # Swish activation

  def forward(self, x):
    x = self.norm(x)
    x = self.act(x)
    x = self.conv(x)
    return x


class ResBlock(nn.Module):
  """
  Residual Block with two ConvBlocks and residual connection.
  As described in Section 2.2 and 2.3 of the paper.
  """

  def __init__(self, in_channels, out_channels, groups=8):
    super().__init__()
    self.conv_block1 = ConvBlock(in_channels, out_channels, groups)
    self.conv_block2 = ConvBlock(out_channels, out_channels, groups)

    # Residual connection
    if in_channels != out_channels:
      self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    else:
      self.residual_conv = nn.Identity()

  def forward(self, x):
    residual = self.residual_conv(x)
    h = self.conv_block1(x)
    h = self.conv_block2(h)
    return h + residual


class DownBlock(nn.Module):
  """
  Downsampling block using 3x3 convolution with stride 2.
  As described in Section 2.2 of the paper.
  """

  def __init__(self, channels):
    super().__init__()
    self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

  def forward(self, x):
    return self.conv(x)


class UpBlock(nn.Module):
  """
  Upsampling block using nearest neighbor upsampling + 1x1 convolution.
  As described in Section 2.3 of the paper.
  """

  def __init__(self, in_channels, out_channels=None):
    super().__init__()
    if out_channels is None:
      out_channels = in_channels
    self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

  def forward(self, x):
    x = self.upsample(x)
    x = self.conv(x)
    return x


class SelfAttention(nn.Module):
  """
  Self-attention module for the bottleneck.
  As mentioned in Section 2.3 of the paper.
  """

  def __init__(self, channels, num_heads=8):
    super().__init__()
    self.channels = channels
    self.num_heads = num_heads
    self.head_dim = channels // num_heads
    assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"

    self.norm = nn.GroupNorm(8, channels)
    self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
    self.proj = nn.Conv2d(channels, channels, kernel_size=1)

  def forward(self, x):
    B, C, H, W = x.shape
    residual = x

    x = self.norm(x)
    qkv = self.qkv(x)  # B, 3*C, H, W
    qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
    qkv = qkv.permute(1, 0, 2, 4, 3)  # 3, B, num_heads, H*W, head_dim
    q, k, v = qkv[0], qkv[1], qkv[2]

    # Attention
    scale = self.head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)

    out = attn @ v  # B, num_heads, H*W, head_dim
    out = out.permute(0, 1, 3, 2)  # B, num_heads, head_dim, H*W
    out = out.reshape(B, C, H, W)
    out = self.proj(out)

    return out + residual


class CrossAttention(nn.Module):
  """
  Cross-attention mechanism for fusing structural and noise features.
  As described in Section 2.3 of the paper: "the last three use a cross-attention
  mechanism to gain similarity-based structure guidance at deeper levels."

  The noise features serve as Query, structural features serve as Key and Value.
  """

  def __init__(self, channels, num_heads=8):
    super().__init__()
    self.channels = channels
    self.num_heads = num_heads
    self.head_dim = channels // num_heads
    assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"

    self.norm_noise = nn.GroupNorm(8, channels)
    self.norm_struct = nn.GroupNorm(8, channels)

    self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
    self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
    self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
    self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

  def forward(self, x_noise, x_struct):
    """
    Args:
        x_noise: Noise features (Query) [B, C, H, W]
        x_struct: Structural features (Key, Value) [B, C, H, W]
    Returns:
        Fused features [B, C, H, W]
    """
    B, C, H, W = x_noise.shape

    # Normalize
    x_noise_norm = self.norm_noise(x_noise)
    x_struct_norm = self.norm_struct(x_struct)

    # Project to Q, K, V
    q = self.q_proj(x_noise_norm)  # B, C, H, W
    k = self.k_proj(x_struct_norm)  # B, C, H, W
    v = self.v_proj(x_struct_norm)  # B, C, H, W

    # Reshape for multi-head attention
    q = q.reshape(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # B, num_heads, H*W, head_dim
    k = k.reshape(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # B, num_heads, H*W, head_dim
    v = v.reshape(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # B, num_heads, H*W, head_dim

    # Attention
    scale = self.head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale  # B, num_heads, H*W, H*W
    attn = F.softmax(attn, dim=-1)

    out = attn @ v  # B, num_heads, H*W, head_dim
    out = out.transpose(2, 3).reshape(B, C, H, W)
    out = self.out_proj(out)

    return out


class StructureEncoder(nn.Module):
  """
  Structure Encoder to extract anatomical information from CT + PTV + OARs.

  As described in Section 2.2 of the paper:
  - 5 operation steps (levels)
  - Each step has: ResBlock + Down block (except last level)
  - ResBlock: 2 ConvBlocks with residual connection
  - Down block: 3x3 Conv with stride 2
  - Pre-trained with L1 loss

  Architecture follows Fig. 2 in the paper.
  """

  def __init__(self, in_channels, base_channels=64, num_blocks_per_level=1):
    super().__init__()

    # Channel progression: 64 -> 128 -> 256 -> 512 -> 1024
    # Following typical UNet structure
    channels = [base_channels, base_channels * 2, base_channels * 4,
                base_channels * 8, base_channels * 16]

    self.num_blocks_per_level = num_blocks_per_level

    # Initial convolution
    self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

    # 5 levels - each level can have multiple ResBlocks
    self.encoder_stages = nn.ModuleList()
    self.down_blocks = nn.ModuleList()

    for i in range(5):
      in_ch = channels[i - 1] if i > 0 else base_channels
      out_ch = channels[i]

      # Create stage with multiple ResBlocks
      stage_blocks = nn.ModuleList()
      for j in range(num_blocks_per_level):
        block_in_ch = in_ch if j == 0 else out_ch
        stage_blocks.append(ResBlock(block_in_ch, out_ch))
      self.encoder_stages.append(stage_blocks)

      # Down block for first 4 levels (no downsampling at last level)
      if i < 4:
        self.down_blocks.append(DownBlock(out_ch))
      else:
        self.down_blocks.append(nn.Identity())

  def forward(self, x):
    """
    Args:
        x: Structure image (CT + PTV + OARs) [B, in_channels, H, W]
    Returns:
        List of feature maps at different levels [feat_0, feat_1, ..., feat_4]
    """
    features = []

    x = self.conv_in(x)

    for stage_blocks, down_block in zip(self.encoder_stages, self.down_blocks):
      # Apply all ResBlocks in this stage
      for res_block in stage_blocks:
        x = res_block(x)
      features.append(x)
      x = down_block(x)

    return features


class NoisePredictor(nn.Module):
  """
  Noise Predictor - 6-level UNet with timestep conditioning.

  As described in Section 2.3 and Fig. 2 of the paper:
  - 6-level UNet architecture
  - Encoder: 6 down-sampling operations
  - Bottleneck: 2 ResBlocks + self-attention
  - Decoder: 5 up-sampling operations
  - Fusion with structure encoder features:
    - First 2 encoder levels (0, 1): element-wise addition
    - Last 3 encoder levels (2, 3, 4): cross-attention
  - Skip connections between encoder and decoder
  - Timestep conditioning via adaptive group normalization
  """

  def __init__(self, structure_encoder, base_channels=64, num_heads=8, num_blocks_per_level=1):
    super().__init__()

    self.structure_encoder = structure_encoder
    self.num_heads = num_heads
    self.num_blocks_per_level = num_blocks_per_level

    # Channel progression for 6 levels: 64 -> 128 -> 256 -> 512 -> 1024 -> 2048
    # Note: Paper doesn't specify exact channels, using standard progression
    channels = [base_channels, base_channels * 2, base_channels * 4,
                base_channels * 8, base_channels * 16, base_channels * 32]

    # Timestep embedding
    time_embed_dim = base_channels * 4
    self.time_embed = nn.Sequential(
      SinusoidalPositionEmbedding(base_channels),
      nn.Linear(base_channels, time_embed_dim),
      nn.SiLU(),
      nn.Linear(time_embed_dim, time_embed_dim)
    )

    # Time embedding projections for each encoder level
    # We need one projection per ResBlock per level
    self.time_projs_encoder = nn.ModuleList()
    for i in range(6):
      level_projs = nn.ModuleList([
        nn.Linear(time_embed_dim, channels[i] * 2) for _ in range(num_blocks_per_level)
      ])
      self.time_projs_encoder.append(level_projs)

    # Initial convolution for noisy dose
    self.conv_in = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)

    # Encoder: 6 levels - each level can have multiple ResBlocks
    self.encoder_stages = nn.ModuleList()
    self.encoder_downs = nn.ModuleList()

    for i in range(6):
      in_ch = channels[i - 1] if i > 0 else base_channels
      out_ch = channels[i]

      # Create stage with multiple ResBlocks
      stage_blocks = nn.ModuleList()
      for j in range(num_blocks_per_level):
        block_in_ch = in_ch if j == 0 else out_ch
        stage_blocks.append(ResBlock(block_in_ch, out_ch))
      self.encoder_stages.append(stage_blocks)

      # Downsampling for first 5 levels
      if i < 5:
        self.encoder_downs.append(DownBlock(out_ch))
      else:
        self.encoder_downs.append(nn.Identity())

    # Fusion strategy from paper Section 2.3:
    # "The following two down-sampling operations retain the addition operation
    # to complete information fusion, while the last three use a cross-attention mechanism"
    # Structure encoder has 5 levels (0-4), noise predictor has 6 levels (0-5)
    # First 2 levels (0, 1) use element-wise addition
    # Last 3 levels (2, 3, 4) use cross-attention
    self.cross_attentions = nn.ModuleList([
      None,  # Level 0: addition
      None,  # Level 1: addition
      CrossAttention(channels[2], num_heads),  # Level 2: cross-attention
      CrossAttention(channels[3], num_heads),  # Level 3: cross-attention
      CrossAttention(channels[4], num_heads),  # Level 4: cross-attention
      # Level 5 doesn't need fusion (bottleneck level, no structure encoder feature)
    ])

    # Bottleneck: 2 ResBlocks + self-attention
    self.bottleneck = nn.ModuleList([
      ResBlock(channels[5], channels[5]),
      SelfAttention(channels[5], num_heads),
      ResBlock(channels[5], channels[5])
    ])

    # Time embedding projections for bottleneck
    self.time_projs_bottleneck = nn.ModuleList([
      nn.Linear(time_embed_dim, channels[5] * 2) for _ in range(2)
    ])

    # Decoder: 5 levels - each level can have multiple ResBlocks
    self.decoder_stages = nn.ModuleList()
    self.decoder_ups = nn.ModuleList()
    self.time_projs_decoder = nn.ModuleList()

    # Track output channels from previous level (starts at bottleneck)
    prev_out_ch = channels[5]  # 2048 from bottleneck

    for i in range(5):
      # Skip connection channels from encoder at this level
      skip_ch = channels[5 - i - 1]  # 1024, 512, 256, 128, 64
      # Output channels after ResBlocks
      next_ch = channels[5 - i - 2] if i < 4 else base_channels  # 512, 256, 128, 64, 64

      # Upsampling: prev_out_ch → skip_ch (to match skip connection for addition)
      self.decoder_ups.append(UpBlock(prev_out_ch, skip_ch))

      # Multiple ResBlocks per decoder level: skip_ch → next_ch
      stage_blocks = nn.ModuleList()
      stage_time_projs = nn.ModuleList()
      for j in range(num_blocks_per_level):
        block_in_ch = skip_ch if j == 0 else next_ch
        stage_blocks.append(ResBlock(block_in_ch, next_ch))
        stage_time_projs.append(nn.Linear(time_embed_dim, next_ch * 2))

      self.decoder_stages.append(stage_blocks)
      self.time_projs_decoder.append(stage_time_projs)

      # Update for next iteration
      prev_out_ch = next_ch

    # Output
    self.norm_out = nn.GroupNorm(8, base_channels)
    self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

  def _apply_time_conditioning(self, x, time_emb, time_proj):
    """Apply timestep conditioning via adaptive group normalization."""
    t_emb = time_proj(time_emb)
    while len(t_emb.shape) < len(x.shape):
      t_emb = t_emb[..., None]
    scale, shift = t_emb.chunk(2, dim=1)
    return x * (1 + scale) + shift

  def forward(self, x_noisy, x_struct_features, time_emb):
    """
    Args:
        x_noisy: Noisy dose map [B, 1, H, W]
        x_struct_features: List of structural features from structure encoder
        time_emb: Timestep embedding [B, time_embed_dim]
    Returns:
        Predicted noise [B, 1, H, W]
    """
    # Initial convolution
    h = self.conv_in(x_noisy)

    # Encoder with fusion
    encoder_features = []

    for level in range(6):
      # Apply all ResBlocks in this encoder stage
      for block_idx, block in enumerate(self.encoder_stages[level]):
        # Apply ResBlock first
        h = block(h)
        # Then apply timestep conditioning (with correct output channels)
        h = self._apply_time_conditioning(h, time_emb, self.time_projs_encoder[level][block_idx])

      # Fuse with structural features (only for first 5 levels)
      if level < 5:
        struct_feat = x_struct_features[level]

        # First 2 levels (0, 1): element-wise addition
        # Last 3 levels (2, 3, 4): cross-attention
        if level < 2:
          # Element-wise addition
          h = h + struct_feat
        else:
          # Cross-attention fusion
          h = h + self.cross_attentions[level](h, struct_feat)

      # Store for skip connections
      encoder_features.append(h)

      # Downsample
      h = self.encoder_downs[level](h)

    # Bottleneck
    for i, block in enumerate(self.bottleneck):
      h = block(h)
      # Apply timestep conditioning after ResBlocks (not after self-attention)
      if isinstance(block, ResBlock):
        idx = i // 2  # 0 for first ResBlock, 1 for second (skipping self-attention at idx 1)
        if idx < len(self.time_projs_bottleneck):
          h = self._apply_time_conditioning(h, time_emb, self.time_projs_bottleneck[idx])

    # Decoder with skip connections
    for level in range(5):
      # Upsample
      h = self.decoder_ups[level](h)

      # Skip connection from encoder
      # Level i of decoder connects to encoder level (5 - i - 1)
      skip = encoder_features[5 - level - 1]
      h = h + skip

      # Multiple ResBlocks with timestep conditioning
      for i, res_block in enumerate(self.decoder_stages[level]):
        # Apply ResBlock first
        h = res_block(h)
        # Then apply timestep conditioning (with correct output channels)
        h = self._apply_time_conditioning(h, time_emb, self.time_projs_decoder[level][i])

    # Output
    h = self.norm_out(h)
    h = F.silu(h)
    h = self.conv_out(h)

    return h


class DiffDP(nn.Module):
  """
  DiffDP: Radiotherapy Dose Prediction via a Diffusion Model

  Complete model combining Structure Encoder and Noise Predictor.
  Implements the DDPM framework for dose prediction as described in the paper.

  The model:
  1. Extracts anatomical features using Structure Encoder
  2. Predicts noise using Noise Predictor with timestep conditioning
  3. Fuses features using element-wise addition (early levels) and
     cross-attention (deeper levels)
  """

  def __init__(self, structure_selector, base_channels=64, num_heads=8, num_blocks_per_level=1):
    super().__init__()

    self.structure_selector = structure_selector
    in_channels = structure_selector.total_channels

    # Structure Encoder
    self.structure_encoder = StructureEncoder(in_channels, base_channels, num_blocks_per_level)

    # Noise Predictor
    self.noise_predictor = NoisePredictor(self.structure_encoder, base_channels, num_heads, num_blocks_per_level)

  def forward(self, x_noisy, condition, timesteps, cache=None):
    """
    Forward pass compatible with DiffusionModelTrainerModule.

    Args:
        x_noisy: Noisy dose map [B, 1, H, W]
        condition: Structural information (CT + PTV + OARs) [B, C, H, W]
        timesteps: Diffusion timesteps [B]
        cache: DeepCache (unused, for compatibility)

    Returns:
        Predicted noise [B, 1, H, W]
    """
    # Timestep embedding
    time_emb = self.noise_predictor.time_embed(timesteps)

    # Extract structural features
    struct_features = self.structure_encoder(condition)

    # Predict noise
    noise_pred = self.noise_predictor(x_noisy, struct_features, time_emb)

    return noise_pred


def build_diffdp(config, structure_selector):
  """
  Build DiffDP model from configuration.

  Default values correspond to the configuration from the paper:
  - Base channels: 64
  - Number of ResBlocks per level: 2 (Section 2.3: "each contains an Up block and two ResBlock")
  - 6-level UNet for noise predictor
  - 5-level structure encoder
  - 8 attention heads for cross-attention and self-attention
  - Element-wise addition for first 2 encoder levels (0, 1)
  - Cross-attention for last 3 encoder levels (2, 3, 4)

  Training settings from paper Section 2.5:
  - T = 1000 diffusion steps
  - Beta schedule: linear from 1e-4 to 0.02
  - Batch size: 16
  - Optimizer: Adam with learning rate 1e-4
  - Learning rate decay: 1e-4 → 5e-5 after epoch 1200
  - Total epochs: 1500
  - Image size: 256x256

  Args:
      config: DiffDP configuration dict
      structure_selector: StructureSelector instance

  Returns:
      DiffDP model
  """
  model = DiffDP(
    structure_selector=structure_selector,
    base_channels=config.get('base_channels', 64),
    num_heads=config.get('num_heads', 8),
    num_blocks_per_level=config.get('num_blocks_per_level', 2)
  )
  return model
