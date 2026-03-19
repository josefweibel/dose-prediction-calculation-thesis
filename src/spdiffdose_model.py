"""
SP-DiffDose: A Conditional Diffusion Model for Radiation Dose Prediction
Based on Multi-Scale Fusion of Anatomical Structures, Guided by SwinTransformer and Projector

Based on: "SP-DiffDose: A Conditional Diffusion Model for Radiation Dose Prediction
Based on Multi-Scale Fusion of Anatomical Structures, Guided by SwinTransformer and Projector"
Fu et al., IEEE Transactions and Journals Template, 2023

Key Features:
1. Diffusion-based dose prediction using conditional DDPM framework
2. Structure encoder combining Conv-ResNet and Swin-ResNet blocks
3. Denoising network with SwinTransformer for capturing multi-scale features
4. Projector module (MLPdown + GELU + MLPup) for optimizing fused features
5. Multi-scale fusion: element-wise addition for first 4 levels, cross-attention for last 2 levels
6. Progressive feature refinement from simple to complex

Architecture (from paper Section III.B):
- Structure Encoder:
  - 1 Conv-ResNet block (3 conv blocks + 1 down) for early local features
  - 4 Swin-ResNet blocks (3 SwinTransformer blocks + 1 down) for global features
  - 5 downsampling stages total

- Denoising Network:
  - Encoder: 5 levels with Conv (first) + Swin (last 4) blocks
  - 2 intermediate SwinTransformer blocks (no up/downsample)
  - Decoder: 5 levels with Conv + Swin blocks
  - Fusion: Element-wise addition for levels 0-3, cross-attention for levels 4-5
  - Projector after each fusion

Training settings (from paper Section IV.B):
- T = 1000 diffusion steps
- Beta schedule: α_t ∈ (0, 1), α̅_t = ∏ᵗₛ₌₁ αₛ
- Batch size: 8
- Optimizer: Adam with initial learning rate 1e-4
- Learning rate decay: linear from 1e-4 to 0 after epoch 100-200
- Total epochs: 200
- Image size: 256x256
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re


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


class Projector(nn.Module):
  """
  Projector module for optimizing fused features.

  As described in Section III.B.3 of the paper:
  "Our projector is composed of merely two multi-layer perceptrons (MLPs)
  and a Gaussian error linear unit (GELU) activation function."

  Equation 14: F_o = MLP_up(GELU(MLP_down(F_i)))

  MLPdown projects features to a lower dimension (down_ratio)
  MLPup projects back to match transformer feature dimensions
  """

  def __init__(self, channels, down_ratio=4):
    super().__init__()
    hidden_dim = channels // down_ratio

    self.mlp_down = nn.Linear(channels, hidden_dim)
    self.gelu = nn.GELU()
    self.mlp_up = nn.Linear(hidden_dim, channels)

  def forward(self, x):
    """
    Args:
        x: Input features [B, C, H, W]
    Returns:
        Projected features [B, C, H, W]
    """
    B, C, H, W = x.shape

    # Reshape to [B, H, W, C] for MLPs
    x = x.permute(0, 2, 3, 1)

    # Apply projector: MLPdown -> GELU -> MLPup
    x = self.mlp_down(x)
    x = self.gelu(x)
    x = self.mlp_up(x)

    # Reshape back to [B, C, H, W]
    x = x.permute(0, 3, 1, 2)

    return x


class SwinTransformerBlock(nn.Module):
  """
  Swin Transformer Block with shifted window attention.

  As described in Section III.B.1 of the paper:
  - Divides input into non-overlapping windows of size W×W
  - Calculates self-attention within each window
  - Uses cyclic shifting between consecutive blocks
  - Followed by MLP with ReLU activation

  Equations 3-8 from the paper:
  Q_i = W_q · x_i
  K_i = W_k · x_i
  V_i = W_v · x_i
  a_ij = softmax(Q_i · K_j^T / √d_k) · V_j
  h_i = Σ_j a_ij · V_j
  y_i = W_2 · relu(W_1 · h_i + b_1) + b_2
  """

  def __init__(self, channels, window_size=7, num_heads=8, shift_size=0, mlp_ratio=4.0):
    super().__init__()
    self.channels = channels
    self.window_size = window_size
    self.shift_size = shift_size
    self.num_heads = num_heads
    self.head_dim = channels // num_heads

    assert channels % num_heads == 0, "channels must be divisible by num_heads"

    # Layer normalization
    self.norm1 = nn.LayerNorm(channels)
    self.norm2 = nn.LayerNorm(channels)

    # Window-based multi-head self attention
    self.qkv = nn.Linear(channels, channels * 3)
    self.proj = nn.Linear(channels, channels)

    # MLP
    mlp_hidden_dim = int(channels * mlp_ratio)
    self.mlp = nn.Sequential(
      nn.Linear(channels, mlp_hidden_dim),
      nn.ReLU(),  # Paper uses relu in Equation 8
      nn.Linear(mlp_hidden_dim, channels)
    )

  def forward(self, x):
    """
    Args:
        x: Input features [B, C, H, W]
    Returns:
        Output features [B, C, H, W]
    """
    B, C, H, W = x.shape

    # Reshape to [B, H, W, C]
    x = x.permute(0, 2, 3, 1)

    # Save for residual connection
    shortcut = x

    # Layer norm
    x = self.norm1(x)

    # Cyclic shift (if shift_size > 0)
    if self.shift_size > 0:
      x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

    # Partition into windows
    x_windows, (H_pad, W_pad) = self.window_partition(x, self.window_size)  # [B*num_windows, window_size, window_size, C]
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [B*num_windows, N, C]

    # Window-based multi-head self attention
    N = self.window_size * self.window_size
    qkv = self.qkv(x_windows)  # [B*num_windows, N, 3*C]
    qkv = qkv.reshape(-1, N, 3, self.num_heads, self.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*num_windows, num_heads, N, head_dim]
    q, k, v = qkv[0], qkv[1], qkv[2]

    # Attention (Equation 6)
    scale = self.head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale  # [B*num_windows, num_heads, N, N]
    attn = F.softmax(attn, dim=-1)

    # Apply attention to values (Equation 7)
    x_windows = (attn @ v).transpose(1, 2).reshape(-1, N, C)
    x_windows = self.proj(x_windows)

    # Merge windows back
    x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
    x = self.window_reverse(x_windows, self.window_size, H, W, H_pad, W_pad)

    # Reverse cyclic shift
    if self.shift_size > 0:
      x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

    # Residual connection
    x = shortcut + x

    # MLP with residual connection (Equation 8)
    x = x + self.mlp(self.norm2(x))

    # Reshape back to [B, C, H, W]
    x = x.permute(0, 3, 1, 2)

    return x

  def window_partition(self, x, window_size):
    """
    Partition input into non-overlapping windows.
    If H or W is not divisible by window_size, pad the feature map.

    Args:
        x: [B, H, W, C]
        window_size: Window size
    Returns:
        windows: [B*num_windows, window_size, window_size, C]
        (H_pad, W_pad): Padded dimensions
    """
    B, H, W, C = x.shape

    # Calculate padding needed
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    # Pad if needed
    if pad_h > 0 or pad_w > 0:
      x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # Pad right and bottom

    H_pad, W_pad = H + pad_h, W + pad_w

    # Partition into windows
    x = x.view(B, H_pad // window_size, window_size, W_pad // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (H_pad, W_pad)

  def window_reverse(self, windows, window_size, H, W, H_pad, W_pad):
    """
    Reverse window partition and remove padding.

    Args:
        windows: [B*num_windows, window_size, window_size, C]
        window_size: Window size
        H: Original height
        W: Original width
        H_pad: Padded height
        W_pad: Padded width
    Returns:
        x: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H_pad * W_pad / window_size / window_size))
    x = windows.view(B, H_pad // window_size, W_pad // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, -1)

    # Remove padding
    if H_pad > H or W_pad > W:
      x = x[:, :H, :W, :]

    return x


class ConvBlock(nn.Module):
  """
  Convolutional block with GroupNorm and SiLU activation.
  Used in Conv-ResNet blocks.
  """

  def __init__(self, in_channels, out_channels, groups=8):
    super().__init__()
    self.norm = nn.GroupNorm(groups, in_channels)
    self.act = nn.SiLU()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

  def forward(self, x):
    x = self.norm(x)
    x = self.act(x)
    x = self.conv(x)
    return x


class ResNetBlock(nn.Module):
  """
  ResNet block with GroupNorm and SiLU activation.
  Includes timestep conditioning via adaptive scaling and shifting.
  """

  def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=8):
    super().__init__()

    self.conv_block1 = ConvBlock(in_channels, out_channels, groups)
    self.conv_block2 = ConvBlock(out_channels, out_channels, groups)

    # Timestep embedding projection
    if time_emb_dim is not None:
      self.time_emb_proj = nn.Linear(time_emb_dim, out_channels * 2)
    else:
      self.time_emb_proj = None

    # Residual connection
    if in_channels != out_channels:
      self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    else:
      self.residual_conv = nn.Identity()

  def forward(self, x, time_emb=None):
    residual = self.residual_conv(x)

    h = self.conv_block1(x)

    # Apply timestep conditioning
    if time_emb is not None and self.time_emb_proj is not None:
      time_emb = self.time_emb_proj(time_emb)
      while len(time_emb.shape) < len(h.shape):
        time_emb = time_emb[..., None]
      scale, shift = time_emb.chunk(2, dim=1)
      h = h * (1 + scale) + shift

    h = self.conv_block2(h)

    return h + residual


class CrossAttention(nn.Module):
  """
  Cross-attention mechanism for fusing structural and noise features.

  As described in Section III.B.2 of the paper:
  "To enhance the dose prediction distribution for organs at risk, SP-DiffDose
  utilizes cross-attention mechanisms in the last two layers."

  Equations 9-13 from the paper:
  Q_A = A·W_Q^A
  K_B = B·W_K^B
  V_B = B·W_V^B
  AttentionWeights = Softmax(Q_A·K_B^T / √d_k)
  Output = AttentionWeights × V_B
  """

  def __init__(self, channels, num_heads=8):
    super().__init__()
    self.channels = channels
    self.num_heads = num_heads
    self.head_dim = channels // num_heads

    assert channels % num_heads == 0, "channels must be divisible by num_heads"

    self.norm_noise = nn.GroupNorm(8, channels)
    self.norm_struct = nn.GroupNorm(8, channels)

    # Q from noise (A), K and V from structure (B)
    self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
    self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
    self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
    self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

  def forward(self, x_noise, x_struct):
    """
    Args:
        x_noise: Noise features (Query, A) [B, C, H, W]
        x_struct: Structural features (Key, Value, B) [B, C, H, W]
    Returns:
        Fused features [B, C, H, W]
    """
    B, C, H, W = x_noise.shape

    # Normalize (Equations 9-11)
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

    # Attention (Equation 12)
    scale = self.head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale  # B, num_heads, H*W, H*W
    attn = F.softmax(attn, dim=-1)

    # Apply attention to values (Equation 13)
    out = attn @ v  # B, num_heads, H*W, head_dim
    out = out.transpose(2, 3).reshape(B, C, H, W)
    out = self.out_proj(out)

    return out


class MultiHeadSelfAttention(nn.Module):
  """Multi-Head Self-Attention module for use in ResNet-based encoder/decoder."""

  def __init__(self, channels, num_heads=8):
    super().__init__()
    assert channels % num_heads == 0
    self.channels = channels
    self.num_heads = num_heads
    self.head_dim = channels // num_heads

    self.norm = nn.GroupNorm(8, channels)
    self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
    self.proj = nn.Conv2d(channels, channels, kernel_size=1)

  def forward(self, x):
    B, C, H, W = x.shape
    residual = x

    x = self.norm(x)
    qkv = self.qkv(x)

    qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
    qkv = qkv.permute(1, 0, 2, 4, 3)  # 3, B, num_heads, H*W, head_dim
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    attn = F.softmax(attn, dim=-1)

    out = torch.matmul(attn, v)  # B, num_heads, H*W, head_dim
    out = out.transpose(2, 3).reshape(B, C, H, W)
    out = self.proj(out)

    return out + residual


class StructureEncoder(nn.Module):
  """
  Configurable Structure Encoder to extract anatomical information.

  Supports:
  - use_swin=True: Level 0 uses ConvBlocks, levels 1+ use SwinTransformer (original)
  - use_swin=False: All levels use ConvBlocks + optional self-attention
  - Configurable number of levels (via channels list length)
  - Configurable number of blocks per level
  """

  def __init__(self, in_channels, channels, num_blocks=3, use_swin=True,
               attention_levels=(), num_heads=8, window_size=7):
    super().__init__()

    num_levels = len(channels)
    attention_set = set(attention_levels)

    self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

    self.channel_matches = nn.ModuleList()
    self.level_blocks = nn.ModuleList()
    self.downs = nn.ModuleList()

    for i in range(num_levels):
      in_ch = channels[i - 1] if i > 0 else channels[0]
      out_ch = channels[i]

      # Channel matching
      self.channel_matches.append(
        nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
      )

      # Blocks: level 0 always uses conv, higher levels use swin if enabled
      blocks = nn.ModuleList()
      if use_swin and i > 0:
        for j in range(num_blocks):
          blocks.append(SwinTransformerBlock(
            out_ch, window_size, num_heads,
            shift_size=0 if j % 2 == 0 else window_size // 2
          ))
      else:
        for _ in range(num_blocks):
          blocks.append(ConvBlock(out_ch, out_ch))
        if i in attention_set:
          blocks.append(MultiHeadSelfAttention(out_ch, num_heads))

      self.level_blocks.append(blocks)

      # Downsampling (none at last level)
      self.downs.append(
        nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1) if i < num_levels - 1
        else nn.Identity()
      )

  def forward(self, x):
    features = []
    x = self.conv_in(x)

    for ch_match, blocks, down in zip(self.channel_matches, self.level_blocks, self.downs):
      x = ch_match(x)
      for block in blocks:
        x = block(x)
      features.append(x)
      x = down(x)

    return features


class DenoisingNetwork(nn.Module):
  """
  Configurable Denoising Network.

  Supports:
  - use_swin: Swin blocks (original) or ResNet + optional self-attention
  - Configurable fusion: cross-attention levels, optional projectors
  - Configurable depth (num_levels via channels list) and width (channel values)
  - Configurable bottleneck attention
  - When struct_features=None in forward (merged mode), skips fusion entirely
  """

  def __init__(self, in_channels, channels, num_blocks=3, num_heads=8,
               use_swin=True, attention_levels=(), cross_attention_levels=(4,),
               use_projector=True, projector_levels=None,
               bottleneck_attention=True, window_size=7):
    super().__init__()

    num_levels = len(channels)
    base_channels = channels[0]
    time_embed_dim = base_channels * 4
    attention_set = set(attention_levels)
    self.cross_attention_levels = set(cross_attention_levels)

    # Determine which levels get projectors
    if projector_levels is not None:
      self.projector_levels = set(projector_levels)
    elif use_projector:
      self.projector_levels = set(range(num_levels))
    else:
      self.projector_levels = set()

    # Timestep embedding
    self.time_embed = nn.Sequential(
      SinusoidalPositionEmbedding(base_channels),
      nn.Linear(base_channels, time_embed_dim),
      nn.SiLU(),
      nn.Linear(time_embed_dim, time_embed_dim)
    )

    # Initial convolution
    self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

    # Encoder
    self.encoder_channel_matches = nn.ModuleList()
    self.encoder_blocks = nn.ModuleList()
    self.encoder_downs = nn.ModuleList()

    for i in range(num_levels):
      in_ch = channels[i - 1] if i > 0 else base_channels
      out_ch = channels[i]

      self.encoder_channel_matches.append(
        nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
      )

      blocks = nn.ModuleList()
      if use_swin and i > 0:
        for j in range(num_blocks):
          blocks.append(SwinTransformerBlock(
            out_ch, window_size, num_heads,
            shift_size=0 if j % 2 == 0 else window_size // 2
          ))
      else:
        for _ in range(num_blocks):
          blocks.append(ResNetBlock(out_ch, out_ch, time_embed_dim))
        if i in attention_set:
          blocks.append(MultiHeadSelfAttention(out_ch, num_heads))

      self.encoder_blocks.append(blocks)
      self.encoder_downs.append(
        nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1) if i < num_levels - 1
        else nn.Identity()
      )

    # Fusion modules (only used in separate mode)
    self.cross_attentions = nn.ModuleList([
      CrossAttention(channels[i], num_heads) if i in self.cross_attention_levels else None
      for i in range(num_levels)
    ])
    self.projectors = nn.ModuleList([
      Projector(channels[i]) if i in self.projector_levels else None
      for i in range(num_levels)
    ])

    # Middle block
    mid_ch = channels[-1]
    self.middle = nn.ModuleList()
    if use_swin and bottleneck_attention:
      self.middle.append(SwinTransformerBlock(mid_ch, window_size, num_heads, shift_size=0))
      self.middle.append(SwinTransformerBlock(mid_ch, window_size, num_heads, shift_size=window_size // 2))
    elif not use_swin:
      self.middle.append(ResNetBlock(mid_ch, mid_ch, time_embed_dim))
      if bottleneck_attention:
        self.middle.append(MultiHeadSelfAttention(mid_ch, num_heads))
      self.middle.append(ResNetBlock(mid_ch, mid_ch, time_embed_dim))
    else:
      # use_swin=True but bottleneck_attention=False
      self.middle.append(ResNetBlock(mid_ch, mid_ch, time_embed_dim))
      self.middle.append(ResNetBlock(mid_ch, mid_ch, time_embed_dim))

    # Decoder
    self.decoder_blocks = nn.ModuleList()
    self.decoder_ups = nn.ModuleList()

    for i in range(num_levels):
      level = num_levels - 1 - i
      cur_ch = channels[level]
      next_ch = channels[level - 1] if level > 0 else base_channels

      blocks = nn.ModuleList()
      if use_swin and level > 0:
        for j in range(num_blocks):
          blocks.append(SwinTransformerBlock(
            cur_ch, window_size, num_heads,
            shift_size=0 if j % 2 == 0 else window_size // 2
          ))
      else:
        for _ in range(num_blocks):
          blocks.append(ResNetBlock(cur_ch, cur_ch, time_embed_dim))
        if level in attention_set:
          blocks.append(MultiHeadSelfAttention(cur_ch, num_heads))

      self.decoder_blocks.append(blocks)
      self.decoder_ups.append(
        nn.ConvTranspose2d(cur_ch, next_ch, 4, stride=2, padding=1) if i < num_levels - 1
        else nn.Identity()
      )

    # Output
    self.norm_out = nn.GroupNorm(8, base_channels)
    self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

  def forward(self, x, timesteps, struct_features=None):
    """
    Args:
        x: Input tensor [B, in_channels, H, W] (noisy dose or merged input)
        timesteps: Diffusion timesteps [B]
        struct_features: Optional list of structure features for fusion (separate mode)
    Returns:
        Predicted noise [B, 1, H, W]
    """
    time_emb = self.time_embed(timesteps)
    h = self.conv_in(x)

    # Encoder
    encoder_features = []
    encoder_sizes = []
    num_levels = len(self.encoder_blocks)

    for level in range(num_levels):
      h = self.encoder_channel_matches[level](h)

      for block in self.encoder_blocks[level]:
        if isinstance(block, ResNetBlock):
          h = block(h, time_emb)
        else:
          h = block(h)

      # Fusion with structure features (separate mode only)
      if struct_features is not None:
        sf = struct_features[level]
        if self.cross_attentions[level] is not None:
          h = h + self.cross_attentions[level](h, sf)
        else:
          h = h + sf

        if self.projectors[level] is not None:
          h = self.projectors[level](h)

      encoder_features.append(h)
      encoder_sizes.append(h.shape[2:])

      if level < num_levels - 1:
        h = self.encoder_downs[level](h)

    # Middle
    for block in self.middle:
      if isinstance(block, ResNetBlock):
        h = block(h, time_emb)
      else:
        h = block(h)

    # Decoder
    for i in range(num_levels):
      skip = encoder_features[num_levels - 1 - i]
      target_size = encoder_sizes[num_levels - 1 - i]

      if h.shape[2:] != target_size:
        h = F.interpolate(h, size=target_size, mode='bilinear', align_corners=False)

      h = h + skip

      for block in self.decoder_blocks[i]:
        if isinstance(block, ResNetBlock):
          h = block(h, time_emb)
        else:
          h = block(h)

      if i < num_levels - 1:
        next_size = encoder_sizes[num_levels - 2 - i]
        h = self.decoder_ups[i](h)
        if h.shape[2:] != next_size:
          h = F.interpolate(h, size=next_size, mode='bilinear', align_corners=False)

    h = self.norm_out(h)
    h = F.silu(h)
    h = self.conv_out(h)
    return h


class SPDiffDose(nn.Module):
  """
  SP-DiffDose with configurable architecture for ablation studies.

  Encoder modes:
  - 'separate': Dedicated StructureEncoder + DenoisingNetwork with fusion (original)
  - 'merged': Concatenate [noisy, condition] and use a single encoder (faster)

  Block types:
  - use_swin=True: SwinTransformer blocks at levels 1+ (original)
  - use_swin=False: ResNet blocks + optional self-attention (faster)
  """

  def __init__(self, structure_selector, base_channels=64, channel_mult=(1, 2, 4, 8, 8),
               num_blocks=3, num_heads=8, use_swin=True, attention_levels=(),
               cross_attention_levels=(4,), use_projector=True, projector_levels=None,
               encoder_mode='separate', bottleneck_attention=True, window_size=7):
    super().__init__()

    self.structure_selector = structure_selector
    self.encoder_mode = encoder_mode

    channels = [base_channels * m for m in channel_mult]
    in_channels_struct = structure_selector.total_channels

    if encoder_mode == 'separate':
      self.structure_encoder = StructureEncoder(
        in_channels_struct, channels, num_blocks, use_swin,
        attention_levels, num_heads, window_size
      )
      denoising_in = 1
    else:  # merged
      self.structure_encoder = None
      denoising_in = 1 + in_channels_struct
      # No fusion needed in merged mode
      cross_attention_levels = []
      use_projector = False
      projector_levels = None

    self.denoising_network = DenoisingNetwork(
      denoising_in, channels, num_blocks, num_heads,
      use_swin, attention_levels, cross_attention_levels,
      use_projector, projector_levels,
      bottleneck_attention, window_size
    )

  def forward(self, x_noisy, condition, timesteps, cache=None):
    """
    Args:
        x_noisy: Noisy dose map [B, 1, H, W]
        condition: Structural information [B, C, H, W]
        timesteps: Diffusion timesteps [B]
        cache: DeepCache (unused, for compatibility)
    Returns:
        Predicted noise [B, 1, H, W]
    """
    if self.encoder_mode == 'separate':
      struct_features = self.structure_encoder(condition)
      return self.denoising_network(x_noisy, timesteps, struct_features)
    else:
      x_merged = torch.cat([x_noisy, condition], dim=1)
      return self.denoising_network(x_merged, timesteps)


def build_spdiffdose(config, structure_selector, out_channels=1):
  """
  Build SP-DiffDose model from configuration.

  Config keys:
    - base_channels: Base channel count (default: 64)
    - channel_mult: Channel multipliers per level (default: [1, 2, 4, 8, 8])
      Length determines number of levels (e.g. [1, 2, 4, 8] = 4 levels)
    - num_blocks: Blocks per encoder/decoder level (default: 3)
    - num_heads: Attention heads (default: 8)
    - use_swin: Use SwinTransformer blocks (default: true)
    - attention_levels: Levels with self-attention, ResNet mode only (default: [])
    - cross_attention_levels: Levels using cross-attention fusion (default: [4])
    - use_projector: Use projector after fusion (default: true)
    - projector_levels: Specific levels for projectors (default: all)
    - encoder_mode: 'separate' or 'merged' (default: 'separate')
    - bottleneck_attention: Attention in middle block (default: true)
    - window_size: Swin window size (default: 7)
  """
  if out_channels != 1:
    raise ValueError("SPDiffDose only supports out_channels=1 for noise prediction")

  model = SPDiffDose(
    structure_selector=structure_selector,
    base_channels=config.get('base_channels', 64),
    channel_mult=tuple(config.get('channel_mult', [1, 2, 4, 8, 8])),
    num_blocks=config.get('num_blocks', 3),
    num_heads=config.get('num_heads', 8),
    use_swin=config.get('use_swin', True),
    attention_levels=tuple(config.get('attention_levels', [])),
    cross_attention_levels=config.get('cross_attention_levels', [4]),
    use_projector=config.get('use_projector', True),
    projector_levels=config.get('projector_levels', None),
    encoder_mode=config.get('encoder_mode', 'separate'),
    bottleneck_attention=config.get('bottleneck_attention', True),
    window_size=config.get('window_size', 7),
  )
  return model


def migrate_spdiffdose_state_dict(state_dict):
  """
  Migrate state dict from old SPDiffDose key naming to new.

  Handles:
  - Removing duplicated denoising_network.structure_encoder.* keys
  - Renaming structure_encoder internal keys (level0_blocks, swin_levels, etc.)
  - Renaming denoising_network internal keys (encoder_stages, decoder_stages, etc.)

  Returns a new state dict with updated keys. If already in new format, returns as-is.
  """
  # Detect if migration is needed
  if not any(k.startswith('denoising_network.structure_encoder.') for k in state_dict):
    return state_dict

  new_state = {}
  se = 'structure_encoder.'
  dn = 'denoising_network.'

  for key, value in state_dict.items():
    # Skip duplicated structure_encoder under denoising_network
    if key.startswith(f'{dn}structure_encoder.'):
      continue

    new_key = key

    if key.startswith(se):
      rest = key[len(se):]

      if rest.startswith('level0_blocks.'):
        # level0_blocks.X.suffix -> level_blocks.0.X.suffix
        new_key = f'{se}level_blocks.0.{rest[len("level0_blocks."):]}'

      elif rest.startswith('level0_down.'):
        # level0_down.suffix -> downs.0.suffix
        new_key = f'{se}downs.0.{rest[len("level0_down."):]}'

      elif rest.startswith('swin_levels.'):
        m = re.match(r'swin_levels\.(\d+)\.(\d+)\.(.*)', rest)
        if m:
          level, sub_idx, suffix = int(m.group(1)), int(m.group(2)), m.group(3)
          if sub_idx == 0:
            # channel_match -> channel_matches.{level+1}
            new_key = f'{se}channel_matches.{level + 1}.{suffix}'
          else:
            # swin_levels.X.1.Y.suffix -> level_blocks.{X+1}.Y.suffix
            new_key = f'{se}level_blocks.{level + 1}.{suffix}'

      elif rest.startswith('down_blocks.'):
        m = re.match(r'down_blocks\.(\d+)\.(.*)', rest)
        if m:
          idx, suffix = int(m.group(1)), m.group(2)
          new_key = f'{se}downs.{idx + 1}.{suffix}'

    elif key.startswith(dn):
      rest = key[len(dn):]

      if rest.startswith('encoder_stages.'):
        m = re.match(r'encoder_stages\.(\d+)\.(\d+)\.(.*)', rest)
        if m:
          level, sub_idx, suffix = int(m.group(1)), int(m.group(2)), m.group(3)
          if level == 0:
            # Level 0: all ResNetBlocks -> encoder_blocks.0.X.suffix
            new_key = f'{dn}encoder_blocks.0.{sub_idx}.{suffix}'
          elif sub_idx == 0:
            # channel_match -> encoder_channel_matches.X
            new_key = f'{dn}encoder_channel_matches.{level}.{suffix}'
          else:
            # swin blocks -> encoder_blocks.X.{sub_idx-1}
            new_key = f'{dn}encoder_blocks.{level}.{sub_idx - 1}.{suffix}'

      elif rest.startswith('decoder_stages.'):
        new_key = key.replace('decoder_stages', 'decoder_blocks')

    new_state[new_key] = value

  return new_state
