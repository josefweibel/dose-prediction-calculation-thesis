"""
BeamDiff: Beam Field Guided Diffusion Model for Liver Cancer Radiotherapy
Based on: "Beam field guided diffusion model for liver cancer radiotherapy dose distribution prediction"
Cao et al., Medical Physics, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model_cache import DeepCache


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
  """Convert timestep to embedding and generate adaptive parameters."""

  def __init__(self, timestep_dim, embed_dim):
    super().__init__()
    self.linear_1 = nn.Linear(timestep_dim, embed_dim)
    self.act = nn.SiLU()
    self.linear_2 = nn.Linear(embed_dim, embed_dim * 2)  # For scale and shift

  def forward(self, t):
    t_emb = self.linear_1(t)
    t_emb = self.act(t_emb)
    t_emb = self.linear_2(t_emb)
    return t_emb


class ResNetBlock(nn.Module):
  """
  ResNet block with Group Normalization and timestep conditioning.
  Includes adaptive scaling and shifting based on timestep.
  """

  def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=8):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

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

    # Apply timestep conditioning (adaptive scale and shift)
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


class MultiHeadSelfAttention(nn.Module):
  """Multi-Head Self-Attention module."""

  def __init__(self, channels, num_heads=4):
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
    qkv = self.qkv(x)  # B, 3*C, H, W

    # Reshape for multi-head attention
    qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
    qkv = qkv.permute(1, 0, 2, 4, 3)  # 3, B, num_heads, H*W, head_dim
    q, k, v = qkv[0], qkv[1], qkv[2]

    # Attention
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    attn = F.softmax(attn, dim=-1)

    # Apply attention to values
    out = torch.matmul(attn, v)  # B, num_heads, H*W, head_dim
    out = out.transpose(2, 3).reshape(B, C, H, W)

    out = self.proj(out)
    return out + residual


class MultiConditionAggregationModule(nn.Module):
  """
  Multi-Condition Aggregation Module (MAM)
  Captures global associations between conditional information and noisy image
  using Transformer encoder with multi-head attention.
  """

  def __init__(self, channels=512, patch_size=4, embed_dim=1024, num_heads=8, dropout=0.1):
    super().__init__()
    self.channels = channels
    self.patch_size = patch_size
    self.embed_dim = embed_dim
    self.num_heads = num_heads

    # Patch embedding dimension
    self.patch_dim = channels * patch_size * patch_size

    # Linear projection for patches
    self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

    # Positional encoding (learned)
    self.num_patches = (128 // patch_size) ** 2  # Assuming 128x128 input at this stage
    self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    # Transformer encoder layer
    self.norm1 = nn.LayerNorm(embed_dim)
    self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.ffn = nn.Sequential(
      nn.Linear(embed_dim, embed_dim * 2),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(embed_dim * 2, embed_dim),
      nn.Dropout(dropout)
    )

    # Projection back to image
    self.proj_back = nn.Linear(embed_dim, self.patch_dim)

  def forward(self, xq, xk, xv):
    """
    Args:
        xq: Query features (noisy image) [B, C, H, W]
        xk: Key features (condition 1) [B, C, H, W]
        xv: Value features (condition 2) [B, C, H, W]
    """
    B, C, H, W = xq.shape

    # Pad to make dimensions divisible by patch_size
    ps = self.patch_size
    pad_h = (ps - H % ps) % ps
    pad_w = (ps - W % ps) % ps
    if pad_h > 0 or pad_w > 0:
      xq = F.pad(xq, (0, pad_w, 0, pad_h))
      xk = F.pad(xk, (0, pad_w, 0, pad_h))
      xv = F.pad(xv, (0, pad_w, 0, pad_h))
    H_pad, W_pad = H + pad_h, W + pad_w

    # Split into patches
    xq_patches = self._to_patches(xq)  # B, num_patches, patch_dim
    xk_patches = self._to_patches(xk)
    xv_patches = self._to_patches(xv)

    # Embed patches
    xq_emb = self.patch_embed(xq_patches) + self.pos_embedding[:, :xq_patches.size(1), :]
    xk_emb = self.patch_embed(xk_patches) + self.pos_embedding[:, :xk_patches.size(1), :]
    xv_emb = self.patch_embed(xv_patches) + self.pos_embedding[:, :xv_patches.size(1), :]

    # Transformer encoder
    # Multi-head attention
    normed_q = self.norm1(xq_emb)
    normed_k = self.norm1(xk_emb)
    normed_v = self.norm1(xv_emb)

    attn_out, _ = self.mha(normed_q, normed_k, normed_v)
    x = xq_emb + attn_out

    # Feed-forward network
    x = x + self.ffn(self.norm2(x))

    # Project back to image space
    x = self.proj_back(x)
    x = self._from_patches(x, H_pad, W_pad)

    # Remove padding
    if pad_h > 0 or pad_w > 0:
      x = x[:, :, :H, :W]

    return x

  def _to_patches(self, x):
    """Convert image to patches. Assumes H and W are divisible by patch_size."""
    B, C, H, W = x.shape
    ps = self.patch_size
    x = x.reshape(B, C, H // ps, ps, W // ps, ps)
    x = x.permute(0, 2, 4, 1, 3, 5)  # B, H//ps, W//ps, C, ps, ps
    x = x.reshape(B, (H // ps) * (W // ps), C * ps * ps)
    return x

  def _from_patches(self, x, H, W):
    """Convert patches back to image. H and W should be the padded dimensions."""
    B, num_patches, patch_dim = x.shape
    ps = self.patch_size
    C = self.channels

    x = x.reshape(B, H // ps, W // ps, C, ps, ps)
    x = x.permute(0, 3, 1, 4, 2, 5)  # B, C, H//ps, ps, W//ps, ps
    x = x.reshape(B, C, H, W)
    return x


class AsymmetricFusionModule(nn.Module):
  """
  Asymmetric Fusion Module (AFM)
  Fuses high-level features from encoder with low-level features from decoder.
  Uses Instance Normalization to capture intra-slice structural relationships.
  """

  def __init__(self, channels):
    super().__init__()
    # High-level branch
    self.conv_high = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.norm_high = nn.InstanceNorm2d(channels)
    self.act_high = nn.SiLU()

    # Low-level branch
    self.conv_low = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.norm_low = nn.InstanceNorm2d(channels)
    self.act_low = nn.SiLU()

  def forward(self, x_high, x_low):
    """
    Args:
        x_high: High-level features from encoder
        x_low: Low-level features from decoder
    """
    h_high = self.act_high(self.norm_high(self.conv_high(x_high)))
    h_low = self.act_low(self.norm_low(self.conv_low(x_low)))

    # Sum and concatenate with original encoder features
    fused = h_high + h_low
    out = torch.cat([x_high, fused], dim=1)

    return out


class EncoderStage(nn.Module):
  """Encoder stage with ResNet blocks and optional attention."""

  def __init__(self, in_channels, out_channels, time_emb_dim, num_blocks=2,
               use_attention=False, num_heads=4):
    super().__init__()
    self.blocks = nn.ModuleList([
      ResNetBlock(
        in_channels if i == 0 else out_channels,
        out_channels,
        time_emb_dim
      ) for i in range(num_blocks)
    ])

    self.attention = None
    if use_attention:
      self.attention = nn.ModuleList([
        MultiHeadSelfAttention(out_channels, num_heads)
        for _ in range(num_blocks)
      ])

  def forward(self, x, time_emb):
    for i, block in enumerate(self.blocks):
      x = block(x, time_emb)
      if self.attention is not None:
        x = self.attention[i](x)
    return x


class DecoderStage(nn.Module):
  """Decoder stage with ResNet blocks."""

  def __init__(self, in_channels, out_channels, time_emb_dim, num_blocks=2):
    super().__init__()
    self.blocks = nn.ModuleList([
      ResNetBlock(
        in_channels if i == 0 else out_channels,
        out_channels,
        time_emb_dim
      ) for i in range(num_blocks)
    ])

  def forward(self, x, time_emb):
    for block in self.blocks:
      x = block(x, time_emb)
    return x


class NoisePredictor(nn.Module):
  """
  Noise Predictor Network for BeamDiff
  Multi-branch hybrid encoder with MAM and AFM modules.

  Compatible with the diffusion trainer interface:
  - forward(x_t, condition, t, cache) where:
    - x_t: noisy dose map
    - condition: concatenated structural info (CT + PTV + OARs + optionally beam fields)
    - t: timestep
    - cache: DeepCache (ignored for now)

  Beam field auto-detection:
    The model automatically detects if 'beam_plate' and/or 'angle_plate' are present
    in the conditioning via the structure_selector and extracts them accordingly.
    Both beam fields can be used simultaneously.
  """

  def __init__(
    self,
    structure_selector,    # StructureSelector for extracting components from condition
    base_channels=128,
    channel_mult=(1, 1, 2, 3, 4),
    num_res_blocks=2,
    attention_levels=(4,),  # Apply attention at the last level
    num_heads=4,
    use_mam=True,
    use_afm=True,
  ):
    super().__init__()

    self.use_mam = use_mam
    self.use_afm = use_afm
    self.structure_selector = structure_selector

    # Auto-detect beam fields from structure selector
    self.beam_field_keys = []
    if 'beam_plate' in structure_selector.groups:
      self.beam_field_keys.append('beam_plate')
    if 'angle_plate' in structure_selector.groups:
      self.beam_field_keys.append('angle_plate')

    # Use beam field if we found any beam field keys
    self.use_beam_field = len(self.beam_field_keys) > 0

    # Calculate input channels for each branch
    # Structural branch: all channels except beam fields
    in_channels_struct = structure_selector.total_channels
    in_channels_beam = 0

    for key in self.beam_field_keys:
      start, end = structure_selector.get_channel_indices(key)
      beam_channels = end - start
      in_channels_struct -= beam_channels
      in_channels_beam += beam_channels

    # Timestep embedding
    time_embed_dim = base_channels * 4
    self.time_embed = nn.Sequential(
      SinusoidalPositionEmbedding(base_channels),
      nn.Linear(base_channels, time_embed_dim),
      nn.SiLU(),
      nn.Linear(time_embed_dim, time_embed_dim)
    )

    # Initial convolutions
    self.conv_in_noisy = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)
    self.conv_in_struct = nn.Conv2d(in_channels_struct, base_channels, kernel_size=3, padding=1)
    if self.use_beam_field:
      self.conv_in_beam = nn.Conv2d(in_channels_beam, base_channels, kernel_size=3, padding=1)

    # Encoder stages
    self.encoder_noisy = nn.ModuleList()
    self.encoder_struct = nn.ModuleList()
    if self.use_beam_field:
      self.encoder_beam = nn.ModuleList()
    self.downsamples = nn.ModuleList()

    channels = [base_channels * m for m in channel_mult]
    in_ch = base_channels

    for level, out_ch in enumerate(channels):
      use_attn = (level + 1) in attention_levels

      # Noisy image encoder
      self.encoder_noisy.append(
        EncoderStage(in_ch, out_ch, time_embed_dim, num_res_blocks, use_attn, num_heads)
      )

      # Structural information encoder
      self.encoder_struct.append(
        EncoderStage(in_ch, out_ch, time_embed_dim, num_res_blocks, use_attn, num_heads)
      )

      # Beam field encoder (only if using beam field)
      if self.use_beam_field:
        self.encoder_beam.append(
          EncoderStage(in_ch, out_ch, time_embed_dim, num_res_blocks, use_attn, num_heads)
        )

      # Downsample (except last level)
      if level < len(channels) - 1:
        self.downsamples.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
      else:
        self.downsamples.append(nn.Identity())

      in_ch = out_ch

    # Multi-Condition Aggregation Module (only if using beam field and MAM)
    if self.use_mam and self.use_beam_field:
      self.mam = MultiConditionAggregationModule(
        channels=channels[-1],
        patch_size=4,
        embed_dim=1024,
        num_heads=8
      )

    # Middle block
    self.middle = nn.ModuleList([
      ResNetBlock(channels[-1], channels[-1], time_embed_dim),
      MultiHeadSelfAttention(channels[-1], num_heads),
      ResNetBlock(channels[-1], channels[-1], time_embed_dim)
    ])

    # Decoder stages
    self.decoder = nn.ModuleList()
    self.upsamples = nn.ModuleList()
    self.afm_modules = nn.ModuleList() if use_afm else None

    reversed_channels = list(reversed(channels))
    for level, out_ch in enumerate(reversed_channels):
      in_ch = reversed_channels[level]
      next_ch = reversed_channels[level + 1] if level + 1 < len(reversed_channels) else base_channels

      # AFM for skip connection
      if self.use_afm and level > 0:
        self.afm_modules.append(AsymmetricFusionModule(in_ch))
        decoder_in_ch = in_ch * 2  # After concatenation in AFM
      else:
        decoder_in_ch = in_ch

      self.decoder.append(
        DecoderStage(decoder_in_ch, next_ch, time_embed_dim, num_res_blocks)
      )

      # Upsample (except last level)
      if level < len(reversed_channels) - 1:
        self.upsamples.append(nn.ConvTranspose2d(next_ch, next_ch, kernel_size=4, stride=2, padding=1))
      else:
        self.upsamples.append(nn.Identity())

    # Output
    self.norm_out = nn.GroupNorm(8, base_channels)
    self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

  def forward(self, x_noisy, condition, timesteps, cache: DeepCache):
    """
    Forward pass compatible with DiffusionModelTrainerModule.

    Args:
        x_noisy: Noisy dose map [B, 1, H, W]
        condition: Structural information (CT + PTV + OARs + optionally beam fields) [B, C, H, W]
        timesteps: Diffusion timesteps [B]
        cache: DeepCache for caching encoder and decoder features

    Returns:
        Predicted noise [B, 1, H, W]
    """
    # Timestep embedding
    t_emb = self.time_embed(timesteps)

    # Extract beam fields from condition if present
    if self.use_beam_field:
      beam_components = []
      for key in self.beam_field_keys:
        beam_components.append(self.structure_selector.get_component(condition, key))
      x_beam = torch.cat(beam_components, dim=1) if len(beam_components) > 1 else beam_components[0]

      # Build structural condition without beam fields
      struct_components = []
      for group in self.structure_selector.groups:
        if group not in self.beam_field_keys:
          struct_components.append(self.structure_selector.get_component(condition, group))
      x_struct = torch.cat(struct_components, dim=1)
    else:
      x_struct = condition
      x_beam = None

    # Initial convolutions
    h_noisy = self.conv_in_noisy(x_noisy)
    h_struct = cache.get_or_compute('head_struct', lambda: self.conv_in_struct(x_struct))

    if self.use_beam_field:
      h_beam = cache.get_or_compute('head_beam', lambda: self.conv_in_beam(x_beam))

    # Encoder
    encoder_outputs = []

    for level in range(len(self.encoder_noisy)):
      cache_key_enc = f'enc{level+1}'

      # Check if we can use cached encoder output
      if cache.should_use(cache_key_enc):
        h_noisy = cache.get(cache_key_enc)
        encoder_outputs.append(h_noisy)
        h_noisy = self.downsamples[level](h_noisy)

        # Still need to update struct/beam for next level (cached)
        cache_key_down_struct = f'down{level+1}_struct'
        h_struct = cache.get_or_compute(
          cache_key_down_struct,
          lambda down=self.downsamples[level], inp=h_struct: down(inp)
        )
        if self.use_beam_field:
          cache_key_down_beam = f'down{level+1}_beam'
          h_beam = cache.get_or_compute(
            cache_key_down_beam,
            lambda down=self.downsamples[level], inp=h_beam: down(inp)
          )
        continue

      # Process noisy branch
      h_noisy = self.encoder_noisy[level](h_noisy, t_emb)

      # Process structural branch (can be cached)
      cache_key_struct = f'enc{level+1}_struct'
      h_struct = cache.get_or_compute(
        cache_key_struct,
        lambda enc=self.encoder_struct[level], inp=h_struct: enc(inp, t_emb)
      )

      if self.use_beam_field:
        # Process beam field branch (can be cached)
        cache_key_beam = f'enc{level+1}_beam'
        h_beam = cache.get_or_compute(
          cache_key_beam,
          lambda enc=self.encoder_beam[level], inp=h_beam: enc(inp, t_emb)
        )

        # Fuse all three branches (except last level)
        if level < len(self.encoder_noisy) - 1:
          h_noisy = h_noisy + h_struct + h_beam
      else:
        # Fuse only noisy and structural branches (except last level)
        if level < len(self.encoder_noisy) - 1:
          h_noisy = h_noisy + h_struct

      # Cache and store for skip connections
      cache.set(cache_key_enc, h_noisy)
      encoder_outputs.append(h_noisy)

      # Downsample
      h_noisy = self.downsamples[level](h_noisy)

      cache_key_down_struct = f'down{level+1}_struct'
      h_struct = cache.get_or_compute(
        cache_key_down_struct,
        lambda down=self.downsamples[level], inp=h_struct: down(inp)
      )

      if self.use_beam_field:
        cache_key_down_beam = f'down{level+1}_beam'
        h_beam = cache.get_or_compute(
          cache_key_down_beam,
          lambda down=self.downsamples[level], inp=h_beam: down(inp)
        )

    # Multi-Condition Aggregation Module or simple fusion
    if self.use_beam_field and self.use_mam:
      h = self.mam(h_noisy, h_struct, h_beam)
    elif self.use_beam_field:
      h = h_noisy + h_struct + h_beam
    else:
      h = h_noisy + h_struct

    # Middle block
    for module in self.middle:
      if isinstance(module, ResNetBlock):
        h = module(h, t_emb)
      else:
        h = module(h)

    # Decoder
    for level in range(len(self.decoder)):
      # Skip connection with AFM
      skip = encoder_outputs[-(level + 1)]

      cache_key = f'dec{len(self.decoder) - level}'
      if cache.should_use(cache_key):
        h = cache.get(cache_key)
        h = self.upsamples[level](h)
        continue

      if level > 0:
        if self.use_afm:
          h = self.afm_modules[level - 1](skip, h)
        else:
          # Fallback: simple addition-based skip connection
          h = h + skip

      h = self.decoder[level](h, t_emb)
      cache.set(cache_key, h)
      h = self.upsamples[level](h)

    # Output
    h = self.norm_out(h)
    h = F.silu(h)
    h = self.conv_out(h)

    return h


def build_beamdiff(config, structure_selector):
  """
  Build BeamDiff model from configuration.

  Args:
      config: BeamDiff configuration dict
      structure_selector: StructureSelector instance for extracting components from conditioning tensor

  Returns:
      NoisePredictor model (not wrapped in BeamDiff, since trainer handles diffusion)
  """
  model = NoisePredictor(
    structure_selector=structure_selector,
    base_channels=config.get('base_channels', 128),
    channel_mult=tuple(config.get('channel_mult', [1, 1, 2, 3, 4])),
    num_res_blocks=config.get('num_res_blocks', 2),
    attention_levels=tuple(config.get('attention_levels', [4])),
    num_heads=config.get('num_heads', 4),
    use_mam=config.get('use_mam', True),
    use_afm=config.get('use_afm', True),
  )
  return model
