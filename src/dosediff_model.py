"""
DoseDiff: Distance-aware Diffusion Model for Dose Prediction in Radiotherapy

Based on: "DoseDiff: Distance-aware Diffusion Model for Dose Prediction in Radiotherapy"
Zhang et al., IEEE Transactions on Medical Imaging, 2023

Key Features:
1. Signed Distance Maps (PSDM) in physical space for distance information
2. Multi-encoder Multi-scale Fusion Network (MMFNet)
3. FusionFormer for global information fusion at bottleneck
4. Multi-scale feature fusion between CT and SDM encoders
5. Compatible with DDIM for accelerated inference

Usage:
  The model expects conditioning with:
  - CT image (group name: 'ct')
  - SDM for targets and OARs (group names containing 'sdm': 'ptv_sdm', 'oar_sdm', etc.)
  - Optional: Beam fields (group names: 'beam_plate', 'angle_plate')

  Example configuration (standard):
    conditioning:
      types:
        - ct           # CT image
        - ptv_sdm      # Signed Distance Map for PTV
        - oar_sdm      # Signed Distance Map for OARs

  Example configuration (with beam fields):
    conditioning:
      types:
        - ct           # CT image
        - ptv_sdm      # Signed Distance Map for PTV
        - oar_sdm      # Signed Distance Map for OARs
        - beam_plate   # Beam field information (optional)
        - angle_plate  # Beam angle information (optional)

  The model automatically detects and separates CT, SDM, and beam groups using the structure_selector.
  Beam fields are optional and enhance the model's understanding of radiation delivery patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


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


class AdaptiveGroupNorm(nn.Module):
  """
  Adaptive Group Normalization (AdaGN)
  Applies affine transformation conditioned on timestep embedding.
  """

  def __init__(self, num_groups, num_channels, embed_dim):
    super().__init__()
    self.norm = nn.GroupNorm(num_groups, num_channels)
    self.scale_shift_proj = nn.Linear(embed_dim, num_channels * 2)

  def forward(self, x, time_emb):
    """
    Args:
        x: Input features [B, C, H, W]
        time_emb: Time embedding [B, embed_dim]
    Returns:
        Normalized and conditioned features
    """
    x = self.norm(x)

    # Project time embedding to scale and shift
    scale_shift = self.scale_shift_proj(time_emb)
    while len(scale_shift.shape) < len(x.shape):
      scale_shift = scale_shift[..., None]

    scale, shift = scale_shift.chunk(2, dim=1)
    return x * (1 + scale) + shift


class ResBlock(nn.Module):
  """
  Residual block with Group Normalization and timestep conditioning via AdaGN.
  """

  def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
    super().__init__()

    self.norm1 = nn.GroupNorm(groups, in_channels)
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    self.ada_gn = AdaptiveGroupNorm(groups, out_channels, time_emb_dim)
    self.act = nn.SiLU()
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    # Residual connection
    if in_channels != out_channels:
      self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    else:
      self.residual_conv = nn.Identity()

  def forward(self, x, time_emb):
    residual = self.residual_conv(x)

    # First conv
    h = self.norm1(x)
    h = self.act(h)
    h = self.conv1(h)

    # Second conv with AdaGN conditioning
    h = self.ada_gn(h, time_emb)
    h = self.act(h)
    h = self.conv2(h)

    return h + residual


class FusionFormer(nn.Module):
  """
  Transformer-based Fusion Module for global information fusion.
  Uses multi-head attention where CT and SDM (optionally with beam) features serve as query/key,
  and noisy dose features serve as value.

  Supports optional 4th input (beam features) which are concatenated with SDM for key.
  """

  def __init__(self, channels, patch_size=2, embed_dim=None, num_heads=8, dropout=0.1, use_beam=False):
    super().__init__()
    self.channels = channels
    self.patch_size = patch_size
    self.embed_dim = embed_dim or channels
    self.num_heads = num_heads
    self.use_beam = use_beam

    # Patch embedding dimension
    self.patch_dim = channels * patch_size * patch_size

    # Embedding layers for each input
    self.embed_q = nn.Sequential(
      nn.Linear(self.patch_dim, self.embed_dim),
      nn.LayerNorm(self.embed_dim)
    )
    self.embed_k = nn.Sequential(
      nn.Linear(self.patch_dim, self.embed_dim),
      nn.LayerNorm(self.embed_dim)
    )
    self.embed_v = nn.Sequential(
      nn.Linear(self.patch_dim, self.embed_dim),
      nn.LayerNorm(self.embed_dim)
    )
    # Optional beam embedding
    if use_beam:
      self.embed_beam = nn.Sequential(
        nn.Linear(self.patch_dim, self.embed_dim),
        nn.LayerNorm(self.embed_dim)
      )

    # Multi-head attention
    self.norm1 = nn.LayerNorm(self.embed_dim)
    self.mha = nn.MultiheadAttention(self.embed_dim, num_heads, dropout=dropout, batch_first=True)

    # Feed-forward network
    self.norm2 = nn.LayerNorm(self.embed_dim)
    self.ffn = nn.Sequential(
      nn.Linear(self.embed_dim, self.embed_dim * 2),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(self.embed_dim * 2, self.embed_dim),
      nn.Dropout(dropout)
    )

    # Projection back to image
    self.proj_back = nn.Linear(self.embed_dim, self.patch_dim)

  def _to_patches(self, x):
    """Convert image to patches."""
    B, C, H, W = x.shape
    ps = self.patch_size
    x = x.reshape(B, C, H // ps, ps, W // ps, ps)
    x = x.permute(0, 2, 4, 1, 3, 5)  # B, H//ps, W//ps, C, ps, ps
    x = x.reshape(B, (H // ps) * (W // ps), C * ps * ps)
    return x

  def _from_patches(self, x, H, W):
    """Convert patches back to image."""
    B, num_patches, patch_dim = x.shape
    ps = self.patch_size
    C = self.channels

    x = x.reshape(B, H // ps, W // ps, C, ps, ps)
    x = x.permute(0, 3, 1, 4, 2, 5)  # B, C, H//ps, ps, W//ps, ps
    x = x.reshape(B, C, H, W)
    return x

  def forward(self, f_ct, f_sdm, f_noisy, f_beam=None):
    """
    Args:
        f_ct: CT features [B, C, H, W]
        f_sdm: SDM features [B, C, H, W]
        f_noisy: Noisy dose features [B, C, H, W]
        f_beam: Optional beam features [B, C, H, W]
    Returns:
        Fused features [B, C, H, W]
    """
    B, C, H, W = f_noisy.shape

    # Convert to patches
    patches_ct = self._to_patches(f_ct)      # B, N, patch_dim
    patches_sdm = self._to_patches(f_sdm)    # B, N, patch_dim
    patches_noisy = self._to_patches(f_noisy)  # B, N, patch_dim

    # Embed patches
    q = self.embed_q(patches_ct)      # Query from CT
    k = self.embed_k(patches_sdm)     # Key from SDM
    v = self.embed_v(patches_noisy)   # Value from noisy dose

    # If beam features are provided, incorporate them into the key
    if f_beam is not None and self.use_beam:
      patches_beam = self._to_patches(f_beam)  # B, N, patch_dim
      beam_emb = self.embed_beam(patches_beam)
      # Average beam embedding with SDM key
      k = (k + beam_emb) / 2.0

    # Multi-head attention: CT & SDM (+ optional beam) guide the attention on noisy features
    normed_q = self.norm1(q)
    normed_k = self.norm1(k)
    normed_v = self.norm1(v)

    attn_out, _ = self.mha(normed_q, normed_k, normed_v)
    x = q + attn_out

    # Feed-forward network
    x = x + self.ffn(self.norm2(x))

    # Project back to image space
    x = self.proj_back(x)
    x = self._from_patches(x, H, W)

    return x


class EncoderStage(nn.Module):
  """Encoder stage with multiple ResBlocks."""

  def __init__(self, in_channels, out_channels, time_emb_dim, num_blocks=2):
    super().__init__()
    self.blocks = nn.ModuleList([
      ResBlock(
        in_channels if i == 0 else out_channels,
        out_channels,
        time_emb_dim
      ) for i in range(num_blocks)
    ])

  def forward(self, x, time_emb):
    for block in self.blocks:
      x = block(x, time_emb)
    return x


class DecoderStage(nn.Module):
  """Decoder stage with multiple ResBlocks."""

  def __init__(self, in_channels, out_channels, time_emb_dim, num_blocks=2):
    super().__init__()
    self.blocks = nn.ModuleList([
      ResBlock(
        in_channels if i == 0 else out_channels,
        out_channels,
        time_emb_dim
      ) for i in range(num_blocks)
    ])

  def forward(self, x, time_emb):
    for block in self.blocks:
      x = block(x, time_emb)
    return x


class MMFNet(nn.Module):
  """
  Multi-encoder Multi-scale Fusion Network (MMFNet)

  Architecture:
  - Three independent encoders for CT, SDM, and noisy dose
  - Multi-scale fusion via element-wise addition at each encoder level
  - FusionFormer at the bottleneck level for global fusion
  - Single decoder with skip connections

  Default configuration matches the best performing DoseDiff from the paper:
  - 4 down-sampling levels with channel mult [1, 2, 3, 4] → [128, 256, 384, 512]
  - 2 ResBlocks per stage
  - FusionFormer at bottleneck with 8 attention heads
  - Base channels: 128 (as per paper's implementation)
  """

  def __init__(
    self,
    structure_selector,
    base_channels=128,
    channel_mult=(1, 2, 3, 4),
    num_res_blocks=2,
    use_fusion_former=True,
    fusion_former_heads=8,
    fusion_former_patch_size=2,
  ):
    super().__init__()

    self.structure_selector = structure_selector
    self.use_fusion_former = use_fusion_former
    self.num_levels = len(channel_mult)

    # Detect CT, SDM, and beam groups from structure_selector
    # CT group (typically just 'ct')
    self.ct_groups = []
    # SDM groups (ptv_sdm, oar_sdm, etc.)
    self.sdm_groups = []
    # Beam field groups (beam_plate, angle_plate) - optional extension
    self.beam_groups = []

    for group in structure_selector.groups:
      group_lower = group.lower()
      if 'sdm' in group_lower:
        self.sdm_groups.append(group)
      elif group_lower == 'ct':
        self.ct_groups.append(group)
      elif 'beam_plate' in group_lower or 'angle_plate' in group_lower:
        self.beam_groups.append(group)

    # Check if beam fields are available
    self.use_beam_fields = len(self.beam_groups) > 0

    # Calculate input channels dynamically
    in_channels_ct = 0
    for group in self.ct_groups:
      start, end = structure_selector.get_channel_indices(group)
      in_channels_ct += (end - start)

    in_channels_sdm = 0
    for group in self.sdm_groups:
      start, end = structure_selector.get_channel_indices(group)
      in_channels_sdm += (end - start)

    in_channels_beam = 0
    if self.use_beam_fields:
      for group in self.beam_groups:
        start, end = structure_selector.get_channel_indices(group)
        in_channels_beam += (end - start)

    # Timestep embedding
    time_embed_dim = base_channels * 4
    self.time_embed = nn.Sequential(
      SinusoidalPositionEmbedding(base_channels),
      nn.Linear(base_channels, time_embed_dim),
      nn.SiLU(),
      nn.Linear(time_embed_dim, time_embed_dim)
    )

    # Initial convolutions for each branch
    self.conv_in_ct = nn.Conv2d(in_channels_ct, base_channels, kernel_size=3, padding=1)
    self.conv_in_sdm = nn.Conv2d(in_channels_sdm, base_channels, kernel_size=3, padding=1)
    self.conv_in_noisy = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)
    if self.use_beam_fields:
      self.conv_in_beam = nn.Conv2d(in_channels_beam, base_channels, kernel_size=3, padding=1)

    # Encoder stages for each branch
    self.encoder_ct = nn.ModuleList()
    self.encoder_sdm = nn.ModuleList()
    self.encoder_noisy = nn.ModuleList()
    if self.use_beam_fields:
      self.encoder_beam = nn.ModuleList()
    self.downsamples = nn.ModuleList()

    channels = [base_channels * m for m in channel_mult]
    in_ch = base_channels

    for level, out_ch in enumerate(channels):
      # CT encoder
      self.encoder_ct.append(
        EncoderStage(in_ch, out_ch, time_embed_dim, num_res_blocks)
      )

      # SDM encoder
      self.encoder_sdm.append(
        EncoderStage(in_ch, out_ch, time_embed_dim, num_res_blocks)
      )

      # Noisy dose encoder
      self.encoder_noisy.append(
        EncoderStage(in_ch, out_ch, time_embed_dim, num_res_blocks)
      )

      # Beam encoder (if using beam fields)
      if self.use_beam_fields:
        self.encoder_beam.append(
          EncoderStage(in_ch, out_ch, time_embed_dim, num_res_blocks)
        )

      # Downsample (except last level)
      if level < len(channels) - 1:
        self.downsamples.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
      else:
        self.downsamples.append(nn.Identity())

      in_ch = out_ch

    # FusionFormer at bottleneck (lowest resolution)
    if use_fusion_former:
      self.fusion_former = FusionFormer(
        channels=channels[-1],
        patch_size=fusion_former_patch_size,
        embed_dim=channels[-1],
        num_heads=fusion_former_heads,
        use_beam=self.use_beam_fields
      )

    # Middle block
    self.middle = nn.ModuleList([
      ResBlock(channels[-1], channels[-1], time_embed_dim),
      ResBlock(channels[-1], channels[-1], time_embed_dim)
    ])

    # Decoder
    self.decoder = nn.ModuleList()
    self.upsamples = nn.ModuleList()

    reversed_channels = list(reversed(channels))
    for level, out_ch in enumerate(reversed_channels):
      next_ch = reversed_channels[level + 1] if level + 1 < len(reversed_channels) else base_channels

      self.decoder.append(
        DecoderStage(out_ch, next_ch, time_embed_dim, num_res_blocks)
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
    Forward pass compatible with DiffusionModelTrainerModule.

    Args:
        x_noisy: Noisy dose map [B, 1, H, W]
        condition: Concatenated CT and SDM [B, C, H, W]
        timesteps: Diffusion timesteps [B]
        cache: DeepCache (unused, for compatibility)

    Returns:
        Predicted noise [B, 1, H, W]
    """
    # Extract CT, SDM, and optional beam fields from condition using structure_selector
    ct_components = []
    for group in self.ct_groups:
      ct_components.append(self.structure_selector.get_component(condition, group))
    x_ct = torch.cat(ct_components, dim=1) if len(ct_components) > 1 else ct_components[0]

    sdm_components = []
    for group in self.sdm_groups:
      sdm_components.append(self.structure_selector.get_component(condition, group))
    x_sdm = torch.cat(sdm_components, dim=1) if len(sdm_components) > 1 else sdm_components[0]

    # Extract beam fields if available
    if self.use_beam_fields:
      beam_components = []
      for group in self.beam_groups:
        beam_components.append(self.structure_selector.get_component(condition, group))
      x_beam = torch.cat(beam_components, dim=1) if len(beam_components) > 1 else beam_components[0]

    # Timestep embedding
    t_emb = self.time_embed(timesteps)

    # Initial convolutions
    h_ct = self.conv_in_ct(x_ct)
    h_sdm = self.conv_in_sdm(x_sdm)
    h_noisy = self.conv_in_noisy(x_noisy)
    if self.use_beam_fields:
      h_beam = self.conv_in_beam(x_beam)

    # Encoder with multi-scale fusion
    skip_connections = []

    for level in range(self.num_levels):
      # Process each branch
      h_ct = self.encoder_ct[level](h_ct, t_emb)
      h_sdm = self.encoder_sdm[level](h_sdm, t_emb)
      h_noisy = self.encoder_noisy[level](h_noisy, t_emb)

      if self.use_beam_fields:
        h_beam = self.encoder_beam[level](h_beam, t_emb)

      # Multi-scale fusion: element-wise addition (except at bottleneck if using FusionFormer)
      if level < self.num_levels - 1 or not self.use_fusion_former:
        if self.use_beam_fields:
          h_fused = h_ct + h_sdm + h_noisy + h_beam
        else:
          h_fused = h_ct + h_sdm + h_noisy
      else:
        # At bottleneck, save for FusionFormer
        h_fused = h_noisy  # Temporarily use noisy features

      # Store for skip connections
      skip_connections.append(h_fused)

      # Downsample
      h_ct = self.downsamples[level](h_ct)
      h_sdm = self.downsamples[level](h_sdm)
      h_noisy = self.downsamples[level](h_noisy)
      if self.use_beam_fields:
        h_beam = self.downsamples[level](h_beam)

    # FusionFormer at bottleneck
    if self.use_fusion_former:
      if self.use_beam_fields:
        h = self.fusion_former(h_ct, h_sdm, h_noisy, h_beam)
      else:
        h = self.fusion_former(h_ct, h_sdm, h_noisy)
    else:
      if self.use_beam_fields:
        h = h_ct + h_sdm + h_noisy + h_beam
      else:
        h = h_ct + h_sdm + h_noisy

    # Middle block
    for block in self.middle:
      h = block(h, t_emb)

    # Decoder with skip connections
    for level in range(len(self.decoder)):
      # Skip connection
      skip = skip_connections[-(level + 1)]
      h = h + skip

      h = self.decoder[level](h, t_emb)
      h = self.upsamples[level](h)

    # Output
    h = self.norm_out(h)
    h = F.silu(h)
    h = self.conv_out(h)

    return h


def build_dosediff(config, structure_selector):
  """
  Build DoseDiff model from configuration.

  Default values correspond to the best configuration from the paper (Table I):
  - Base channels: 128
  - Channel multipliers: [1, 2, 3, 4] → [128, 256, 384, 512]
  - Num ResBlocks: 2 per stage (4 encoder/decoder levels)
  - FusionFormer: Enabled at bottleneck
  - Multi-scale fusion: Enabled at all levels
  - PSDM: Used as input (handled via preprocessing)

  Configuration options:
  - base_channels: Base number of channels (default: 128)
  - channel_mult: Channel multipliers for each level (default: [1, 2, 3, 4])
  - num_res_blocks: Number of ResBlocks per stage (default: 2)
  - use_fusion_former: Whether to use FusionFormer at bottleneck (default: True)
  - fusion_former_heads: Number of attention heads in FusionFormer (default: 8)
  - fusion_former_patch_size: Patch size for FusionFormer (default: 2)

  Args:
      config: DoseDiff configuration dict
      structure_selector: StructureSelector instance

  Returns:
      MMFNet model
  """
  model = MMFNet(
    structure_selector=structure_selector,
    base_channels=config.get('base_channels', 128),
    channel_mult=tuple(config.get('channel_mult', [1, 2, 3, 4])),
    num_res_blocks=config.get('num_res_blocks', 2),
    use_fusion_former=config.get('use_fusion_former', True),
    fusion_former_heads=config.get('fusion_former_heads', 8),
    fusion_former_patch_size=config.get('fusion_former_patch_size', 2),
  )
  return model
