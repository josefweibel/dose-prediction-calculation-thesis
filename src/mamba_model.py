import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List
import re

from model_cache import DeepCache


class TimeEmbedding(nn.Module):
  def __init__(self, embed_dim, scale=30.0):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :].to(x.device) * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class TimeEmbeddingProjection(nn.Module):
  def __init__(self, in_size, out_size):
    super().__init__()
    self.act = Swish()
    self.proj = nn.Linear(in_size, out_size)

  def forward(self, x):
    x = self.act(x)
    return self.proj(x)


class Swish(nn.Module):
  def forward(self, x):
    return x * torch.sigmoid(x)


class Upsample(nn.Module):
  def __init__(self, n_feat):
    super(Upsample, self).__init__()

    self.body = nn.Sequential(
      nn.Conv2d(
        n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
      ),
      nn.PixelShuffle(2),
    )

  def forward(self, x):
    return self.body(x)


class MambaUnet(nn.Module):
  def __init__(
    self,
    img_size,
    in_channels=1,
    in_struc_channels=5,
    out_channels=1,
    bias=False,
    num_layers=4,
    base_channels=24,
    base_embed_dim=48,
    depths: Optional[List[int]] = None,
  ):
    """
    Args:
      img_size: Input image size (H, W)
      in_channels: Number of input channels
      in_struc_channels: Number of structure input channels
      out_channels: Number of output channels
      bias: Whether to use bias in convolutions
      num_layers: Number of encoder/decoder layers (default: 4)
      base_channels: Base number of channels (default: 24)
      base_embed_dim: Base embedding dimension (default: 48)
      depths: List of depths for each layer. If None, uses default pattern
    """
    super(MambaUnet, self).__init__()

    from mamba_lib import VisionMamba

    self.num_layers = num_layers

    # Default depths pattern if not provided
    if depths is None:
      depths = [4] + [6] * (num_layers - 2) + [8] if num_layers > 1 else [4]

    assert len(depths) == num_layers, f"depths length ({len(depths)}) must match num_layers ({num_layers})"

    # Time embedding
    tdim = base_channels * 4
    self.time_embedding = TimeEmbedding(tdim)

    # Create time projections for each encoder and decoder layer
    self.t_proj_enc = nn.ModuleList()
    self.t_proj_dec = nn.ModuleList()

    for i in range(num_layers):
      embed_dim = base_embed_dim * (2 ** i)
      self.t_proj_enc.append(TimeEmbeddingProjection(tdim, embed_dim))
      self.t_proj_dec.append(TimeEmbeddingProjection(tdim, max(embed_dim // 2, base_embed_dim)))

    # Input heads
    self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
    self.head_struc = nn.Conv2d(in_struc_channels, base_channels, kernel_size=3, stride=1, padding=1)

    # Create encoder layers
    self.encoders = nn.ModuleList()
    self.encoders_struc = nn.ModuleList()

    for i in range(num_layers):
      embed_dim = base_embed_dim * (2 ** i)
      channels = base_channels * (2 ** i)
      curr_img_size = tuple(x // (2 ** i) for x in img_size)

      self.encoders.append(
        VisionMamba(
          img_size=curr_img_size,
          patch_size=2,
          embed_dim=embed_dim,
          depth=depths[i],
          channels=channels,
          final_pool_type='all'
        )
      )
      self.encoders_struc.append(
        VisionMamba(
          img_size=curr_img_size,
          patch_size=2,
          embed_dim=embed_dim,
          depth=depths[i],
          channels=channels,
          final_pool_type='all'
        )
      )

    # Create decoder layers
    self.upsamplers = nn.ModuleList()
    self.reduce_channels = nn.ModuleList()
    self.decoders = nn.ModuleList()

    for i in range(num_layers - 1, -1, -1):
      embed_dim = base_embed_dim * (2 ** i)

      # Upsampler
      self.upsamplers.append(Upsample(embed_dim))

      if i > 0:
        # Channel reduction after concatenation
        self.reduce_channels.append(nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1, bias=bias))

      # Decoder
      curr_img_size = tuple(x // (2 ** i) for x in img_size)
      dec_depth = 6 if i > 1 else 4

      self.decoders.append(
        VisionMamba(
          img_size=curr_img_size,
          patch_size=1,
          embed_dim=max(embed_dim // 2, base_embed_dim),
          depth=dec_depth,
          channels=max(embed_dim // 2, base_embed_dim),
          final_pool_type='all'
        )
      )

    # Output head
    self.out = nn.Conv2d(base_embed_dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

  def forward(self, x, structure, t, cache: Optional[DeepCache] = None):
    # Time embedding
    t_emb = self.time_embedding(t)

    # Generate time projections
    t_enc = [proj(t_emb)[:, :, None, None] for proj in self.t_proj_enc]
    t_dec = [proj(t_emb)[:, :, None, None] for proj in self.t_proj_dec]

    # Process structure through encoder
    structure = cache.get_or_compute('head_struc', lambda: self.head_struc(structure))

    enc_struc_features = []
    out_struc = structure
    for i, enc_struc in enumerate(self.encoders_struc):
      cache_key = f'enc{i+1}_struc'
      out_struc = cache.get_or_compute(cache_key, lambda enc=enc_struc, inp=out_struc: enc(inp, return_features=True))
      enc_struc_features.append(out_struc)

    # Process input through encoder
    x = self.head(x)

    enc_features = []
    out = x
    for i, encoder in enumerate(self.encoders):
      cache_key = f'enc{i+1}'
      out = cache.get_or_compute(cache_key, lambda enc=encoder, inp=out: enc(inp, return_features=True))
      out = out + enc_struc_features[i] + t_enc[i]
      enc_features.append(out)

    # Decoder path
    out_dec = enc_features[-1]

    for i in range(self.num_layers - 1):
      cache_key = f'dec{self.num_layers - i}'
      if cache.should_use(cache_key):
        out_dec = cache.get(cache_key)
      else:
        dec_idx = i
        enc_idx = self.num_layers - 2 - i

        # Upsample
        out_dec = self.upsamplers[dec_idx](out_dec)

        # Match spatial dimensions if needed
        if out_dec.shape[2:] != enc_features[enc_idx].shape[2:]:
          out_dec = torch.nn.functional.interpolate(
            out_dec,
            size=enc_features[enc_idx].shape[2:],
            mode='bilinear',
            align_corners=False
          )

        # Skip connection
        out_dec = torch.cat([out_dec, enc_features[enc_idx]], 1)

        # Reduce channels
        out_dec = self.reduce_channels[dec_idx](out_dec)

        # Decoder block
        out_dec = self.decoders[dec_idx](out_dec, return_features=True)

        # Add time embedding
        out_dec = out_dec + t_dec[self.num_layers - 1 - i]
        cache.set(cache_key, out_dec)

    # Final decoder (no upsampling before)
    final_dec_idx = self.num_layers - 1
    out_dec = self.upsamplers[final_dec_idx](out_dec)

    # Match spatial dimensions if needed
    if out_dec.shape[2:] != x.shape[2:]:
      out_dec = torch.nn.functional.interpolate(
        out_dec,
        size=x.shape[2:],
        mode='bilinear',
        align_corners=False
      )

    out_dec = torch.cat([out_dec, x], 1)
    out_dec = self.decoders[final_dec_idx](out_dec, return_features=True)
    out_dec = out_dec + t_dec[0]

    return self.out(out_dec)


def map_old_to_new_key(old_key: str) -> str:
  mapping_rules = [
    # time projections
    (r"^t_proj_enc1", "t_proj_enc.0"),
    (r"^t_proj_enc2", "t_proj_enc.1"),
    (r"^t_proj_enc3", "t_proj_enc.2"),
    (r"^t_proj_enc4", "t_proj_enc.3"),
    (r"^t_proj_dec4", "t_proj_dec.3"),
    (r"^t_proj_dec3", "t_proj_dec.2"),
    (r"^t_proj_dec2", "t_proj_dec.1"),
    (r"^t_proj_dec1", "t_proj_dec.0"),

    # encoders
    (r"^enc1_struc", "encoders_struc.0"),
    (r"^enc2_struc", "encoders_struc.1"),
    (r"^enc3_struc", "encoders_struc.2"),
    (r"^enc4_struc", "encoders_struc.3"),
    (r"^enc1", "encoders.0"),
    (r"^enc2", "encoders.1"),
    (r"^enc3", "encoders.2"),
    (r"^enc4", "encoders.3"),

    # upsample & reduce
    (r"^up4", "upsamplers.0"),
    (r"^up3", "upsamplers.1"),
    (r"^up2", "upsamplers.2"),
    (r"^up1", "upsamplers.3"),
    (r"^reduce_channels4", "reduce_channels.0"),
    (r"^reduce_channels3", "reduce_channels.1"),
    (r"^reduce_channels2", "reduce_channels.2"),

    # decoders
    (r"^dec4", "decoders.0"),
    (r"^dec3", "decoders.1"),
    (r"^dec2", "decoders.2"),
    (r"^dec1", "decoders.3"),
  ]

  new_key = old_key
  for pattern, replacement in mapping_rules:
    new_key = re.sub(pattern, replacement, new_key)
  return new_key


def remap_state_dict(old_state_dict):
  new_state_dict = {}
  unmatched = []

  for k, v in old_state_dict.items():
    new_k = map_old_to_new_key(k)
    if new_k == k:
      unmatched.append(k)
    new_state_dict[new_k] = v

  # if unmatched:
  #   print("Warning: The following keys were not remapped:")
  #   for key in unmatched:
  #     print(f"  {key}")

  return new_state_dict
