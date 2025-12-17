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
    x_windows = self.window_partition(x, self.window_size)  # [B*num_windows, ws_h, ws_w, C] or [B, H, W, C] if single window

    # Handle single window case (when feature map is smaller than window size)
    is_single_window = (x_windows.shape == x.shape)

    if is_single_window:
      # Treat entire feature map as one window
      ws_h, ws_w = H, W
      x_windows = x_windows.view(B, ws_h * ws_w, C)
    else:
      # Get actual window dimensions from partitioned tensor
      ws_h = x_windows.shape[1]
      ws_w = x_windows.shape[2]
      x_windows = x_windows.view(-1, ws_h * ws_w, C)  # [B*num_windows, ws_h*ws_w, C]

    # Window-based multi-head self attention
    N = ws_h * ws_w
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
    if is_single_window:
      x = x_windows.view(B, H, W, C)
    else:
      x_windows = x_windows.view(-1, ws_h, ws_w, C)
      x = self.window_reverse(x_windows, self.window_size, H, W)

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
    If H or W is not divisible by window_size, use adaptive window size.

    Args:
        x: [B, H, W, C]
        window_size: Window size
    Returns:
        windows: [B*num_windows, window_size', window_size', C]
    """
    B, H, W, C = x.shape

    # Adaptive window size if needed
    ws_h = min(window_size, H)
    ws_w = min(window_size, W)

    # If dimensions are not divisible, adjust to use full feature map
    if H % ws_h != 0:
      ws_h = H
    if W % ws_w != 0:
      ws_w = W

    # If window size equals feature map size, treat as single window
    if ws_h == H and ws_w == W:
      return x.view(B, H, W, C)

    x = x.view(B, H // ws_h, ws_h, W // ws_w, ws_w, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws_h, ws_w, C)
    return windows

  def window_reverse(self, windows, window_size, H, W):
    """
    Reverse window partition.

    Args:
        windows: [B*num_windows, window_size', window_size', C]
        window_size: Original window size
        H: Height of image
        W: Width of image
    Returns:
        x: [B, H, W, C]
    """
    # Get actual window size from windows tensor
    ws_h = windows.shape[1]
    ws_w = windows.shape[2]

    # If window size equals feature map size, reverse single window case
    if ws_h == H and ws_w == W:
      B = windows.shape[0]
      return windows.view(B, H, W, -1)

    B = int(windows.shape[0] / (H * W / ws_h / ws_w))
    x = windows.view(B, H // ws_h, W // ws_w, ws_h, ws_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
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


class StructureEncoder(nn.Module):
  """
  Structure Encoder to extract anatomical information from CT + PTV + OARs.

  As described in Section III.B.1 of the paper:
  - First level: Conv-ResNet block (3 conv blocks + 1 down) for early local features
  - Levels 2-5: Swin-ResNet blocks (3 SwinTransformer blocks + 1 down) for global features
  - Total: 5 downsampling stages
  - All downsampling operations use residual connections
  """

  def __init__(self, in_channels, base_channels=64):
    super().__init__()

    # Channel progression: 64 -> 128 -> 256 -> 512 -> 512
    # From paper architecture diagram (Fig. 2)
    channels = [base_channels, base_channels * 2, base_channels * 4,
                base_channels * 8, base_channels * 8]

    # Initial convolution (3x3 with stride 1)
    self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

    # Level 0: Conv-ResNet block (3 conv blocks + 1 down)
    self.level0_blocks = nn.ModuleList([
      ConvBlock(base_channels, channels[0]),
      ConvBlock(channels[0], channels[0]),
      ConvBlock(channels[0], channels[0])
    ])
    self.level0_down = nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=2, padding=1)

    # Levels 1-4: Swin-ResNet blocks (3 SwinTransformer blocks + 1 down)
    self.swin_levels = nn.ModuleList()
    self.down_blocks = nn.ModuleList()

    for i in range(1, 5):
      in_ch = channels[i - 1]
      out_ch = channels[i]

      # Match channels first if needed
      if in_ch != out_ch:
        channel_match = nn.Conv2d(in_ch, out_ch, kernel_size=1)
      else:
        channel_match = nn.Identity()

      # 3 SwinTransformer blocks with alternating shift (all use out_ch)
      swin_blocks = nn.ModuleList([
        SwinTransformerBlock(out_ch, window_size=7, num_heads=8,
                            shift_size=0 if j % 2 == 0 else 3)
        for j in range(3)
      ])

      self.swin_levels.append(nn.ModuleList([channel_match, swin_blocks]))

      # Downsampling with residual connection
      if i < 4:  # No downsampling at last level
        self.down_blocks.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
      else:
        self.down_blocks.append(nn.Identity())

  def forward(self, x):
    """
    Args:
        x: Structure image (CT + PTV + OARs) [B, in_channels, H, W]
    Returns:
        List of feature maps at different levels [f_0, f_1, f_2, f_3, f_4]
    """
    features = []

    # Initial convolution
    x = self.conv_in(x)

    # Level 0: Conv-ResNet block
    for block in self.level0_blocks:
      x = block(x)
    features.append(x)
    x = self.level0_down(x)

    # Levels 1-4: Swin-ResNet blocks
    for i, (level, down) in enumerate(zip(self.swin_levels, self.down_blocks)):
      channel_match, swin_blocks = level

      # Match channels
      x = channel_match(x)

      # Apply SwinTransformer blocks
      for swin_block in swin_blocks:
        x = swin_block(x)

      features.append(x)
      x = down(x)

    return features


class DenoisingNetwork(nn.Module):
  """
  Denoising Network - converts noise to dose distribution map.

  As described in Section III.B.2 of the paper:
  - Encoder: 5 levels (Conv at first, Swin at levels 2-5)
  - 2 intermediate SwinTransformer blocks (no up/downsample)
  - Decoder: 5 levels (Conv + Swin)
  - Fusion: Element-wise addition for levels 0-3, cross-attention for levels 4-5
  - Projector after each fusion
  - Timestep conditioning via sinusoidal embedding
  """

  def __init__(self, structure_encoder, base_channels=64, num_heads=8, cross_attention_levels=None):
    super().__init__()

    self.structure_encoder = structure_encoder

    # Channel progression: 64 -> 128 -> 256 -> 512 -> 512
    channels = [base_channels, base_channels * 2, base_channels * 4,
                base_channels * 8, base_channels * 8]

    # Fusion strategy: which levels use cross-attention
    # Default: [4] - only last level (conservative, based on best ablation results)
    # Paper mentions: [2, 3, 4] - "last three use cross-attention"
    # Best ablation (Table IV): [3, 4] - "attn-last2" achieves best results
    if cross_attention_levels is None:
      cross_attention_levels = [4]  # Conservative default
    self.cross_attention_levels = set(cross_attention_levels)

    # Timestep embedding
    time_embed_dim = base_channels * 4
    self.time_embed = nn.Sequential(
      SinusoidalPositionEmbedding(base_channels),
      nn.Linear(base_channels, time_embed_dim),
      nn.SiLU(),
      nn.Linear(time_embed_dim, time_embed_dim)
    )

    # Initial convolution for noisy dose
    self.conv_in = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)

    # Encoder: Level 0 uses Conv, Levels 1-4 use Swin
    self.encoder_stages = nn.ModuleList()
    self.encoder_downs = nn.ModuleList()

    for i in range(5):
      in_ch = channels[i - 1] if i > 0 else base_channels
      out_ch = channels[i]

      if i == 0:
        # Level 0: Conv blocks
        stage = nn.ModuleList([
          ResNetBlock(in_ch, out_ch, time_embed_dim),
          ResNetBlock(out_ch, out_ch, time_embed_dim),
          ResNetBlock(out_ch, out_ch, time_embed_dim)
        ])
      else:
        # Channel matching if needed (do this first)
        if in_ch != out_ch:
          channel_match = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
          channel_match = nn.Identity()

        # Levels 1-4: SwinTransformer blocks (all use out_ch)
        swin_blocks = nn.ModuleList([
          SwinTransformerBlock(out_ch, window_size=7, num_heads=num_heads,
                              shift_size=0 if j % 2 == 0 else 3)
          for j in range(3)
        ])

        stage = nn.ModuleList([channel_match] + list(swin_blocks))

      self.encoder_stages.append(stage)

      # Downsampling
      if i < 4:
        self.encoder_downs.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
      else:
        self.encoder_downs.append(nn.Identity())

    # Fusion strategy: configurable cross-attention levels
    # Create cross-attention modules only for specified levels
    self.cross_attentions = nn.ModuleList([
      CrossAttention(channels[i], num_heads) if i in self.cross_attention_levels else None
      for i in range(5)
    ])

    # Projectors after each fusion
    self.projectors = nn.ModuleList([
      Projector(channels[i]) for i in range(5)
    ])

    # 2 intermediate SwinTransformer blocks (no up/downsample)
    self.middle = nn.ModuleList([
      SwinTransformerBlock(channels[4], window_size=7, num_heads=num_heads, shift_size=0),
      SwinTransformerBlock(channels[4], window_size=7, num_heads=num_heads, shift_size=3)
    ])

    # Decoder: 5 levels
    self.decoder_stages = nn.ModuleList()
    self.decoder_ups = nn.ModuleList()

    for i in range(5):
      level = 4 - i  # Reverse order
      cur_ch = channels[level]  # Current channels (from skip connection)
      next_ch = channels[level - 1] if level > 0 else base_channels  # Channels for next level

      # Decoder blocks process at current resolution with current channels
      if level == 0:
        # Last decoder level (corresponds to encoder level 0): Conv blocks
        stage = nn.ModuleList([
          ResNetBlock(cur_ch, cur_ch, time_embed_dim),
          ResNetBlock(cur_ch, cur_ch, time_embed_dim),
          ResNetBlock(cur_ch, cur_ch, time_embed_dim)
        ])
      else:
        # Other levels: SwinTransformer blocks
        stage = nn.ModuleList([
          SwinTransformerBlock(cur_ch, window_size=7, num_heads=num_heads,
                              shift_size=0 if j % 2 == 0 else 3)
          for j in range(3)
        ])

      self.decoder_stages.append(stage)

      # Upsampling to next level (changes channels and resolution)
      if i < 4:  # Don't upsample after last decoder level
        self.decoder_ups.append(nn.ConvTranspose2d(cur_ch, next_ch, kernel_size=4, stride=2, padding=1))
      else:
        self.decoder_ups.append(nn.Identity())

    # Output
    self.norm_out = nn.GroupNorm(8, base_channels)
    self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

  def forward(self, x_noisy, x_struct, timesteps):
    """
    Args:
        x_noisy: Noisy dose map [B, 1, H, W]
        x_struct: Structural information [B, C, H, W]
        timesteps: Diffusion timesteps [B]
    Returns:
        Predicted noise [B, 1, H, W]
    """
    # Timestep embedding
    time_emb = self.time_embed(timesteps)

    # Extract structural features
    struct_features = self.structure_encoder(x_struct)

    # Initial convolution
    h = self.conv_in(x_noisy)

    # Encoder with fusion
    encoder_features = []

    for level in range(5):
      stage = self.encoder_stages[level]

      # Apply blocks
      for block in stage:
        if isinstance(block, ResNetBlock):
          h = block(h, time_emb)
        else:
          h = block(h)

      # Fuse with structural features
      struct_feat = struct_features[level]

      if self.cross_attentions[level] is not None:
        # Cross-attention fusion
        h = h + self.cross_attentions[level](h, struct_feat)
      else:
        # Element-wise addition fusion
        h = h + struct_feat

      # Apply projector
      h = self.projectors[level](h)

      # Store for skip connections (before downsampling)
      encoder_features.append(h)

      # Downsample for next level
      if level < 4:  # Don't downsample after last encoder level
        h = self.encoder_downs[level](h)

    # 2 intermediate SwinTransformer blocks
    for block in self.middle:
      h = block(h)

    # Decoder with skip connections
    for level in range(5):
      # Skip connection from encoder at same resolution
      skip = encoder_features[4 - level]

      # Add skip connection BEFORE upsampling (UNet-style)
      h = h + skip

      # Apply decoder blocks at current resolution
      stage = self.decoder_stages[level]
      for block in stage:
        if isinstance(block, ResNetBlock):
          h = block(h, time_emb)
        else:
          h = block(h)

      # Upsample to next resolution (except at last level)
      if level < 4:
        h = self.decoder_ups[level](h)

    # Output
    h = self.norm_out(h)
    h = F.silu(h)
    h = self.conv_out(h)

    return h


class SPDiffDose(nn.Module):
  """
  SP-DiffDose: Complete model combining Structure Encoder and Denoising Network.

  Implements the conditional DDPM framework for dose prediction as described in the paper.

  The model:
  1. Extracts anatomical features using Structure Encoder (Conv-ResNet + Swin-ResNet)
  2. Predicts noise using Denoising Network with timestep conditioning
  3. Fuses features using element-wise addition (levels 0-3) and cross-attention (levels 4-5)
  4. Optimizes fused features using Projector modules
  5. Uses SwinTransformer to capture multi-scale features
  """

  def __init__(self, structure_selector, base_channels=64, num_heads=8, cross_attention_levels=None):
    super().__init__()

    self.structure_selector = structure_selector
    in_channels = structure_selector.total_channels

    # Structure Encoder
    self.structure_encoder = StructureEncoder(in_channels, base_channels)

    # Denoising Network
    self.denoising_network = DenoisingNetwork(
      self.structure_encoder, base_channels, num_heads, cross_attention_levels
    )

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
    # Predict noise
    noise_pred = self.denoising_network(x_noisy, condition, timesteps)

    return noise_pred


def build_spdiffdose(config, structure_selector):
  """
  Build SP-DiffDose model from configuration.

  Default values correspond to the best configuration from the paper (Section IV):
  - Base channels: 64 (inferred from architecture diagram)
  - Window size: 7 for SwinTransformer blocks
  - Number of attention heads: 8
  - 5 encoder/decoder levels
  - Conv-ResNet for first encoder level, Swin-ResNet for others
  - Cross-attention levels: configurable (default [4], best ablation [3, 4])
  - Projector after each fusion
  - 2 intermediate SwinTransformer blocks

  Cross-attention configuration:
  - Default: [4] - only last level (conservative)
  - Paper text: [2, 3, 4] - "last three use cross-attention"
  - Best ablation (Table IV): [3, 4] - "attn-last2" achieves:
    Dose Score=1.901, DVH Score=1.533, HI=0.278

  Training settings from paper Section IV.B:
  - T = 1000 diffusion steps
  - Beta schedule: β_1 = 0.0001, β_T = 0.02 (linear)
  - Batch size: 8
  - Optimizer: Adam with initial learning rate 1e-4
  - Learning rate decay: linear from 1e-4 to 0 (epoch 100-200)
  - Total epochs: 200
  - Image size: 256x256

  Args:
      config: SP-DiffDose configuration dict with optional keys:
        - base_channels: Base channel count (default: 64)
        - num_heads: Number of attention heads (default: 8)
        - cross_attention_levels: List of levels to use cross-attention (default: [4])
          Examples:
            - [4]: Only last level (conservative)
            - [3, 4]: Last 2 levels (best ablation result - "attn-last2")
            - [2, 3, 4]: Last 3 levels (paper text - "last three")
      structure_selector: StructureSelector instance

  Returns:
      SPDiffDose model
  """
  # Get cross-attention levels from config
  # If not specified, use [4] as conservative default
  # Paper's best ablation uses [3, 4] ("attn-last2")
  cross_attention_levels = config.get('cross_attention_levels', [4])

  model = SPDiffDose(
    structure_selector=structure_selector,
    base_channels=config.get('base_channels', 64),
    num_heads=config.get('num_heads', 8),
    cross_attention_levels=cross_attention_levels
  )
  return model
