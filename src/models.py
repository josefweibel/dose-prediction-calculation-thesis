import torch
import torch.nn as nn
import numpy as np

from models_mamba import VisionMamba


class TimeEmbedding(nn.Module):
  def __init__(self, embed_dim, scale=30.0):
    super().__init__()
    # Randomly sample weights druing initialization. These weights are fixed
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
    in_channels=1,
    in_struc_channels=5,
    out_channels=1,
    bias=False,
  ):
    super(MambaUnet, self).__init__()

    tdim = 24 * 4
    self.time_embedding = TimeEmbedding(tdim)

    self.t_proj_enc1 = TimeEmbeddingProjection(tdim, 48)
    self.t_proj_enc2 = TimeEmbeddingProjection(tdim, 96)
    self.t_proj_enc3 = TimeEmbeddingProjection(tdim, 192)
    self.t_proj_enc4 = TimeEmbeddingProjection(tdim, 384)

    self.t_proj_dec4 = TimeEmbeddingProjection(tdim, 192)
    self.t_proj_dec3 = TimeEmbeddingProjection(tdim, 96)
    self.t_proj_dec2 = TimeEmbeddingProjection(tdim, 48)
    self.t_proj_dec1 = TimeEmbeddingProjection(tdim, 48)

    self.head = nn.Conv2d(in_channels, 24, kernel_size=3, stride=1, padding=1)
    self.head_struc = nn.Conv2d(in_struc_channels, 24, kernel_size=3, stride=1, padding=1)

    self.enc1 = VisionMamba(img_size=128, patch_size=2, embed_dim=48, depth=4, channels=24)
    self.enc1_enc1 = VisionMamba(img_size=128, patch_size=2, embed_dim=48, depth=4, channels=24)

    self.enc2 = VisionMamba(img_size=64, patch_size=2, embed_dim=96, depth=6, channels=48)
    self.enc2_struc = VisionMamba(img_size=64, patch_size=2, embed_dim=96, depth=6, channels=48)

    self.enc3 = VisionMamba(img_size=32, patch_size=2, embed_dim=192, depth=6, channels=96)
    self.enc3_struc = VisionMamba(img_size=32, patch_size=2, embed_dim=192, depth=6, channels=96)

    self.enc4 = VisionMamba(img_size=16, patch_size=2, embed_dim=384, depth=8, channels=192)
    self.enc4_struc = VisionMamba(img_size=16, patch_size=2, embed_dim=384, depth=8, channels=192)

    self.up4 = Upsample(384)
    self.reduce_channels4 = nn.Conv2d(384, 192, kernel_size=1, bias=bias)
    self.dec4 = VisionMamba(img_size=16, patch_size=1, embed_dim=192, depth=6, channels=192)

    self.up3 = Upsample(192)
    self.reduce_channels3 = nn.Conv2d(192, 96, kernel_size=1, bias=bias)
    self.dec3 = VisionMamba(img_size=32, patch_size=1, embed_dim=96, depth=6, channels=96)

    self.up2 = Upsample(96)
    self.reduce_channels2 = nn.Conv2d(96, 48, kernel_size=1, bias=bias)
    self.dec2 = VisionMamba(img_size=64, patch_size=1, embed_dim=48, depth=4, channels=48)

    self.up1 = Upsample(48)
    self.dec1 = VisionMamba(img_size=128, patch_size=1, embed_dim=48, depth=4, channels=48)

    self.out = nn.Conv2d(48, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


  def forward(self, x, structure, t):
    t_emb = self.time_embedding(t)

    t_for_enc1 = self.t_proj_enc1(t_emb)[:, :, None, None]
    t_for_enc2 = self.t_proj_enc2(t_emb)[:, :, None, None]
    t_for_enc3 = self.t_proj_enc3(t_emb)[:, :, None, None]
    t_for_enc4 = self.t_proj_enc4(t_emb)[:, :, None, None]
    t_for_dec4 = self.t_proj_dec4(t_emb)[:, :, None, None]
    t_for_dec3 = self.t_proj_dec3(t_emb)[:, :, None, None]
    t_for_dec2 = self.t_proj_dec2(t_emb)[:, :, None, None]
    t_for_dec1 = self.t_proj_dec1(t_emb)[:, :, None, None]

    x = self.head(x)
    structure = self.head_struc(structure)

    out_enc1 = self.enc1(x)
    out_enc1_struc = self.enc1_struc(structure)
    out_enc1 = out_enc1 + out_enc1_struc + t_for_enc1

    out_enc2 = self.enc2(out_enc1)
    out_enc2_struc = self.enc2_struc(out_enc1_struc)
    out_enc2 = out_enc2 + out_enc2_struc + t_for_enc2

    out_enc3 = self.enc3(out_enc2)
    out_enc3_struc = self.enc3_struc(out_enc2_struc)
    out_enc3 = out_enc3 + out_enc3_struc + t_for_enc3

    out_enc4 = self.enc4(out_enc3)
    out_enc4_struc = self.enc4_struc(out_enc3_struc)
    out_enc4 = out_enc4 + out_enc4_struc + t_for_enc4

    in_dec4 = self.up4(out_enc4)
    in_dec4 = torch.cat([in_dec4, out_enc3], 1)
    in_dec4 = self.reduce_channels4(in_dec4)
    out_dec4 = self.dec4(in_dec4)
    out_dec4 = out_dec4 + t_for_dec4

    in_dec3 = self.up3(out_dec4)
    in_dec3 = torch.cat([in_dec3, out_enc2], 1)
    in_dec3 = self.reduce_channels3(in_dec3)
    out_dec3 = self.dec3(in_dec3)
    out_dec3 = out_dec3 + t_for_dec3

    in_dec2 = self.up2(out_dec3)
    in_dec2 = torch.cat([in_dec2, out_enc1], 1)
    in_dec2 = self.reduce_channels2(in_dec2)
    out_dec2 = self.dec2(in_dec2)
    out_dec2 = out_dec2 + t_for_dec2

    in_dec1 = self.up1(out_dec2)
    in_dec1 = torch.cat([in_dec1, x], 1)
    out_dec1 = self.d_1(in_dec1)
    out_dec1 = out_dec1 + t_for_dec1

    return self.out(out_dec1)

