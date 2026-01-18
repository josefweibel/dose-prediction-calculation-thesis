"""
Progressive Distillation Trainer Module.

Implements Algorithm 2 from "Progressive Distillation for Fast Sampling of Diffusion Models"
(Salimans & Ho, ICLR 2022).

The key insight:
- Teacher takes 2 DDIM steps: z_t -> z_t' -> z_t''
- Student learns to match z_t'' in 1 step from z_t
- This halves the number of sampling steps each round
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch.distributed as dist
import contextlib

from data import create_gdphmm_dataset
from models import build_model
from utils.checkpointing import CHECKPOINTS_PATH, cleanup_checkpoints
from model_cache import DeepCache, CacheConfig
from diffusion_utils import (
  predict_x0,
  ddim_step,
  compute_distillation_target,
  compute_snr_weights,
)


class ProgressiveDistillationTrainerModule(nn.Module):
  """
  Progressive distillation training module.

  During training:
  1. Sample student timestep index i from [0, ..., student_T-1]
  2. Map to teacher timesteps: t (start), t' (mid), t'' (end)
  3. Add noise to get z_t
  4. Teacher (frozen) takes 2 DDIM steps: z_t -> z_t' -> z_t''
  5. Compute distillation target x_tilde from z_t and z_t''
  6. Student predicts x0 from z_t at the student's step
  7. Loss = MSE(student_x0_pred, x_tilde)
  """

  def __init__(
    self,
    student_model,
    teacher_model,
    beta_1,
    beta_T,
    train_T,
    teacher_T,
    student_T,
    parameterization='epsilon',
    loss_weighting='snr_plus_one',
    spatial_dims=2,
  ):
    """
    Args:
      student_model: The student denoising model (trainable)
      teacher_model: The teacher denoising model (frozen)
      beta_1: Starting noise level
      beta_T: Ending noise level
      train_T: Original training timesteps (e.g., 1000)
      teacher_T: Number of steps teacher uses (e.g., 512)
      student_T: Number of steps student learns (teacher_T / 2, e.g., 256)
      parameterization: 'epsilon' or 'v'
      loss_weighting: 'uniform', 'snr', 'snr_plus_one', or 'truncated_snr'
      spatial_dims: Number of spatial dimensions (2 for 2D models, 3 for 3D)
    """
    super().__init__()

    assert teacher_T == 2 * student_T, "Teacher T must be exactly 2x student T"
    assert spatial_dims in (2, 3), "spatial_dims must be 2 or 3"

    self.student_model = student_model
    self.teacher_model = teacher_model
    self.train_T = train_T
    self.teacher_T = teacher_T
    self.student_T = student_T
    self.parameterization = parameterization
    self.loss_weighting = loss_weighting
    self.spatial_dims = spatial_dims
    # Broadcast shape for noise schedule parameters: [B, 1, 1, 1] for 2D, [B, 1, 1, 1, 1] for 3D
    self.broadcast_shape = (-1,) + (1,) * (spatial_dims + 1)

    # Freeze teacher
    for param in self.teacher_model.parameters():
      param.requires_grad = False

    # Compute noise schedule
    betas = torch.linspace(beta_1, beta_T, train_T).float()
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
    self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1.0 - alphas_bar))

    # Compute teacher timestep indices (linearly spaced in [0, train_T-1])
    # Teacher uses teacher_T steps, so indices are spaced by train_T / teacher_T
    teacher_indices = torch.linspace(train_T - 1, 0, teacher_T + 1).long()
    self.register_buffer('teacher_indices', teacher_indices)

    # Precompute alpha and sigma at teacher timesteps
    teacher_alphas = self.sqrt_alphas_bar[teacher_indices]
    teacher_sigmas = self.sqrt_one_minus_alphas_bar[teacher_indices]
    self.register_buffer('teacher_alphas', teacher_alphas)
    self.register_buffer('teacher_sigmas', teacher_sigmas)

    # Compute loss weights based on student timesteps
    # Student step i corresponds to teacher step 2*i
    student_timestep_indices = torch.arange(0, teacher_T, 2)  # [0, 2, 4, ..., teacher_T-2]
    student_alphas = teacher_alphas[student_timestep_indices]
    student_sigmas = teacher_sigmas[student_timestep_indices]

    weights = compute_snr_weights(student_alphas, student_sigmas, loss_weighting)
    weights = weights / weights.mean()  # Normalize
    self.register_buffer('loss_weights', weights)

    # Cache for model inference
    self.teacher_cache = DeepCache(CacheConfig(intervals={}, warmup=0, layers=[], sampling_steps=train_T))
    self.student_cache = DeepCache(CacheConfig(intervals={}, warmup=0, layers=[], sampling_steps=train_T))

  def get_teacher_timesteps(self, student_step_idx):
    """
    Map student step index to teacher timesteps.

    Student step i spans teacher steps [2i, 2i+1, 2i+2]
    - t: start (2i)
    - t': middle (2i+1)
    - t'': end (2i+2)

    Args:
      student_step_idx: Student step index [B]

    Returns:
      t_idx, t_prime_idx, t_double_prime_idx: Indices into teacher schedule
    """
    t_idx = 2 * student_step_idx
    t_prime_idx = t_idx + 1
    t_double_prime_idx = t_idx + 2

    return t_idx, t_prime_idx, t_double_prime_idx

  def teacher_ddim_step(self, z, alpha_from, sigma_from, alpha_to, sigma_to,
                        t_discrete, condition):
    """
    Single deterministic DDIM step using the teacher model.

    Args:
      z: Current noisy sample [B, ...]
      alpha_from, sigma_from: Noise schedule params at current timestep
      alpha_to, sigma_to: Noise schedule params at target timestep
      t_discrete: Discrete timestep for model conditioning [B]
      condition: Conditioning information [B, ...]

    Returns:
      z_next: Sample at next timestep [B, ...]
    """
    with torch.no_grad():
      # Get teacher prediction
      model_output = self.teacher_model(z, condition, t_discrete, self.teacher_cache)

      # Convert to x0 prediction
      x0_pred = predict_x0(
        model_output, z,
        alpha_from.view(self.broadcast_shape),
        sigma_from.view(self.broadcast_shape),
        self.parameterization
      )

      # DDIM step
      z_next = ddim_step(
        z, x0_pred,
        alpha_from.view(self.broadcast_shape),
        sigma_from.view(self.broadcast_shape),
        alpha_to.view(self.broadcast_shape),
        sigma_to.view(self.broadcast_shape),
        eta=0.0  # Deterministic
      )

    return z_next

  def forward(self, x_0, condition):
    """
    Distillation training step.

    Args:
      x_0: Clean samples [B, C, D, H, W]
      condition: Conditioning [B, C, D, H, W]

    Returns:
      loss: Scalar distillation loss
    """
    batch_size = x_0.shape[0]
    device = x_0.device

    # Sample random student step indices
    student_step_idx = torch.randint(0, self.student_T, (batch_size,), device=device)

    # Map to teacher timesteps
    t_idx, t_prime_idx, t_double_prime_idx = self.get_teacher_timesteps(student_step_idx)

    # Get noise schedule parameters at each teacher timestep
    alpha_t = self.teacher_alphas[t_idx]
    sigma_t = self.teacher_sigmas[t_idx]
    alpha_t_prime = self.teacher_alphas[t_prime_idx]
    sigma_t_prime = self.teacher_sigmas[t_prime_idx]
    alpha_t_double_prime = self.teacher_alphas[t_double_prime_idx]
    sigma_t_double_prime = self.teacher_sigmas[t_double_prime_idx]

    # Get discrete timesteps for model conditioning
    t_discrete = self.teacher_indices[t_idx]
    t_prime_discrete = self.teacher_indices[t_prime_idx]

    # Add noise to get z_t
    noise = torch.randn_like(x_0)
    z_t = (
      alpha_t.view(self.broadcast_shape) * x_0 +
      sigma_t.view(self.broadcast_shape) * noise
    )

    # Teacher takes 2 DDIM steps (frozen, no gradients)
    z_t_prime = self.teacher_ddim_step(
      z_t, alpha_t, sigma_t, alpha_t_prime, sigma_t_prime,
      t_discrete, condition
    )
    z_t_double_prime = self.teacher_ddim_step(
      z_t_prime, alpha_t_prime, sigma_t_prime, alpha_t_double_prime, sigma_t_double_prime,
      t_prime_discrete, condition
    )

    # Compute distillation target: what x0 would the student need to predict
    # such that one DDIM step from z_t arrives at z_t''?
    x_tilde = compute_distillation_target(
      z_t, z_t_double_prime,
      alpha_t.view(self.broadcast_shape),
      sigma_t.view(self.broadcast_shape),
      alpha_t_double_prime.view(self.broadcast_shape),
      sigma_t_double_prime.view(self.broadcast_shape),
    )

    # Student predicts x0 from z_t (uses student's timestep mapping)
    # For the student, step i corresponds to the same noise level as teacher step 2*i
    student_output = self.student_model(z_t, condition, t_discrete, self.student_cache)

    # Convert student output to x0 prediction
    student_x0_pred = predict_x0(
      student_output, z_t,
      alpha_t.view(self.broadcast_shape),
      sigma_t.view(self.broadcast_shape),
      self.parameterization
    )

    # Weighted MSE loss
    weights = self.loss_weights[student_step_idx].view(self.broadcast_shape)
    loss = weights * F.mse_loss(student_x0_pred, x_tilde, reduction='none')

    return loss.mean()


class DistillationTrainer:
  """
  Trainer for a single distillation round.

  Handles:
  - Loading teacher checkpoint
  - Creating student (copy of teacher)
  - Training loop with distillation loss
  - Saving student checkpoints
  """

  def __init__(
    self,
    model_config,
    distill_config,
    teacher_T,
    student_T,
    round_idx,
    teacher_checkpoint_path,
    device,
    rank=None,
    world_size=None,
    local_rank=None,
  ):
    self.model_config = model_config
    self.distill_config = distill_config
    self.teacher_T = teacher_T
    self.student_T = student_T
    self.round_idx = round_idx
    self.device = device
    self.rank = rank
    self.local_rank = local_rank
    self.world_size = world_size if world_size is not None else 1
    self.is_distributed = rank is not None and world_size is not None
    self.is_main_process = not self.is_distributed or rank == 0

    # Create dataset
    conditioning_groups = model_config['conditioning']['types']
    train_dataset = create_gdphmm_dataset(model_config, 'train', structure_groups=conditioning_groups)

    if self.is_main_process:
      print(f'Training dataset size: {len(train_dataset)}')

    # Create dataloader
    train_sampler = None
    if self.is_distributed:
      train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
      )

    batch_size = distill_config['round'].get('batch_size', model_config.get('batch_size', 4))
    self.train_dataloader = DataLoader(
      train_dataset,
      batch_size=batch_size // self.world_size,
      shuffle=train_sampler is None,
      sampler=train_sampler,
      num_workers=model_config.get('num_workers', 4),
      pin_memory=True,
    )
    self.train_sampler = train_sampler

    # Build teacher model and load checkpoint
    self.teacher_model = build_model(model_config, train_dataset.selector).to(device)
    teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location=device)

    # Load EMA weights if available and configured
    if distill_config['teacher'].get('use_ema', True) and 'ema_model_state_dict' in teacher_checkpoint:
      self.teacher_model.load_state_dict(teacher_checkpoint['ema_model_state_dict'])
      if self.is_main_process:
        print('Loaded teacher EMA weights')
    else:
      self.teacher_model.load_state_dict(teacher_checkpoint['net_model_state_dict'])
      if self.is_main_process:
        print('Loaded teacher model weights')

    self.teacher_model.eval()

    # Build student model (copy from teacher)
    self.student_model = build_model(model_config, train_dataset.selector).to(device)
    if distill_config.get('student_init', 'copy_teacher') == 'copy_teacher':
      self.student_model.load_state_dict(self.teacher_model.state_dict())
      if self.is_main_process:
        print('Initialized student from teacher weights')

    # Create distillation trainer module
    parameterization = model_config['diffusion'].get('parameterization', 'epsilon')
    self.model = ProgressiveDistillationTrainerModule(
      student_model=self.student_model,
      teacher_model=self.teacher_model,
      beta_1=model_config['diffusion']['beta_1'],
      beta_T=model_config['diffusion']['beta_T'],
      train_T=model_config['diffusion']['T'],
      teacher_T=teacher_T,
      student_T=student_T,
      parameterization=parameterization,
      loss_weighting=distill_config['round'].get('loss_weighting', 'snr_plus_one'),
      spatial_dims=2,
    ).to(device)

    # EMA of student
    self.ema_model = copy.deepcopy(self.student_model).to(device)

    # Wrap for distributed
    if self.is_distributed:
      self.model = DDP(
        self.model,
        device_ids=[0] if device.type == 'cuda' else None,
        output_device=0 if device.type == 'cuda' else None,
        find_unused_parameters=False,
      )

    # Optimizer
    lr = distill_config['round'].get('lr', 0.00005)
    self.optim = torch.optim.AdamW(self.student_model.parameters(), lr=lr)

    # LR scheduler with warmup
    warmup_epochs = distill_config['round'].get('warmup_epochs', 2)
    total_epochs = distill_config['round'].get('epochs', 50)
    total_steps = len(self.train_dataloader) * total_epochs
    warmup_steps = len(self.train_dataloader) * warmup_epochs

    def lr_lambda(step):
      if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
      else:
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())

    self.lr_scheduler = LambdaLR(self.optim, lr_lambda=lr_lambda)

    # Mixed precision
    self.use_mixed_precision = device.type == 'cuda' and distill_config.get('mixed_precision', True)
    self.scaler = torch.amp.GradScaler('cuda') if self.use_mixed_precision else None

    # Grad clipping
    self.grad_clip = distill_config.get('grad_clip', 1.0)

    # EMA decay
    self.ema_decay = distill_config.get('ema_decay', 0.9999)

    # Save paths
    self.checkpoint_dir = (
      CHECKPOINTS_PATH / model_config['name'] /
      f"distill-{distill_config['name']}" /
      f"round-{round_idx}-{teacher_T}-to-{student_T}"
    )
    self.checkpoint_model_name = (
      f"{model_config['name']}/distill-{distill_config['name']}/"
      f"round-{round_idx}-{teacher_T}-to-{student_T}"
    )
    self.checkpoint_interval = distill_config['round'].get('checkpoint_interval', 10)

  def train(self, n_epochs, wandb_run=None):
    """
    Run distillation training for n_epochs.

    Args:
      n_epochs: Number of epochs
      wandb_run: Optional wandb run for logging

    Returns:
      Path to final checkpoint
    """
    pbar_epochs = tqdm(range(n_epochs), desc='Distillation', disable=not self.is_main_process)
    for epoch in pbar_epochs:
      if self.is_distributed and self.train_sampler is not None:
        self.train_sampler.set_epoch(epoch)

      self.model.train()

      pbar_batches = tqdm(
        self.train_dataloader,
        desc=f'Epoch {epoch}',
        leave=False,
        disable=not self.is_main_process
      )

      for structure, target, _ in pbar_batches:
        self.optim.zero_grad()
        structure = structure.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        condition = structure

        with torch.amp.autocast('cuda', dtype=torch.float16) if self.use_mixed_precision else contextlib.nullcontext():
          loss = self.model(target, condition)

        current_lr = self.optim.param_groups[0]['lr']

        if self.scaler is not None:
          self.scaler.scale(loss).backward()
          self.scaler.unscale_(self.optim)
          torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.grad_clip)
          self.scaler.step(self.optim)
          self.scaler.update()
        else:
          loss.backward()
          torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.grad_clip)
          self.optim.step()

        self.lr_scheduler.step()

        # Update EMA
        self._update_ema_weights(self.student_model, self.ema_model, self.ema_decay)

        if self.is_main_process:
          pbar_epochs.set_postfix(loss=loss.item())
          if wandb_run:
            wandb_run.log({
              'distill/loss': loss.item(),
              'distill/lr': current_lr,
              'distill/epoch': epoch,
              'distill/round': self.round_idx,
            })

      if self.is_distributed:
        dist.barrier()

      # Save checkpoint after each epoch
      if self.is_main_process:
        self._save_checkpoint(epoch, wandb_run)
        cleanup_checkpoints(self.checkpoint_model_name, self.checkpoint_interval)

      if self.is_distributed:
        dist.barrier()

    # Return path to final checkpoint
    return self.checkpoint_dir / f'{n_epochs - 1}.pt'

  def _update_ema_weights(self, source, target, decay):
    """Update EMA model weights."""
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
      target_dict[key].data.copy_(
        target_dict[key].data * decay + source_dict[key].data * (1 - decay)
      )

  def _save_checkpoint(self, epoch, wandb_run=None):
    """Save checkpoint."""
    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
      'epoch': epoch,
      'net_model_state_dict': self.student_model.state_dict(),
      'ema_model_state_dict': self.ema_model.state_dict(),
      'optimizer_state_dict': self.optim.state_dict(),
      'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
      'teacher_T': self.teacher_T,
      'student_T': self.student_T,
      'round_idx': self.round_idx,
    }

    if wandb_run:
      checkpoint_data['wandb_run_id'] = wandb_run.id

    if self.scaler is not None:
      checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()

    checkpoint_path = self.checkpoint_dir / f'{epoch}.pt'
    torch.save(checkpoint_data, checkpoint_path)
    print(f'Saved checkpoint: {checkpoint_path}')

  def finish(self):
    if hasattr(self, 'train_dataloader') and self.train_dataloader is not None:
      self.train_dataloader.dataset.finish()
    torch.cuda.empty_cache()
