"""
Progressive Distillation Orchestrator.

Manages the progressive distillation process:
T -> T/2 -> T/4 -> ... -> target_T steps

Initial T is taken from model config (diffusion.T).
Target T is specified in distillation config (distillation.target_T).

Each round:
- Teacher is the previous student (or original model for round 0)
- Student halves the number of steps
- All intermediate models are saved for comparison

Features:
- Wandb logging per round (grouped by model name)
- Automatic evaluation after each round
- Resume from any round
"""

import wandb
import torch
from torch.utils.data import DataLoader, DistributedSampler

from utils.checkpointing import CHECKPOINTS_PATH, load_last_checkpoint, find_checkpoint
from distillation_trainer import DistillationTrainer
from evaluator import Evaluator
from data import create_gdphmm_dataset


class ProgressiveDistillationOrchestrator:
  """
  Manages progressive distillation: T -> T/2 -> T/4 -> ... -> target_T steps.

  Each round: teacher becomes previous student, student halves steps.
  """

  def __init__(
    self,
    model_config,
    distill_config,
    device,
    rank=None,
    world_size=None,
    local_rank=None,
  ):
    """
    Args:
      model_config: Model configuration dict
      distill_config: Distillation configuration dict
      device: PyTorch device
      rank: Distributed rank (None for single GPU)
      world_size: Number of processes
      local_rank: Local rank for multi-GPU
    """
    self.model_config = model_config
    self.distill_config = distill_config
    self.device = device
    self.rank = rank
    self.world_size = world_size
    self.local_rank = local_rank
    self.is_distributed = rank is not None and world_size is not None
    self.is_main_process = not self.is_distributed or rank == 0

    # Compute distillation rounds
    self.rounds = self._compute_rounds()

    # Wandb settings
    self.wandb_project = 'dose-prediction-thesis'
    self.wandb_entity = 'joseph-weibel'
    self.run_group = f"{model_config['name']}-distill-{distill_config['name']}"

    if self.is_main_process:
      print(f"\nProgressive Distillation Plan:")
      print(f"  Model: {model_config['name']}")
      print(f"  Config: {distill_config['name']}")
      print(f"  Rounds: {len(self.rounds)}")
      for i, (t, s) in enumerate(self.rounds):
        print(f"    Round {i}: {t} -> {s} steps")
      print()

  def _compute_rounds(self):
    """Compute distillation rounds from initial_T to target_T."""
    initial_T = self.model_config['diffusion']['T']
    target_T = self.distill_config['distillation']['target_T']

    rounds = []
    current_T = initial_T
    while current_T > target_T:
      rounds.append((current_T, current_T // 2))
      current_T //= 2

    return rounds

  def _get_initial_teacher_checkpoint(self):
    """Get the path to the initial teacher checkpoint."""
    epoch = self.distill_config['teacher'].get('checkpoint_epoch', 'latest')

    if epoch == 'latest':
      checkpoint_info = load_last_checkpoint(self.model_config['name'])
      if checkpoint_info is None:
        raise ValueError(f"No checkpoint found for model {self.model_config['name']}")
      # load_last_checkpoint returns the checkpoint dict, we need the path
      checkpoint_dir = CHECKPOINTS_PATH / self.model_config['name']
      # Find the latest checkpoint file
      checkpoints = sorted(checkpoint_dir.glob('*.pt'))
      if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
      return checkpoints[-1]
    else:
      return find_checkpoint(self.model_config['name'], epoch)

  def _get_round_checkpoint_dir(self, round_idx, teacher_T, student_T):
    """Get checkpoint directory for a distillation round."""
    return (
      CHECKPOINTS_PATH / self.model_config['name'] /
      f"distill-{self.distill_config['name']}" /
      f"round-{round_idx}-{teacher_T}-to-{student_T}"
    )

  def _find_round_checkpoint(self, round_idx, teacher_T, student_T):
    """Find the latest checkpoint for a round, if exists."""
    checkpoint_dir = self._get_round_checkpoint_dir(round_idx, teacher_T, student_T)
    if not checkpoint_dir.exists():
      return None

    checkpoints = sorted(checkpoint_dir.glob('*.pt'))
    if not checkpoints:
      return None

    return checkpoints[-1]

  def _is_round_complete(self, round_idx, teacher_T, student_T):
    """Check if a round has been completed (reached target epochs)."""
    checkpoint = self._find_round_checkpoint(round_idx, teacher_T, student_T)
    if checkpoint is None:
      return False

    # Extract epoch from filename (e.g., "49.pt" -> 49)
    epoch = int(checkpoint.stem)
    target_epochs = self.distill_config['round'].get('epochs', 50)
    return epoch >= target_epochs - 1

  def _find_resume_point(self):
    """
    Find the round and checkpoint to resume from.

    Returns:
      tuple: (round_idx, teacher_checkpoint_path, student_checkpoint_path)
        - round_idx: The round to resume/start from
        - teacher_checkpoint_path: Path to teacher checkpoint for this round
        - student_checkpoint_path: Path to student checkpoint if resuming mid-round, else None
    """
    # Check each round from the end to find where to resume
    for round_idx in range(len(self.rounds) - 1, -1, -1):
      teacher_T, student_T = self.rounds[round_idx]
      student_ckpt = self._find_round_checkpoint(round_idx, teacher_T, student_T)

      if student_ckpt is not None:
        # Found a checkpoint for this round
        if self._is_round_complete(round_idx, teacher_T, student_T):
          # Round is complete, start from next round
          if round_idx + 1 < len(self.rounds):
            # Use this completed student as teacher for next round
            return round_idx + 1, student_ckpt, None
          else:
            # All rounds complete
            return len(self.rounds), student_ckpt, None
        else:
          # Round was interrupted, resume it
          # Need to find the teacher checkpoint for this round
          if round_idx == 0:
            teacher_ckpt = self._get_initial_teacher_checkpoint()
          else:
            prev_teacher_T, prev_student_T = self.rounds[round_idx - 1]
            teacher_ckpt = self._find_round_checkpoint(round_idx - 1, prev_teacher_T, prev_student_T)
            if teacher_ckpt is None:
              raise ValueError(f"Cannot find teacher checkpoint for interrupted round {round_idx}")
          return round_idx, teacher_ckpt, student_ckpt

    # No checkpoints found, start from round 0
    return 0, self._get_initial_teacher_checkpoint(), None

  def run_round(self, round_idx, teacher_checkpoint_path, student_checkpoint_path=None):
    """
    Execute a single distillation round with wandb logging.

    Args:
      round_idx: Round index (0-based)
      teacher_checkpoint_path: Path to teacher checkpoint
      student_checkpoint_path: Optional path to student checkpoint for resuming

    Returns:
      Path to student checkpoint after training
    """
    teacher_T, student_T = self.rounds[round_idx]

    if self.is_main_process:
      print(f"\n{'='*60}")
      print(f"Round {round_idx}: {teacher_T} -> {student_T} steps")
      print(f"Teacher checkpoint: {teacher_checkpoint_path}")
      if student_checkpoint_path:
        print(f"Resuming from student checkpoint: {student_checkpoint_path}")
      print(f"{'='*60}\n")

    # Initialize wandb for this round
    wandb_run = None
    if self.is_main_process:
      wandb.login()

      # Check if we should resume an existing wandb run
      wandb_run_id = None
      if student_checkpoint_path is not None:
        checkpoint = torch.load(student_checkpoint_path, map_location='cpu')
        wandb_run_id = checkpoint.get('wandb_run_id')
        if wandb_run_id:
          print(f"Resuming wandb run: {wandb_run_id}")

      wandb_run = wandb.init(
        project=self.wandb_project,
        entity=self.wandb_entity,
        name=f"{self.run_group}-round{round_idx}-{teacher_T}to{student_T}",
        config={
          'model': self.model_config['name'],
          'distill_config': self.distill_config['name'],
          'round': round_idx,
          'teacher_T': teacher_T,
          'student_T': student_T,
          **self.distill_config['round'],
        },
        group=self.run_group,
        reinit=True,
        id=wandb_run_id,
        resume='must' if wandb_run_id else None,
      )

    # Create distillation trainer
    trainer = DistillationTrainer(
      model_config=self.model_config,
      distill_config=self.distill_config,
      teacher_T=teacher_T,
      student_T=student_T,
      round_idx=round_idx,
      teacher_checkpoint_path=teacher_checkpoint_path,
      student_checkpoint_path=student_checkpoint_path,
      device=self.device,
      rank=self.rank,
      world_size=self.world_size,
      local_rank=self.local_rank,
    )

    # Train
    n_epochs = self.distill_config['round'].get('epochs', 50)
    student_checkpoint_path = trainer.train(n_epochs, wandb_run)

    # Evaluate finished student
    if self.distill_config.get('eval_after_round', True):
      self._evaluate_student(trainer.ema_model, student_T, wandb_run)

    # Cleanup
    trainer.finish()

    if wandb_run:
      wandb_run.finish()

    return student_checkpoint_path

  def _evaluate_student(self, model, student_T, wandb_run):
    """
    Run evaluation on the finished student model.

    Args:
      model: Student model (EMA weights)
      student_T: Number of sampling steps for the student
      wandb_run: Wandb run for logging
    """
    if self.is_main_process:
      print(f"\nEvaluating student model with T={student_T}...")

    # Create validation dataset
    val_dataset = create_gdphmm_dataset(self.model_config, 'valid', structure_groups=None)

    val_sampler = None
    if self.is_distributed:
      val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=self.world_size,
        rank=self.rank,
        shuffle=False
      )

    val_dataloader = DataLoader(
      val_dataset,
      batch_size=self.model_config.get('batch_size', 4),
      shuffle=False,
      sampler=val_sampler,
      num_workers=self.model_config.get('num_workers', 4),
      pin_memory=True,
    )

    # Create sampler config for evaluation
    # Use DDIM with student_T steps
    sampler_config = {
      'name': f'ddim-{student_T}',
      'sampler': 'ddim',
      'T': student_T,
      'eta': 0,
      'schedule': 'linear',
    }

    # Create evaluator
    evaluator = Evaluator(
      val_dataloader,
      self.model_config,
      sampler_config,
      self.device,
      rank=self.rank,
      world_size=self.world_size,
    )

    # Run evaluation
    model.eval()
    _, metrics = evaluator.evaluate(model)

    # Log to wandb
    if self.is_main_process and wandb_run:
      eval_metrics = {f'eval/{k}': v for k, v in metrics.to_dict().items()}
      eval_metrics['eval/sampling_steps'] = student_T
      wandb_run.log(eval_metrics)
      print(f"Evaluation (T={student_T}): {metrics}")

    # Cleanup
    val_dataset.finish()

  def run_all(self, start_round=None, end_round=None, auto_resume=True):
    """
    Execute all distillation rounds sequentially.

    Args:
      start_round: Round to start from (0-based). If None and auto_resume=True,
                   automatically detects from checkpoints.
      end_round: Round to end at (exclusive, None for all)
      auto_resume: If True and start_round is None, automatically detect
                   resume point from existing checkpoints.
    """
    if end_round is None:
      end_round = len(self.rounds)

    # Determine starting point
    student_ckpt_resume = None
    if auto_resume:
      # Auto-detect resume point
      start_round, teacher_ckpt, student_ckpt_resume = self._find_resume_point()

      if start_round >= len(self.rounds):
        if self.is_main_process:
          print(f"\nAll {len(self.rounds)} distillation rounds already complete.")
          print(f"Final model: {teacher_ckpt}")
        return teacher_ckpt

      if self.is_main_process:
        if student_ckpt_resume:
          print(f"\nResuming interrupted round {start_round}")
        else:
          print(f"\nResuming from round {start_round}")
    elif start_round is None:
      start_round = 0
      teacher_ckpt = self._get_initial_teacher_checkpoint()
    elif start_round == 0:
      teacher_ckpt = self._get_initial_teacher_checkpoint()
    else:
      # Load from previous round's checkpoint
      prev_teacher_T, prev_student_T = self.rounds[start_round - 1]
      prev_ckpt = self._find_round_checkpoint(start_round - 1, prev_teacher_T, prev_student_T)
      if prev_ckpt is None:
        raise ValueError(
          f"Cannot resume from round {start_round}: "
          f"no checkpoint found for round {start_round - 1}"
        )
      teacher_ckpt = prev_ckpt

    if self.is_main_process:
      print(f"\nStarting distillation from round {start_round}")
      print(f"Initial teacher: {teacher_ckpt}")
      if student_ckpt_resume:
        print(f"Resuming student from: {student_ckpt_resume}")
      print()

    # Run each round
    for round_idx in range(start_round, end_round):
      # Only pass student checkpoint for the first round if resuming
      student_ckpt_to_pass = student_ckpt_resume if round_idx == start_round else None
      student_ckpt = self.run_round(round_idx, teacher_ckpt, student_ckpt_to_pass)
      teacher_ckpt = student_ckpt  # Student becomes teacher for next round

    if self.is_main_process:
      print(f"\n{'='*60}")
      print(f"Distillation complete!")
      print(f"Final model: {teacher_ckpt}")
      print(f"{'='*60}\n")

    return teacher_ckpt
