import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import numpy as np

from config import read_model_config
from data import create_gdphmm_dataset, StructureSelector
from evaluator import Evaluator, compute_volume_metrics
from mamba_model import remap_state_dict
from utils.mp import detect_device, PrefetchLoader
from utils.checkpointing import load_last_checkpoint, load_checkpoint
from results import SAMPLES_PATH
from models import build_model
from slice_combinator import SliceCombinator


class Sampler:
  def __init__(
    self,
    model_name,
    sampling_config,
    data_split,
    epoch=None,
    rank=None,
    world_size=None,
    local_rank=None,
  ):
    self.model_name = model_name
    self.model_config = read_model_config(model_name)
    self.sampling_config = sampling_config
    self.data_split = data_split
    self.rank = rank
    self.world_size = world_size if world_size is not None else 1
    self.is_distributed = rank is not None and world_size is not None
    self.is_main_process = not self.is_distributed or rank == 0

    self.device = detect_device(local_rank)

    self.dataset = create_gdphmm_dataset(self.model_config, data_split, structure_groups=None)

    conditioning_groups = self.model_config.get('conditioning', {}).get('types', ['ct', 'ptv', 'oar'])
    conditioning_selector = StructureSelector(groups=conditioning_groups)
    self.conditioning_total_channels = conditioning_selector.total_channels

    if self.is_main_process:
      print('Dataset size', len(self.dataset))

    self.sampler = None
    if self.is_distributed:
      self.sampler = DistributedSampler(
        self.dataset,
        num_replicas=world_size,
        rank=rank,
      )

    self.dataloader = DataLoader(
      self.dataset,
      batch_size=self.sampling_config['batch_size_per_gpu'],
      shuffle=False,
      sampler=self.sampler,
      num_workers=self.sampling_config['num_workers'],
      pin_memory=True,
      prefetch_factor=self.sampling_config['prefetch_factor'],
      persistent_workers=self.sampling_config['num_workers'] > 0,
    )

    if self.sampling_config['prefetch_on_gpu'] and self.device.type == 'cuda':
      self.dataloader = PrefetchLoader(self.dataloader, self.device)

    self.ema_model = build_model(self.model_config, self.conditioning_total_channels).to(self.device)

    if epoch is None or epoch == 'latest':
      checkpoint = load_last_checkpoint(model_name)
    else:
      checkpoint = load_checkpoint(model_name, epoch)

    self.epoch = checkpoint['epoch']

    model_state_dict = checkpoint['ema_model_state_dict']
    if "t_proj_enc1.proj.weight" in model_state_dict:
      model_state_dict = remap_state_dict(model_state_dict)
    self.ema_model.load_state_dict(model_state_dict)

    self.evaluator = Evaluator(
      self.dataloader,
      self.model_config,
      self.sampling_config,
      self.device,
      rank=self.rank,
      world_size=self.world_size,
    )

  def sample(self):
    print('Rank', self.rank, 'starts sampling')
    self.set_seed(self.sampling_config['seed'])

    dir = SAMPLES_PATH / self.model_name / self.sampling_config['name'] / f'{self.data_split}_{self.epoch}'

    self.ema_model.eval()
    _ = self.evaluator.evaluate(self.ema_model, dir=dir)

    if self.is_distributed:
      dist.barrier()

    if self.is_main_process:
      print('\nCombining slices into volumes...')
      combinator = SliceCombinator(
        model_name=self.model_name,
        sampling_name=self.sampling_config['name'],
        data_split=self.data_split,
        epoch=self.epoch,
        delete_slices=True,
        rank=self.rank,
        world_size=self.world_size,
      )
      combinator.combine()
      combinator.finish()

    if self.is_distributed:
      dist.barrier()

    print(f'\nRank {self.rank}: Computing volume metrics...')
    volume_metrics = compute_volume_metrics(
      dir,
      self.dataset,
      self.device,
      rank=self.rank,
      world_size=self.world_size
    )

    if self.is_distributed:
      dist.barrier()

    if self.is_main_process:
      if len(volume_metrics) > 0:
        overall_volume_metrics = volume_metrics.mean(numeric_only=True)

        volume_metrics.to_csv(dir / '_volume_metrics.csv', index=False)
        overall_volume_metrics.to_frame().to_csv(dir / '_overall_volume_metrics.csv')

        print(f'Volume metrics saved to {dir}')
      else:
        print('No volume metrics computed')

    if self.is_distributed:
      dist.barrier()

  def finish(self):
    print('finishing sampler instance', flush=True)

    self.dataset.finish()
    self.dataloader = None

    torch.cuda.empty_cache()

  def set_seed(self, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

