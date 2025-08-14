from time import time
import torch
from torchmetrics import MeanSquaredError, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from src.metrics import DVHScore, DoseScore, HomogeneityIndex
from src.sampler import GaussianDiffusionSampler

class Evaluator:
  def __init__(self, dataloader, config, device):
    self.dataloader = dataloader
    self.config = config
    self.device = device

    self.metrics = {
      'mse': MeanSquaredError(),
      'psnr': PeakSignalNoiseRatio(),
      'ssim': StructuralSimilarityIndexMeasure(),
      'dvh': DVHScore(),
      'dose_score': DoseScore(),
      'homogeneity_index': HomogeneityIndex()
    }

    self.metrics = {k: v.to(device) for k, v in self.metrics.items()}

  def evaluate(self, model):
    model.eval()
    sampler = GaussianDiffusionSampler(
      model,
      self.config['beta_1'],
      self.config['beta_T'],
      self.config['T']
    ).to(self.device)

    with torch.no_grad():
      for metric in self.metrics.values():
        metric.reset()

      sum_duration = 0.0
      n_duration = 0

      for structure, target, info in self.dataloader:
        structure, target = structure.to(self.device), target.to(self.device)

        start = time()
        x_T = torch.randn_like(target, device=self.device)
        pred = sampler(x_T, structure)
        sum_duration += time() - start
        n_duration += target.shape[0]

        for metric in self.metrics.values():
          metric.update(pred, target)

    return {
      'duration': sum_duration / n_duration,
      **{k: m.compute().item() for k, m in self.metrics.items()}
    }

