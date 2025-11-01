import torch
from dataclasses import dataclass
from typing import Dict


@dataclass
class CacheConfig:
  sampling_steps: int
  intervals: Dict[str, int] # e.g. {"enc4": 4, "enc3": 2}
  warmup: int = 0  # number of initial steps with no cache (optional)


class DeepCache:
  def __init__(self, cfg: CacheConfig):
    self.cfg = cfg
    self.cache = {}  # layer_name -> (feature, last_step)
    self.current_step = None
    self.cached_steps = self._compute_cached_steps()  # layer_name -> list of step indices where cache is used

  def clear(self):
    self.cache.clear()

  def get(self, layer: str):
    return self.cache.get(layer, (None, None))[0]

  def set(self, layer: str, feat: torch.Tensor):
    self.cache[layer] = (feat.detach(), self.current_step)

  def set_current_step(self, step: int):
    self.current_step = step

  def should_use(self, layer: str) -> bool:
    return self.current_step in self.cached_steps.get(layer, [])

  def get_or_compute(self, layer: str, compute_fn):
    if self.should_use(layer):
      return self.get(layer)
    else:
      feat = compute_fn()
      self.set(layer, feat)
      return feat

  def _compute_cached_steps(self) -> Dict[str, list]:
    """Compute which step indices each layer uses the cache (vs. computing new features)."""
    cached_steps = {}

    for layer, interval in self.cfg.intervals.items():
      layer_cached_steps = []

      for step in range(self.cfg.sampling_steps):
        # Skip warmup steps (cache never used)
        if step < self.cfg.warmup:
          continue

        # Adjust step to account for warmup
        adjusted_step = step - self.cfg.warmup

        # Cache is used when step is NOT a multiple of the interval (and not the first step)
        # First step after warmup (adjusted_step == 0) is always computed
        if adjusted_step > 0 and adjusted_step % interval != 0:
          layer_cached_steps.append(step)

      cached_steps[layer] = layer_cached_steps

    return cached_steps
