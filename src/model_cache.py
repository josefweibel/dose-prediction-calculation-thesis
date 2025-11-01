import torch
from dataclasses import dataclass
from typing import Dict, Optional, Literal, List
import numpy as np


@dataclass
class CacheConfig:
  sampling_steps: int
  warmup: int = 0  # number of initial steps with no cache (optional)
  layers: List[str]  # list of layer names to cache
  mode: Literal["uniform", "nonuniform"] = "uniform"  # caching strategy mode

  intervals: Optional[Dict[str, int]] # e.g. {"enc4": 4, "enc3": 2}

  # Non-uniform 1:N parameters (required when mode="nonuniform")
  c: Optional[float] = None  # center timestep around which sampling is concentrated
  p: Optional[float] = None  # power value controlling the distribution curve
  k: Optional[int] = None    # number of cache updates (⌈T/N⌉)


class DeepCache:
  def __init__(self, cfg: CacheConfig):
    self.cfg = cfg
    self.cache = {}  # layer_name -> (feature, last_step)
    self.current_step = None
    self.cached_steps = self.__compute_cached_steps()  # layer_name -> list of step indices where cache is used

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

  def __compute_cached_steps(self) -> Dict[str, list]:
    """Compute which step indices each layer uses the cache (vs. computing new features)."""
    cached_steps = {}

    for layer in self.cfg.layers:
      if self.cfg.mode == "nonuniform":
        layer_cached_steps = self.__compute_nonuniform_cached_steps()
      else:  # uniform mode (default)
        layer_cached_steps = self.__compute_uniform_cached_steps(layer)

      cached_steps[layer] = layer_cached_steps

    return cached_steps

  def __compute_uniform_cached_steps(self, layer: str) -> list:
    """Compute cached steps using uniform 1:N strategy."""

    if self.cfg.intervals is None or layer not in self.cfg.intervals:
      raise ValueError(f"Uniform mode requires interval for layer '{layer}' to be set in cfg.intervals")

    layer_cached_steps = []

    for step in range(self.cfg.sampling_steps):
      # Skip warmup steps (cache never used)
      if step < self.cfg.warmup:
        continue

      # Adjust step to account for warmup
      adjusted_step = step - self.cfg.warmup

      # Cache is used when step is NOT a multiple of the interval (and not the first step)
      # First step after warmup (adjusted_step == 0) is always computed
      if adjusted_step > 0 and adjusted_step % self.cfg.intervals[layer] != 0:
        layer_cached_steps.append(step)

    return layer_cached_steps

  def __compute_nonuniform_cached_steps(self) -> list:
    """Compute cached steps using non-uniform 1:N strategy.

    The non-uniform strategy concentrates full inference around timestep c,
    with frequency changing according to power p.

    Args:
      interval: N value for 1:N inference (cache reuse ratio)

    Returns:
      List of step indices where cache should be used
    """
    if self.cfg.c is None or self.cfg.p is None or self.cfg.k is None:
        raise ValueError("Non-uniform mode requires c, p, and k parameters to be set")

    c = self.cfg.c
    p = self.cfg.p
    k = self.cfg.k
    T = self.cfg.sampling_steps

    # Generate non-uniform update indices using power-law transformation
    l_k = np.linspace((-c) ** (1/p), (T - c) ** (1/p), k)
    update_indices = (l_k ** p) + c

    # Keep only valid indices (within range and after warmup)
    update_indices = np.unique(update_indices)
    update_indices = update_indices[(update_indices >= self.cfg.warmup) & (update_indices < T)]

    # Cache is used on all steps NOT in update_indices
    return [step for step in range(T) if step not in update_indices]
