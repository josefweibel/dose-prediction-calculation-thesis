# Dose Prediction & Dose Calculation Thesis

Master's thesis code on **radiotherapy dose prediction and dose calculation for head-and-neck cancer with diffusion models**.
Given a planning CT together with the delineated structures (body, PTVs, OARs, signed distance maps, beam
plate), the models predict the 3D dose distribution slice by slice.

The focus is on making diffusion-based dose prediction and dose calculation **fast enough to be practical**: the repo compares
diffusion backbones and baselines (U-Net-style regression models, flow matching, InDI), and studies
step-count reduction, alternative samplers (DDPM/DDIM/DEIS), feature caching (DeepCache-style), mixed
precision/quantization, and progressive distillation — all evaluated with clinically motivated metrics
(dose score, DVH score, conformity/homogeneity/gradient index, gamma index, constraint satisfaction).

## Setup

Python 3.13, dependencies managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

Scripts import from `src/`, so run them with `PYTHONPATH` set:

```bash
PYTHONPATH=src python scripts/train.py --config 101-beamdiff
```

For the GPU cluster, `Dockerfile` builds the image (pushed to `ghcr.io/josefweibel/dose-prediction-thesis`)
that the `*_on_slurm.sh` wrappers pull and run through Singularity.

### Data

Raw GDP-HMM head-and-neck plans go to `data/gdphmm-hn-vmat/{train,valid,test}` (not in git).
`scripts/preprocess_gdphmm_slices.py` converts the volumes into preprocessed 2D slices under
`data/gdphmm-hn-vmat_slices/`, which is what training and sampling read.

## Typical workflow

```bash
# 1. preprocess volumes into slices
PYTHONPATH=src python scripts/preprocess_gdphmm_slices.py

# 2. train a model (config name = file in configs/models without .yaml)
PYTHONPATH=src python scripts/train.py --config 101-beamdiff

# 3. sample the test split with a given sampler config, writing per-patient volumes + metrics
PYTHONPATH=src python scripts/sample.py --model 101-beamdiff --config ddpm-1000 --epoch latest
```

Each script has an `*_on_slurm.sh` counterpart that submits the same command as an sbatch job.
Training runs are logged to Weights & Biases; checkpoints land in `checkpoints/<model>/`, samples and
their metrics in `samples/<model>/<sampler>/<split>_<epoch>/`.

## Layout

| Path | Contents |
| --- | --- |
| `src/` | library code: data, models, training, sampling, metrics |
| `scripts/` | CLI entry points (`train.py`, `sample.py`, `distill.py`, preprocessing, DICOM conversion) plus SLURM wrappers |
| `configs/models/` | one YAML per training run (architecture, data, diffusion schedule, conditioning) |
| `configs/samplers/` | one YAML per inference setting (sampler type, step count, caching, precision, seed) |
| `configs/distillation/` | progressive distillation schedules |
| `data/`, `samples/`, `checkpoints/`, `wandb/` | generated artifacts, git-ignored |

### Key modules in `src/`

- `data.py` — dataset, structure lists, normalization ranges, `StructureSelector` for conditioning channels
- `config.py` — config loading
- `models.py` — `build_model`, dispatching to the architectures: `beamdiff_model.py`, `spdiffdose_model.py`,
  `dosediff_model.py`, `diffdp_model.py`, `c3d_model.py`, `mednext.py`, `hdunet_model.py`,
  `deeplabv3plus_model.py`, `mamba_model.py`
- `trainer.py` / `sampler.py` — distributed training and sampling loops; `diffusion_utils.py` holds the schedules
- `model_cache.py` — DeepCache-style feature caching used by the `ddpm-cache-*` sampler configs
- `distillation.py`, `distillation_trainer.py` — progressive distillation
- `metrics.py`, `evaluator.py`, `metrics_recalculator.py` — the metric implementations and their evaluation harness
- `results.py` — loading metric tables and the plotting helpers used by the notebooks
