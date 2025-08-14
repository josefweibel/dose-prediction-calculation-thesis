FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

COPY --from=ghcr.io/josefweibel/dose-prediction-thesis-builder:latest /opt/conda/lib/python3.11/site-packages /opt/conda/lib/python3.11/site-packages

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
  numpy \
  pandas \
  matplotlib \
  rope \
  scipy \
  seaborn \
  timm \
  tqdm \
  jupyter \
  PyYAML \
  wandb \
  torchmetrics

ENV PATH=/opt/conda/bin:$PATH

WORKDIR /work
