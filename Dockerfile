FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

RUN apt-get update && apt-get install -y \
    git \
    rdma-core \
    libibverbs-dev \
    librdmacm-dev \
    ibutils \
    perftest \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mamba-ssm
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
  torchmetrics \
  python-dotenv

ENV PATH=/opt/conda/bin:$PATH

WORKDIR /work
