FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

ENV TZ=Europe/Zurich
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y software-properties-common && rm -rf /var/lib/apt/lists/*
RUN add-apt-repository multiverse
RUN apt-get update && apt-get install -y --allow-change-held-packages \
    wget \
    git \
    rdma-core \
    libibverbs-dev \
    librdmacm-dev \
    ibutils \
    perftest \
    ibverbs-providers \
    ibverbs-utils \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA CUDA network repository (contains NCCL builds with IB plugin)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm cuda-keyring_1.0-1_all.deb && \
    apt-get update

# Install NCCL runtime and dev packages
# This will also pull the version matched to your CUDA toolkit (12.x)
RUN apt-get install -y --allow-change-held-packages \
    libnccl2 \
    libnccl-dev \
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
  python-dotenv \
  ipykernel \
  jupyterlab

RUN git clone https://github.com/MIC-DKFZ/MedNeXt.git /tmp/mednext && \
    cd /tmp/mednext && \
    pip install . && \
    cd / && \
    rm -rf /tmp/mednext

ENV PATH=/opt/conda/bin:$PATH

WORKDIR /work
