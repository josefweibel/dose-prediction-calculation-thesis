FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

RUN pip install --no-cache-dir mamba-ssm

RUN python -c "import mamba_ssm; print(mamba_ssm.__version__)"
