ARG BASE_IMAGE=harbor.aicloud.szu.edu.cn:5000/training/pytorch:2.8.0-cuda12.6-cudnn9-py311-ubuntu22.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies required by DeepSpeed multi-node launcher
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       pdsh \
       openssh-client \
       ninja-build \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies
COPY requirements_docker.txt /tmp/requirements.txt
RUN conda install -y gpustat \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip install --use-pep517 --no-build-isolation flash-attn \
    && conda clean -afy \
    && rm -f /tmp/requirements.txt
