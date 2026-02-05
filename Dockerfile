ARG BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    SHELL=/bin/bash

WORKDIR /code

# System deps:
# - openssh-server: required by the platform for multi-node DeepSpeed
# - build tools: required to compile DeepSpeed ops
# - audio deps: required by librosa/soundfile
# - mpi: required by mpi4py
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    git \
    curl \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    libaio-dev \
    libsndfile1 \
    ffmpeg \
    openmpi-bin \
    libopenmpi-dev \
    tzdata \
    locales \
    vim \
    wget \
    iputils-ping \
    iproute2 \
    net-tools \
    dnsutils \
    tmux \
    htop \
    unzip \
    less \
  && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
  && echo $TZ > /etc/timezone \
  && locale-gen en_US.UTF-8 \
  && rm -rf /var/lib/apt/lists/*

# Enable sshd (generate host keys so sshd can start)
RUN mkdir -p /var/run/sshd \
  && ssh-keygen -A \
  && sed -ri 's/^#?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config \
  && sed -ri 's/^#?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config

# Install Python deps
COPY requirements_docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt --break-system-packages

# Entrypoint keeps container alive and starts sshd for platform plugins
RUN printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -e' \
  'if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then ssh-keygen -A; fi' \
  '/usr/sbin/sshd' \
  'exec "$@"' \
  > /usr/local/bin/entrypoint.sh \
  && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# Keep container alive for dev plugins; platform can override the command.
CMD ["/bin/bash", "-lc", "sleep infinity"]
