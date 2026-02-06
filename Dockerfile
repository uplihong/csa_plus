ARG BASE_IMAGE=harbor.aicloud.szu.edu.cn:5000/training/pytorch:2.8.0-cuda12.6-cudnn9-py311-ubuntu22.04
FROM ${BASE_IMAGE}

# Install Python deps
COPY requirements_docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
