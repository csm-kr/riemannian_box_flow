# ────────────────────────────────────────────────
# Base: NVIDIA CUDA 13.0 + cuDNN + Ubuntu 22.04
# PyTorch: 2.11.0 (CUDA 13.0)
# ────────────────────────────────────────────────
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ────────────────────────────────────────────────
# System dependencies
# ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git wget curl \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20 LTS (NodeSource) — apt 기본 nodejs는 v12로 Claude Code 설치 불가
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# ────────────────────────────────────────────────
# Claude Code CLI 설치
# ────────────────────────────────────────────────
RUN npm install -g @anthropic-ai/claude-code

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# ────────────────────────────────────────────────
# PyTorch (CUDA 13.0)
# ────────────────────────────────────────────────
RUN pip install --upgrade pip && \
    pip install torch==2.11.0 torchvision \
        --index-url https://download.pytorch.org/whl/cu130

# ────────────────────────────────────────────────
# Python dependencies
# ────────────────────────────────────────────────
WORKDIR /workspace/riemannian_box_flow

RUN pip install \
    opencv-python \
    numpy \
    imageio \
    Pillow \
    matplotlib \
    tqdm \
    tensorboard \
    scipy

# ────────────────────────────────────────────────
# Project code
# ────────────────────────────────────────────────
COPY . .

# ────────────────────────────────────────────────
# Non-root user
# ────────────────────────────────────────────────
RUN useradd -m -s /bin/bash docker_user && \
    chown -R docker_user:docker_user /workspace/riemannian_box_flow

USER docker_user

RUN mkdir -p data outputs/checkpoints outputs/logs outputs/figures

EXPOSE 6006

CMD ["bash"]
