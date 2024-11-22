# Use a base image with CUDA 11.7
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Install Python tools, system dependencies, and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential git zip htop screen libgl1-mesa-glx libglib2.0-0 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    apt-get clean

# Install PyTorch and related packages
RUN pip3 install torch==1.13.0+cu117 \
    torchaudio==0.13.0+cu117 \
    torchvision==0.14.0+cu117 \
    -f https://download.pytorch.org/whl/cu117/torch_stable.html

# Set environment variables for CUDA
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    FORCE_CUDA="1"

# Set working directory
WORKDIR /YoloR

# Copy YOLOR code to the container
COPY . .

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    seaborn \
    thop

# Install mish-cuda (if needed)
RUN git clone https://github.com/JunnYu/mish-cuda /tmp/mish-cuda && \
    cd /tmp/mish-cuda && \
    python setup.py build install && \
    cd / && rm -rf /tmp/mish-cuda

# Install pytorch_wavelets (if needed)
RUN git clone https://github.com/fbcotter/pytorch_wavelets /tmp/pytorch_wavelets && \
    cd /tmp/pytorch_wavelets && \
    pip install . && \
    cd / && rm -rf /tmp/pytorch_wavelets

# Volume mounts (optional, can also be set when running the container)
VOLUME ["/data", "/yolor"]

# Expose a default port (for inference if needed)
EXPOSE 5000

# Default command to keep the container running
CMD ["/bin/bash"]
