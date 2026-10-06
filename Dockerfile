# FlashCart runtime image (CUDA 12.8).
#
#   docker build -t flashcart:cu128 .
#   docker run --gpus all -it -v $PWD:/work -w /work flashcart:cu128 flashcart-train config.yaml

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN python3 -m pip install --no-cache-dir --break-system-packages \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch

COPY . /opt/flashcart
RUN python3 -m pip install --no-cache-dir --break-system-packages /opt/flashcart
