FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ENV PYTHONUNBUFFERED=1

# Set up the working directory
WORKDIR /

RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    gpg-agent \
    build-essential apt-utils \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get install --reinstall ca-certificates

# PYTHON 3.11
RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt update --yes --quiet

RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.11 \
    python3.11-dev \
    libglib2.0-0 ffmpeg libsm6 libxext6 \
    python3.11-distutils \
    python3.11-lib2to3 \
    python3.11-gdbm \
    python3.11-tk \
    pip \
    wget \
    curl \
    git \
    unzip

RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Add your file
ADD . .

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install runpod

# Setup model directory
# RUN mkdir -p /models

# Download the model https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth?download=true
RUN wget -O /models/model.pth https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth?download=true
RUN wget -O /models/config.json https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json?download=true
RUN wget -O /models/vocab.json https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json?download=true


# CMD ["mistral"]
RUN chmod +x /start.sh

# Start the container
CMD /start.sh
