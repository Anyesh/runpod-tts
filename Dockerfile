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
    python3.11-distutils \
    python3.11-lib2to3 \
    python3.11-gdbm \
    python3.11-tk \
    pip \
    wget \
    curl \
    git \
    unzip

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 999 \
    && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --upgrade pip

# Add your file
ADD . .

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install runpod

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
