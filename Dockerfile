FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ENV PYTHONUNBUFFERED=1

# Set up the working directory
WORKDIR /

RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    gpg-agent \
    build-essential apt-utils \
    python3-pip \
    python3-dev \
    libglib2.0-0 ffmpeg libsm6 libxext6 \
    wget

RUN apt-get install --reinstall ca-certificates

RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*
RUN pip3 install networkx


RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .

RUN pip3 install -r requirements.txt

ADD . .

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
