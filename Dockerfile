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
    pip

RUN apt-get install --yes --quiet --no-install-recommends \
    curl wget
# Install pip using get-pip.py to ensure it’s correctly installed
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.11 get-pip.py && rm get-pip.py

# Install html5lib if it’s a required dependency
RUN python3.11 -m pip install html5lib

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN pip install --upgrade pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .

RUN pip install -r requirements.txt

# Setup model directory
# RUN mkdir -p /models

ADD . .

RUN wget -O /models/model.pth https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth?download=true
RUN wget -O /models/config.json https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json?download=true
RUN wget -O /models/vocab.json https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json?download=true

# CMD ["mistral"]
RUN chmod +x /start.sh

# Start the container
CMD /start.sh
