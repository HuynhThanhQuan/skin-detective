#!/bin/sh
FROM --platform=linux/amd64 pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
# nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Install and update packages
RUN apt-get update

# Config tzdata and install opencv lib
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install -y ffmpeg libsm6 libxext6
    
# Install python3.8 and pip
RUN apt-get install -y \
    python3.8-dev \
    python3-pip \
    python3-dev
RUN apt-get install -y \
    software-properties-common \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update

RUN apt-get install -y \
    sudo \
    wget \ 
    vim \
    zip \
    unzip \
    git \
    make \
    gcc \
    nano
    
# Install and update packages
RUN apt-get update

# Add ubuntu user with default password is 1
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1001 ubuntu
RUN echo 'ubuntu:1' | chpasswd

# Switch to ubuntu user and change directory
USER ubuntu
WORKDIR /home/ubuntu

# Pip upgrade and PATH environment
ENV PATH="$PATH:/home/ubuntu/.local/bin"
RUN pip install --upgrade pip

# Install gdrive download
RUN pip install \
    gdown \
    numpy \
    cython

# Copy skin-detective source code
WORKDIR /home/ubuntu/skin-detective
COPY --chown=ubuntu:root . .
USER root
RUN chown ubuntu:root /home/ubuntu/skin-detective
USER ubuntu

# Clone and build pycocotools
WORKDIR /home/ubuntu
RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR /home/ubuntu/cocoapi/PythonAPI
RUN make
WORKDIR /home/ubuntu
RUN mv cocoapi/PythonAPI/pycocotools skin-detective/

# Set Environment variables
ENV DATA_ID=1MDAqxciP7Rm9Hs25OL-3sW_cXmJwf0B8
ENV MODEL_ID=1tofvtca8kc9iOtWRfXunXt3vGyAIz2yg

# Download skin data
WORKDIR /home/ubuntu/skin-detective/data
RUN gdown $DATA_ID
RUN unzip data.zip

# Download models
WORKDIR /home/ubuntu/skin-detective/models
RUN gdown $MODEL_ID

WORKDIR /home/ubuntu/skin-detective

RUN pip install jupyterlab==3.4.5 \
    torch \
    tensorflow

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8080", "--LabApp.token=''"]
