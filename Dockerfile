#!/bin/sh
# FROM =>Select a base image
FROM --platform=linux/amd64 et2010/jupyter-tensorflow-pytorch-gpu

RUN apt-get update
RUN apt-get install -y \
    wget \ 
    vim \
    zip \
    unzip \
    git \
    make \
    gcc \
    nano

# Install opencv lib
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install -y ffmpeg libsm6 libxext6

#pip upgrade
RUN pip install --upgrade pip

#install gdrive download
RUN pip install gdown

# Clone skin-detective repo
WORKDIR /opt/program
RUN git clone https://github.com/HuynhThanhQuan/skin-detective.git

# Clone and build pycocotools
WORKDIR /opt/program
RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR /opt/program/cocoapi/PythonAPI
RUN make
RUN mv cocoapi/PythonAPI/pycocotools skin-detective/

# Set Environment variables
ENV DATA_ID=1MDAqxciP7Rm9Hs25OL-3sW_cXmJwf0B8
ENV MODEL_ID=1tofvtca8kc9iOtWRfXunXt3vGyAIz2yg

# Download skin data
WORKDIR /opt/program/skin-detective/data
RUN gdown $DATA_ID
RUN unzip data.zip

# Download models
WORKDIR /opt/program/skin-detective/models
RUN gdown $MODEL_ID

WORKDIR /opt/program/skin-detective

#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8080","--allow-root", "--LabApp.token=''"]
