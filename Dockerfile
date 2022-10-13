#!/bin/sh
# FROM =>Select a base image
FROM --platform=linux/amd64 et2010/jupyter-tensorflow-pytorch-gpu

USER root

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

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
RUN mv /opt/program/cocoapi/PythonAPI/pycocotools /opt/program/skin-detective/

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

RUN pip install jupyterlab 

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8080","--allow-root", "--LabApp.token=''"]
