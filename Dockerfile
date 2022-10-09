#!/bin/sh
# FROM =>Select a base image
FROM --platform=linux/amd64 ubuntu:20.04

# RUN =>Call a command
# apt-Upgrade get and install the required packages
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

#　WORKDIR =>Create and move an arbitrary directory directly under the root on the container side
WORKDIR /opt

#Install anaconda3 and delete the original executable file
#　wget =>Specify the URL to download the file
#　sh =>Run a shell script
#　-b =>Avoid interactive operations
#  -p =>Specify the installation destination
#  rm =>Delete the specified file
#  -f =>Forcibly execute
RUN wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh && \
    sh /opt/Anaconda3-2022.05-Linux-x86_64.sh -b -p /opt/anaconda3 && \
    rm -f Anaconda3-2022.05-Linux-x86_64.sh

#PATH of anaconda3
#  ENV =>Change environment variables
ENV PATH /opt/anaconda3/bin:$PATH

#pip upgrade
RUN pip install --upgrade pip

#install gdrive download
RUN pip install gdown

# Clone repo
RUN git clone https://github.com/HuynhThanhQuan/skin-detective.git

# Clone and build pycocotools
WORKDIR /opt/program
RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR /opt/program/cocoapi/PythonAPI
RUN make

# Set Environment variables
ENV DATA_ID=1MDAqxciP7Rm9Hs25OL-3sW_cXmJwf0B8
ENV MODEL_ID=1tofvtca8kc9iOtWRfXunXt3vGyAIz2yg

# Return to workdir-data folder
WORKDIR /opt/program/data

# Execute download data
RUN gdown $DATA_ID

# Unzip data
RUN unzip data.zip

# Return to workdir-model folder
WORKDIR /opt/program/models

# Download model
RUN gdown $MODEL_ID

#Return directly under root
WORKDIR /opt/program

# Install package (broken)
RUN pip install -r requirements.txt

#Open jupyter lab when container starts
#  CMD =>Specify the command to be executed when the container starts
#  "jupyter", "lab" =>Launch jupyter lab
#  "--ip=0.0.0.0" =>Remove ip restrictions
#  "--allow-root" =>Allow root user, not good for security
#  "LabApp.token=''" = >It can be started without a token. Not good for security
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8080","--allow-root", "--LabApp.token=''"]
