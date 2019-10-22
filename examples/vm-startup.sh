#!/usr/bin/env bash

#install pre-requisites
sudo apt-get update && sudo apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    python-tk \
    git
    
# install packages for python
sudo pip3 install --upgrade pip
sudo pip3 install virtualenv
sudo pip3 install wheel