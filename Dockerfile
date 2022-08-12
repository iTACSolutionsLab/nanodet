FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
ENV PYTHONUNBUFFERED=1
WORKDIR /code
# https://github.com/NVIDIA/nvidia-docker/issues/1631#issuecomment-1112561097
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# for opencv
RUN apt-get update && apt-get install -y libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/
RUN python setup.py develop 