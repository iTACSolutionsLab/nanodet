# build image
# sudo docker build . -t nanodet
# run container
# sudo docker run
# docker run --gpus all --rm -ti --ipc=host -v /home/lingxt/github/nanodet:/code -v /home/lingxt/dataset:/code/dataset -p 6006:6006 nanodet:latest
# tensorboard
# cd <YOUR_SAVE_DIR>
# tensorboard --logdir ./ --host=0.0.0.0
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
ENV PYTHONUNBUFFERED=1
WORKDIR /code
COPY requirements.txt /code/
#for opencv
RUN apt-get update && apt-get install -y libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip install -r requirements.txt
COPY . /code/
RUN python setup.py develop 