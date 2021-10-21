#!/usr/bin/env bash
# Add Python Path
# export PYTHONPATH=$PYTHONPATH:/detectron/lib
# for NVCC
ls /usr/local/
# export PATH=$PATH:/usr/local/cuda-10.0/bin
# env setting
sudo ln -s /usr/local/lib /usr/local/lib/python3.6/dist-packages/lib
sudo chmod -R 777 /usr/local/lib/python3.6/dist-packages/
sudo chmod -R 777 /usr/local/bin/
nvcc --version
nvidia-smi
# update gcc/g++
gcc -v
sudo apt-get install -y python-software-properties
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y g++-7
cd /usr/bin
sudo rm -r gcc
sudo ln -sf gcc-7 gcc
sudo rm -r g++
sudo ln -sf g++-7 g++
cd /
gcc -v

# install mmcv-full
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
# Install Easydict and other libs
# pip install numpy==1.15.4 # specific numpy version for PHOTO_METRIC_DISTORT
# pip install easydict cython matplotlib scipy
# pip install opencv-python pyyaml packaging pycocotools tensorboardX cffi shapely
# pip install Polygon3 json_tricks munkres ninja pandas paramiko requests scikit-image tqdm yacs
# pip install transformers==2.11.0
# pip install filelock iopath
# pip install torch==1.7.0
# pip install timm

#Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library. Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
#solution: https://python.libhunt.com/pytorch-latest-version
export MKL_THREADING_LAYER=GNU

# make file
sudo chmod -R 777 /mmdetection
cd /mmdetection
pip install -r requirements/build.txt
pip install -v -e .

# link dataset, start use blob, don't link data now
cd /mmdetection
mkdir -p datasets
sudo ln -s /blob/data/pretrained_models/detectron2_related_models ./datasets/pretrained_models

#DOTA_devkit
sudo apt-get install swig
cd DOTA_devkit/polyiou
swig -c++ -python csrc/polyiou.i
python setup.py build_ext --inplace

cd /mmdetection
