#! /bin/bash
ENV_NAME="test"
PROJECT_PATH="set-path-here"
conda activate $ENV_NAME
module load cuda/10.1
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
cd $PROJECT_PATH/setup
python setup_cuda.py install
# https://varhowto.com/install-pytorch-cuda-10-1/