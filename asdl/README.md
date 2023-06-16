# How to setup, step-by-step

1. conda create --name asdl python=3.8 -y
2. conda activate asdl
3. conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
4. pip install timm wandb
5. pip install -e .
6. module load cuda/11.7
7. python setup_cuda.py install

