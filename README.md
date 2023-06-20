# Error Feedback Can Accurately Compress Preconditioners

The official project repository for the [EFCP paper](https://arxiv.org/abs/2306.06098) from DAS Lab @ Institute of Science and Technology Austria.

# Installing custom CUDA kernels for M-FAC

Suppose the project is in the home directory `~/EFCP`, the CUDA kernels can be installed using the following commands:

```shell
$ cd ~/EFCP/cuda/mfac_kernel
$ python setup_cuda.py install
```

We used M-FAC on RTX-3090 and A6000 GPUs.

# Reproducing experiments
We provide a shell script to reproduce all our experiments and we recommend using WandB to track the results.

## ImageNet
For this experiment we build on top of the [FFCV repository](https://github.com/libffcv/ffcv-imagenet) and add a few more features to the parameters (see the `custom` section).

<strong>Dataset generation</strong>. The FFCV repository uses the ImageNet dataset to pre-process it in order to obtain the FFCV dataset. Make sure you set the correct paths in `~/EFCP/ffcv-imagenet/write_imagenet.sh` before running this script file.

<strong>Image scaling.</strong> Comment out the section `resolution` in the `yaml` config.

<strong>Running the experiment.</strong> Run the following commands after replacing the parameter values starting with prefix `@` with your own values.

```shell
$ export EFCP_ROOT=~/EFCP # the root folder will be added as a library path
$ cd ~/EFCP/ffcv-imagenet
$ bash write_imagenet.sh
$ CUDA_VISIBLE_DEVICES=0 python train_imagenet.py \
    --data.train_dataset @TRAIN_PATH \
    --data.val_dataset @VALIDATION-PATH \
    --logging.folder @LOGGING_FOLDER \
    --wandb.project @WANDB_PROJECT \
    --wandb.group @WANDB_GROUP\
    --wandb.job_type @WANDB_JOB_TYPE \
    --wandb.name @WANDB_NAME \
    --data.num_workers 12 \
    --data.in_memory 1 \
    --config-file rn18_configs/rn18_88_epochs.yaml \
    --training.optimizer kgmfac \
    --training.batch_size 1024 \
    --training.momentum 0 \
    --training.weight_decay 1e-05 \
    --lr.lr 0.001 \
    --lr.lr_schedule_type linear \
    --custom.damp 1e-07 \
    --custom.k 0.01 \
    --custom.seed @SEED \
    --custom.wd_type wd
```

## ASDL

For this experiment we build on top of the [ASDL repository](https://github.com/kazukiosawa/asdl). We integrate our M-FAC implementations in the following files:

- `~/EFCP/asdl/asdl/precondition/mfac.py` for Dense M-FAC
- `~/EFCP/asdl/asdl/precondition/sparse_mfac.py` for Sparse M-FAC

<strong>Features added.</strong> We added the following new parameters to the existing repository:

- `clip_type` - specifies whether clipping should be performed by value or by norm (`val`, `norm`)
- `clip_bound` - the value used in clipping. Set it to `0` to disable clipping, regardless of the value of `clip_type`
- `ignore_bn_ln_type` - used to perform BN/LN ablation. Possible values are `none`, `all`, `modules`

```shell
$ export EFCP_ROOT=~/EFCP # the root folder will be added as a library path
$ cd ~/EFCP/asdl/examples/arxiv_results
$ CUDA_VISIBLE_DEVICES=0 python train.py \
    --wandb_project @WANDB_PROJECT \
    --wandb_group @WANDB_GROUP\
    --wandb_job_type @WANDB_JOB_TYPE \
    --wandb_name @WANDB_NAME \
    --folder @LOGGING_FOLDER \
    --ngrads 1024 \
    --momentum 0 \
    --dataset cifar10 \
    --optim kgmfac \
    --k 0.01 \
    --epochs 20 \
    --batch_size 32 \
    --model rn18 \
    --weight_decay 0.0005 \
    --ignore_bn_ln_type all \
    --lr 0.03 \
    --clip_type norm \
    --clip_bound 10 \
    --damp 1e-05 \
    --seed 1
```

## BERT training

We use the [HuggingFace repository](https://github.com/huggingface) stated in the original [M-FAC paper](https://arxiv.org/abs/2107.03356) and integrate Sparse M-FAC to experiment with [Question Answering](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) and [Text Classification](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification). The following commands can be used to reproduce our experiments for QA and GLUE using the parameters from <strong>Appendix D</strong> in our paper.

<strong>Instructions for GLUE/MNLI.</strong> Run Sparse-MFAC on BERT-Base:
```shell
$ export EFCP_ROOT=~/EFCP # the root folder will be added as a library path
$ cd ~/EFCP/huggingface/examples/MFAC_optim
python run_glue.py \
    --wandb_project @WANDB_PROJECT \
    --wandb_group @WANDB_GROUP\
    --wandb_job_type @WANDB_JOB_TYPE \
    --wandb_name @WANDB_NAME \
    --output_dir @OUTPUT_DIR \
    --seed @SEED \
    --logging_strategy steps \
    --logging_steps 10 \
    --model_name_or_path bert-base \
    --task_name mnli \
    --optim kgmfac \
    --num_train_epochs 3 \
    --lr 2e-5 \
    --damp 5e-5 \
    --ngrads 1024 \
    --k 0.01
```

<strong>Instructions for QA/SquadV2.</strong> Run Sparse-MFAC on BERT-Base:
```shell
$ export EFCP_ROOT=~/EFCP # the root folder will be added as a library path
$ cd ~/EFCP/huggingface/examples/MFAC_optim
python run_qa.py \
    --wandb_project @WANDB_PROJECT \
    --wandb_group @WANDB_GROUP\
    --wandb_job_type @WANDB_JOB_TYPE \
    --wandb_name @WANDB_NAME \
    --output_dir @OUTPUT_DIR \
    --seed @SEED \
    --logging_strategy steps \
    --logging_steps 10 \
    --model_name_or_path bert-base \
    --optim kgmfac \
    --num_train_epochs 2 \
    --lr 3e-5 \
    --damp 5e-5 \
    --ngrads 1024 \
    --k 0.01
```



## Own training pipeline

We use our own training pipeline to train a small ResNet-20 on CIFAR-10 and for our linear probing experiment that uses Logistic Regression on a synthetic dataset. The notations for the hyper-parameters are introduced in the first paragraph of the <strong>Appendix</strong>.

<strong>CIFAR-10 / ResNet-20 (272k params).</strong> For these particular experiments, check the parameters in the <strong>Appendix C</strong> of the paper and match them with the ones in `~/EFCP/args/args_mfac.py`


```shell
$ export EFCP_ROOT=~/EFCP # the root folder will be added as a library path
$ cd ~/EFCP
python main.py \
    --wandb_project @WANDB_PROJECT \
    --wandb_group @WANDB_GROUP\
    --wandb_job_type @WANDB_JOB_TYPE \
    --wandb_name @WANDB_NAME \
    --seed @SEED \
    --root_folder @EXPERIMENT_FOLDER \
    --dataset_path @PATH_TO_DATASET \
    --dataset_name cifar10 \
    --optim kgmfac \
    --model rn20 \
    --epochs 164 \
    --batch_size 128 \
    --lr_sched step \
    --k 0.01 \
    --ngrads 1024 \
    --lr 1e-3 \
    --damp 1e-4 \
    --weight_decay 1e-4 \
    --momentum 0 \
    --wd_type wd
```

<strong>Logistic Regression / Synthetic Data.</strong> For this experiment we use the same script `main.py` using the hyper-parameters from the <strong>Appendix A</strong> in our paper. The dataset we used is publicly available [here](https://seafile.ist.ac.at/lib/7ae0eddc-4f66-4103-8aba-37ea22d34901/file/NeurIPS2023-EFCP/RN50x16-openai-imagenet1k.zip?dl=1). Below we only present the script to run `Sparse GGT`. In order to run other optimizers, please have a look at the method `get_optimizer` from [`helpers/training.py`](https://github.com/IST-DASLab/EFCP/blob/main/helpers/training.py) file and at the method `get_arg_parse` from [args/args_mfac.py](https://github.com/IST-DASLab/EFCP/blob/main/args/args_mfac.py) that stores command line arguments.

```shell
$ export EFCP_ROOT=~/EFCP # the root folder will be added as a library path
$ CUDA_VISIBLE_DEVICES=0 python main.py \
    --wandb_project @WANDB_PROJECT \
    --wandb_group @WANDB_GROUP\
    --wandb_job_type @WANDB_JOB_TYPE \
    --wandb_name @WANDB_NAME \
    --seed @SEED \
    --root_folder @EXPERIMENT_FOLDER \
    --dataset_path @PATH_TO_RN50x16-openai-imagenet1k \
    --dataset_name rn50x16openai \
    --optim ksggt \
    --model logreg \
    --epochs 10 \
    --batch_size 128 \
    --lr_sched cos \
    --k 0.01 \
    --ngrads 100 \
    --lr 1 \
    --weight_decay 0 \
    --ggt_beta1 0 \
    --ggt_beta2 1 \
    --ggt_eps 1e-05
```

## Quantify Preconditioning

We describe the preconditioning quantification in <strong>Section 6</strong> of [our paper](https://arxiv.org/abs/2306.06098). We use [quantify_preconditioning](https://github.com/IST-DASLab/EFCP/blob/main/helpers/optim.py#L103) method to compute the metrics for scaling and rotation, which requires the raw gradient `g` and the preconditioned gradient `u`. We would like to mention that calling this method at each time step for large models (such as BERT-Base) slows down training by a lot because the operations are performed using large tensors. Moreover, the quantiles are computed in `numpy` because `pytorch` raises an error when calling `quantile` function for large tensors.

<strong></strong>