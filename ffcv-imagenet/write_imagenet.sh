#!/bin/bash

a=500
b=0.50
c=90

IMAGENET_DIR=PATH_TO_RAW_IMAGENET_DATASET_HERE
WRITE_DIR=PATH_WHERE_TO_WRITE_FFCV_DATASET_TO

write_dataset () {
    write_path=$WRITE_DIR/${1}_${2}_${3}_${4}.ffcv
    echo "Writing ImageNet ${1} dataset to ${write_path}"
    python write_imagenet.py \
        --cfg.dataset=imagenet \
        --cfg.split=${1} \
        --cfg.data_dir=$IMAGENET_DIR/${1} \
        --cfg.write_path=$write_path \
        --cfg.max_resolution=${2} \
        --cfg.write_mode=proportion \
        --cfg.compress_probability=${3} \
        --cfg.jpeg_quality=$4
}

write_dataset train $a $b $c
write_dataset val $a $b $c
