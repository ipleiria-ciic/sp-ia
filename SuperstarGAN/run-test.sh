#!/bin/sh

python src/main.py \
       --mode test \
       --dataset ImageNet \
       --test_iters 500000 \
       --imagenet_image_dir ../../Datasets/ImageNet5/Images-Test \
       --c_dim 2