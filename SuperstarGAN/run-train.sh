#!/bin/sh

python main.py --dataset ImageNet \
    --attr_path ../Datasets/ImageNet5/image_data.txt \
    --c_dim 2 \
    --log_dir imagenet5/logs \
    --model_save_dir imagenet5/models \
    --sample_dir imagenet5/samples \
    --result_dir imagenet5/results
