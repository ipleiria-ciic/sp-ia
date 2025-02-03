#!/bin/sh

# Training session. It produces logs to 'arguments_log.json'
python src/main.py --use_tensorboard False

echo "[ INFO ] Preparing for the testing session... Entering sleep mode." ; sleep 30s

# Testing session
python src/main.py --mode test --imagenet_image_dir ../../Datasets/ImageNet5/Images-Test

echo "[ INFO ] Preparing for the evaluation session... Entering sleep mode." ; sleep 30s

# Evaluation session. It produces logs to 'classification_log.json'
python src/evaluation.py