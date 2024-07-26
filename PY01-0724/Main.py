"""
File: Main.py
Author: José Areia
Date: 2024-07-23
"""

# Utils
from Utils import *

# Miscellaneous
import os
import gc
import time

# Torch
import torch

# Global constants
dataset = '../Datasets/ImageNet100/Train'
uap_per = 'UAP/UAP_VGG16.npy'
model_n = 'VGG16'
save_path = 'Dataset/Intermediate'
batch_size = 2
num_images = 50000
chunk_size = 2000

# Check if path exists
os.makedirs(save_path, exist_ok=True)

# Device definition
device = use_device()

# Get UAP
start_time = time.time()
uap = get_uap(uap_per, device)
print(f"[INFO] UAP perturbation loaded. Time taken: {time.time() - start_time:.2f} seconds")

# Get model
start_time = time.time()
model = prepare_model(model_n, device)
print(f"[INFO] Model loaded. Time taken: {time.time() - start_time:.2f} seconds")

# Create a dataloader
start_time = time.time()
dataloader, class_names = get_dataloader(dataset, batch_size, num_images, shuffle=True)
print(f"[INFO] Dataloader created. Time taken: {time.time() - start_time:.2f} seconds")

# Generate the adversarial images in chunks cause... memory problems!
start_time = time.time()
delta = torch.clamp(uap, -10/255, 10/255)

adv_images = []
adv_classes = []
ori_classes = []
batch_count = 0
image_count = 0

for batch in dataloader:
    adv_images_batch, adv_classes_batch, ori_classes_batch = get_adversarial_images(model, delta, [batch], device)
    adv_images.extend(adv_images_batch)
    adv_classes.extend(adv_classes_batch)
    ori_classes.extend(ori_classes_batch)

    if len(adv_images) * batch_size >= chunk_size:
        torch.save(torch.cat(adv_images), os.path.join(save_path, f'Adv_Images_{batch_count}.pt'))
        torch.save(torch.cat(adv_classes), os.path.join(save_path, f'Adv_Classes_{batch_count}.pt'))
        torch.save(torch.cat(ori_classes), os.path.join(save_path, f'Ori_Classes_{batch_count}.pt'))

        adv_images.clear()
        adv_classes.clear()
        ori_classes.clear()

        batch_count += 1

        gc.collect()
        torch.cuda.empty_cache()
    
    image_count += batch_size

print(f"[INFO] Number of adversarial images processed: {image_count}. Time taken: {time.time() - start_time:.2f} seconds")

if adv_images:
    torch.save(torch.cat(adv_images), os.path.join(save_path, f'Adv_Images_{batch_count}.pt'))
    torch.save(torch.cat(adv_classes), os.path.join(save_path, f'Adv_Classes_{batch_count}.pt'))
    torch.save(torch.cat(ori_classes), os.path.join(save_path, f'Ori_Classes_{batch_count}.pt'))