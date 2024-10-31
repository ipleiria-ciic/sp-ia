"""
File: Join.py
Author: José Areia
Date: 2024-07-24
"""

# Utils
from Utils import *

# Miscellaneous
import os
import torch

save_path = 'Dataset/Intermediate'

# Load and concatenate all chunks
adv_images_tensor = load_chunks(save_path, 'Adv_Images')
adv_classes_tensor = load_chunks(save_path, 'Adv_Classes')
ori_classes_tensor = load_chunks(save_path, 'Ori_Classes')

# Save the concatenated tensors to disk
final_save_path = 'Torch_Data'
os.makedirs(final_save_path, exist_ok=True)
torch.save(adv_images_tensor, os.path.join(final_save_path, 'Adv_Images.pt'))
torch.save(adv_classes_tensor, os.path.join(final_save_path, 'Adv_Classes.pt'))
torch.save(ori_classes_tensor, os.path.join(final_save_path, 'Ori_Classes.pt'))

print("[INFO] All chunks concatenated and saved successfully!")