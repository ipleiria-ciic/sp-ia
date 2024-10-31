#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:21:20 2024

@author: joseareia
"""

import os
import csv
import shutil

base_path = '../Datasets/ImageNet5'
new_folder = '../Datasets/ImageNet5/Images'

csv_data = []
txt_data = []

image_counter = 1

for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    
    if os.path.isdir(folder_path) and folder_name != "Images":
        original_value = 1 if folder_name != "9999" else -1
        perturbation_value = -original_value

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            
            if os.path.isfile(image_path):
                new_image_name = f"{image_counter:05}.jpg"
                new_image_path = os.path.join(new_folder, new_image_name)
                
                shutil.copy(image_path, new_image_path)
                
                csv_data.append([new_image_name, original_value, perturbation_value])
                txt_data.append(f"{new_image_name} {original_value} {perturbation_value}")
                
                image_counter += 1

# Write CSV file
csv_path = os.path.join(base_path, "image_data.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_number", "original", "perturbation"])
    writer.writerows(csv_data)
    
# Write TXT file
txt_path = os.path.join(base_path, "image_data.txt")
with open(txt_path, mode="w") as file:
    file.write(f"{image_counter}\n")
    file.write("original perturbation\n")
    file.write("\n".join(txt_data))

print("Images moved and both CSV and TXT files created successfully.")