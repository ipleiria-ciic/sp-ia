"""
    Script to take an image folder, remove a partial threshold
    from the images, and save the results to a defined location.
"""

import os
import numpy as np
from PIL import Image

input_folder = '../results/imagenet'
output_folder = '../results-transformed/imagenet'

os.makedirs(output_folder, exist_ok=True)

def remove_black_border(image_path, output_path):
    image = Image.open(image_path)
    gray_image = image.convert("L")
    np_image = np.array(gray_image)
    
    mask = np_image > 12  # Adjust threshold if needed
    
    coords = np.argwhere(mask)
    if coords.size > 0:
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1
        cropped_image = image.crop((y0, x0, y1, x1))
        cropped_image.save(output_path)
    else:
        image.save(output_path)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        remove_black_border(input_path, output_path)

print("Finished processing images.")