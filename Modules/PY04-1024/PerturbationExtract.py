from PIL import Image
import torch
import numpy as np
import os
import random

# Get UAP
uap = np.load('UAP/UAP_VGG16.npy')
uap = torch.tensor(uap, device='cpu')

# Remove the batch dimension if it exists
if uap.shape[0] == 1:
    uap = uap.squeeze(0)

print("Shape of UAP after squeezing:", uap.shape)

uap_numpy = uap.numpy()
uap_numpy = uap_numpy.transpose((1, 2, 0))
uap_processed = (uap_numpy - uap_numpy.min()) / (uap_numpy.max() - uap_numpy.min())
uap_image = (uap_processed * 255).astype(np.uint8)

image = Image.fromarray(uap_image)

output_dir = '9999'
os.makedirs(output_dir, exist_ok=True)

num_images = 1300

# Data augmentation functions
def augment_image(img):
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    rotation_angle = random.randint(0, 360)
    img = img.rotate(rotation_angle, expand=True)

    scale_factor = random.uniform(1.2, 1.5)
    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
    img = img.resize(new_size, Image.LANCZOS)

    left = (img.width - 178) / 2
    top = (img.height - 178) / 2
    right = (img.width + 178) / 2
    bottom = (img.height + 178) / 2
    img = img.crop((left, top, right, bottom))

    return img

# Generate augmented images
for i in range(num_images):
    augmented_image = augment_image(image)
    augmented_image.save(os.path.join(output_dir, f'9999_{i + 1}.png'))

print(f'Generated {num_images} augmented images in the "{output_dir}" directory.')