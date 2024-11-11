"""
    File containing various reusable functions that can be utilised across
    multiple scripts.

    Functions:
    - fetch_images(img_original_path, img_adversarial_path, img_to_class)
    - transforms()
    - process_image(img, transpose=False)
    - def use_device()
"""

import os
from PIL import Image

import torch
from torchvision.transforms import v2 as T

def fetch_images(img_original_path, img_adversarial_path, img_to_class):
    # Mapping the image names to a class.
    image_to_class_mapping = {}
    with open(img_to_class, 'r') as f:
        for line in f:
            if " -- " not in line:
                continue
            img_name, img_class = line.strip().split(' -- ')
            image_to_class_mapping[img_name] = img_class

    # Get the adversarial images. 
    adversarial_images = set(os.listdir(img_adversarial_path))

    # Filter only the images that also exist in the original folder.
    common_images = [img for img in adversarial_images if os.path.exists(os.path.join(img_original_path, img))]

    print(f"[ INFO ] Number of images: {len(common_images)}.")

    original_images = []
    adversarial_images = []
    mapping_classes = []

    # Populate the arrays with the correct information for each.
    for img_name in common_images:
        original_images.append(Image.open(os.path.join(img_original_path, img_name)))
        adversarial_images.append(Image.open(os.path.join(img_adversarial_path, img_name)))
        img_class = image_to_class_mapping.get(img_name)
        mapping_classes.append(img_class)

    return original_images, adversarial_images, mapping_classes

def transforms():
    data_transforms = {
        'train': T.Compose([
            T.ToImage(), # Convert to tensor, because the image comes has PIL
            T.Resize(size=(128, 128)),
            # T.RandomResizedCrop(size=(128, 128), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomGrayscale(p=0.1),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': T.Compose([
            T.ToImage(), # Convert to tensor, because the image comes has PIL
            T.Resize(size=(128, 128)),
            T.CenterCrop(128),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def process_image(img, transpose=False):
    if transpose:
        img = img.numpy().transpose((1, 2, 0)) # Transform (X, Y, Z) shape
    img = (img - img.min()) / (img.max() - img.min()) # Clip the image to [0, 255] values
    return img

def use_device():
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return device