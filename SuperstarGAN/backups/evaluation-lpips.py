"""
    Script to evaluate the results of the
    SuperstarGAN with the LPIPS metric.
"""

import lpips
import torch
import warnings

from PIL import Image
from torchvision import transforms

# Disable the warning from torch (built-in in LPIPS).
warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings("ignore", module="lpips")

# Load the VGG pre-trained network.
loss_fn = lpips.LPIPS(net='vgg')

# Load and pre-process images.
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

img1 = transform(Image.open('../test/image1.jpg')).unsqueeze(0)
img2 = transform(Image.open('../test/image2.jpg')).unsqueeze(0)

# Calculate the learned perceptual image patch similatiry.
d = loss_fn(img1, img2)

print(f"[ RESULTS ] LPIPS similarity: {d.item():.02f}.")