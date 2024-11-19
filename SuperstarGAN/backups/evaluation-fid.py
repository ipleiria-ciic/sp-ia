"""
    Script to evaluate the results of the
    SuperstarGAN with the FID metric.
"""

import os
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from scipy.linalg import sqrtm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datetime import datetime, timedelta

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('jpg', 'png', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        self.inception = models.inception_v3(weights='DEFAULT', transform_input=False)
        self.inception.fc = nn.Identity()

    def forward(self, x):
        return self.inception(x)

# Function to compute activations for an entire DataLoader.
def get_activations(data_loader, model, device):
    model = model.to(device)
    all_activations = []
    with torch.no_grad():
        for images in data_loader:
            images = images.to(device)
            activations = model(images)
            all_activations.append(activations.cpu())
    return torch.cat(all_activations, dim=0).numpy()

def calculate_fid(mu1, sigma1, mu2, sigma2):
    # Compute the FID score between two Gaussian distributions
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    # If the result contains imaginary numbers, discard the imaginary part
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

start_time = time.time()

model = InceptionV3FeatureExtractor().eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(f"[ INFO ] Processing the dataset.")

real_dataset = ImageDataset(image_dir='../../../Datasets/ImageNet5/Images', transform=transform)
generated_dataset = ImageDataset(image_dir='../results-transformed/imagenet', transform=transform)

real_loader = DataLoader(real_dataset, batch_size=8, shuffle=False, num_workers=4)
generated_loader = DataLoader(generated_dataset, batch_size=8, shuffle=False, num_workers=4)

# Calculate activations for real and generated images.
print(f"[ INFO ] Calculating the activations of the real images.")
real_activations = get_activations(real_loader, model, device)

print(f"[ INFO ] Calculating the activations of the generated images.")
generated_activations = get_activations(generated_loader, model, device)

# Calculate mean and covariance.
mu_real = np.mean(real_activations, axis=0)
sigma_real = np.cov(real_activations, rowvar=False)
mu_gen = np.mean(generated_activations, axis=0)
sigma_gen = np.cov(generated_activations, rowvar=False)

# Compute FID.
fid_score = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)

elapsed = time.time() - start_time

print(f"[ RESULTS ] FID Score: {fid_score:.02f}. Completed in '{str(timedelta(seconds=int(elapsed)))}'.")
