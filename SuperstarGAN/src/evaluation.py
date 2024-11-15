"""
    Script to evaluate the results of the SuperstarGAN
    with both FID and LPIPS metrics.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader

class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        self.inception = models.inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()

    def forward(self, x):
        return self.inception(x)

model = InceptionV3FeatureExtractor().eval()

transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Required size for Inception
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calculate_activation_statistics(images, model, device):
    model = model.to(device)
    images = images.to(device)
    with torch.no_grad():
        activations = model(images).cpu().numpy()
    mean = np.mean(activations, axis=0)
    covariance = np.cov(activations, rowvar=False)
    return mean, covariance

def calculate_fid(mu1, sigma1, mu2, sigma2):
    # Compute the FID score between two Gaussian distributions.
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    # If the result contains imaginary numbers, discard the imaginary part.
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mu_real, sigma_real = calculate_activation_statistics(real_images, model, device)
mu_gen, sigma_gen = calculate_activation_statistics(generated_images, model, device)

fid_score = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
print(f"FID Score: {fid_score}")
