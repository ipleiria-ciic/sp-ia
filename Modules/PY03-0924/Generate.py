import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.utils as vutils

from dcgan import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='model/model_final.pth', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=4, help='Number of generated outputs')
args = parser.parse_args()

state_dict = torch.load(args.load_path)

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

params = state_dict['params']

netG = Generator(params).to(device)
netG.load_state_dict(state_dict['generator'])
# print(netG)

print("[INFO] Number of images to generate: " + args.num_output)
noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)

print("[INFO] Generating...")
with torch.no_grad():
    generated_img = netG(noise).detach().cpu()

print("[INFO] Image(s) generated!")
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1,2,0)))
plt.show()