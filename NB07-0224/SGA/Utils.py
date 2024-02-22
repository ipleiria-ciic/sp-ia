import numpy as np
import torch
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torchvision.datasets import ImageFolder

# MEAN and STD for the ImageNet dataset
IMGWOOF_MEAN = [0.485, 0.456, 0.406]
IMGWOOF_STD = [0.229, 0.224, 0.225]

# Normalization wrapper
class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        
    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)
    
    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)
    
# Differentiable version of the PyTorch normalization method
def normalize_fn(tensor, mean, std):
    """
    Differentiable version of torchvision.functional.normalize.
    
    Returns:
    - tensor: A tensor with empty mean and std.
    """
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)
 
# Model definition for the ImageWoof dataset
def model_def(model_name):
    model = eval("torchvision.models.{}(weights='DEFAULT')".format(model_name))
    model = nn.DataParallel(model).cuda()
    # Normalization wrapper, so that we don't have to normalize adversarial perturbations
    normalize = Normalizer(mean=IMGWOOF_MEAN, std=IMGWOOF_STD)
    model = nn.Sequential(normalize, model)
    model = model.cuda()
    print("[INFO] Model loading complete!")
    return model

# Dataloader for a given validation set
def dataloader_validation(dir_data, batch_size=16, model_dimension=256,center_crop=224):
    """
    Dataloader for a given validation set.

    Parameters:
    - dir_data: Directory of the **validation** set of a given dataset.
    - batch_size: Batch size (DEFAULT=16).
    - model_dimension: Model acceptable input dimension (DEFAULT=256).
    - center_crop: Crop ensuring the that central region of the image is preserved (DEFAULT=224). 
    
    Returns:
    - dataloader: A dataloader for the **validation** set of a given dataset.
    """
    # Define the transformations you want to apply to the images
    data_transforms = {
        'val': T.Compose([
            T.ToImage(), # Convert to tensor, because the image comes has PIL
            T.Resize(model_dimension),
            T.CenterCrop(center_crop),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(IMGWOOF_MEAN, IMGWOOF_STD)
        ]),
    }

    val_dataset = ImageFolder(dir_data, data_transforms['val'])
    dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader

# Evaluate model on data with/without UAP.
def evaluate(model, loader, uap=None, batch_size=None, device=None):
    """
    Evaluate model on data with/without UAP.

    Parameters:
    - model: A previous pre-trained model.
    - loader: Dataloader of a given validation set.
    - uap: Torch model of the UAP attack (DEFAULT=None).
    - batch_size: Batch size (DEFAULT=None).
    - device: Device used to compute the evaluation (DEFAULT=None). 
    
    Returns:
    - top: Top 5 predictions for each example in a NumPy array partion.
    - top_probs: Top 5 predictions for each example in Float32.
    - top1acc: Best accuracy value.
    - top5acc: Fifth accuracy value. 
    - outputs: Top 5 predictions for each example in a simple array.
    - labels: Labels for each example.
    - y_outputs: N/A.
    """
    probs, labels, y_out= [], [], []
    model.eval()
    
    if uap is not None:
        batch_size = batch_size
        uap = uap.unsqueeze(0).repeat([batch_size, 1, 1, 1]).to(device)
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            x_val = data[0].to(device)
            y_val = data[1].to(device)
            if uap is None:
                out = torch.nn.functional.softmax(model(x_val), dim = 1)
            else:
                y_ori = torch.nn.functional.softmax(model(x_val), dim = 1)
                perturbed = torch.clamp((x_val + uap), 0, 1) # Clamp to [0, 1]
                out = torch.nn.functional.softmax(model(perturbed), dim = 1)

            probs.append(out.cpu().numpy())
            labels.append(y_val.cpu())
            y_out.append(y_ori.cpu().numpy())

    # Convert batches to single numpy arrays
    probs = np.array([p for l in probs for p in l])
    labels = np.array([t for l in labels for t in l])
    y_out = np.array([s for l in y_out for s in l])

    # Extract top 5 predictions for each example
    top = np.argpartition(-probs, 5, axis = 1)[:,:5]
    top_probs = probs[np.arange(probs.shape[0])[:, None], top].astype(np.float32)
    top1acc = top[range(len(top)), np.argmax(top_probs, axis = 1)] == labels
    top5acc = [labels[i] in row for i, row in enumerate(top)]
    outputs = top[range(len(top)), np.argmax(top_probs, axis = 1)]

    y_top = np.argpartition(-y_out, 5, axis=1)[:, :5]
    y_top_probs = y_out[np.arange(y_out.shape[0])[:, None], y_top].astype(np.float32)
    y_outputs = y_top[range(len(y_top)), np.argmax(y_top_probs, axis=1)]
        
    return top, top_probs, top1acc, top5acc, outputs, labels, y_outputs