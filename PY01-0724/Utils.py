"""
File: Utils.py
Author: José Areia
Date: 2024-07-23
"""

# Utils
import os
import time
import numpy as np

# Torch
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from torchvision import datasets
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader, Subset

def use_device():
    """
    Checks the available device and returns it.

    Returns:
    - device: Available device.       
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def process_image(img, transpose=False, numpy=False):
    """
    Preprocess the images to displays them in a plot.

    Parameters:
    - img: The image to preprocess.
    - transpose: Bool value if the img needs to be transposed.
    - numpy: Flag to identify if the image is already in a numpy array.

    Returns:
    - img: The image preprocessed.       
    """        
    if transpose:
        if numpy is False:
            img = img.numpy()
        img = img.transpose((1, 2, 0)) # Transform (X, Y, Z) shape
    img = (img - img.min()) / (img.max() - img.min()) # Clip the image to [0, 255] values
    return img


def obj_variance(obj, type=None):
    """
    Display the maximum and minimum value in a given object.

    Parameters:
    - obj: A given object.
    - type: Object type. Can be "tensor" and "numpy".
    """

    if type == "tensor":
        tensor_min = torch.min(obj)
        tensor_max = torch.max(obj)
        
        print("Maximum value:", "{:.5f}".format(tensor_max.item()))
        print("Minimum value:", "{:.5f}".format(tensor_min.item()))
    else:
        numpy_min = "{:.5f}".format(np.max(obj))
        numpy_max = "{:.5f}".format(np.min(obj))
        
        print("Maximum value:", numpy_max)
        print("Minimum value:", numpy_min)

def normalize(image):
    """
    Normalise a given image.

    Parameters:
    - img: The image to be normalised.

    Returns:
    - img: The image normalised.       
    """

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    return (image - mean.type_as(image)[None,:,None,None]) / std.type_as(image)[None,:,None,None]

def get_uap(path, device):
    """
    Get the UAP perturbation file.

    Parameters:
    - path: Directory path of a given a UAP perturbation.
    - device: The device (CPU/GPU) to store the perturbation that will be loaded.

    Returns:
    - uap: UAP perturbation.       
    """

    uap = np.load(path)
    uap = torch.tensor(uap, device=device)
    return uap

def prepare_model(model_name, device):
    """
    Prepares a model for later usage.

    Parameters:
    - model_name: Name of the model that is going to be used.
    - device: The device (CPU/GPU) to store the model that will be loaded.

    Returns:
    - model: A pre-trained model ready to use.       
    """
     
    model = getattr(models, model_name.lower())(pretrained=True).to(device)
    return model

def get_dataloader(dataset, batch_size, num_images, shuffle=True):
    """
    Creates a PyTorch Dataloader for a given dataset.

    Parameters:
    - dataset: Direcorty path of the dataset that is going to be loaded.
    - batch_size: Batch size number.
    - num_images: Number of images that are going to be loaded.
    - shuffle: If the images are going to be shuffle or not (DEFAULT=True).

    Returns:
    - dataloader: A dataloader of a given dataset.
    - class_to_name: A dictionary mapping class indices to class names.
    """

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    val_dataset =  datasets.ImageFolder(dataset, transform=transform)
    num_classes = len(val_dataset.classes)
    class_to_name = {i: class_name for i, class_name in enumerate(val_dataset.classes)}
    
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(val_dataset.samples):
        class_indices[label].append(idx)
        
    samples_per_class = num_images // num_classes
    subset_indices = []
    
    for i in range(num_classes):
        class_subset_indices = np.random.choice(class_indices[i], samples_per_class, replace=False).tolist()
        subset_indices.extend(class_subset_indices)
        
    val_subset = Subset(val_dataset, subset_indices)
    dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    
    return dataloader, class_to_name

def get_adversarial_images(model, delta, dataloader, device):
    """
    Generates the adverasial images for a given model and dataset.

    Parameters:
    - model: Model to be use.
    - delta: Intensity of the perturbation.
    - dataloader: Dataloader to be use.
    - device: Device to be used to store in memory the adversarial images.

    Returns:
    - adv_dataset: Advesarial images that are missclassified.       
    - adv_classes: Classes of the missclassified advesarial images.
    - ori_classes: Original classes of the images.    
    """

    adv_dataset = []
    adv_classes = []
    ori_images = []
    ori_classes = []
    
    model.eval()    
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(normalize(images))
            _, predicted = torch.max(outputs.data, 1)

            adv_images = torch.add(delta, images).clamp(0, 1)
            adv_outputs = model(normalize(adv_images))

            _, adv_predicted = torch.max(adv_outputs.data, 1)
                
            misclassified_indices = (predicted != adv_predicted).nonzero(as_tuple=True)[0]
            if misclassified_indices.numel() > 0:
                adv_dataset.append(adv_images[misclassified_indices].cpu())
                adv_classes.append(adv_predicted[misclassified_indices].cpu())
                ori_classes.append(labels[misclassified_indices].cpu())
                ori_images.append(images[misclassified_indices].cpu())
                
            del images, labels, outputs, adv_images, adv_outputs, predicted, adv_predicted
            torch.cuda.empty_cache()
                
    return adv_dataset, adv_classes, ori_classes, ori_images


def load_chunks(path, prefix):
    """
    Load separated chunks into one.

    Parameters:
    - path: Path of the chunks.
    - prefix: Prefix name of the chunk (EX: ADV_01, ADV_02, ADV_XX).

    Returns:
    - tensors: One chunk in torch tensor format.       
    """

    tensors = []
    chunk_idx = 0

    while True:
        file_path = os.path.join(path, f'{prefix}_{chunk_idx}.pt')
        if not os.path.exists(file_path):
            break
        tensors.append(torch.load(file_path))
        chunk_idx += 1
    return torch.cat(tensors)

def export_images(batch_index, output_path, global_counter):
    """
    Export the images of a given Torch tensor.

    Parameters:
    - batch_index: Index of the batch to be processed.
    - output_path: Path to output the images extracted.
    - global_counter: Global counter of the images.
    
    Returns:
    - global_counter: Global counter incremented for the next batch.      
    """
    
    start_time = time.time()

    adv_images_path = f'Dataset/Intermediate-2/ADV_Images/Adv_Images_{batch_index}.pt'
    adv_classes_path = f'Dataset/Intermediate-2/ADV_Classes/Adv_Classes_{batch_index}.pt'
    ori_classes_path = f'Dataset/Intermediate-2/ORI_Classes/Ori_Classes_{batch_index}.pt'
    ori_images_path = f'Dataset/Intermediate-2/ORI_Images/Ori_Images_{batch_index}.pt'

    AdversarialImages = torch.load(adv_images_path)
    AdversarialClasses = torch.load(adv_classes_path)
    OriginalClasses = torch.load(ori_classes_path)
    OriginalImages = torch.load(ori_images_path)

    with open('Dataset/AdversarialClasses_Mapping-2.txt', 'a') as f:  # Append mode
        for adv_image, ori_class, adv_class, ori_image in zip(AdversarialImages, OriginalClasses, AdversarialClasses, OriginalImages):
            img_adv = ToPILImage()(adv_image) # Adversarial image
            img_ori = ToPILImage()(ori_image) # Original image

            # Original image process
            name_ori_image = f'ORI_{global_counter:05d}_{ori_class.item():04d}'
            extension_ori_image = 'png'
            filename_ori_image = f'{name_ori_image}.{extension_ori_image}'
            img_path_ori = os.path.join(output_path, filename_ori_image)
            img_ori.save(img_path_ori)

            # Adversarial image process 
            name_adv_image = f'ADV_{global_counter:05d}_{adv_class.item():04d}'
            extension_adv_image = 'png'
            filename_adv_image = f'{name_adv_image}.{extension_adv_image}'
            img_path_adv = os.path.join(output_path, filename_adv_image)
            img_adv.save(img_path_adv)

            f.write(f'{name_ori_image}\n')
            f.write(f'{name_adv_image}::{ori_class.item():04d}\n')
            
            global_counter += 1

    print(f"[INFO] Batch {batch_index} processed. Time taken: {time.time() - start_time:.2f} seconds")
    return global_counter