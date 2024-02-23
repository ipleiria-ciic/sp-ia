import os
import torch
import torchvision
import numpy as np

from PIL import Image
from torchvision.transforms import v2 as T

# MEAN and STD for the ImageNet dataset
IMGWOOF_MEAN = [0.485, 0.456, 0.406]
IMGWOOF_STD = [0.229, 0.224, 0.225]

def preprocess_image(image_paths=None, model_dimension=256, center_crop=224):
    """
    Preprocess the images with transformers.

    Parameters:
    - image_paths: Path to a given image (DEFAULT=None).
    - model_dimension: Model acceptable input dimension (DEFAULT=256).
    - center_crop: Crop ensuring the that central region of the image is preserved (DEFAULT=224).

    Returns:
    - img: Image preprocessed. 
    """
    img = Image.open(image_paths)
    train_transform = T.Compose([
        T.ToImage(), # Convert to tensor, because the image comes has PIL
        T.Resize(model_dimension),
        T.CenterCrop(center_crop),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMGWOOF_MEAN, IMGWOOF_STD)
    ])
    img = train_transform(img)

    return img

def create_dataset_npy(train_path, num_classes, len_batch=10000, model_dimension=256, center_crop=224):
    """
    Generate a dataset in the form of a NumPy array for a given dataset.

    Parameters:
    - train_path: The directory path where the training images are stored.
    - num_classes: Number of classes in the given dataset.
    - len_batch: Length of the batch to fetch from a given dataset (DEFAULT=10000).
    - model_dimension: Model acceptable input dimension (DEFAULT=256).
    - center_crop: Crop ensuring the that central region of the image is preserved (DEFAULT=224).

    Return:
    - imageset: Preprocessed images from the given dataset in a NumPy array.
    """
    sz_img = [center_crop, center_crop]
    num_channels = 3

    im_array = np.zeros([len_batch] + [num_channels]+sz_img, dtype=np.float32)
    num_imgs_per_batch = int(len_batch / num_classes)

    dirs = [x[0] for x in os.walk(train_path)]
    dirs = dirs[1:]

    # Sort the directory in alphabetical order
    dirs = sorted(dirs)
    it = 0
    Matrix = [0 for x in range(len_batch)]

    for d in dirs:
        for _, _, filename in os.walk(d):
            Matrix[it] = filename
        it = it+1

    it = 0

    # Load images, pre-process, and save
    print('[INFO] Processing images...')
    for k in range(num_classes):
        for u in range(num_imgs_per_batch):
            print(it)
            path_img = os.path.join(dirs[k], Matrix[k][u])
            image = preprocess_image(path_img, model_dimension, center_crop)
            im_array[it:(it+1), :, :, :] = image
            it = it + 1

    print('[INFO] Processing images phase completed!')
    imageset = im_array
    return imageset