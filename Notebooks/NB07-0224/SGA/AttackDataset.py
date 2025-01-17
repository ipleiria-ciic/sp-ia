import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt

from Attack import uap_sga
from Utils import model_def
from PrepareData import create_dataset_npy

def main() -> None:
    start_time = datetime.datetime.now()

    # Initial parameters
    dir_uap = './UAP/SGA/'
    # Always the absolute path.. relative path will not work!
    dir_data = '/home/joseareia/Documents/SPIA/NB07-0224/Datasets/Imagewoof/train/'
    model_name = 'vgg16'
    model_dimension = 256
    center_crop = 224
    num_images = 5000
    batch_size = 8
    minibatch = 4
    epoch = 20
    alpha = 10
    beta = 10
    step_decay = 0.1
    cross_loss = 0
    iter = 4
    momentum = 0

    # Display some parameters
    print("***************************************")
    print("[INFO] Model name:", model_name.upper())
    print("[INFO] Number of images:", num_images)
    print("[INFO] Batch size:", batch_size)
    print("[INFO] Alpha:", alpha)
    print("[INFO] Beta:", beta)
    print("***************************************\n")
    
    # Load dataset
    X = create_dataset_npy(dir_data, num_classes=10, len_batch=num_images, model_dimension=model_dimension, center_crop=center_crop)
    torch.manual_seed(0)

    # Dataloader creation for both training and validation sets
    loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=True, num_workers=4)
    loader_eval = torch.utils.data.DataLoader(X, batch_size=2, shuffle=True, num_workers=4)

    # Model definition
    model = model_def(model_name)

    nb_epoch = epoch
    eps = alpha / 255
    
    # SGA attack implementation
    delta ,losses = uap_sga(model, loader, nb_epoch, eps, beta, step_decay, loss_function=cross_loss, batch_size=batch_size,
                       minibatch=minibatch, loader_eval=loader_eval, dir_uap=dir_uap, center_crop=center_crop, iter=iter, 
                       momentum=momentum, img_num=num_images)
    
    # Save delta value
    torch.save(delta, dir_uap + "Delta_%d_%d_Epochs_%d_Batch.pth" % (num_images, epoch, batch_size))
    print("\n[INFO] Saving delta values")

    # Save loss value
    np.save(dir_uap + "Losses.npy", losses)
    print("[INFO] Saving loss values")
    
    # Plot creation within attack performance
    plt.plot(losses)
    plt.savefig(dir_uap + model_name.upper() +'_loss_epoch.png')
    print("[INFO] Saving loss values")

    # Time consumed
    end_time = datetime.datetime.now()
    print("[INFO] Time Consumed: ", end_time - start_time)

if __name__ == '__main__':
    main()