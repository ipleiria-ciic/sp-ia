import torch
import datetime

from Utils import dataloader_validation, model_def, evaluate

def main() -> None:
    start_time = datetime.datetime.now()

    # Initial parameters
    device = torch.device("cuda:0")
    dir_uap = './UAP/SGA/Delta_100_11_Epochs_4_Batch.pth'
    # Always the absolute path.. relative path will not work!
    dir_data = '/home/joseareia/Documents/SPIA/NB07-0224/Datasets/Imagewoof/train/'
    model_name = 'vgg16'
    model_dimension = 256
    center_crop = 224
    num_images = 1000
    batch_size = 8

    # Dataloader creation for the validation set
    loader = dataloader_validation(dir_data, num_images=num_images, batch_size=batch_size, 
                                   model_dimension=model_dimension, center_crop=center_crop)

    # Model definition
    model = model_def(model_name)

    # UAP delta load
    uap = torch.load(dir_uap)

    # Evaluation of the SGA attack within the validation set
    _, _, _, _, outputs, labels, y_outputs = evaluate(model, loader, uap=uap, batch_size=batch_size, device=device)
    
    # Output of the results
    print('[INFO] True image accuracy:', sum(y_outputs == labels) / len(labels))
    print('[INFO] Adversarial image accuracy:', sum(outputs == labels) / len(labels))
    print('[INFO] Fooling rate:', 1-sum(outputs == labels) / len(labels))
    print('[INFO] Fooling ratio:', 1-sum(y_outputs == outputs) / len(labels))
    
    # Time consumed
    end_time = datetime.datetime.now()
    print("[INFO] Time Consumed:", end_time - start_time)

if __name__ == '__main__':
    main()