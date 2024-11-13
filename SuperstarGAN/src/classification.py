"""
    Script to classify both original and adversarial 
    images and export the accuracy results for each trial.
"""

import os
import time
import torch
import utils
from datetime import datetime, timedelta

img_original_path = "../../../Datasets/ImageNet5/Images"
img_adversarial_path = "../results-transformed/imagenet"
img_to_class = "../../../Datasets/ImageNet5/image_to_class.txt"
log_path = "../logs/classification"

def classifier(model, device, img_given):
    was_training = model.training
    model.eval()
    
    data_transforms = utils.transforms()
    
    img = data_transforms['val'](img_given)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        model.train(mode=was_training)

    return preds[0]+1

# Start time.
start_time = time.time()

# Get the device available to use.
device = utils.use_device()

# Get the classification model.
model_path = '../models/vgg16/modelvgg16-imagenet4.pt'
model = torch.jit.load(model_path)

# Fetch all the images (original and adversarial) and the mapping classes.
original_images, adversarial_images, mapping_classes = utils.fetch_images(img_original_path, img_adversarial_path, img_to_class)

# Counters.
total_images = len(original_images)
acc_original = 0
acc_adversarial = 0

# Classify both original and adversarial image if it is correct.
for i, img in enumerate(original_images):
    # Reject the adversarial images (only noise images).
    if mapping_classes[i] == '9999':
        continue
    
    # Classification of the original image.
    r = classifier(model, device, img)
    
    # If the classification of the original image is correct, the adversarial image will be tested.
    if r == int(mapping_classes[i]):
        # Debugger. Can be deleted later.
        print(f"[ INFO ] Classification of the image '{i:04}'. [ORI: {int(mapping_classes[i])}; ADV: {r+1}] [ OK ]")
       
        # Accuracy of original images counter.
        acc_original += 1
        
        # Classification of the adversarial image.
        r_adv = classifier(model, device, adversarial_images[i])
        if r_adv != r:
            acc_adversarial += 1
    else:
        # Debugger. Can be deleted later.
        print(f"[ INFO ] Classification of the image '{i:04}'. [ORI: {int(mapping_classes[i])}; ADV: {r+1}] [ NOK ]")

print(f"[ RESULTS ] Total images classified: {total_images}.")
print(f"[ RESULTS ] Original accuracy classification: {((acc_original*100)/total_images):.2f}%")
print(f"[ RESULTS ] Adversarial accuracy classification (in a total of {acc_original} images): {((acc_adversarial*100)/acc_original):.2f}%")
print(f"[ RESULTS ] A total of {acc_adversarial} adversarial images successfully fooled the classifier!")

# End time.
elapsed = time.time() - start_time

# Create log file with the following information: datetime, total_images, acc_original and acc_adversarial.
os.makedirs(log_path, exist_ok=True)
txt_class_path = os.path.join(log_path, "log.txt")
with open(txt_class_path, mode="a") as file:
    file.write(f"Classification created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n")
    file.write(f"Total images classified: {total_images}.\n")
    file.write(f"Original accuracy classification: {((acc_original*100)/total_images):.2f}%\n")
    file.write(f"Adversarial accuracy classification (in a total of {acc_original} images): {((acc_adversarial*100)/acc_original):.2f}%\n")
    file.write(f"Total adversarial images that fooled the classifier: {acc_adversarial}.\n")
    file.write(f"Classification completed in '{str(timedelta(seconds=int(elapsed)))}'\n")
    file.write(f"---\n")

print(f"[ INFO ] Classification completed in '{str(timedelta(seconds=int(elapsed)))}'.")