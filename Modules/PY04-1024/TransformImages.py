"""
File: TransformImages.py
Author: José Areia
Date: 2024-10-24
"""

# Miscellaneous
import os
import gc
import time
from PIL import Image

# Global constants
dataset = '../Datasets/ImageNet-Pre'
save_path = '../Datasets/ImageNet5'

# Check if path exists
os.makedirs(save_path, exist_ok=True)

def resize_and_save_images(input_dir, output_dir, image_size=(128, 128)):
    # Walk through the dataset directory
    for root, dirs, files in os.walk(input_dir):
        class_number = os.path.basename(root)
        
        # Create corresponding class folder in output directory
        class_save_path = os.path.join(output_dir, class_number)
        os.makedirs(class_save_path, exist_ok=True)
        
        image_id_counter = 1
        for file in files:
            if file.endswith(('.jpg', '.JPEG', '.jpeg', '.png')):
                try:
                    # Open and resize the image
                    img_path = os.path.join(root, file)
                    with Image.open(img_path) as img:
                        img = img.resize(image_size)

                        # Ensure the image has 3 channels and shape [3, 128, 128]
                        if img.mode == 'RGB':
                            # Save the image with new naming convention
                            new_image_name = f"{class_number}_{image_id_counter}.jpg"
                            img.save(os.path.join(class_save_path, new_image_name))
                            
                            print(f"Processed and saved: {new_image_name}")
                            image_id_counter += 1
                        else:
                            print(f"Skipped {file}: Not in RGB mode")

                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    continue
                
            # Cleanup memory
            gc.collect()
    
    print("All images processed.")

# Start the image transformation process
start_time = time.time()
resize_and_save_images(dataset, save_path)
end_time = time.time()

print(f"Completed in {end_time - start_time} seconds.")
