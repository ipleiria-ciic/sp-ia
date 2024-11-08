import os
import shutil

base_path = '../../../Datasets/ImageNet5'
new_folder = '../../../Datasets/ImageNet5/Test-Images'

def clean_and_create_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

clean_and_create_dir(new_folder)

txt_data = []
image_counter = 1
images_break = 20

for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    
    if os.path.isdir(folder_path) and folder_name not in ["Images", "Test-Images", "9999"]:
        for i, image_name in enumerate(os.listdir(folder_path)):
            if i >= images_break:
                break

            original_class = image_name[:4]
            image_path = os.path.join(folder_path, image_name)
            
            if os.path.isfile(image_path):
                new_image_name = f"{image_counter:02}-{original_class}.jpg"
                new_image_path = os.path.join(new_folder, new_image_name)
                
                shutil.copy(image_path, new_image_path)
                txt_data.append(f"{new_image_name} -- {original_class}")
                
                image_counter += 1
                print(f"Image {new_image_name} saved.")
    
# Write TXT file
txt_path = os.path.join(base_path, "test_class_data.txt")
with open(txt_path, mode="w") as file:
    file.write(f"{image_counter-1}\n")
    file.write("\n".join(txt_data))

print("Images moved and TXT file created successfully.")