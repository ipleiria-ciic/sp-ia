import os
import csv
import shutil

base_path = '../../../Datasets/ImageNet5'
new_folder = '../../../Datasets/ImageNet5/Images-Test'

def clean_and_create_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

clean_and_create_dir(new_folder)

csv_data = []
txt_data = []
txt_class_data = []

image_counter = 1

for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    
    if os.path.isdir(folder_path) and folder_name not in ["Images", "Images-Test"]:
        original_value = 1 if folder_name != "9999" else -1
        perturbation_value = -original_value

        for i, image_name in enumerate(os.listdir(folder_path)):
            image_path = os.path.join(folder_path, image_name)

            # Get the original class of the image.
            original_class = image_name[:4]
            
            if os.path.isfile(image_path):
                new_image_name = f"{image_counter:05}.jpg"
                new_image_path = os.path.join(new_folder, new_image_name)
                
                shutil.copy(image_path, new_image_path)
                
                csv_data.append([new_image_name, original_value, perturbation_value])
                txt_data.append(f"{new_image_name} {original_value} {perturbation_value}")

                # Append the new image name and the original class name.
                txt_class_data.append(f"{new_image_name} -- {original_class}")
                
                image_counter += 1

                print(f"Image {new_image_name} saved.")

# Write csv file.
csv_path = os.path.join(base_path, "image_data.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_number", "original", "perturbation"])
    writer.writerows(csv_data)
    
# Write txt file.
txt_path = os.path.join(base_path, "image_data.txt")
with open(txt_path, mode="w") as file:
    file.write(f"{image_counter-1}\n")
    file.write("original perturbation\n")
    file.write("\n".join(txt_data))

# Write txt class file.
txt_class_path = os.path.join(base_path, "image_to_class.txt")
with open(txt_class_path, mode="w") as file:
    file.write(f"{image_counter-1}\n")
    file.write("\n".join(txt_class_data))

print("Images moved and both CSV and TXT files created successfully.")