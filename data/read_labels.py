# This scrip shows how to load the table polygon labels
# The file ./labels/tabletop_labels.dat contains a python 
# 3D list where indexes correspond to [frame][table_instance][coordinate]:
# coordinates x,y represent polygon vertices around a table instance

import os
import pickle
import matplotlib.pyplot as plt
import re

# path = "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/harvard_c5/hv_c5_1"
# path = "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/harvard_c6/hv_c6_1"
# path = "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/mit_76_studyroom/76-1studyroom2"
# path = "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/mit_32_d507/d507_2"
# path = "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/harvard_c11/hv_c11_2"
# path = "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika"
# path = "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/mit_76_459/76-459b"
# path = "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/mit_gym_z_squash/gym_z_squash_scan1_oct_26_2012_erika"
# path = "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/harvard_tea_2/hv_tea2_2"

path = "./coursework2_groupJ/data/RealSense/ucl_data"

label_path = "/labels/tabletop_labels.dat"

# img_path = "/image/"
img_path = "/rgb/"

with open(path+label_path, 'rb') as label_file:
    tabletop_labels = pickle.load(label_file)
    label_file.close()

# Get the list of image files
all_files = os.listdir(path+img_path)

# Filter for only image files 
valid_image_extensions = ['.jpg', '.png']
img_list = [f for f in all_files if os.path.splitext(f.lower())[1] in valid_image_extensions]

# Sort the images 
def extract_number(filename):
    # Extract the number from the filename using regex
    numbers = re.findall(r'\d+', filename)
    # If numbers found, return the first one as an integer
    return int(numbers[0]) if numbers else 0

# Sort the images
img_list.sort(key=extract_number)

# Make sure we have as many labels as images
if len(tabletop_labels) != len(img_list):
    print(f"Warning: Number of label entries ({len(tabletop_labels)}) doesn't match number of images ({len(img_list)})")
    # Only process the smaller number of items
    num_to_process = min(len(tabletop_labels), len(img_list))
else:
    num_to_process = len(img_list)

# Iterate through images and labels
for i in range(num_to_process):
    polygon_list = tabletop_labels[i]
    img_name = img_list[i]
    
    try:
        img = plt.imread(path+img_path+img_name)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f"Image: {img_name}")
        
        for polygon in polygon_list:
            plt.plot(polygon[0]+polygon[0][0:1], polygon[1]+polygon[1][0:1], 'r', linewidth=2)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error processing image {img_name}: {e}")
        continue