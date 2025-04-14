import argparse
import cv2
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
import types

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# To handle the local DINOv2 path 
original_load = torch.hub.load

# Local path to the DINOv2 model
def patched_load(repo_or_dir, model, *args, **kwargs):
    if repo_or_dir == 'torchhub/facebookresearch_dinov2_main' and kwargs.get('source') == 'local':
        dinov2_path = os.path.join(project_root, "models/Depth-Anything/torchhub/facebookresearch_dinov2_main")
        print(f"Using local DINOv2 from: {dinov2_path}")
        return original_load(dinov2_path, model, *args, **kwargs)
    return original_load(repo_or_dir, model, *args, **kwargs)

torch.hub.load = patched_load
torch.hub.set_dir(os.path.join(project_root, "models/Depth-Anything"))

# import from depth_anything
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process folders of images with Depth Anything')
parser.add_argument('--encoder', default='vits', choices=['vits', 'vitb', 'vitl'], help='Encoder type')
parser.add_argument('--visualize', action='store_true', help='Show results during processing')
parser.add_argument('--dataset', choices=['training', 'test1', 'test2', 'all'], default='all', 
                    help='Which dataset to process: training, test1, test2, or all')
parser.add_argument('--custom_dirs', nargs='*', help='Custom input directories (optional)')
args = parser.parse_args()

# Dataset definitions
training_sets = ["mit_32_d507/d507_2", "mit_76_459/76-459b", "mit_76_studyroom/76-1studyroom2", 
                "mit_gym_z_squash/gym_z_squash_scan1_oct_26_2012_erika", "mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika"]
test_sets_1 = ["harvard_c5/hv_c5_1", "harvard_c6/hv_c6_1", "harvard_c11/hv_c11_2", "harvard_tea_2/hv_tea2_2"]
test_sets_2 = ['ucl_data']
base_dataset_path1 = "./coursework2_groupJ/data/CW2-Dataset/sun3D_data"
base_dataset_path2 = "./coursework2_groupJ/data/RealSense"
rgb_path1 = '/image/'
rgb_path2 = '/rgb/'

# Build input directories based on dataset selection
input_dirs = []

# Load datatset paths
def build_dataset_paths(dataset_list, base_path, rgb_folder):
    paths = []
    for dataset in dataset_list:
        full_path = os.path.join(base_path, dataset, rgb_folder.strip('/'))
        if os.path.exists(full_path):
            paths.append((full_path, os.path.join(dataset, rgb_folder.strip('/'))))  # Store both full path and relative path
        else:
            print(f"Warning: Path does not exist: {full_path}")
    return paths

# Process datasets based on user selection
if args.dataset in ['training', 'all']:
    input_dirs.extend(build_dataset_paths(training_sets, base_dataset_path1, rgb_path1))
    
if args.dataset in ['test1', 'all']:
    input_dirs.extend(build_dataset_paths(test_sets_1, base_dataset_path1, rgb_path1))
    
if args.dataset in ['test2', 'all']:
    input_dirs.extend(build_dataset_paths(test_sets_2, base_dataset_path2, rgb_path2))

# Add any custom directories if specified
if args.custom_dirs:
    for custom_dir in args.custom_dirs:
        rel_path = os.path.relpath(custom_dir, "./coursework2_groupJ/data")
        input_dirs.append((custom_dir, rel_path))

if not input_dirs:
    parser.error("No valid input directories found. Please check dataset paths.")

encoder = args.encoder

print(f"Processing {len(input_dirs)} directories:")
for dir_path, rel_path in input_dirs:
    print(f"  - {dir_path}")

# Set up the output directory
output_dir = os.path.join("./coursework2_groupJ/results/predictions")
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Determine device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load depth model
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE)

total_params = sum(param.numel() for param in depth_anything.parameters())
print('Total parameters: {:.2f}M'.format(total_params / 1e6))

depth_anything.eval()

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

def process_image(image_path):
    """Process a single image and return depth map"""
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    raw_image = cv2.resize(raw_image, (640, 480))
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    
    h, w = image.shape[:2]
    
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        depth = depth_anything(image)
    
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    
    depth = depth.cpu().numpy().astype(np.uint8)
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    
    return depth_color

# Process each input directory
for input_dir, rel_path in input_dirs:
    # Determine which dataset type this directory belongs to
    dataset_type = None
    if any(training_set.split('/')[0] in rel_path for training_set in training_sets):
        dataset_type = "training_sets"
    elif any(test_set.split('/')[0] in rel_path for test_set in test_sets_1):
        dataset_type = "test_sets_1"
    elif any(test_set in rel_path for test_set in test_sets_2):
        dataset_type = "test_sets_2"
    else:
        dataset_type = "other"
    
    # Extract the top folder name from the relative path
    # This gives us names like "mit_32_d507", "harvard_c5", "ucl_data"
    top_folder = rel_path.split('/')[0]
    
    # Create output subdirectory under the appropriate dataset type folder
    output_subdir = os.path.join(output_dir, dataset_type, top_folder)
    os.makedirs(output_subdir, exist_ok=True)
    
    # Get all image files in the directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        continue
    
    print(f"Processing {len(image_files)} images from {input_dir}")
    print(f"Saving to {output_subdir}")
    
    # Process each image
    for image_file in tqdm(image_files):
        image_path = os.path.join(input_dir, image_file)
        
        # Process the image
        depth_map = process_image(image_path)
        if depth_map is None:
            continue
        
        # Save the depth map to the output directory with the same name as the original image
        output_filename = os.path.splitext(image_file)[0] + '.png'
        depth_path = os.path.join(output_subdir, output_filename)
        cv2.imwrite(depth_path, depth_map)
        
        # Display if requested
        if args.visualize:
            cv2.imshow('Depth Anything', depth_map)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if args.visualize:
    cv2.destroyAllWindows()

print(f"All processing complete. Results saved to {output_dir}")