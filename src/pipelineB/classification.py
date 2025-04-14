import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import json
from sklearn.cluster import KMeans

# Add this near the top of your file after the imports
import os

# Create the plots directory
plots_dir = "./coursework2_groupJ/results/plots/pipelineB"
os.makedirs(plots_dir, exist_ok=True)
print(f"Saving plots to: {plots_dir}")

# Create the weights directory
weights_dir = "./coursework2_groupJ/weights"
os.makedirs(weights_dir, exist_ok=True)
print(f"Saving model to: {weights_dir}")

# Create dataset structure
depth_maps_dir = "./coursework2_groupJ/results/predictions"
classes = ['table', 'non_table']

# Configuration
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

class DepthMapDataset(Dataset):
    def __init__(self, depth_paths, labels, transform=None):
        self.depth_paths = depth_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.depth_paths)
        
    def __getitem__(self, idx):
        depth_map = cv2.imread(self.depth_paths[idx])
        label = self.labels[idx]
        
        if self.transform:
            depth_map = self.transform(depth_map)
            
        return depth_map, label
        
# Define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create model
class DepthClassifier(nn.Module):
    def __init__(self):
        super(DepthClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Calculate the size after convolutions and pooling
        final_size = IMAGE_SIZE // (2**4)  # 4 pooling layers
        self.fc1 = nn.Linear(256 * final_size * final_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, len(classes))
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def extract_depth_map_features(depth_map):
    """Extract features from a depth map for KNN classification"""
    # Convert to grayscale if it's color (the INFERNO colormap)
    if len(depth_map.shape) == 3:
        gray = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
    else:
        gray = depth_map
    
    # Resize to ensure consistent feature dimensions
    resized = cv2.resize(gray, (200, 150))
    
    # Split the image into regions
    h, w = resized.shape
    features = []
    
    # Divide the image into a 5x5 grid and compute statistics for each cell
    cell_h, cell_w = h // 5, w // 5
    
    for i in range(5):
        for j in range(5):
            cell = resized[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            
            # Extract statistical features from each cell
            mean_depth = np.mean(cell)
            std_depth = np.std(cell)
            min_depth = np.min(cell)
            max_depth = np.max(cell)
            
            features.extend([mean_depth, std_depth, min_depth, max_depth])
    
    # Global features
    edges = cv2.Canny(resized, 50, 150)
    edge_count = np.sum(edges > 0) / (h * w)  # Normalized edge count
    
    # Horizontal and vertical gradients
    sobelx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
    
    grad_x_mean = np.mean(np.abs(sobelx))
    grad_y_mean = np.mean(np.abs(sobely))
    grad_ratio = grad_x_mean / (grad_y_mean + 1e-5)  # Ratio of horizontal to vertical gradients
    
    features.extend([edge_count, grad_x_mean, grad_y_mean, grad_ratio])
    
    # Add depth histogram features (10 bins)
    hist, _ = np.histogram(resized, bins=10, range=(0, 255))
    hist_norm = hist / np.sum(hist)  # Normalize
    features.extend(hist_norm)
    
    return np.array(features)

def detect_table_region(image):
    """Detect table region using edge detection and contour analysis"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate to connect edges
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    min_area = image.shape[0] * image.shape[1] * 0.05  # At least 5% of image
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not large_contours:
        # Fall back to the default centered rectangle if no suitable contour found
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        width, height = w // 3, h // 3
        return np.array([
            [center_x - width//2, center_x + width//2, center_x + width//2, center_x - width//2],
            [center_y - height//2, center_y - height//2, center_y + height//2, center_y + height//2]
        ])
    
    # Sort contours by area (largest first)
    largest_contour = sorted(large_contours, key=cv2.contourArea, reverse=True)[0]
    
    # Get a bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Make the box a bit smaller to better fit the table
    padding = 0.1  # 10% padding
    x = int(x + w * padding)
    y = int(y + h * padding)
    w = int(w * (1 - 2 * padding))
    h = int(h * (1 - 2 * padding))
    
    # Create polygon coordinates in the format expected by the reader script
    x_coords = [x, x+w, x+w, x]
    y_coords = [y, y, y+h, y+h]
    
    return [x_coords, y_coords]

def detect_table_from_depth(depth_map):
    """Detect table surface using depth information"""
    # Normalize depth for visualization
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = np.uint8(depth_normalized)
    
    # Apply morphological operations to clean up the depth map
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(depth_normalized, cv2.MORPH_OPEN, kernel)
    
    # Apply threshold to segment potential flat surfaces
    _, thresh = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours on the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area and shape
    min_area = depth_map.shape[0] * depth_map.shape[1] * 0.05
    table_contour = None
    max_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Check if the contour is roughly rectangular
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            if len(approx) >= 4 and len(approx) <= 6:
                if area > max_area:
                    max_area = area
                    table_contour = approx
    
    if table_contour is not None:
        # Use the polygon directly
        # Flatten the points and separate into x and y coordinates
        points = table_contour.reshape(-1, 2)
        x_coords = points[:, 0].tolist()
        y_coords = points[:, 1].tolist()
        return [x_coords, y_coords]
    else:
        # Fallback to default
        h, w = depth_map.shape[:2]
        center_x, center_y = w // 2, h // 2
        width, height = w // 3, h // 3
        return [
            [center_x - width//2, center_x + width//2, center_x + width//2, center_x - width//2],
            [center_y - height//2, center_y - height//2, center_y + height//2, center_y + height//2]
        ]

def load_ground_truth_labels():
    """Load ground truth table labels from pickle files"""
    import pickle
    import re
    
    # Create a mapping of dataset paths to their label files
    dataset_label_mapping = {
        "mit_32_d507/d507_2": "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/mit_32_d507/d507_2/labels/tabletop_labels.dat",
        "mit_76_459/76-459b": "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/mit_76_459/76-459b/labels/tabletop_labels.dat",
        "mit_76_studyroom/76-1studyroom2": "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/mit_76_studyroom/76-1studyroom2/labels/tabletop_labels.dat",
        "mit_gym_z_squash/gym_z_squash_scan1_oct_26_2012_erika": "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/mit_gym_z_squash/gym_z_squash_scan1_oct_26_2012_erika/labels/tabletop_labels.dat",
        "mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika": "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika/labels/tabletop_labels.dat",
        "harvard_c5/hv_c5_1": "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/harvard_c5/hv_c5_1/labels/tabletop_labels.dat",
        "harvard_c6/hv_c6_1": "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/harvard_c6/hv_c6_1/labels/tabletop_labels.dat",
        "harvard_c11/hv_c11_2": "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/harvard_c11/hv_c11_2/labels/tabletop_labels.dat",
        "harvard_tea_2/hv_tea2_2": "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/harvard_tea_2/hv_tea2_2/labels/tabletop_labels.dat",
    }
    
    # Function to extract image number for matching
    def extract_number(filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0
    
    # Dictionary to store results
    image_labels = {}
    
    # Process each dataset
    for dataset_key, label_file in dataset_label_mapping.items():
        # Get the base folder name
        base_folder = dataset_key.split('/')[-1]
        
        # Check if label file exists
        if not os.path.exists(label_file):
            print(f"Warning: Label file {label_file} not found")
            continue
            
        # Load polygon labels
        try:
            with open(label_file, 'rb') as f:
                tabletop_labels = pickle.load(f)
            
            # Determine if this is a CW2 dataset (has /image/) or RealSense (has /rgb/)
            if "mit" in dataset_key or "harvard" in dataset_key:
                img_path = os.path.join("./coursework2_groupJ/data/CW2-Dataset/sun3d_data", dataset_key, "image")
            else:
                img_path = os.path.join("./coursework2_groupJ/data/RealSense", dataset_key, "rgb")
            
            # Get all image files
            if os.path.exists(img_path):
                all_files = os.listdir(img_path)
                img_list = [f for f in all_files if f.lower().endswith(('.jpg', '.png'))]
                img_list.sort(key=extract_number)
                
                # Match images to labels
                if len(tabletop_labels) != len(img_list):
                    print(f"Warning: {dataset_key} - Number of labels ({len(tabletop_labels)}) doesn't match images ({len(img_list)})")
                    num_to_process = min(len(tabletop_labels), len(img_list))
                else:
                    num_to_process = len(img_list)
                
                # Create label for each image
                for i in range(num_to_process):
                    polygon_list = tabletop_labels[i]
                    img_name = img_list[i]
                    
                    # Depth map path will be based on the dataset and image name
                    if "mit" in dataset_key:
                        depth_path_part = os.path.join("training_sets", dataset_key.split('/')[0], os.path.splitext(img_name)[0] + ".png")
                    elif "harvard" in dataset_key:
                        depth_path_part = os.path.join("test_sets_1", dataset_key.split('/')[0], os.path.splitext(img_name)[0] + ".png")
                    else:
                        depth_path_part = os.path.join("test_sets_2", dataset_key.split('/')[0], os.path.splitext(img_name)[0] + ".png")
                    
                    # Full depth map path
                    depth_map_path = os.path.join(depth_maps_dir, depth_path_part)
                    
                    # Has table if polygon list is not empty
                    has_table = len(polygon_list) > 0
                    
                    # Store label (0 for table, 1 for non-table)
                    image_labels[depth_map_path] = 0 if has_table else 1
                
            else:
                print(f"Warning: Image path {img_path} not found")
        
        except Exception as e:
            print(f"Error processing dataset {dataset_key}: {e}")
    
    # Print summary
    table_count = list(image_labels.values()).count(0)
    non_table_count = list(image_labels.values()).count(1)
    print(f"Loaded {len(image_labels)} image labels: {table_count} tables, {non_table_count} non-tables")
    
    return image_labels

def cluster_and_label():
    """Label depth maps using ground truth labels when available, fall back to clustering for unlabeled data"""
    # Set up the directory structure
    depth_maps_dir = "./coursework2_groupJ/results/predictions"
    
    # Define the category folders
    folders = ["training_sets", "test_sets_1", "test_sets_2"]
    
    # Step 1: Collect all depth map paths
    all_paths = []
    
    print("Collecting depth maps for analysis...")
    
    for folder in folders:
        folder_dir = os.path.join(depth_maps_dir, folder)
        if not os.path.exists(folder_dir):
            print(f"Warning: {folder} directory not found")
            continue
            
        for dataset_dir in os.listdir(folder_dir):
            dataset_path = os.path.join(folder_dir, dataset_dir)
            if not os.path.isdir(dataset_path):
                continue
                
            for filename in os.listdir(dataset_path):
                if filename.endswith('.png') and not filename.endswith('_visualization.png'):
                    full_path = os.path.join(dataset_path, filename)
                    all_paths.append(full_path)
    
    print(f"Found {len(all_paths)} depth maps for analysis")
    
    # Step 2: Load ground truth labels
    ground_truth_labels = load_ground_truth_labels()
    print(f"Loaded {len(ground_truth_labels)} ground truth labels")
    
    # Step 3: Process each image - use ground truth when available, clustering otherwise
    labels = []
    unlabeled_paths = []
    unlabeled_indices = []
    
    # First pass: use ground truth labels when available
    for i, path in enumerate(all_paths):
        if path in ground_truth_labels:
            # Use the ground truth label
            labels.append(ground_truth_labels[path])
        else:
            # Mark for clustering
            labels.append(-1)  # Temporary placeholder
            unlabeled_paths.append(path)
            unlabeled_indices.append(i)
    
    # If we have unlabeled images, use clustering to label them
    if unlabeled_paths:
        print(f"{len(unlabeled_paths)} images don't have ground truth labels. Using clustering.")
        
        # Extract features for unlabeled images
        print("Extracting features from unlabeled depth maps...")
        unlabeled_features = []
        for path in tqdm(unlabeled_paths):
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not read {path}")
                continue
            features = extract_depth_map_features(img)
            unlabeled_features.append(features)
        
        # Standardize features
        unlabeled_features = np.array(unlabeled_features)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(unlabeled_features)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Determine which cluster is likely tables
        cluster_0_features = scaled_features[cluster_labels == 0]
        cluster_1_features = scaled_features[cluster_labels == 1]
        
        # Use features that might indicate tables
        table_indicator_indices = [104, 105]
        
        # Compare the clusters (if both have elements)
        if len(cluster_0_features) > 0 and len(cluster_1_features) > 0:
            cluster_0_indicators = np.mean(cluster_0_features[:, table_indicator_indices], axis=0)
            cluster_1_indicators = np.mean(cluster_1_features[:, table_indicator_indices], axis=0)
            
            # Higher horizontal-to-vertical gradient ratio might indicate tables
            if cluster_0_indicators[1] > cluster_1_indicators[1]:
                table_cluster = 0
                non_table_cluster = 1
            else:
                table_cluster = 1
                non_table_cluster = 0
                
            # Convert cluster labels to our class labels (0=table, 1=non-table)
            for i, idx in enumerate(unlabeled_indices):
                if cluster_labels[i] == table_cluster:
                    labels[idx] = 0  # Table
                else:
                    labels[idx] = 1  # Non-table
        else:
            # If one cluster is empty, assign all to non-tables
            for i, idx in enumerate(unlabeled_indices):
                labels[idx] = 1  # Default to non-table
    
    # Step 4: Visualize results
    tables_count = labels.count(0)
    non_tables_count = labels.count(1)
    print(f"Final dataset: {tables_count} tables and {non_tables_count} non-tables")
    
    # Save all results
    results = {}
    for path, label in zip(all_paths, labels):
        results[path] = int(label)
    
    # Save to JSON
    with open('table_detection_labels.json', 'w') as f:
        json.dump(results, f)
    
    # Save to the plots directory
    json_path = os.path.join(plots_dir, 'table_detection_labels.json')
    with open(json_path, 'w') as f:
        json.dump(results, f)
    print(f"Saved label data to {json_path}")
    
    
    return all_paths, labels, results

def create_visual_report(all_paths, labels, filename='table_detection_report.png', max_images=50):
    """Create a visual grid showing images and their table/non-table classification"""
    # Limit the number of images to display
    sample_size = min(max_images, len(all_paths))
    
    # Sample indices evenly across the dataset
    indices = np.linspace(0, len(all_paths)-1, sample_size, dtype=int)
    
    # Create figure
    cols = 5
    rows = (sample_size + cols - 1) // cols  # Ceiling division
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()
    
    # Plot each image with its label
    for i, idx in enumerate(indices):
        if i < len(axes):
            img = cv2.imread(all_paths[idx])
            if img is not None:
                axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
                # Extract just the filename without the full path
                image_filename = os.path.basename(all_paths[idx])
                
                # Add Table/Non-Table at the top
                label = "Table" if labels[idx] == 0 else "Non-Table"
                axes[i].set_title(label, fontsize=10, color='green' if labels[idx] == 0 else 'red')
                
                # Add the image filename as a caption below the image
                axes[i].text(0.5, -0.15, image_filename, fontsize=6, ha='center', 
                             transform=axes[i].transAxes, wrap=True)
                
                axes[i].axis('off')
                
                # Add a colored border - green for table, red for non-table
                color = 'green' if labels[idx] == 0 else 'red'
                for spine in axes[i].spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(4)
    
    # Turn off any unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    # Save to the plots directory with the original filename
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path, dpi=150)
    print(f"Saved visual report to {save_path}")

def evaluate_against_ground_truth(predictions, ground_truth):
    """Evaluate predictions against ground truth labels"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Get common keys (images that have both predictions and ground truth)
    common_paths = set(predictions.keys()) & set(ground_truth.keys())
    
    if not common_paths:
        print("No common images found between predictions and ground truth")
        return
    
    # Extract the labels
    y_true = [ground_truth[path] for path in common_paths]
    y_pred = [predictions[path] for path in common_paths]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Evaluation against ground truth ({len(common_paths)} images):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return accuracy, precision, recall, f1

def label_ucl_data():
    """Save the UCL data labels to a .dat file in the RealSense directory"""
    import pickle
    
    # Find all UCL data paths and their classifications
    ucl_paths = []
    ucl_labels = []
    ucl_polygon_lists = []
    
    # Get predictions from our model
    all_paths, labels, _ = cluster_and_label()
    
    # Find UCL data
    for i, path in enumerate(all_paths):
        if 'ucl_data' in path.lower():
            ucl_paths.append(path)
            label = labels[i]
            ucl_labels.append(label)
            
            if label == 0:  # Table
                # Load the image
                img = cv2.imread(path)
                if img is not None:
                    # Get image dimensions
                    h, w = img.shape[:2]
                    
                    # Use table detection if available, otherwise fallback to default rectangle
                    try:
                        # Try to detect the actual table
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                        edges = cv2.Canny(blurred, 50, 150)
                        
                        # Dilate to connect edges
                        kernel = np.ones((5,5), np.uint8)
                        dilated = cv2.dilate(edges, kernel, iterations=2)
                        
                        # Find contours
                        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Filter by area
                        min_area = h * w * 0.05  # At least 5% of image
                        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
                        
                        if large_contours:
                            # Get largest contour
                            largest_contour = max(large_contours, key=cv2.contourArea)
                            
                            # Approximate the contour to get a polygon
                            peri = cv2.arcLength(largest_contour, True)
                            approx = cv2.approxPolyDP(largest_contour, 0.05 * peri, True)
                            
                            # If too many points, simplify to a rectangle
                            if len(approx) > 6:
                                x, y, w_rect, h_rect = cv2.boundingRect(approx)
                                # Create rectangle corners
                                x_coords = [x, x + w_rect, x + w_rect, x]
                                y_coords = [y, y, y + h_rect, y + h_rect]
                            else:
                                # Use the approximated polygon
                                x_coords = [int(point[0][0]) for point in approx]
                                y_coords = [int(point[0][1]) for point in approx]
                        else:
                            # Fallback to default rectangle
                            raise Exception("No large contours found")
                    except:
                        # Default rectangle in the center
                        center_x, center_y = w // 2, h // 2
                        width, height = w // 2, h // 2  # Make the rectangle bigger
                        
                        # Rectangle corners - create x and y coordinate arrays
                        x_coords = [center_x - width//2, center_x + width//2, center_x + width//2, center_x - width//2]
                        y_coords = [center_y - height//2, center_y - height//2, center_y + height//2, center_y + height//2]
                    
                    # THIS IS THE CRITICAL PART - store x_coords and y_coords separately
                    # This matches the expected format for read_labels.py
                    ucl_polygon_lists.append([[x_coords, y_coords]])
                else:
                    ucl_polygon_lists.append([])  # Empty if image can't be read
            else:
                # Non-table gets empty polygon list
                ucl_polygon_lists.append([])
    
    # Create the output directory and save files
    output_dir = "./coursework2_groupJ/data/RealSense/ucl_data/labels"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the labels to a .dat file
    output_path = os.path.join(output_dir, "tabletop_labels.dat")
    with open(output_path, 'wb') as f:
        pickle.dump(ucl_polygon_lists, f)
    
    print(f"Saved {len(ucl_paths)} UCL data labels to {output_path}")

    
    return ucl_paths, ucl_labels, ucl_polygon_lists

def main():
    # Use ground truth labels when available
    all_paths, labels, label_dict = cluster_and_label()
    
    # Load ground truth labels for evaluation
    ground_truth_labels = load_ground_truth_labels()
    
    # Instead of random split, use your predefined dataset definitions
    training_sets = ["mit_32_d507", "mit_76_459", "mit_76_studyroom", 
                    "mit_gym_z_squash", "mit_lab_hj"]
    test_sets_1 = ["harvard_c5", "harvard_c6", "harvard_c11", "harvard_tea_2"]
    test_sets_2 = ['ucl_data']
    
    # Separate paths based on dataset definitions
    train_indices = []
    test1_indices = []
    test2_indices = []
    
    # Categorize each path
    for i, path in enumerate(all_paths):
        path_lower = path.lower()
        
        # Check which dataset it belongs to
        if any(train_set in path_lower for train_set in training_sets):
            train_indices.append(i)
        elif any(test_set in path_lower for test_set in test_sets_1):
            test1_indices.append(i)
        elif any(test_set in path_lower for test_set in test_sets_2):
            test2_indices.append(i)
        else:
            train_indices.append(i)
    
    # Prepare data for each dataset
    train_features = np.array([extract_depth_map_features(cv2.imread(all_paths[i])) for i in train_indices])
    train_labels = np.array([labels[i] for i in train_indices])
    train_paths = [all_paths[i] for i in train_indices]
    
    test1_features = np.array([extract_depth_map_features(cv2.imread(all_paths[i])) for i in test1_indices])
    test1_labels = np.array([labels[i] for i in test1_indices])
    test1_paths = [all_paths[i] for i in test1_indices]
    
    test2_features = np.array([extract_depth_map_features(cv2.imread(all_paths[i])) for i in test2_indices])
    test2_labels = np.array([labels[i] for i in test2_indices])
    test2_paths = [all_paths[i] for i in test2_indices]
    
    # Print dataset sizes
    print(f"Training set: {len(train_features)} images")
    print(f"Test set 1: {len(test1_features)} images")  
    print(f"Test set 2: {len(test2_features)} images")
    
    # Create pipeline with scaling and KNN
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
    ])
    
    # Train model on the training data
    pipeline.fit(train_features, train_labels)
    
    # Evaluate on test set 1
    test1_accuracy = pipeline.score(test1_features, test1_labels)
    print(f"Test set 1 accuracy: {test1_accuracy:.4f}")
    
    # Evaluate on test set 2
    test2_accuracy = pipeline.score(test2_features, test2_labels)
    print(f"Test set 2 accuracy: {test2_accuracy:.4f}")
    
    # Find optimal K using test set 1 as validation
    k_values = list(range(1, 21, 2))
    val_accuracies = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        pipeline = Pipeline([('scaler', StandardScaler()), ('knn', knn)])
        pipeline.fit(train_features, train_labels)
        val_acc = pipeline.score(test1_features, test1_labels)
        val_accuracies.append(val_acc)
        print(f"K={k}, Accuracy: {val_acc:.4f}")
    
    # Plot K vs. accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, val_accuracies, marker='o')
    plt.title('KNN: Effect of K Value on Accuracy')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'knn_k_selection.png'))

    # Use best K
    best_k = k_values[np.argmax(val_accuracies)]
    print(f"Best K value: {best_k}")
    
    # Final model with best K
    best_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=best_k, weights='distance'))
    ])
    best_pipeline.fit(train_features, train_labels)
    
    # Save the model
    import joblib
    weights_dir = "./coursework2_groupJ/weights"
    os.makedirs(weights_dir, exist_ok=True)  # Create directory if it doesn't exist
    model_path = os.path.join(weights_dir, 'pipeline_b_knn.pkl')
    joblib.dump(best_pipeline, model_path)
    print(f"Saved trained KNN model to {model_path}")
    
    # Generate a visual report
    create_visual_report(all_paths, labels, filename='all_images_report.png')

    create_visual_report([all_paths[i] for i in train_indices], 
                          [labels[i] for i in train_indices], 
                          filename='training_set_report.png',
                          max_images=25)

    create_visual_report([all_paths[i] for i in test1_indices], 
                          [labels[i] for i in test1_indices], 
                          filename='test1_set_report.png',
                          max_images=25)

    create_visual_report([all_paths[i] for i in test2_indices], 
                          [labels[i] for i in test2_indices], 
                          filename='test2_set_report.png',
                          max_images=25)

    # Create reports for each individual dataset
    dataset_groups = {
        "mit_32_d507": "MIT 32 D507",
        "mit_76_459": "MIT 76 459",
        "mit_76_studyroom": "MIT 76 Studyroom",
        "mit_gym_z_squash": "MIT Gym Z Squash",
        "mit_lab_hj": "MIT Lab HJ",
        "harvard_c5": "Harvard C5",
        "harvard_c6": "Harvard C6",
        "harvard_c11": "Harvard C11",
        "harvard_tea_2": "Harvard Tea 2",
        "ucl_data": "UCL Data"
    }

    # Process each dataset individually
    for dataset_key, dataset_name in dataset_groups.items():
        # Find images belonging to this dataset
        dataset_indices = []
        for i, path in enumerate(all_paths):
            if dataset_key in path:
                dataset_indices.append(i)
        
        # Skip if no images found
        if not dataset_indices:
            print(f"No images found for dataset: {dataset_name}")
            continue
        
        # Create report for this dataset
        filename = f"{dataset_key}_report.png"
        report_title = f"{dataset_name} Dataset"
        
        print(f"Creating report for {dataset_name} with {len(dataset_indices)} images")
        
        # Get paths and labels for this dataset
        dataset_paths = [all_paths[i] for i in dataset_indices]
        dataset_labels = [labels[i] for i in dataset_indices]
        
        # Create a visual report for this dataset
        create_visual_report(dataset_paths, dataset_labels, 
                             filename=filename, 
                             max_images=min(len(dataset_indices), 25))
        
        # Calculate stats for this dataset
        table_count = dataset_labels.count(0)
        non_table_count = dataset_labels.count(1)
        
        print(f"  - {dataset_name}: {table_count} tables, {non_table_count} non-tables")

    # After training the model, evaluate against ground truth
    if ground_truth_labels:
        print("\nEvaluating model against ground truth:")
        
        # Create predictions for all paths
        all_features = [extract_depth_map_features(cv2.imread(path)) for path in all_paths]
        all_predictions = {}
        for path, feature in zip(all_paths, all_features):
            prediction = best_pipeline.predict([feature])[0]
            all_predictions[path] = int(prediction)
        
        # Evaluate
        evaluate_against_ground_truth(all_predictions, ground_truth_labels)

    # Label UCL data
    print("\nLabeling UCL data and saving to .dat file...")
    ucl_paths, ucl_labels, ucl_polygons = label_ucl_data()
    
    # Print UCL data statistics
    table_count = ucl_labels.count(0)
    non_table_count = ucl_labels.count(1)
    print(f"\nUCL Dataset Statistics:")
    print(f"Total images: {len(ucl_labels)}")
    print(f"Tables: {table_count} ({table_count/len(ucl_labels)*100:.1f}%)")
    print(f"Non-tables: {non_table_count} ({non_table_count/len(ucl_labels)*100:.1f}%)")

if __name__ == "__main__":
    main()
