import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import joblib
import pickle
import re

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
classes = ['non_table', 'table'] 

# Configuration
IMAGE_SIZE = 224

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

def extract_depth_map_features(depth_map):
    """Extract features from a depth map for classification"""
    # Convert to grayscale 
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

def extract_enhanced_table_features(depth_map):
    """Enhanced feature extraction with specific focus on table characteristics"""
    # Get the baseline features
    features = extract_depth_map_features(depth_map)
    
    # Convert to grayscale 
    if len(depth_map.shape) == 3:
        gray = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
    else:
        gray = depth_map
    
    resized = cv2.resize(gray, (200, 150))
    h, w = resized.shape
    
    # 1. Horizontality score - tables have strong horizontal edges
    sobely = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=5)
    horizontal_ratio = np.sum(np.abs(sobely) < 10) / (h * w)
    
    # 2. Planarity score - tables are planar objects
    std_by_region = []
    for i in range(3):
        for j in range(3):
            region = resized[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
            std_by_region.append(np.std(region))
    planarity = np.mean(std_by_region)  # Lower means more planar
    
    # 3. Height consistency - tables are typically at consistent height
    height_hist, _ = np.histogram(resized, bins=5)
    height_concentration = np.max(height_hist) / np.sum(height_hist) if np.sum(height_hist) > 0 else 0
    
    # 4. Surface continuity - tables have continuous surfaces
    # Use Laplacian for edge detection (identifies discontinuities)
    laplacian = cv2.Laplacian(resized, cv2.CV_64F)
    continuity = np.sum(np.abs(laplacian) < 10) / (h * w)
    
    # 5. Regional depth consistency - compute variance of each region's mean depth
    region_means = []
    for i in range(3):
        for j in range(3):
            region = resized[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
            region_means.append(np.mean(region))
    depth_consistency = np.var(region_means)  # Lower means more consistent
    
    # Add table-specific features
    table_features = [horizontal_ratio, planarity, height_concentration, 
                      continuity, depth_consistency]
    
    return np.concatenate([features, table_features])

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
    # Check if it's a color image and convert to grayscale if needed
    if len(depth_map.shape) == 3:
        gray = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
    else:
        gray = depth_map
    
    # Normalize depth for visualization
    depth_normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = np.uint8(depth_normalized)
    
    # Apply morphological operations to clean up the depth map
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(depth_normalized, cv2.MORPH_OPEN, kernel)
    
    # Apply threshold to segment potential flat surfaces
    _, thresh = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY)
    
    # Ensure thresh is single-channel before finding contours
    if len(thresh.shape) > 2:
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    
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
    # For datasets containing tables: create a mapping of dataset paths to their label files
    dataset_label_mapping = {
        "mit_32_d507/d507_2": "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/mit_32_d507/d507_2/labels/tabletop_labels.dat",
        "mit_76_459/76-459b": "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/mit_76_459/76-459b/labels/tabletop_labels.dat",
        "mit_76_studyroom/76-1studyroom2": "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/mit_76_studyroom/76-1studyroom2/labels/tabletop_labels.dat",
        "mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika": "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika/labels/tabletop_labels.dat",
        "harvard_c5/hv_c5_1": "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/harvard_c5/hv_c5_1/labels/tabletop_labels.dat",
        "harvard_c6/hv_c6_1": "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/harvard_c6/hv_c6_1/labels/tabletop_labels.dat",
        "harvard_c11/hv_c11_2": "./coursework2_groupJ/data/CW2-Dataset/sun3d_data/harvard_c11/hv_c11_2/labels/tabletop_labels.dat",
    }
    
    # Function to extract image number for matching
    def extract_number(filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0
    
    # Dictionary to store results
    image_labels = {}
    
    # For datasets known to contain NO tables: Label all images as non-table 
    no_table_datasets = [
        "mit_gym_z_squash",
        "harvard_tea_2"
    ]
    
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
                    
                    # Store label (1 for table, 0 for non-table)
                    image_labels[depth_map_path] = 1 if has_table else 0
                
            else:
                print(f"Warning: Image path {img_path} not found")
        
        except Exception as e:
            print(f"Error processing dataset {dataset_key}: {e}")
        
        # After loading labels, check if this is a known no-table dataset
        if any(no_table_key in dataset_key for no_table_key in no_table_datasets):
            print(f"Overriding {dataset_key} labels - known to contain NO tables")
            # Find all paths for this dataset and mark them as non-tables (0)
            for key in list(image_labels.keys()):
                if dataset_key in key:
                    image_labels[key] = 0  # Mark as non-table
    
    # Print summary
    table_count = list(image_labels.values()).count(1)
    non_table_count = list(image_labels.values()).count(0)
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
            # Unlabeled handling
            labels.append(-1) 
            unlabeled_paths.append(path)
            unlabeled_indices.append(i)

    if unlabeled_paths:
        ucl_indices = [i for i, path in enumerate(unlabeled_paths) if "ucl_data" in path.lower()]
        non_ucl_indices = [i for i, path in enumerate(unlabeled_paths) if "ucl_data" not in path.lower()]
        
        print(f"  - {len(ucl_indices)} UCL images (should contain tables)")
        print(f"  - {len(non_ucl_indices)} other unlabeled images (assuming non-tables)")
        
        # Mark UCL images as tables (1)
        for i in ucl_indices:
            idx = unlabeled_indices[i]
            labels[idx] = 1 
        
        # Mark other unlabeled images(negative samples) as non-tables (0)
        for i in non_ucl_indices:
            idx = unlabeled_indices[i]
            labels[idx] = 0 
    
    # Visualize results
    tables_count = labels.count(1) 
    non_tables_count = labels.count(0) 
    print(f"Final dataset: {tables_count} tables and {non_tables_count} non-tables")
    
    # Create results dictionary 
    results = {}
    for path, label in zip(all_paths, labels):
        results[path] = int(label)
    
    return all_paths, labels, results

def create_visual_report(all_paths, labels, filename, max_images=None):
    """Create a visual grid showing images and their table/non-table classification"""
    # If max_images is None, show all images
    sample_size = len(all_paths) if max_images is None else min(max_images, len(all_paths))
    
    # Sample indices evenly across the dataset
    indices = np.linspace(0, len(all_paths)-1, sample_size, dtype=int)
    
    # Create figure - adjust grid for more images
    if sample_size <= 36:
        cols = 6  # 6×6 grid for up to 36 images
    else:
        cols = 7  # 7×x grid for larger sets
        
    rows = (sample_size + cols - 1) // cols  # Ceiling division
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))  # Slightly smaller images to fit more
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
                label = "Table" if labels[idx] == 1 else "Non-Table"
                axes[i].set_title(label, fontsize=10, color='green' if labels[idx] == 1 else 'red')
                
                # Add the image filename as a caption below the image
                axes[i].text(0.5, -0.15, image_filename, fontsize=6, ha='center', 
                             transform=axes[i].transAxes, wrap=True)
                
                axes[i].axis('off')
                
                # Add a colored border - green for table, red for non-table
                color = 'green' if labels[idx] == 1 else 'red'
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
    precision = precision_score(y_true, y_pred, zero_division=0, pos_label=1)
    recall = recall_score(y_true, y_pred, zero_division=0, pos_label=1)
    f1 = f1_score(y_true, y_pred, zero_division=0, pos_label=1)
    
    print(f"Evaluation against ground truth ({len(common_paths)} images):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return accuracy, precision, recall, f1

def create_ensemble_classifier(train_features, train_labels):
    """Create ensemble of multiple classifiers for robust performance"""
    # Create base classifiers
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm = SVC(probability=True, C=10, gamma='scale', random_state=42)
    
    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf), 
            ('svm', svm)
        ],
        voting='soft'  # Use probability estimates
    )
    
    # Create pipeline with scaling
    ensemble_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=20)), 
        ('ensemble', ensemble)
    ])
    
    # Train the ensemble
    print("Training ensemble classifier...")
    ensemble_pipeline.fit(train_features, train_labels)
    print("Ensemble classifier trained successfully")
    
    return ensemble_pipeline

def adapt_to_ucl_domain(model, train_features, train_labels, ucl_features):
    """Adapt the model to UCL domain using confidence-based self-training"""
    # Initial predictions on UCL data
    print("Adapting model to UCL domain...")
    initial_probs = model.predict_proba(ucl_features)
    
    # Find high-confidence predictions
    confidence_threshold = 0.85
    high_conf_indices = [i for i, probs in enumerate(initial_probs) 
                        if np.max(probs) >= confidence_threshold]
    
    if len(high_conf_indices) > 5:  # If we have enough confident examples
        # Get labels and features for confident predictions
        high_conf_features = [ucl_features[i] for i in high_conf_indices]
        high_conf_preds = model.predict([ucl_features[i] for i in high_conf_indices])
        
        # Combine with original training data (with higher weight on original data)
        augmented_features = np.vstack([train_features] + [train_features] + [high_conf_features])
        augmented_labels = np.concatenate([train_labels] + [train_labels] + [high_conf_preds])
        
        # Retrain model on augmented dataset
        model.fit(augmented_features, augmented_labels)
        print(f"Adapted model using {len(high_conf_indices)} high-confidence UCL samples")
    
    return model

def find_optimal_threshold(model, ucl_features):
    """Find optimal decision threshold for UCL data classification"""
    # Get probability predictions
    ucl_probs = model.predict_proba(ucl_features)
    
    # Try different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = []
    
    for threshold in thresholds:
        # Apply threshold to probabilities
        ucl_preds = [1 if prob[1] >= threshold else 0 for prob in ucl_probs]
        table_count = sum(ucl_preds)
        table_ratio = table_count / len(ucl_preds) if ucl_preds else 0
        
        results.append((threshold, table_ratio))
        print(f"Threshold {threshold}: {table_count} tables ({table_ratio:.1%})")
    
    # Since we expect most UCL images to be tables, choose threshold that gives ~85% tables
    best_threshold = min(results, key=lambda x: abs(x[1] - 0.85))[0]
    print(f"Selected threshold {best_threshold} based on domain expectations")
    
    return best_threshold

def label_ucl_data(trained_model, threshold=0.5):
    """Enhanced UCL data classification with lower threshold to detect more tables"""
    import pickle
    
    # Find all UCL data paths
    ucl_paths = []
    depth_maps_dir = "./coursework2_groupJ/results/predictions"
    ucl_dir = os.path.join(depth_maps_dir, "test_sets_2", "ucl_data")
    
    if os.path.exists(ucl_dir):
        for filename in os.listdir(ucl_dir):
            if filename.endswith('.png'):
                full_path = os.path.join(ucl_dir, filename)
                ucl_paths.append(full_path)
    
    print(f"Found {len(ucl_paths)} UCL depth maps for classification")
    
    # Extract features from UCL data - use enhanced feature extraction
    ucl_features = []
    for path in tqdm(ucl_paths):
        img = cv2.imread(path)
        if img is not None:
            features = extract_enhanced_table_features(img)
            ucl_features.append(features)
        else:
            ucl_features.append(None) 
    
    # Find optimal threshold for UCL data
    optimal_threshold = find_optimal_threshold(trained_model, [f for f in ucl_features if f is not None])
            
    # Use the trained model to predict labels with the optimal threshold
    ucl_labels = []
    ucl_polygon_lists = []
    
    for i, (path, features) in enumerate(zip(ucl_paths, ucl_features)):
        if features is not None:
            # Get probability estimate
            prob = trained_model.predict_proba([features])[0]
            
            # Apply threshold
            label = 1 if prob[1] >= optimal_threshold else 0
            ucl_labels.append(label)
            
            # Generate polygon data based on prediction
            img = cv2.imread(path)
            if label == 1:  # Table detected
                # Use detect_table_from_depth to get polygon coordinates
                try:
                    # Using depth-based detection instead of RGB-based detection
                    polygon = detect_table_from_depth(img)
                    ucl_polygon_lists.append([polygon])
                except Exception as e:
                    print(f"Error detecting table in {path}: {e}")
                    # Fallback to detect_table_region if depth-based detection fails
                    try:
                        polygon = detect_table_region(img)
                        ucl_polygon_lists.append([polygon])
                        print(f"Fallback to RGB detection for {path}")
                    except:
                        # Ultimate fallback to default rectangle
                        h, w = img.shape[:2]
                        center_x, center_y = w // 2, h // 2
                        width, height = w // 3, h // 3
                        default_polygon = [
                            [center_x - width//2, center_x + width//2, center_x + width//2, center_x - width//2],
                            [center_y - height//2, center_y - height//2, center_y + height//2, center_y + height//2]
                        ]
                        ucl_polygon_lists.append([default_polygon])
                        print(f"Using default rectangle for {path}")
            else:
                # No table, empty polygon list
                ucl_polygon_lists.append([])
        else:
            # Handle failed image reads
            ucl_labels.append(0)  # Default to non-table 
            ucl_polygon_lists.append([])
    
    # Save the labels to a .dat file
    output_dir = "./coursework2_groupJ/data/RealSense/ucl_data/labels"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "tabletop_labels.dat")
    with open(output_path, 'wb') as f:
        pickle.dump(ucl_polygon_lists, f)
    
    table_count = ucl_labels.count(1)  
    non_table_count = ucl_labels.count(0) 
    print(f"UCL classification results: {table_count} tables, {non_table_count} non-tables")
    print(f"Saved {len(ucl_paths)} UCL data labels to {output_path}")
    
    return ucl_paths, ucl_labels, ucl_polygon_lists

def main():
    # Use ground truth labels when available
    all_paths, labels, label_dict = cluster_and_label()
    ground_truth_labels = load_ground_truth_labels()

    # Define dataset paths
    training_sets = ["mit_32_d507", "mit_76_459", "mit_76_studyroom", 
                    "mit_gym_z_squash", "mit_lab_hj"]
    test_sets_1 = ["harvard_c5", "harvard_c6", "harvard_c11", "harvard_tea_2"]
    test_sets_2 = ['ucl_data']
    
    # Separate dataset paths
    train_indices = []
    test1_indices = []
    test2_indices = []
    
    # Categorize each path
    for i, path in enumerate(all_paths):
        path_lower = path.lower()
        if any(train_set in path_lower for train_set in training_sets):
            train_indices.append(i)
        elif any(test_set in path_lower for test_set in test_sets_1):
            test1_indices.append(i)
        elif any(test_set in path_lower for test_set in test_sets_2):
            test2_indices.append(i)
        else:
            train_indices.append(i)
    
    # Prepare data for each dataset
    print("\nExtracting features from training set...")
    train_features = np.array([extract_enhanced_table_features(cv2.imread(all_paths[i])) for i in train_indices])
    train_labels = np.array([labels[i] for i in train_indices])
    train_paths = [all_paths[i] for i in train_indices]

    print("Extracting features from test set 1...")
    test1_features = np.array([extract_enhanced_table_features(cv2.imread(all_paths[i])) for i in test1_indices])
    test1_labels = np.array([labels[i] for i in test1_indices])
    test1_paths = [all_paths[i] for i in test1_indices] 

    print("Extracting features from test set 2...")
    test2_features = np.array([extract_enhanced_table_features(cv2.imread(all_paths[i])) for i in test2_indices])
    test2_labels = np.array([labels[i] for i in test2_indices])
    test2_paths = [all_paths[i] for i in test2_indices]
    
    # Print dataset sizes
    print(f"Training set: {len(train_features)} images")
    print(f"Test set 1: {len(test1_features)} images")  
    print(f"Test set 2: {len(test2_features)} images")
    
    # Create and train ensemble model
    print("\nTraining ensemble model for robust performance...")
    ensemble = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, C=10, gamma='scale', random_state=42))
        ],
        voting='soft'
    )
    ensemble_pipeline = Pipeline([
        ('selector', SelectKBest(f_classif, k=20)),  # First select features
        ('scaler', StandardScaler()),                # Then scale the selected features
        ('ensemble', ensemble)
    ])

    # First fit the ensemble pipeline
    ensemble_pipeline.fit(train_features, train_labels)
    print("Ensemble model trained successfully")

    # Then adapt it to UCL domain
    adapted_ensemble = adapt_to_ucl_domain(
        ensemble_pipeline, train_features, train_labels, 
        test2_features
    )
    
    # Save the model
    ensemble_path = os.path.join(weights_dir, 'pipeline_b_ensemble.pkl')
    joblib.dump(adapted_ensemble, ensemble_path)
    print(f"Saved trained ensemble model to {ensemble_path}")
    
    # After training the model, evaluate against ground truth
    if ground_truth_labels:
        print("\nEvaluating ensemble model against ground truth:")
        # Create predictions for all paths - USE ENHANCED FEATURES
        all_features = []
        for path in all_paths:
            img = cv2.imread(path)
            if img is not None:
                # Use enhanced features but DON'T pre-select
                features = extract_enhanced_table_features(img)
                all_features.append(features)
            else:
                all_features.append(None)

        # The pipeline will handle selection and scaling internally
        ensemble_predictions = {}
        for path, feature in zip(all_paths, all_features):
            if feature is not None:
                prediction = adapted_ensemble.predict([feature])[0] 
                ensemble_predictions[path] = int(prediction)
                
        ensemble_metrics = evaluate_against_ground_truth(ensemble_predictions, ground_truth_labels)

    # Label UCL data with the ensemble model
    print("\nClassifying UCL data with adapted ensemble model...")
    ucl_paths_ens, ucl_labels_ens, ucl_polygons_ens = label_ucl_data(adapted_ensemble)
    
    # Create visual report
    create_visual_report(ucl_paths_ens, ucl_labels_ens, filename="ucl_data_report_ensemble.png")
    
    # Print UCL data statistics
    ens_table_count = ucl_labels_ens.count(1)
    ens_non_table_count = ucl_labels_ens.count(0)
    
    print(f"\nUCL Dataset Statistics:")
    print(f"Total images: {len(ucl_labels_ens)}")
    print(f"Ensemble Classification: {ens_table_count} tables ({ens_table_count/len(ucl_labels_ens)*100:.1f}%), {ens_non_table_count} non-tables")

if __name__ == "__main__":
    main()