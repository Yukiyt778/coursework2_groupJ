"""
Enhanced script to debug H5 files, particularly for the UCL dataset
"""
import h5py
import argparse
import numpy as np
import os
import sys
import torch

# Fix import paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))  # Get the project root
sys.path.append(ROOT_DIR)  # Add the project root to the path
sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))  # Add path to data_utils

# Try to import the dataset
try:
    from data_utils.sun3d_dataset_pytorch import SUN3DDataset
    print("Successfully imported SUN3DDataset")
except Exception as e:
    print(f"Error importing SUN3DDataset: {e}")

def parse_args():
    parser = argparse.ArgumentParser('H5Py Test')
    parser.add_argument('--h5_file', type=str, default='./coursework2_groupJ/data/ucl_data_fixed.h5', 
                        help='Path to h5 file to test')
    parser.add_argument('--detailed', action='store_true', help='Show detailed dataset information')
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points per cloud')
    return parser.parse_args()

def explore_h5_structure(file_path, detailed=False):
    """Explore the structure of an H5 file in detail"""
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\n==== H5 FILE STRUCTURE: {file_path} ====")
            
            # Get file attributes
            print("\nFile attributes:")
            for key in f.attrs.keys():
                print(f"  {key}: {f.attrs[key]}")
            
            # List all datasets and groups
            print("\nDatasets and Groups:")
            
            def print_group_items(name, obj):
                if isinstance(obj, h5py.Dataset):
                    item_type = "Dataset"
                    shape_info = f"shape={obj.shape}, dtype={obj.dtype}"
                    
                    # Print sample data for small datasets or if detailed mode is on
                    sample_info = ""
                    if detailed or (len(obj.shape) > 0 and obj.shape[0] < 10):
                        if len(obj.shape) == 1 and len(obj) > 0:
                            sample_info = f", First item: {obj[0]}"
                        elif len(obj.shape) > 1 and obj.shape[0] > 0:
                            sample_info = f", First item shape: {obj[0].shape}"
                    
                    print(f"  {name} ({item_type}): {shape_info}{sample_info}")
                else:
                    print(f"  {name} (Group)")
            
            # Traverse the H5 file hierarchy
            f.visititems(print_group_items)
            
            # For detailed analysis, examine main datasets more thoroughly
            if detailed:
                if 'data' in f:
                    data = f['data']
                    print("\nDetailed 'data' examination:")
                    print(f"  Shape: {data.shape}")
                    print(f"  Dtype: {data.dtype}")
                    if len(data.shape) > 0 and data.shape[0] > 0:
                        print(f"  First item shape: {data[0].shape if len(data.shape) > 1 else 'N/A'}")
                        if data.shape[0] < 10:
                            print("  All items:")
                            for i in range(data.shape[0]):
                                print(f"    Item {i}: {data[i].shape if len(data.shape) > 1 else data[i]}")
                        else:
                            print("  Sample items:")
                            for i in range(min(5, data.shape[0])):
                                print(f"    Item {i}: {data[i].shape if len(data.shape) > 1 else data[i]}")
                            
                if 'label' in f:
                    label = f['label']
                    print("\nDetailed 'label' examination:")
                    print(f"  Shape: {label.shape}")
                    print(f"  Dtype: {label.dtype}")
                    if len(label.shape) > 0 and label.shape[0] > 0:
                        if label.shape[0] < 10:
                            print("  All items:")
                            for i in range(label.shape[0]):
                                labels = label[i]
                                unique_vals, counts = np.unique(labels, return_counts=True)
                                print(f"    Item {i}: Unique labels={unique_vals}, Counts={counts}")
                        else:
                            print("  Sample items:")
                            for i in range(min(5, label.shape[0])):
                                labels = label[i]
                                unique_vals, counts = np.unique(labels, return_counts=True)
                                print(f"    Item {i}: Unique labels={unique_vals}, Counts={counts}")

            return True
    except Exception as e:
        print(f"Error examining H5 file: {e}")
        return False

def test_sun3d_dataset(file_path, num_points):
    """Test loading the H5 file as a SUN3DDataset"""
    try:
        # Try to load as SUN3DDataset
        print("\n==== TESTING AS SUN3DDATASET ====")
        
        # See if the dataset loads
        dataset = SUN3DDataset(file_path, num_points=num_points, split='test')
        print(f"  Successfully loaded dataset with {len(dataset)} samples")
        
        # Check the actual implementation
        print("\nExamining dataset class implementation:")
        
        # Show the SUN3DDataset __init__ parameters
        print(f"  Dataset initialization parameters:")
        print(f"    h5_file: {dataset.h5_file}")
        print(f"    num_points: {dataset.num_points}")
        print(f"    split: {dataset.split}")
        
        # Check data loading
        print("\nChecking data structure:")
        with h5py.File(file_path, 'r') as h5:
            data_shape = h5['data'].shape
            label_shape = h5['label'].shape
            print(f"  H5 data shape: {data_shape}")
            print(f"  H5 label shape: {label_shape}")
            
            # Calculate expected number of frames
            total_points = data_shape[0]
            expected_frames = total_points // num_points
            print(f"  Total points in dataset: {total_points}")
            print(f"  Points per cloud: {num_points}")
            print(f"  Expected # of complete point clouds: {expected_frames}")
            
            # Calculate train/test split
            expected_train = int(expected_frames * 0.8)
            expected_test = expected_frames - expected_train
            print(f"  Expected # of training samples (80%): {expected_train}")
            print(f"  Expected # of test samples (20%): {expected_test}")
        
        # Check frames and splitting logic
        if hasattr(dataset, 'frames'):
            print(f"\nLooking at frame construction:")
            total_frames = len(dataset.frames)
            print(f"  Total frames created: {total_frames}")
            if total_frames > 0:
                print(f"  First few frames:")
                for i in range(min(3, len(dataset.frames))):
                    start, end = dataset.frames[i]
                    print(f"    Frame {i}: points {start}-{end} ({end-start} points)")
            
            # Check split procedure
            print(f"\nLooking at test/train split:")
            split_idx = int(len(dataset.frames) * 0.8)
            print(f"  Split index: {split_idx}")
            print(f"  Train frames: {split_idx}")
            print(f"  Test frames: {len(dataset.frames) - split_idx}")
            
            # Actual frame indices assigned to this split
            print(f"\nThis dataset ({dataset.split}) has {len(dataset.frame_indices)} frames:")
            for i in range(min(len(dataset.frame_indices), 5)):
                start, end = dataset.frame_indices[i]
                print(f"  Frame {i}: points {start}-{end} ({end-start} points)")
        
        # Verify indexing works
        print("\nTesting dataset indexing:")
        for i in range(min(len(dataset), 3)):
            points, label = dataset[i]
            print(f"  Sample {i}:")
            print(f"    Points shape: {points.shape}")
            print(f"    Label shape: {label.shape}")
            unique_labels, counts = np.unique(label.numpy(), return_counts=True)
            print(f"    Label distribution: {dict(zip(unique_labels, counts))}")
            
            # Get percentage of each label
            for lbl in np.unique(label.numpy()):
                percent = counts[np.where(unique_labels == lbl)[0][0]] / len(label) * 100
                print(f"      Label {lbl}: {percent:.2f}%")
        
        # EXPLAIN THE SOLUTION
        print("\n==== EXPLAINING THE POINT CLOUD COUNT ISSUE ====")
        print(f"Dataset has {len(dataset)} point clouds in the {dataset.split} split, but why?")
        print(f"The H5 file contains {total_points} individual 3D points.")
        print(f"These points are being grouped into point clouds of {num_points} points each.")
        print(f"This gives {expected_frames} complete point clouds (frames).")
        print(f"These are then split 80/20 into training and test sets.")
        print(f"- Training set has {expected_train} point clouds")
        print(f"- Test set has {expected_test} point clouds")
        
        # Add explanation about the 33 vs 4 issue
        if data_shape[0] >= 33000:
            expected_using_all_points = data_shape[0] // 1024
            print(f"\nIf you expected 33 point clouds, it may be because:")
            print(f"1. You're considering that {data_shape[0]} points should generate {expected_using_all_points} clouds")
            print(f"2. But these are grouped by {num_points} points per cloud, then split 80/20 into train/test")
            print(f"3. So the test set ends up with only {expected_test} clouds")
            print("\nAre these 33 different scenes? Let's check by analyzing the distribution:")
            
            # Analyze if these might be 33 separate scenes
            point_stats = {}
            scene_boundaries = []
            prev_label = None
            boundary_count = 0
            
            with h5py.File(file_path, 'r') as h5:
                labels = h5['label'][:]
                # Look for label changes that might indicate scene boundaries
                for i in range(len(labels)):
                    if i % 1000 == 0:  # Print progress
                        print(f"  Analyzing point {i}/{len(labels)}...", end="\r")
                    
                    curr_label = labels[i]
                    if prev_label is not None and curr_label != prev_label:
                        boundary_count += 1
                        scene_boundaries.append(i)
                    prev_label = curr_label
                
                print("\nFound label transitions that might indicate separate scenes:")
                print(f"  Label transitions: {boundary_count}")
                if len(scene_boundaries) > 0 and len(scene_boundaries) < 50:  # Only print if reasonable number
                    print(f"  Potential scene boundaries at points: {scene_boundaries}")
                
                # Count unique labels
                unique_labels, counts = np.unique(labels, return_counts=True)
                print(f"\nOverall label distribution:")
                for lbl, count in zip(unique_labels, counts):
                    print(f"  Label {lbl}: {count} points ({count/len(labels)*100:.2f}%)")
                
            print("\nSUGGESTION:")
            print("If you need to retain all 33 'images', you may need to modify the dataset class to:")
            print("1. Not group points by fixed sizes of 1024")
            print("2. Use scene boundaries or other indicators to properly segment the data")
            print("3. Or create a separate dataloader specific to this dataset's structure")
        
        return True
    except Exception as e:
        print(f"Error testing as SUN3DDataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(args):
    print(f"H5py version: {h5py.__version__}")
    print(f"H5py location: {h5py.__file__}")
    print(f"\nTesting H5 file: {args.h5_file}")
    
    # Explore the H5 file structure
    if explore_h5_structure(args.h5_file, args.detailed):
        # Test loading as SUN3DDataset
        test_sun3d_dataset(args.h5_file, args.num_points)
    else:
        print("Could not properly examine H5 file structure.")

if __name__ == '__main__':
    args = parse_args()
    main(args) 