"""
Author: Based on Benny's test_semseg.py
Date: Apr 2024
Description: Testing script for SUN3D binary segmentation (table vs background)
"""
import argparse
import os
import torch
import numpy as np
import importlib
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
import h5py

# Fix import paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))  # Get the project root (two levels up from pipelineC)
sys.path.append(ROOT_DIR)  # Add the project root to the path
sys.path.append(os.path.join(ROOT_DIR, 'models/pointnet2'))  # Add path to pointnet2 models
sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))  # Add path to data_utils
sys.path.append(os.path.join(ROOT_DIR, 'src/pipelineA'))  # Add path to pipelineA for provider

# Now import from data_utils
from data_utils.sun3d_dataset_pytorch import SUN3DDataset, get_data_loaders

# Binary segmentation: 0 = background, 1 = table
classes = ['background', 'table']
NUM_CLASSES = len(classes)

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg', help='model name [default: pointnet2_sem_seg]')
    parser.add_argument('--h5_file', type=str, default='/cs/student/projects1/rai/2024/jiawyang/coursework2_groupJ/data/sun3d_test_fixed.h5', help='Path to test h5 file with point clouds and labels')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--log_dir', type=str, default='/cs/student/projects1/rai/2024/jiawyang/coursework2_groupJ/results/pipelineC/sun3d_radius_tuned/sun3d_training_2025-04-18_17-01-56', help='Experiment root')
    parser.add_argument('--npoint', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--visual', action='store_true', default=False, help='Whether to visualize result [default: False]')
    return parser.parse_args()

def main(args):
    def log_string(str):
        print(str)

    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Update experiment_dir to use the absolute path
    experiment_dir = args.log_dir
    visual_dir = os.path.join(experiment_dir, 'visual')
    if args.visual and not os.path.exists(visual_dir):
        os.makedirs(visual_dir)

    # Create test dataset and loader
    if "ucl_data" in args.h5_file:
        # For UCL dataset, create a dataset that uses ALL points without any train/test split
        print("UCL dataset detected - using ALL point clouds without train/test split")
        
        # Create a completely custom dataset that manually creates point cloud frames
        class UCLCompleteDataset(torch.utils.data.Dataset):
            def __init__(self, h5_file, num_points=1024):
                self.h5_file = h5_file
                self.num_points = num_points
                
                # Load data directly from h5 file
                with h5py.File(h5_file, 'r') as f:
                    self.data = f['data'][:]
                    self.label = f['label'][:]
                
                # Calculate how many complete point clouds we can form
                self.total_points = self.data.shape[0]
                self.num_clouds = self.total_points // self.num_points
                
                print(f"UCL dataset loaded: {self.total_points} points")
                print(f"Creating {self.num_clouds} complete point clouds with {self.num_points} points each")
                
                # Normalize point clouds
                self._normalize_data()
            
            def _normalize_data(self):
                """Normalize point clouds to have zero mean and unit variance"""
                # Center each point cloud
                centroid = np.mean(self.data, axis=0)
                self.data = self.data - centroid
                
                # Scale to unit sphere
                m = np.max(np.sqrt(np.sum(self.data**2, axis=1)))
                self.data = self.data / m
            
            def __len__(self):
                return self.num_clouds
            
            def __getitem__(self, idx):
                start_idx = idx * self.num_points
                end_idx = start_idx + self.num_points
                
                # Get points and labels for this cloud
                points = self.data[start_idx:end_idx]
                labels = self.label[start_idx:end_idx]
                
                # Convert to torch tensors
                points = torch.from_numpy(points).float()
                labels = torch.from_numpy(labels).long()
                
                # Reshape points to [C, N] format expected by PointNet++
                # C = 3 (xyz coordinates), N = num_points
                points = points.transpose(0, 1)  # Shape: [3, num_points]
                
                return points, labels
        
        # Create dataset and loader with the custom class
        test_dataset = UCLCompleteDataset(args.h5_file, num_points=args.npoint)
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Testing on entire UCL dataset: {len(test_dataset)} point clouds")
    else:
        # For regular datasets, use the standard test split
        _, test_loader = get_data_loaders(
            args.h5_file,
            batch_size=args.batch_size,
            num_points=args.npoint,
            num_workers=4
        )
    
    log_string("The number of test data is: %d" % len(test_loader.dataset))

    # Load model
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()

    checkpoint = torch.load(os.path.join(experiment_dir, 'checkpoints/best_model.pth'))
    classifier.load_state_dict(checkpoint['model_state_dict'])
    
    with torch.no_grad():
        classifier = classifier.eval()
        num_batches = len(test_loader)
        total_correct = 0
        total_seen = 0
        labelweights = np.zeros(NUM_CLASSES)
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION ----')
        
        for i, (points, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
            # Points are already in [B, C, N] format from the dataset loader
            points, target = points.float().cuda(), target.long().cuda()
            
            # No need for additional transposition since data is already in correct format
            seg_pred, _ = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.cpu().data.numpy()
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += (args.batch_size * args.npoint)
            tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
            labelweights += tmp

            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            # Visualization - remove batch limit to visualize all clouds
            if args.visual:
                for b in range(points.shape[0]):
                    pts = points[b, :, :].transpose(1, 0).contiguous().cpu().data.numpy()
                    pred_label = pred_val[b]
                    gt_label = batch_label[b]
                    
                    # Save to file for visualization
                    np.savetxt(os.path.join(visual_dir, f'pts_{i}_{b}.txt'), pts[:, :3])
                    np.savetxt(os.path.join(visual_dir, f'pred_{i}_{b}.txt'), pred_label)
                    np.savetxt(os.path.join(visual_dir, f'gt_{i}_{b}.txt'), gt_label)
                    
                    # Log the visualization progress
                    if b == 0:  # Only log once per batch to avoid flooding
                        print(f"Visualizing batch {i}, point cloud {b}...")

        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class) + 1e-6))
        log_string('Test point accuracy: %f' % (total_correct / float(total_seen)))
        log_string('Test point avg class IoU: %f' % (mIoU))
        log_string('Test point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class) + 1e-6))))
        
        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s IoU: %.3f \n' % (
                classes[l], total_correct_class[l] / float(total_iou_deno_class[l] + 1e-6))
        log_string(iou_per_class_str)
        
        acc_per_class_str = '------- Acc --------\n'
        for l in range(NUM_CLASSES):
            acc_per_class_str += 'class %s Acc: %.3f \n' % (
                classes[l], total_correct_class[l] / float(total_seen_class[l] + 1e-6))
        log_string(acc_per_class_str)


if __name__ == '__main__':
    args = parse_args()
    main(args) 