"""
Pipeline A: Testing script for binary classification of point clouds (table vs. no table)
"""

import os
import sys
import torch
import numpy as np
import logging
import importlib
import argparse
from tqdm import tqdm

# Add necessary paths - FIX ROOT_DIR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))  # Up 3 directories to /UCL/Object Detection and Classification
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# Import the dataset
from data_loader import TableClassificationDataset

def parse_args():
    parser = argparse.ArgumentParser('Binary Classification Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size for testing')
    parser.add_argument('--num_point', type=int, default=1024, help='point number')
    parser.add_argument('--log_dir', type=str, default='pipelineA', help='experiment root')
    parser.add_argument('--model', default='coursework2_groupJ.models.pointnet2.pointnet2_cls_ssg', help='model name')
    parser.add_argument('--test_file', type=str, default='./coursework2_groupJ/data/sun3d_test.h5', help='path to test file')
    parser.add_argument('--ucl_file', type=str, default='./coursework2_groupJ/data/ucl_data.h5', help='path to UCL data file')
    return parser.parse_args()

def test(model, loader, label_name='dataset', log_func=print):
    with torch.no_grad():
        mean_correct = []
        classifier = model.eval()
        
        all_preds = []
        all_targets = []
        table_correct = []
        non_table_correct = []
        total_tables = 0
        total_non_tables = 0
        
        # Get the device that the model is on
        device = next(model.parameters()).device
        
        log_func(f"Testing on {label_name}...")
        for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
            # Move data to the same device as the model
            points = points.to(device)
            target = target.to(device)

            points = points.transpose(2, 1)
            pred, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]
            
            all_preds.extend(pred_choice.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Overall accuracy
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            
            # Per-class accuracy
            for i in range(len(target)):
                if target[i] == 1:  # Table
                    if pred_choice[i] == 1:
                        table_correct.append(1.0)
                    else:
                        table_correct.append(0.0)
                    total_tables += 1
                else:  # Non-table
                    if pred_choice[i] == 0:
                        non_table_correct.append(1.0)
                    else:
                        non_table_correct.append(0.0)
                    total_non_tables += 1

        instance_acc = np.mean(mean_correct)
        
        # Calculate accuracy for each class (0: no table, 1: table)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Per class accuracy using our tracked metrics
        table_acc = np.mean(table_correct) if total_tables > 0 else float('nan')
        non_table_acc = np.mean(non_table_correct) if total_non_tables > 0 else float('nan')
        
        log_func(f'Test Instance Accuracy on {label_name}: {instance_acc:.4f}')
        log_func(f'Table Accuracy on {label_name}: {table_acc:.4f} ({total_tables} samples)')
        log_func(f'Non-Table Accuracy on {label_name}: {non_table_acc:.4f} ({total_non_tables} samples)')
        log_func(f'Confusion Matrix:')
        log_func(f'Pred\\True | No Table | Table')
        log_func(f'No Table  | {np.sum((all_preds == 0) & (all_targets == 0)):8d} | {np.sum((all_preds == 0) & (all_targets == 1)):5d}')
        log_func(f'Table     | {np.sum((all_preds == 1) & (all_targets == 0)):8d} | {np.sum((all_preds == 1) & (all_targets == 1)):5d}')
        
        return instance_acc, table_acc, non_table_acc

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Set the device (CPU or GPU) with proper fallback
    device = torch.device('cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu')
    
    '''CREATE DIR'''
    experiment_dir = './coursework2_groupJ/results/log/pipelineA'
    os.makedirs(experiment_dir, exist_ok=True)  # Ensure the directory exists
    
    '''LOG'''
    # Initialize logger before defining log_string
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/test_results.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    def log_string(str):
        logger.info(str)
        print(str)
    
    log_string(f"Using device: {device}")
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    
    # Load the standard test set
    test_dataset = TableClassificationDataset(args.test_file, args.num_point)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                                 shuffle=False, num_workers=4)
    
    # Load the UCL dataset (which should all contain tables)
    log_string('Load UCL dataset ...')
    ucl_dataset = TableClassificationDataset(args.ucl_file, args.num_point)
    uclDataLoader = torch.utils.data.DataLoader(ucl_dataset, batch_size=args.batch_size, 
                                              shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    num_class = 2  # Binary classification: table or no table
    model = importlib.import_module(args.model)
    classifier = model.get_model(num_class, normal_channel=False)
    
    # Move model to the appropriate device
    classifier = classifier.to(device)

    checkpoint_path = os.path.join('./coursework2_groupJ/weights/pipelineA/checkpoints', 'best_model.pth')
    
    # Load checkpoint with appropriate device mapping
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    
    log_string(f"Model loaded from {checkpoint_path} to {device}")
    
    # Log metrics if they exist in the checkpoint
    if 'instance_acc' in checkpoint:
        log_string(f"Test instance accuracy reported during training: {checkpoint['instance_acc']:.4f}")
    if 'table_acc' in checkpoint:
        log_string(f"Test table accuracy reported during training: {checkpoint['table_acc']:.4f}")
    if 'non_table_acc' in checkpoint:
        log_string(f"Test non-table accuracy reported during training: {checkpoint['non_table_acc']:.4f}")
    if 'f1_score' in checkpoint:
        log_string(f"F1 score reported during training: {checkpoint['f1_score']:.4f}")
    if 'epoch' in checkpoint:
        log_string(f"Model from epoch: {checkpoint['epoch']}")
    
    # Test on standard dataset
    log_string('\n==== TESTING ON STANDARD TEST SET ====')
    test(classifier, testDataLoader, label_name='sun3d_test', log_func=log_string)
    
    # Test on UCL dataset
    log_string('\n==== TESTING ON UCL DATASET ====')
    test(classifier, uclDataLoader, label_name='ucl_data', log_func=log_string)

if __name__ == '__main__':
    args = parse_args()
    main(args)