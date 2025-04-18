"""
Script to visualize the point cloud segmentation results
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Fix import paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))  # Get the project root
sys.path.append(ROOT_DIR)  # Add the project root to the path
sys.path.append(os.path.join(ROOT_DIR, 'models/pointnet2/visualizer'))

def parse_args():
    parser = argparse.ArgumentParser('Visualization')
    parser.add_argument('--visual_dir', type=str, 
                        default='/cs/student/projects1/rai/2024/jiawyang/coursework2_groupJ/results/pipelineC/sun3d_training_2025-04-18_13-48-16/visual',
                        help='Directory containing visual files')
    parser.add_argument('--output_dir', type=str, 
                        default='/cs/student/projects1/rai/2024/jiawyang/coursework2_groupJ/results/pipelineC/sun3d_training_2025-04-18_13-48-16/visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=-1, 
                        help='Number of samples to visualize, -1 for all [default: -1]')
    return parser.parse_args()

def main(args):
    try:
        # Import visualization module
        from convert_txt_to_html import load_point_cloud, create_visualization, save_visualization, create_index_html
        print("Successfully imported visualization module")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
        
        # List all visualization files
        visual_dir = Path(args.visual_dir)
        pts_files = sorted(list(visual_dir.glob('pts_*.txt')))
        
        # Limit the number of samples to visualize
        if args.num_samples > 0:
            pts_files = pts_files[:args.num_samples]
        
        # Generate visualizations
        print(f"Generating visualizations for {len(pts_files)} samples...")
        visualizations = []
        
        for pts_file in pts_files:
            # Get the corresponding prediction and ground truth files
            file_prefix = pts_file.stem.split('_')[0]
            file_indices = '_'.join(pts_file.stem.split('_')[1:])
            pred_file = visual_dir / f"pred_{file_indices}.txt"
            gt_file = visual_dir / f"gt_{file_indices}.txt"
            
            # Check if files exist
            if not pred_file.exists() or not gt_file.exists():
                print(f"Missing files for {pts_file.stem}, skipping")
                continue
            
            print(f"Processing {pts_file.stem}...")
            
            # Load point cloud data
            points, predictions, ground_truth = load_point_cloud(
                str(pts_file), 
                str(pred_file), 
                str(gt_file)
            )
            
            # Create visualization
            title = f"Sample {file_indices}"
            fig = create_visualization(points, predictions, ground_truth, title)
            
            # Save visualization
            output_file = os.path.join(args.output_dir, f"vis_{file_indices}.html")
            save_visualization(fig, output_file)
            visualizations.append(output_file)
            print(f"  Saved to {output_file}")
        
        # Create index HTML
        create_index_html(args.output_dir, visualizations)
        print(f"Created index.html in {args.output_dir}")
        print("\nVisualization complete. Open the following file in a web browser to view results:")
        print(f"{os.path.join(args.output_dir, 'index.html')}")
        
    except ImportError as e:
        print(f"Error importing visualization module: {e}")
        print("\nMake sure you have installed the required dependencies:")
        print("  pip install plotly")
        
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == '__main__':
    args = parse_args()
    main(args) 