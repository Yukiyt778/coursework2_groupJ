import os
import numpy as np
import h5py
import glob
import cv2
from tqdm import tqdm
import pickle
import json

def load_camera_intrinsics(intrinsics_path):
    """
    Load camera intrinsics for the UCL data.
    
    Args:
        intrinsics_path: Path to the intrinsics.txt file
    
    Returns:
        dict with camera parameters (fx, fy, cx, cy, coeffs, depth_scale)
    """
    try:
        # First check if there's a JSON file with intrinsics (preferred format)
        json_path = os.path.splitext(intrinsics_path)[0] + '.json'
        if os.path.exists(json_path):
            print(f"Loading intrinsics from JSON file: {json_path}")
            with open(json_path, 'r') as f:
                intrinsics_data = json.load(f)
                
                # Extract relevant parameters
                fx = float(intrinsics_data.get('fx', 0))
                fy = float(intrinsics_data.get('fy', 0))  
                cx = float(intrinsics_data.get('cx', 0))
                cy = float(intrinsics_data.get('cy', 0))
                
                # Extract distortion coefficients if available
                coeffs = intrinsics_data.get('distortion_coefficients', [0, 0, 0, 0, 0])
                
                # Get depth scale (convert raw units to meters)
                depth_scale = float(intrinsics_data.get('depth_scale', 0.001))  # Default 1mm = 0.001m
                
                print(f"Loaded intrinsics from JSON: fx={fx}, fy={fy}, cx={cx}, cy={cy}, depth_scale={depth_scale}")
                
                return {
                    'fx': fx,
                    'fy': fy,
                    'cx': cx,
                    'cy': cy,
                    'coeffs': coeffs,
                    'depth_scale': depth_scale
                }
        
        # Fallback to text file parsing
        print(f"Loading intrinsics from text file: {intrinsics_path}")
        with open(intrinsics_path, 'r') as f:
            lines = f.readlines()
            
            # Set defaults
            fx, fy, cx, cy = 0, 0, 0, 0
            coeffs = [0, 0, 0, 0, 0]
            depth_scale = 0.001  # Default for RealSense: 1mm = 0.001m
            
            # Parse the depth intrinsics section
            depth_section = False
            for line in lines:
                if "Depth Intrinsics:" in line or "Camera Intrinsics:" in line:
                    depth_section = True
                    continue
                
                if depth_section:
                    if "PPX:" in line or "Principal Point X:" in line:
                        cx = float(line.split(":")[1].strip())
                    elif "PPY:" in line or "Principal Point Y:" in line:
                        cy = float(line.split(":")[1].strip())
                    elif "Fx:" in line or "Focal Length X:" in line:
                        fx = float(line.split(":")[1].strip())
                    elif "Fy:" in line or "Focal Length Y:" in line:
                        fy = float(line.split(":")[1].strip())
                    elif "Depth Scale:" in line:
                        depth_scale = float(line.split(":")[1].strip())
                    elif "Distortion Coefficients:" in line:
                        # Try to parse distortion coefficients if present
                        try:
                            coeff_line = next(f).strip()
                            coeffs = [float(x) for x in coeff_line.split()]
                        except:
                            # Use default coefficients if parsing fails
                            coeffs = [0, 0, 0, 0, 0]
                    elif "Extrinsics" in line:
                        # We've moved past the depth section
                        break
            
            print(f"Loaded intrinsics from text: fx={fx}, fy={fy}, cx={cx}, cy={cy}, depth_scale={depth_scale}")
            
            # Validate intrinsics - these should never be zero!
            if fx == 0 or fy == 0 or cx == 0 or cy == 0:
                print("WARNING: Invalid intrinsics detected (zeros), using RealSense D455 defaults")
                # RealSense D455 typical defaults (these could still be wrong for your specific device)
                fx = 390.7360534667969
                fy = 390.7360534667969
                cx = 320.08819580078125
                cy = 244.1026153564453
            
            return {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'coeffs': coeffs,
                'depth_scale': depth_scale
            }
    except Exception as e:
        print(f"Error loading intrinsics from {intrinsics_path}: {e}")
        print("Using RealSense D455 default values")
        
        # Return RealSense D455 default values
        return {
            'fx': 390.7360534667969,
            'fy': 390.7360534667969,
            'cx': 320.08819580078125,
            'cy': 244.1026153564453,
            'coeffs': [0, 0, 0, 0, 0],
            'depth_scale': 0.001  # 1mm = 0.001m for RealSense
        }

def undistort_depth_image(depth_img, intrinsics):
    """
    Undistort a depth image using camera intrinsics
    
    Args:
        depth_img: Raw depth image
        intrinsics: Camera intrinsics dictionary
    
    Returns:
        Undistorted depth image
    """
    # Create camera matrix
    camera_matrix = np.array([
        [intrinsics['fx'], 0, intrinsics['cx']],
        [0, intrinsics['fy'], intrinsics['cy']],
        [0, 0, 1]
    ])
    
    # Get distortion coefficients
    dist_coeffs = np.array(intrinsics['coeffs'])
    
    # Check if we need to undistort (skip if all coeffs are 0)
    if np.all(dist_coeffs == 0):
        return depth_img
    
    # Undistort the image
    h, w = depth_img.shape
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    # Use OpenCV to undistort
    undistorted_img = cv2.undistort(depth_img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    return undistorted_img

def depth_to_pointcloud(depth_img, intrinsics):
    """
    Convert depth image to 3D point cloud.
    
    Args:
        depth_img: HxW depth image
        intrinsics: dict with camera parameters (fx, fy, cx, cy, coeffs, depth_scale)
    
    Returns:
        points: Nx3 array of 3D points
    """
    # Scale depth values to meters
    z = depth_img.astype(np.float32) * intrinsics['depth_scale']
    
    # Get image dimensions
    height, width = depth_img.shape
    
    # Create coordinate maps for all pixels
    # Note the ordering: x coordinates correspond to cols (width), y to rows (height)
    pixel_x, pixel_y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert to camera coordinates using the pinhole camera model
    x = (pixel_x - intrinsics['cx']) * z / intrinsics['fx']
    y = (pixel_y - intrinsics['cy']) * z / intrinsics['fy']
    
    # Stack coordinates and remove invalid points (where depth is 0)
    points = np.stack([x, y, z], axis=-1)
    valid_mask = z > 0
    points = points[valid_mask]
    
    # Add debug info
    num_points = len(points)
    print(f"Generated {num_points} 3D points from depth image")
    
    # Check for invalid/extreme values that might indicate problems
    if num_points > 0:
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        print(f"Point cloud bounds: min={min_vals}, max={max_vals}")
        
        # Check if points form a frustum by measuring spread
        x_spread = max_vals[0] - min_vals[0]
        y_spread = max_vals[1] - min_vals[1]
        z_spread = max_vals[2] - min_vals[2]
        print(f"Point cloud spread: X={x_spread:.3f}m, Y={y_spread:.3f}m, Z={z_spread:.3f}m")
        
        # Warn about potential frustum issue
        if x_spread < 0.1 or y_spread < 0.1:
            print("WARNING: Point cloud has very little spread in X/Y dimensions.")
            print("This might indicate a camera frustum problem with the intrinsics.")
    
    return points

def visualize_pointcloud(points, output_file=None):
    """
    Visualize point cloud using Open3D (if available)
    
    Args:
        points: Nx3 array of 3D points
        output_file: Path to save visualization image
    """
    try:
        import open3d as o3d
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Add colors based on height (z value)
        colors = np.zeros((len(points), 3))
        z_min = np.min(points[:, 2])
        z_max = np.max(points[:, 2])
        z_range = max(z_max - z_min, 1e-5)  # Avoid division by zero
        
        # Color based on height (blue->green->red)
        for i in range(len(points)):
            z_norm = (points[i, 2] - z_min) / z_range
            colors[i] = [z_norm, 1-z_norm, 1-z_norm]
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Point Cloud Visualization", width=800, height=600)
        
        # Add point cloud
        vis.add_geometry(pcd)
        
        # Set viewpoint
        view_control = vis.get_view_control()
        view_control.set_front([0, 0, -1])
        view_control.set_up([0, -1, 0])
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis.add_geometry(coord_frame)
        
        # Update and render
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        
        # Save image if output file specified
        if output_file:
            vis.capture_screen_image(output_file)
            print(f"Visualization saved to {output_file}")
        
        # Run visualization (comment out for non-interactive mode)
        # vis.run()
        
        # Close visualizer
        vis.destroy_window()
        
        return True
    except ImportError:
        print("Open3D not available, skipping visualization")
        return False
    except Exception as e:
        print(f"Error visualizing point cloud: {e}")
        return False

def prepare_ucl_data(data_dir, output_file, max_points=1024, visualize=True):
    """
    Process UCL data and save to H5 file.
    
    Args:
        data_dir: Path to UCL data directory
        output_file: Path to output H5 file
        max_points: Maximum number of points per frame
        visualize: Whether to visualize the point clouds
    """
    print(f"Processing UCL data from {data_dir}")
    
    # Load camera intrinsics
    intrinsics_path = os.path.join(data_dir, 'intrinsics.txt')
    intrinsics = load_camera_intrinsics(intrinsics_path)
    print(f"Loaded camera intrinsics: {intrinsics}")
    
    # Define paths
    depth_dir = os.path.join(data_dir, 'depth')
    tsdf_dir = os.path.join(data_dir, 'depthTSDF')
    
    # Look for depth images first, then fall back to TSDF
    if os.path.exists(depth_dir) and len(glob.glob(os.path.join(depth_dir, '*.png'))) > 0:
        print(f"Using raw depth images from {depth_dir}")
        image_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
        use_depth = True
    elif os.path.exists(tsdf_dir) and len(glob.glob(os.path.join(tsdf_dir, '*.png'))) > 0:
        print(f"Using TSDF images from {tsdf_dir}")
        image_files = sorted(glob.glob(os.path.join(tsdf_dir, '*.png')))
        use_depth = False
    else:
        raise FileNotFoundError(f"No depth or TSDF images found in {data_dir}")
    
    print(f"Found {len(image_files)} image files")
    
    # Create visualization directory
    vis_dir = os.path.join(os.path.dirname(output_file), 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Process each frame
    all_points = []
    all_labels = []
    
    # Tracking statistics
    total_frames = 0
    total_table_points = 0
    total_points = 0
    
    for idx, image_file in enumerate(tqdm(image_files, desc="Processing frames")):
        # Load image
        depth_img = cv2.imread(image_file, cv2.IMREAD_ANYDEPTH)
        
        if depth_img is None:
            print(f"Warning: Could not load image {image_file}")
            continue
        
        # Undistort the depth image if using raw depth
        if use_depth:
            depth_img = undistort_depth_image(depth_img, intrinsics)
        
        # Convert to point cloud
        points = depth_to_pointcloud(depth_img, intrinsics)
        
        # Skip if no valid points
        if len(points) == 0:
            print(f"Warning: No valid points in frame {image_file}")
            continue
        
        # IMPORTANT: UCL data contains tables, so label ALL points as tables (label 1)
        labels = np.ones(len(points), dtype=np.int32)
        
        # Update statistics
        total_frames += 1
        frame_points = len(points)
        total_points += frame_points
        total_table_points += frame_points  # All points are labeled as tables
        
        # Visualize first few point clouds
        if visualize and idx < 3:
            vis_file = os.path.join(vis_dir, f"pointcloud_{idx:03d}.png")
            visualize_pointcloud(points, vis_file)
        
        # Randomly sample points if max_points is specified
        if max_points is not None and len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            labels = labels[indices]  # Update labels too
            
            # Update statistics after sampling
            frame_points = len(points)
            
        # Print frame statistics
        print(f"Frame {os.path.basename(image_file)}: {frame_points} points (all labeled as tables)")
        
        all_points.append(points)
        all_labels.append(labels)
    
    # Concatenate all points and labels
    all_points = np.vstack(all_points)
    all_labels = np.concatenate(all_labels)
    
    # Save to HDF5
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('data', data=all_points)
        f.create_dataset('label', data=all_labels)
    
    # Print summary statistics
    print(f"\n=== UCL DATASET SUMMARY ===")
    print(f"Total frames processed: {total_frames}")
    print(f"Total points: {len(all_points)}")
    print(f"Table points: {np.sum(all_labels)}/{len(all_labels)} ({np.sum(all_labels)/len(all_labels)*100:.2f}%)")
    print(f"All points are labeled as tables (1), since UCL data contains tables")
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare UCL data for table segmentation")
    parser.add_argument("--data_dir", type=str, default='dataset/ucl_data', help="Path to UCL data directory")
    parser.add_argument("--output_file", type=str, default='CW2-Dataset/ucl_data_fixed.h5', help="Path to output H5 file")
    parser.add_argument("--max_points", type=int, default=1024, help="Maximum number of points per frame")
    parser.add_argument("--no_vis", action="store_true", help="Disable visualization")
    
    args = parser.parse_args()
    
    # Process the UCL data
    prepare_ucl_data(args.data_dir, args.output_file, args.max_points, not args.no_vis) 