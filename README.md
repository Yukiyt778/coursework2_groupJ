# Table Classification, Segmentation and Point Cloud Processing - Coursework 2 (Group J)

This project focuses on processing both UCL and SUN3D datasets, utilizing both RGB and depth images for point cloud conversion, depth maps estimation, and table classification and segmentation.

## Data Preprocessing

The project uses the SUN3D dataset, processed into HDF5 (.h5) files. The dataset is prepared using scripts in the `data_utils/` directory:
Hint: When pasting CW2-Dataset to test our code, please rename the subfolder to `CW2-Dataset/sun3d_data`.
```
CW2-Dataset/sun3d_data
  ├── sun3d_train_fixed.h5  # Training dataset (MIT scenes)
  ├── sun3d_test_fixed.h5   # Testing dataset (Harvard scenes)
  └── ucl_data_fixed.h5     # Additional UCL dataset
```

The dataset is prepared from the SUN3D dataset using the following process:

1. Process depth maps from SUN3D using DepthTSDF format
2. Generate point clouds from the depth data
3. Combine point clouds and labels into H5 files

If raw depth maps need to be converted to DepthTSDF format:

```bash
# Convert depth to TSDF format for specific building
python ./coursework_groupJ/data_utils/depth_to_tsdf.py --building harvard_tea_2

# Generate the dataset
python ./coursework_groupJ/data_utils/prepare_sun3d_dataset.py --prefix mit --output CW2-Dataset/sun3d_train.h5
python ./coursework_groupJ/data_utils/prepare_sun3d_dataset.py --prefix harvard --output CW2-Dataset/sun3d_test.h5
```

## Models

- **Pipeline A**: Uses PointNet++ SSG (Single Scale Grouping) for classification (`pointnet2_cls_ssg.py`)
- **Pipeline B**: Uses Depth Anything V2 for depth maps estimation(`estimate_depth.py`), uses Ensemble Classification(Random Forest and Support Vector Machine) for classification(`classification.py`)
- **Pipeline C**: Uses PointNet++ for semantic segmentation (`pointnet2_sem_seg.py`)

## Pipeline A: Table Classification

Pipeline A focuses on classifying entire point clouds as either containing a table (1) or not (0).

### Training
For training using K-fold cross-validation:
```bash
python ./coursework_groupJ/src/pipelineA/train.py
```

### Testing
```bash
python ./coursework_groupJ/src/pipelineA/test.py
```

### Output
- Pipeline A uses pre-trained weights in `weights/pipelineA/check_points/best_model.pth`, the trained best model weights are stored in `weights/best_model.pth`
- Pipeline A training curves and metrics can be found in `results/plots/pipelineA`

## Pipeline B: Monocular Depth Estimation and Classification
Depth Anything V2 is state-of-the-art model which has robust performance on a large range of images in various conditions. It takes RGB images as input for monocular depth estimation. The classification model combines Random Forest classifier and Support Vector Machine classifier.

### Depth Maps Estimation
```bash
python ./coursework_groupJ/src/pipelineB/estimate_depth.py
```

### Binary Classification 
```bash
python ./coursework_groupJ/src/pipelineB/classification.py
```
### Label Visualization
```bash
python ./coursework_groupJ/data/read_labels.py
```
### Output
- The predicted depth maps can be found in `results/predictions`
- The predicted labels for UCL data are also stored in `./coursework2_groupJ/data/RealSense/ucl_data/labels/tabletop_labels.dat`

## Pipeline C: Semantic Segmentation
Pipeline C performs point-level semantic segmentation to identify which points in a point cloud belong to tables.

### Training
```bash
python ./coursework_groupJ/src/pipelineC/train.py
```

### Testing
```bash
python ./coursework_groupJ/src/pipelineC/test_sun3d.py
```

### Visualization
```bash
python ./coursework_groupJ/src/pipelineC/visualize_results.py
```

### Output
All the results for pipeline C are in `results/pipelineC`

## Visualization

The repository includes tools for visualizing point clouds, ground truth, and predictions:

1. Run testing with visualization enabled:
```bash
python ./coursework_groupJ/src/pipelineC/test_sun3d.py --log_dir semantic_segmentation_TIMESTAMP --h5_file CW2-Dataset/sun3d_test_fixed.h5 --visual
```

2. Convert to interactive HTML visualizations:
```bash
python ./coursework_groupJ/models/pointnet2/visualizer/convert_txt_to_html.py --log_dir semantic_segmentation_TIMESTAMP --output_dir point_cloud_visualizations
```

3. Package visualizations for download:
```bash
python ./coursework_groupJ/models/pointnet2/visualizer/package_visualizations.py --log_dir semantic_segmentation_TIMESTAMP
```

## Implementation details

Both pipeline A and C support data augmentation during training:
- Random scaling
- Random translation
- Random point dropout
- Random jittering

For pipeline B, 2:1 weighting domain adaptation(training data : UCL data = 2:1) is used to prevent the model from overfitting to potentially noisy UCL characteristics.

