# Image
shape_img = (1024, 1024)
shape_patch = (256, 256)
# shape_patch = (128, 128)

# Threshold
threshold_depth = 0.1 # Valid distance
threshold_diff = 0.01 # is close to GT depth
threshold_shade = 0 # Valid RGB
threshold_valid = 0.1 # Rate of valida pix

# Directory
dir_root = '../../'
dir_root_data = dir_root + 'Data/'
dir_root_model = dir_root + 'Models/'
dir_root_output = dir_root + 'Outputs/'

# Data
dir_synthetic = 'synthetic/'
dir_real = 'real/'
dir_gt = 'gt/' # GT depth
dir_low = 'low/' # Low-res depth
dir_shade = 'shade/' # Shading image
dir_proj = 'proj/' # Pattern projected image
dir_mask = 'mask/' # Valid mask

# Synthetic data info
synthetic_num = 600
synthetic_train = (0, 500)
synthetic_test = (500, 600)

# Real data info


# File
file_log = 'training.csv'
file_model_keras_final = 'model_keras_final.hdf5'
file_model_keras_best = 'model_keras_best.hdf5'
file_model_torch_final = 'model_torch_final.pt'
file_model_torch_best = 'model_keras_torch.pt'

# Inputs
is_input_low = True # Low-res depth
is_input_proj = True # Pattern projected image

# Normalization
is_norm_shade = True # Shading Normalization
is_norm_diff = True # Difference Normalization

# Evaluation
is_patch_norm = False
size_norm_patch = 24
rate_valid_norm_patch = 50

# Plot
depth_range = 0.04
err_range = 0.002

# Save
is_save_ply_gt = True
is_save_ply_low = False

# Camera Parameter
cam_params = {
    'focal_length': 0.037009,
    'pix_x': 1.25e-05,
    'pix_y': 1.2381443057539635e-05,
    'center_x': 702.902,
    'center_y': 512.635
}