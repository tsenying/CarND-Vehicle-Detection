# frame count
count = 0

# number of frames to keep in history for smoothing
nframes = 5

image_shape = { 'width': 1280, 'height': 720 }

settings = {    
    "svc": None, 
    "scaler": None,
    "color_space": 'YCrCb', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb #( HSV, HLS)
    "orient": 12,  # HOG orientations
    "pix_per_cell": 8, # HOG pixels per cell
    "cell_per_block": 2, # HOG cells per block
    "hog_channel": 'ALL', # Can be 0, 1, 2, or "ALL"
    "spatial_size": (32, 32), # Spatial binning dimensions
    "hist_bins": 32,    # Number of histogram bins
    "spatial_feat": True, # Spatial features on or off
    "hist_feat": True, # Histogram features on or off
    "hog_feat": True, # HOG features on or off
    "y_start_stop": [400, 656] # Min and max in y to search in slide_window()
}

scales = [
    { "scale": 1.0, "x_start_stop": [0,1280], "y_start_stop":[400,528]},
    { "scale": 1.5, "x_start_stop": [0,1280], "y_start_stop":[400,592]},
    { "scale": 2.0, "x_start_stop": [0,1280], "y_start_stop":[400,656]},
    { "scale": 3.0, "x_start_stop": [0,1280], "y_start_stop":[400,656]}
]

heatmap_threshold = 3

# debug log file
debug_log = None
debug = False