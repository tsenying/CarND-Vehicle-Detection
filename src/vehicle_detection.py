import numpy as np
import config
from car import Car
import hog_subsample
import heatmap_threshold_detection
from scipy.ndimage.measurements import label

class VehicleDetection():
    """ Vehicle detection state
    """
    def __init__(self, nframes):
        # number of frames in history
        self.nframes = nframes
        
        # bounding boxes for positive detections for each frame in history
        self.bbox_list_history = []
        
        # cars in each frame in history
        self.cars_history = []
        
    def process_image( self, image ):
        """Find cars in image
        
        Args:
            image (numpy.ndarray): Source image. Color channels in RGB order.

        Returns:
            (numpy.ndarray): image decorated with bounding boxes around cars
        """
        
        # 1. detect cars in image at different scales
        scale = 1.5
        detects_image, box_list = hog_subsample.find_cars(image, 
            config.settings["y_start_stop"][0], config.settings["y_start_stop"][1], 
            scale, 
            config.settings["svc"], 
            config.settings["scaler"], 
            config.settings["orient"], 
            config.settings["pix_per_cell"], config.settings["cell_per_block"], 
            config.settings["spatial_size"], config.settings["hist_bins"])
        
        # 2. heat map and threshold
        
        # Make zeros shaped like image
        heat = np.zeros_like(image[:,:,0]).astype(np.float)

        # Add heat for each box in box list
        heat = heatmap_threshold_detection.add_heat(heat, box_list)

        # Apply threshold to help remove false positives
        heat_threshold = 1
        heat = heatmap_threshold_detection.apply_threshold(heat, heat_threshold)

        # Find final boxes from heatmap using label function
        heatmap = np.clip(heat, 0, 255) # only need to clip if there is more than 255 boxes around a point?
        labels = label(heatmap)
        draw_image = heatmap_threshold_detection.draw_labeled_bboxes(np.copy(image), labels)
        
        return draw_image
        
        