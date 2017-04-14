import numpy as np
import cv2
import config
from car import Car
import hog_subsample
import heatmap_threshold_detection
from scipy.ndimage.measurements import label
from collections import deque

class VehicleDetection():
    """ Vehicle detection state
    """
    def __init__(self, nframes):
        # number of frames in history
        self.nframes = nframes
        
        # bounding boxes for positive detections for each frame in history
        self.bbox_list_history = deque([], self.nframes)
        
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
        
        # Modify x/y start stop according to scale, cars appear smaller near horizon
        scales = config.scales
        
        box_list = []
        for scale_item in scales:
            scale = scale_item["scale"]
            detects_image, boxes = hog_subsample.find_cars(image, 
                scale_item["y_start_stop"][0], scale_item["y_start_stop"][1], 
                scale, 
                config.settings["svc"], 
                config.settings["scaler"], 
                config.settings["orient"], 
                config.settings["pix_per_cell"], config.settings["cell_per_block"], 
                config.settings["spatial_size"], config.settings["hist_bins"],
                scale_item["x_start_stop"][0], scale_item["x_start_stop"][1])
            box_list.extend(boxes)
            
        # Update history
        self.bbox_list_history.append( box_list )
        bbox_list_history_list = sum(self.bbox_list_history.copy(), []) # single list of bbox lists in history
        
        # 2. heat map and threshold
        
        # Make zeros shaped like image
        heat = np.zeros_like(image[:,:,0]).astype(np.float)

        # Add heat for each box in box list history
        heat = heatmap_threshold_detection.add_heat(heat, bbox_list_history_list)

        # Apply threshold to help remove false positives
        heat_threshold = config.heatmap_threshold
        heat = heatmap_threshold_detection.apply_threshold(heat, heat_threshold)

        # Find final boxes from heatmap using label function
        heatmap = np.clip(heat, 0, 255) # only need to clip if there is more than 255 boxes around a point?
        labels = label(heatmap)
        boxed_image = heatmap_threshold_detection.draw_labeled_bboxes(np.copy(image), labels)
        
        # frame image annotation
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(boxed_image,"Frame:{}".format(config.count), (10,100), font, 1, (255,255,255), 2 ,cv2.LINE_AA )
        
        return boxed_image
        
        