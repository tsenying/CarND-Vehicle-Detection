import config
from car import Car
import hog_subsample

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
        
    def process_image( self, img ):
        """Find cars in image
        
        Args:
            img (numpy.ndarray): Source image. Color channels in RGB order.

        Returns:
            (numpy.ndarray): image decorated with bounding boxes around cars
        """
        
        # 1. detect cars in image at different scales
        scale = 1.5
        out_img, boxes = hog_subsample.find_cars(img, 
            config.settings["y_start_stop"][0], config.settings["y_start_stop"][1], 
            scale, 
            config.settings["svc"], 
            config.settings["scaler"], 
            config.settings["orient"], 
            config.settings["pix_per_cell"], config.settings["cell_per_block"], 
            config.settings["spatial_size"], config.settings["hist_bins"])
        
        # 2. heat map and threshold
        
        return out_img
        
        