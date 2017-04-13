import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
from feature_extraction_utils import *
import hog_subsample
import config

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap < threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
    
if __name__ == "__main__":
    # Read in the saved StandardScalar and trained classifier
    # and feature detection settings
    # These were saved by running "search_and_classify.py"
    config.settings = pickle.load( open("svc_trained.p", "rb" ) )
    
    # Plot the result
    fig, axes = plt.subplots(6, 3, squeeze=False, figsize=(8.5, 9.5))
    fig.tight_layout()
    plt.suptitle( "Heat map threshold".format() )

    for i in range(1,7):
        print("\nprocessing image " + str(i))
        
        img_path = 'test_images/test{0}.jpg'.format(str(i))
        img = mpimg.imread(img_path)
        
        ##### 1. detect cars in image at different scales #####
        scales = config.scales

        # Modify x/y start stop according to scale, cars appear smaller near horizon
        box_list = []
        for scale_item in scales:
            detects_image, boxes = hog_subsample.find_cars(img, 
                scale_item["y_start_stop"][0], scale_item["y_start_stop"][1], 
                scale_item["scale"], 
                config.settings["svc"], 
                config.settings["scaler"], 
                config.settings["orient"], 
                config.settings["pix_per_cell"], config.settings["cell_per_block"], 
                config.settings["spatial_size"], config.settings["hist_bins"],
                scale_item["x_start_stop"][0], scale_item["x_start_stop"][1])
            box_list.extend(boxes)
    
        print ("boxes count={}, boxes={}".format(len(box_list), box_list ))
        boxes_img = draw_boxes(img, box_list, color=(0, 0, 255), thick=3)  
        
        # plot classifier detection boxes
        row = (i-1)
        col = 0
        
        axes[row][col].imshow(boxes_img)
        axes[row][col].set_title('Image {0} boxes={1}'.format(i, len(box_list)), fontsize=10)
        
        ##### 2. heat map and threshold #####
        
        # Make zeros shaped like image
        heat = np.zeros_like(img[:,:,0]).astype(np.float)

        # Add heat for each box in box list history
        heat = add_heat(heat, box_list)

        # Apply threshold to help remove false positives
        heat_threshold = config.heatmap_threshold
        heat = apply_threshold(heat, heat_threshold)

        # Find final boxes from heatmap using label function
        heatmap = np.clip(heat, 0, 255) # only need to clip if there is more than 255 boxes around a point?
        
        # plot heatmap values
        col = 1
        axes[row][col].imshow(heatmap, cmap='hot')
        axes[row][col].set_title('Heatmap {0}'.format(i), fontsize=10)
        
        # Find and label heatmap values
        labels = label(heatmap)
        heatmap_img = draw_labeled_bboxes(np.copy(img), labels)

        # plot heatmap labeled boxes
        col = 2

        axes[row][col].imshow(heatmap_img)
        axes[row][col].set_title('Heatmap boxes {0}'.format(i), fontsize=10)
        
        
        
    save_plot_path = "output_images/heatmap_threshold.png".format()
    plt.savefig(save_plot_path)
    plt.show()
